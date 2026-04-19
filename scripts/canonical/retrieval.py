"""Shabad retrieval: 4-gram TF-IDF over ±N-token skeleton window + video-memory
prior + optional sequential prior (for sehaj path's linear reading).
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from .gurmukhi_skeleton import skel


@dataclass
class RetrievalConfig:
    ngram_n: int = 4
    margin_low: float = 0.05
    video_prior_weight: float = 0.8
    sequential_current_boost: float = 0.0
    sequential_next_boost: float = 0.0


def _ngrams(s: str, n: int) -> list[str]:
    if len(s) >= n:
        return [s[i : i + n] for i in range(len(s) - n + 1)]
    return [s] if s else []


def score_all_shabads(
    caption_skel: str, shabad_ngrams: dict, df: Counter, cfg: RetrievalConfig
) -> Counter:
    q_grams = Counter(_ngrams(caption_skel, cfg.ngram_n))
    n_shabads = len(shabad_ngrams)
    scores: Counter = Counter()
    for g, q_tf in q_grams.items():
        if g not in df:
            continue
        idf = math.log(1 + n_shabads / df[g])
        for sid, cnt in shabad_ngrams.items():
            if g in cnt:
                scores[sid] += min(q_tf, cnt[g]) * idf
    return scores


def _apply_priors(
    scores: Counter,
    video_hits: Counter,
    nxt_shabad: dict | None,
    cfg: RetrievalConfig,
) -> Counter:
    out: Counter = Counter()
    total_hits = sum(video_hits.values())
    prev_sid = video_hits.most_common(1)[0][0] if video_hits else None
    for sid, sc in scores.items():
        mul = 1.0
        if total_hits > 0:
            mul += cfg.video_prior_weight * (video_hits.get(sid, 0) / total_hits)
        if cfg.sequential_current_boost and prev_sid and sid == prev_sid:
            mul += cfg.sequential_current_boost
        if (
            cfg.sequential_next_boost
            and nxt_shabad
            and prev_sid
            and sid == nxt_shabad.get(prev_sid)
        ):
            mul += cfg.sequential_next_boost
        out[sid] = sc * mul
    return out


def _top2(scores: Counter) -> tuple[str, float, float]:
    top = scores.most_common(2)
    if not top:
        return ("", 0.0, 0.0)
    top_sid, top_score = top[0]
    second = top[1][1] if len(top) > 1 else 0.0
    margin = (top_score - second) / top_score if top_score > 0 else 0.0
    return (top_sid, top_score, margin)


def retrieve_shabad(
    cur_tokens: list[str],
    prev_next_tokens: tuple[list[str], list[str]],
    prev2_next2_tokens: tuple[list[str], list[str]],
    shabad_ngrams: dict,
    df: Counter,
    video_hits: Counter,
    nxt_shabad: dict | None,
    cfg: RetrievalConfig,
) -> tuple[str, float, float, int]:
    """Retrieve best shabad for cur_tokens using a ±1 context window first,
    widening to ±2 if margin is low. Returns (shabad_id, score, margin, window)."""
    prev_t, next_t = prev_next_tokens
    prev2_t, next2_t = prev2_next2_tokens

    ctx1 = prev_t + cur_tokens + next_t
    skel1 = "".join(skel(t) for t in ctx1)
    scores1 = _apply_priors(
        score_all_shabads(skel1, shabad_ngrams, df, cfg),
        video_hits,
        nxt_shabad,
        cfg,
    )
    sid, score, margin = _top2(scores1)
    if margin >= cfg.margin_low:
        return sid, score, margin, 1

    ctx2 = prev2_t + prev_t + cur_tokens + next_t + next2_t
    skel2 = "".join(skel(t) for t in ctx2)
    scores2 = _apply_priors(
        score_all_shabads(skel2, shabad_ngrams, df, cfg),
        video_hits,
        nxt_shabad,
        cfg,
    )
    sid2, score2, margin2 = _top2(scores2)
    return sid2, score2, margin2, 2
