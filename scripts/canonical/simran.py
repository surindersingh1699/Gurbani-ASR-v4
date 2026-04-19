"""Simran detection + quota downsampling.

Kirtan videos (especially AKJ) include long ਵਾਹਿਗੁਰੂ simran sequences. Without
a quota, simran could be 17–25% of the final dataset and bias the ASR decoder
toward always outputting ਵਾਹਿਗੁਰੂ. This module detects simran rows and applies
a two-stage quota: adaptive per-video cap + round-robin global trim to an
absolute target (default 750).
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass

from .gurmukhi_skeleton import skel
from .waheguru import WAHEGURU_SKEL


@dataclass
class SimranConfig:
    min_reps: int = 5
    ratio_threshold: float = 0.70
    target_count: int = 750
    per_video_min: int = 1
    per_video_max: int = 10


def is_simran(tokens: list[str], cfg: SimranConfig) -> bool:
    if not tokens:
        return False
    waheguru_like = [i for i, t in enumerate(tokens) if skel(t) == WAHEGURU_SKEL]
    if len(waheguru_like) < cfg.min_reps:
        return False
    best_run, cur_run = 1, 1
    for a, b in zip(waheguru_like, waheguru_like[1:]):
        if b - a == 1:
            cur_run += 1
            best_run = max(best_run, cur_run)
        else:
            cur_run = 1
    if best_run < cfg.min_reps:
        return False
    gur_total = sum(
        1 for t in tokens if any("\u0A00" <= ch <= "\u0A7F" for ch in t)
    )
    if gur_total == 0:
        return False
    return len(waheguru_like) / gur_total >= cfg.ratio_threshold


def apply_simran_quota(
    rows: list[dict], cfg: SimranConfig, seed: int = 42
) -> list[dict]:
    """Downsample simran rows to ~cfg.target_count with adaptive per-video cap
    and round-robin stratified sampling across videos."""
    rng = random.Random(seed)
    non_simran = [r for r in rows if not r.get("is_simran")]
    simran = [r for r in rows if r.get("is_simran")]

    by_video: dict[str, list[dict]] = defaultdict(list)
    for r in simran:
        by_video[r["video_id"]].append(r)

    n_videos = len(by_video)
    if n_videos == 0:
        return rows

    cap = max(
        cfg.per_video_min,
        min(cfg.per_video_max, math.ceil(cfg.target_count / n_videos)),
    )

    capped: dict[str, list[dict]] = {}
    for vid, group in by_video.items():
        capped[vid] = list(group) if len(group) <= cap else rng.sample(group, cap)

    video_ids = sorted(capped.keys())
    for vid in video_ids:
        rng.shuffle(capped[vid])

    survivors: list[dict] = []
    idx = 0
    while len(survivors) < cfg.target_count and any(capped.values()):
        vid = video_ids[idx % len(video_ids)]
        if capped[vid]:
            survivors.append(capped[vid].pop(0))
        idx += 1
        if idx > cfg.target_count * 2 and not any(capped.values()):
            break

    return non_simran + survivors
