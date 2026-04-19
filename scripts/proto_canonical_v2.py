#!/usr/bin/env python3
# DEPRECATED: kept for regression reference only.
# Production code lives at scripts/canonical/ and is tested in tests/canonical/.
"""Prototype v2: all 4 fixes.

Fix 1: full monotonic Needleman-Wunsch DP (replaces greedy).
Fix 2: retrieval widens to ±2 window if margin low, Tier-2 global lookup otherwise.
Fix 3: consecutive-phrase dedup before retrieval (not before alignment).
Fix 4: min-consonant-length rule on fix ops (both sides ≥ 2 cons, |Δlen| ≤ 1).
"""
from __future__ import annotations

import math
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gurmukhi_converter import ascii_to_unicode, strip_vishraams  # noqa: E402

DB_PATH = Path(__file__).parent.parent / "database.sqlite"

SKELETON_STRIP = re.compile(
    r"[\u0A3C\u0A3E-\u0A4D\u0A51\u0A70\u0A71\u0A75"
    r"\u0A66-\u0A6F"
    r"0-9\s\u200C\u200D।॥॰.,;:!?'\"()\[\]<>]+"
)
_NUKTA_PAIRS = [("ਸ਼", "ਸ"), ("ਖ਼", "ਖ"), ("ਗ਼", "ਗ"), ("ਜ਼", "ਜ"), ("ਫ਼", "ਫ"), ("ਲ਼", "ਲ")]


def _denukta(t: str) -> str:
    for a, b in _NUKTA_PAIRS:
        t = t.replace(a, b)
    return t


def skel(text: str) -> str:
    if not text:
        return ""
    return SKELETON_STRIP.sub("", _denukta(text))


def tokenize(text: str) -> list[str]:
    return [tok for tok in text.split() if tok and tok != ">>"]


def ngrams(s: str, n: int = 4) -> list[str]:
    return [s[i:i + n] for i in range(len(s) - n + 1)] if len(s) >= n else ([s] if s else [])


def lev(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[len(b)]


# ---- Fix 3: consecutive-phrase dedup ---- #

def collapse_consecutive_repeats(tokens: list[str],
                                 min_phrase: int = 3, max_phrase: int = 8,
                                 min_repeats: int = 3) -> list[str]:
    """Collapse N+ consecutive repeats of a phrase to a single copy.

    Only triggers when a phrase of ≥ min_phrase tokens repeats ≥ min_repeats
    times back-to-back. 2-time repetitions (very common in SGGS and kirtan)
    are LEFT INTACT — they provide retrieval signal, not noise.
    """
    out: list[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        collapsed = False
        for L in range(min(max_phrase, (n - i) // min_repeats), min_phrase - 1, -1):
            # check if tokens[i:i+L] repeats min_repeats times
            repeats = 1
            while i + L * (repeats + 1) <= n and tokens[i + L * repeats:i + L * (repeats + 1)] == tokens[i:i + L]:
                repeats += 1
            if repeats >= min_repeats:
                out.extend(tokens[i:i + L])
                i += L * repeats
                collapsed = True
                break
        if not collapsed:
            out.append(tokens[i])
            i += 1
    return out


# ---- Index build ---- #

@dataclass
class SggsLine:
    line_id: str
    shabad_id: str
    ang: int
    unicode: str
    skel: str
    tokens: list[str]
    tok_skels: list[str]


_VISHRAAM_TOKEN_RE = re.compile(r"^[॥।੦-੯0-9.,;:!?'\"()\[\]]+$")


def _clean_token(t: str) -> str:
    return re.sub(r"[॥।੦-੯0-9.]+$", "", t).strip()


def load_sggs(db_path: Path) -> tuple[list[SggsLine], dict[str, list]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT l.id AS line_id, l.shabad_id, l.source_page AS ang, l.gurmukhi, l.order_id
        FROM lines l
        JOIN shabads s ON l.shabad_id = s.id
        WHERE s.source_id = 1 AND l.type_id IN (1, 3, 4)
        ORDER BY l.order_id
    """).fetchall()
    out = []
    # Fix 2b: global token → shabad mapping for Tier-2 fallback
    global_tok_idx: dict[str, list] = defaultdict(list)
    for r in rows:
        uni = ascii_to_unicode(strip_vishraams(r["gurmukhi"])).strip()
        if not uni or len(uni) < 3:
            continue
        raw_tokens = uni.split()
        tokens = [_clean_token(t) for t in raw_tokens if not _VISHRAAM_TOKEN_RE.match(t)]
        tokens = [t for t in tokens if t]
        if not tokens:
            continue
        ln = SggsLine(
            line_id=r["line_id"], shabad_id=r["shabad_id"], ang=r["ang"],
            unicode=" ".join(tokens), skel=skel(" ".join(tokens)),
            tokens=tokens, tok_skels=[skel(t) for t in tokens],
        )
        out.append(ln)
        for tok, ts in zip(tokens, ln.tok_skels):
            global_tok_idx[ts].append((tok, r["line_id"], r["shabad_id"]))
    return out, global_tok_idx


def build_shabad_idx(lines: list[SggsLine]):
    shabad_ngrams: dict[str, Counter[str]] = defaultdict(Counter)
    shabad_lines: dict[str, list[SggsLine]] = defaultdict(list)
    for ln in lines:
        shabad_lines[ln.shabad_id].append(ln)
        for g in ngrams(ln.skel, 4):
            shabad_ngrams[ln.shabad_id][g] += 1
    df: Counter[str] = Counter()
    for sid, cnt in shabad_ngrams.items():
        for g in cnt:
            df[g] += 1
    return shabad_ngrams, shabad_lines, df


def retrieve_shabad(caption_skel: str, shabad_ngrams, df, n_shabads) -> tuple[str, float, float]:
    q_grams = Counter(ngrams(caption_skel, 4))
    scores: Counter[str] = Counter()
    for g, q_tf in q_grams.items():
        if g not in df:
            continue
        idf = math.log(1 + n_shabads / df[g])
        for sid, cnt in shabad_ngrams.items():
            if g in cnt:
                scores[sid] += min(q_tf, cnt[g]) * idf
    top = scores.most_common(2)
    if not top:
        return ("", 0.0, 0.0)
    top_sid, top_score = top[0]
    second_score = top[1][1] if len(top) > 1 else 0.0
    margin = (top_score - second_score) / top_score if top_score > 0 else 0.0
    return (top_sid, top_score, margin)


# ---- Fix 1: Full NW DP ---- #
# ---- Fix 4: min-consonant-length gate ---- #

MAX_OP_EDIT = 1         # primary pass — conservative
MAX_OP_EDIT_RELAX = 2   # Fix 2: secondary pass on orphan runs — marked review
MIN_CONS_LEN = 2
MAX_LEN_DELTA = 1
MAX_SPAN = 3
MIN_ORPHAN_RUN = 3      # Fix 3: realign delete runs ≥ this length
VIDEO_PRIOR_WEIGHT = 0.8  # Fix 1: how much to bias retrieval toward seen shabads

SCORE_MATCH = 10
SCORE_FIX_MATRA = 7    # skel equal, unicode differs
SCORE_FIX_1CONS = 4    # skel differs by 1 consonant
SCORE_MERGE_SPLIT = 4
GAP_CAP = -1           # cost of deleting a caption token (kept)
GAP_SGS = -2           # cost of skipping an sggs token (insert)
FLOOR = -10_000_000


def _fix_eligible(cap_skel: str, sgs_skel: str, max_edit: int = MAX_OP_EDIT) -> bool:
    if len(cap_skel) < MIN_CONS_LEN or len(sgs_skel) < MIN_CONS_LEN:
        return False
    if abs(len(cap_skel) - len(sgs_skel)) > max_edit:
        return False
    return lev(cap_skel, sgs_skel) <= max_edit


def _score_11(cap: str, cs: str, sgs: str, ss: str, max_edit: int = MAX_OP_EDIT):
    if cap == sgs and cs == ss:
        return SCORE_MATCH, "match"
    if cs == ss and cs and len(cs) >= MIN_CONS_LEN:
        return SCORE_FIX_MATRA, "fix"
    if _fix_eligible(cs, ss, max_edit):
        return SCORE_FIX_1CONS, "fix"
    return FLOOR, None


def align_nw(cap_tokens: list[str], shabad_lines: list[SggsLine],
             max_edit: int = MAX_OP_EDIT) -> list[dict]:
    # Flatten shabad to tokens
    sggs_stream_tok: list[str] = []
    sggs_stream_skel: list[str] = []
    sggs_stream_lid: list[str] = []
    for ln in shabad_lines:
        for t, ts in zip(ln.tokens, ln.tok_skels):
            sggs_stream_tok.append(t)
            sggs_stream_skel.append(ts)
            sggs_stream_lid.append(ln.line_id)
    m = len(cap_tokens)
    n = len(sggs_stream_tok)
    if m == 0 or n == 0:
        return [{"op": "delete", "cap": [t], "sggs": [t], "line_ids": []} for t in cap_tokens]
    cap_skels = [skel(t) for t in cap_tokens]

    # DP: dp[i][j] = best score aligning cap[:i] to sgs[:j]
    # bt[i][j] = (op, back_i, back_j, cap_span, sgs_span, line_ids)
    # SEMI-GLOBAL: caption must align fully, but SGGS prefix/suffix are free.
    # => dp[0][j] = 0 for all j (free SGGS prefix skip)
    # => traceback starts at max(dp[m][j]) (free SGGS suffix skip)
    dp = [[FLOOR] * (n + 1) for _ in range(m + 1)]
    bt: list[list] = [[None] * (n + 1) for _ in range(m + 1)]
    for j in range(n + 1):
        dp[0][j] = 0
        bt[0][j] = ("prefix_skip", 0, 0, [], [], []) if j > 0 else None
    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] + GAP_CAP
        bt[i][0] = ("delete", i - 1, 0, [cap_tokens[i - 1]], [cap_tokens[i - 1]], [])

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            best_score = FLOOR
            best_bt = None
            # 1:1
            s11, op11 = _score_11(cap_tokens[i - 1], cap_skels[i - 1],
                                  sggs_stream_tok[j - 1], sggs_stream_skel[j - 1], max_edit)
            if op11 is not None:
                cand = dp[i - 1][j - 1] + s11
                if cand > best_score:
                    best_score = cand
                    sgs_tok_for_op = sggs_stream_tok[j - 1] if op11 == "fix" else cap_tokens[i - 1]
                    best_bt = (op11, i - 1, j - 1,
                               [cap_tokens[i - 1]], [sgs_tok_for_op],
                               [sggs_stream_lid[j - 1]])
            # 1:N SPLIT
            for span in range(2, MAX_SPAN + 1):
                if j - span < 0:
                    break
                sgs_joined = "".join(sggs_stream_skel[j - span:j])
                if _fix_eligible(cap_skels[i - 1], sgs_joined, max_edit):
                    cand = dp[i - 1][j - span] + SCORE_MERGE_SPLIT
                    if cand > best_score:
                        best_score = cand
                        best_bt = ("split", i - 1, j - span,
                                   [cap_tokens[i - 1]],
                                   list(sggs_stream_tok[j - span:j]),
                                   list(sggs_stream_lid[j - span:j]))
            # M:1 MERGE
            for span in range(2, MAX_SPAN + 1):
                if i - span < 0:
                    break
                cap_joined = "".join(cap_skels[i - span:i])
                if _fix_eligible(cap_joined, sggs_stream_skel[j - 1], max_edit):
                    cand = dp[i - span][j - 1] + SCORE_MERGE_SPLIT
                    if cand > best_score:
                        best_score = cand
                        best_bt = ("merge", i - span, j - 1,
                                   list(cap_tokens[i - span:i]),
                                   [sggs_stream_tok[j - 1]],
                                   [sggs_stream_lid[j - 1]])
            # INSERT (skip sggs j)
            cand = dp[i][j - 1] + GAP_SGS
            if cand > best_score:
                best_score = cand
                best_bt = ("insert", i, j - 1, [], [sggs_stream_tok[j - 1]], [sggs_stream_lid[j - 1]])
            # DELETE (skip cap i)
            cand = dp[i - 1][j] + GAP_CAP
            if cand > best_score:
                best_score = cand
                best_bt = ("delete", i - 1, j,
                           [cap_tokens[i - 1]], [cap_tokens[i - 1]], [])

            dp[i][j] = best_score
            bt[i][j] = best_bt

    # Traceback — find best endpoint (any j for i=m, allows trailing sggs unused)
    best_j = max(range(n + 1), key=lambda j: dp[m][j])
    i, j = m, best_j
    ops: list[dict] = []
    while i > 0:
        if bt[i][j] is None:
            break
        op, bi, bj, cap_span, sgs_span, lids = bt[i][j]
        if op not in ("insert", "prefix_skip"):
            ops.append({"op": op, "cap": cap_span, "sggs": sgs_span, "line_ids": lids})
        i, j = bi, bj
    ops.reverse()
    return ops


# ---- Fix 2: retrieval with widen + Tier-2 ---- #

MARGIN_LOW = 0.05


def _score_all_shabads(caption_skel: str, shabad_ngrams, df, n_shabads) -> Counter:
    q_grams = Counter(ngrams(caption_skel, 4))
    scores: Counter[str] = Counter()
    for g, q_tf in q_grams.items():
        if g not in df:
            continue
        idf = math.log(1 + n_shabads / df[g])
        for sid, cnt in shabad_ngrams.items():
            if g in cnt:
                scores[sid] += min(q_tf, cnt[g]) * idf
    return scores


def retrieve_with_fallback(cur_tokens, prev_tokens, next_tokens,
                           prev2_tokens, next2_tokens,
                           idx, video_hits: Counter) -> tuple[str, float, float, int]:
    """Returns (shabad_id, score, margin, window_used). Fix 1: video-level prior."""
    shabad_ngrams, shabad_lines, df = idx
    n_sh = len(shabad_ngrams)

    def _apply_video_prior(scores: Counter) -> Counter:
        if not video_hits:
            return scores
        total = sum(video_hits.values())
        boosted = Counter()
        for sid, sc in scores.items():
            prior = video_hits.get(sid, 0) / total if total else 0
            boosted[sid] = sc * (1 + VIDEO_PRIOR_WEIGHT * prior)
        return boosted

    def _top2(scores: Counter) -> tuple[str, float, float]:
        top = scores.most_common(2)
        if not top:
            return ("", 0.0, 0.0)
        top_sid, top_score = top[0]
        second = top[1][1] if len(top) > 1 else 0.0
        margin = (top_score - second) / top_score if top_score > 0 else 0.0
        return (top_sid, top_score, margin)

    # ±1
    ctx = prev_tokens + cur_tokens + next_tokens
    ctx = collapse_consecutive_repeats(ctx)
    scores1 = _score_all_shabads("".join(skel(t) for t in ctx), shabad_ngrams, df, n_sh)
    sid, score, margin = _top2(_apply_video_prior(scores1))
    if margin >= MARGIN_LOW:
        return sid, score, margin, 1

    # widen to ±2
    ctx2 = prev2_tokens + prev_tokens + cur_tokens + next_tokens + next2_tokens
    ctx2 = collapse_consecutive_repeats(ctx2)
    scores2 = _score_all_shabads("".join(skel(t) for t in ctx2), shabad_ngrams, df, n_sh)
    sid2, score2, margin2 = _top2(_apply_video_prior(scores2))
    return sid2, score2, margin2, 2


def tier2_rescue(cap_tokens, current_shabad, ops, global_tok_idx) -> str | None:
    """If ≥ 3 orphan caption tokens agree on a different shabad, return that shabad_id."""
    orphan_shabads: Counter[str] = Counter()
    orphan_count = 0
    cap_index = 0
    for op in ops:
        if op["op"] == "delete":
            # orphan caption token
            ts = skel(op["cap"][0])
            if ts and len(ts) >= MIN_CONS_LEN:
                candidates = global_tok_idx.get(ts, [])
                for _, _, sid in candidates:
                    if sid != current_shabad:
                        orphan_shabads[sid] += 1
                orphan_count += 1
    if orphan_count >= 3 and orphan_shabads:
        top, n = orphan_shabads.most_common(1)[0]
        if n >= 2:
            return top
    return None


# ---- Driver ---- #

def _realign_orphan_runs(ops: list[dict], shabad_lines: list[SggsLine],
                         max_edit: int = MAX_OP_EDIT) -> tuple[list[dict], bool]:
    """Fix 3: for contiguous delete runs ≥ MIN_ORPHAN_RUN, run a fresh NW on that
    span alone and splice in if it reduces deletes. Fix 2: optionally expand edit.
    Returns (new_ops, used_relaxed_edit).
    """
    used_relaxed = False
    changed = True
    while changed:
        changed = False
        # find first long delete run
        i = 0
        while i < len(ops):
            if ops[i]["op"] == "delete":
                start = i
                while i < len(ops) and ops[i]["op"] == "delete":
                    i += 1
                if i - start >= MIN_ORPHAN_RUN:
                    orphan_cap = [ops[k]["cap"][0] for k in range(start, i)]
                    # Try fresh NW restricted-edit first
                    alt = align_nw(orphan_cap, shabad_lines, max_edit=max_edit)
                    alt_d = sum(1 for op in alt if op["op"] == "delete")
                    if alt_d < (i - start):
                        ops = ops[:start] + alt + ops[i:]
                        changed = True
                        break
                    # Fix 2: escalate to relaxed edit; caller marks as review
                    if max_edit < MAX_OP_EDIT_RELAX:
                        alt2 = align_nw(orphan_cap, shabad_lines, max_edit=MAX_OP_EDIT_RELAX)
                        alt2_d = sum(1 for op in alt2 if op["op"] == "delete")
                        if alt2_d < (i - start):
                            ops = ops[:start] + alt2 + ops[i:]
                            used_relaxed = True
                            changed = True
                            break
            else:
                i += 1
    return ops, used_relaxed


def process_row(caption: str, idx, neighbors, global_tok_idx,
                video_hits: Counter | None = None) -> dict:
    shabad_ngrams, shabad_lines, df = idx
    cap_tokens = tokenize(caption)
    if not cap_tokens:
        return {"sggs_line": None, "final_text": "", "decision": "skip_marker",
                "_debug": {"shabad_id": "", "window": 0, "ops": {}}}

    prev_t = tokenize(neighbors[0])
    next_t = tokenize(neighbors[1])
    prev2_t = tokenize(neighbors[2])
    next2_t = tokenize(neighbors[3])
    if video_hits is None:
        video_hits = Counter()
    sid, score, margin, window_used = retrieve_with_fallback(
        cap_tokens, prev_t, next_t, prev2_t, next2_t, idx, video_hits)

    if not sid or score < 2.0:
        return {"sggs_line": None,
                "final_text": " ".join(cap_tokens),
                "decision": "unchanged",
                "_debug": {"shabad_id": sid, "window": window_used, "ops": {}, "margin": margin}}

    ops = align_nw(cap_tokens, shabad_lines[sid])
    # Fix 3: realign orphan runs (secondary NW). Fix 2: escalate to edit=2 if needed.
    ops, used_relaxed = _realign_orphan_runs(ops, shabad_lines[sid])
    # Tier-2 rescue: if many orphans converge on a different shabad, retry
    rescue = tier2_rescue(cap_tokens, sid, ops, global_tok_idx)
    if rescue:
        ops2 = align_nw(cap_tokens, shabad_lines[rescue])
        # accept rescue if it reduces deletes
        d1 = sum(1 for op in ops if op["op"] == "delete")
        d2 = sum(1 for op in ops2 if op["op"] == "delete")
        if d2 < d1:
            sid = rescue
            ops = ops2

    # Build outputs
    final_parts, sggs_parts, line_ids = [], [], []
    n_match = n_fix = n_merge = n_split = n_del = 0
    for op in ops:
        if op["op"] == "match":
            final_parts.extend(op["cap"])
            sggs_parts.extend(op["sggs"])
            n_match += 1
        elif op["op"] == "fix":
            final_parts.extend(op["sggs"])
            sggs_parts.extend(op["sggs"])
            n_fix += 1
        elif op["op"] == "split":
            final_parts.extend(op["sggs"])
            sggs_parts.extend(op["sggs"])
            n_split += 1
        elif op["op"] == "merge":
            final_parts.extend(op["sggs"])
            sggs_parts.extend(op["sggs"])
            n_merge += 1
        else:
            final_parts.extend(op["cap"])
            n_del += 1
        line_ids.extend(op.get("line_ids", []))

    total_gur = n_match + n_fix + n_merge + n_split + n_del
    score_pct = (n_match + n_fix + n_merge + n_split) / total_gur if total_gur else 0.0
    if score_pct >= 0.92 and n_fix + n_merge + n_split == 0:
        decision = "matched"
    elif score_pct >= 0.92 and not used_relaxed:
        decision = "replaced"
    elif score_pct >= 0.75 or used_relaxed:
        decision = "review"
    else:
        decision = "unchanged"

    # Fix 1: record shabad hit for future rows in this video
    if decision in ("matched", "replaced", "review"):
        video_hits[sid] += 1

    # SAFETY: on unchanged, never apply partial corrections — return caption pass-through
    if decision == "unchanged":
        return {
            "sggs_line": None,
            "final_text": " ".join(cap_tokens),
            "decision": "unchanged",
            "_debug": {"shabad_id": sid, "window": window_used, "margin": round(margin, 3),
                       "ops": {"match": n_match, "fix": n_fix, "merge": n_merge,
                               "split": n_split, "delete": n_del}, "suppressed": True},
        }

    return {
        "sggs_line": " ".join(dict.fromkeys(sggs_parts)) or None,
        "final_text": " ".join(final_parts),
        "decision": decision,
        "_debug": {"shabad_id": sid, "window": window_used, "margin": round(margin, 3),
                   "ops": {"match": n_match, "fix": n_fix, "merge": n_merge,
                           "split": n_split, "delete": n_del},
                   "line_ids": list(dict.fromkeys(line_ids))},
    }


SAMPLE = """\
ਤੇਰੀ ਸਰਣ ਮੇਰੇ ਦੀਨ ਦਇਆਲਾ
ਸੁਖ ਸਾਗਰ ਮੇਰੇ ਗੁਰ ਗੋਪਾਲਾ
ਸੁਖ ਸਾਗਰ ਮੇਰੇ ਗੁਰ ਗੋਪਾਲਾ
ਤੇਰੀ ਸਰਣ ਮੇਰੇ ਦੀਨ ਦਇਆਲਾ
ਸੁਖ ਸਾਗਰ ਮੇਰੇ ਗੁਰ ਗੋਪਾਲਾ
ਸੁਖ ਸਾਗਰ ਮੇਰੇ ਗੁਰ ਗੋਪਾਲਾ
ਕਰਿ ਕਿਰਪਾ ਨਾਨਕ ਗੁਣ ਗਾਵੈ
ਕਰਿ ਕਿਰਪਾ ਨਾਨਕ ਗੁਣ ਗਾਵੈ
ਰਾਖਹੁ ਸਰਮ ਅਸਾੜੀ ਜੀਆ
ਰਾਖਹੁ ਸਰਮ ਅਸਾੜੀ ਜੀਆ
ਤੇਰੀ ਸਰਣ ਮੇਰੇ ਦੀਨ ਦਇਆਲਾ
ਸੁਖ ਸਾਗਰ ਮੇਰੇ ਗੁਣ ਗੋਪਾਲਾ
ਸੁਖ ਸਾਗਰ ਮੇਰੇ ਗੁਨ ਗੋਪਾਲਾ
>>
>> ਜਿਥੈ ਨਾਮੁ ਜਪੀਐ ਪ੍ਰਭ ਪਿਆਰੇ
ਸੇ ਅਸਥਲ ਸੋਇਨ ਚਉਾਰੇ
>>
>> ਜਿਥੈ ਨਾਮੁ ਜਪੀਐ ਪ੍ਰਭ ਪਿਆਰੇ
ਸਿਆਸਤ ਸੋਇਨ ਚਵਾਰੇ
ਜਿਥੈ ਨਾਮੁ ਨ ਜਪੀਐ ਮੇਰੇ ਗੋਇੰਦਾ
>> >> ਜਿਥੈ ਨਾਮ
ਨ ਜਪੀਐ ਮੇਰੇ ਗੋਇੰਦਾ ਸੇਈ ਨਦਰ
ਉਜਾੜੀ ਜੀਉ
ਸੇ ਨਗਰ >>
>> ਉਦਾੜੀ ਦੀਆ ਸੇਈ ਨਗਰ
ਉਦਾੜੀ ਦੀਆ ਸੇਈ ਨਗਰ
ਉਦਾੜੀ ਦੇਆ ਤੇਰੀ ਸਰਣ ਮੇਰੇ
ਦੀਨ ਦਇਆਲਾ ਸੁਖ ਸਾਗਰ
ਮੇਰੇ ਗੁਰ ਗੋਪਾਲਾ ਸੁਖ ਸਾਗਰ
ਮੇਰੇ ਗੁਣ ਗੋਪਾਲਾ ਤੇਰੀ ਸਰਨ ਮੇਰੇ
ਦੀਨ ਦਇਆਲਾ ਸੁਖ ਸਾਗਰ
ਮੇਰੇ ਗੁਰ ਗੋਪਾਲਾ ਸੁਖ ਸਾਗਰ
ਮੇਰੇ ਗੁਰ ਗੋਪਾਲਾ ਸਰਿ
ਰੁਖੀ ਰੋਤੀ ਖਾਇ ਸਮਾਲੇ
ਹਰਿ ਅੰਤਰ ਬਾਹਰ ਨਦਰਿ ਨਿਹਾਲੇ
ਹਰਿ ਅੰਤਰ ਬਾਹਰ ਨਦਰਿ ਨਿਹਾਲੇ
ਖਾਇ ਖਾਇ ਕਰੇ ਮਦ ਫੈਲੀ
>>
>> ਖਾਇ ਖਾਇ ਕਰੇ ਮਦ ਫੈਲੀ
ਜਾਣ ਵਿਸੋ ਕੀ ਵਾੜੀ ਜੀਆ
ਜਾਣ ਵਿਸ ਸੋ ਕੀ ਵਾੜੀ ਜੀਆ
ਜਾਣ ਵਿਸੋ ਕੀ ਵਾੜੀ ਜੀਆ
ਜਾਣ ਵਿਸੋ ਕੀ ਵਾੜੀ ਜੀਆ
ਤੇਰੀ ਸਰਨ ਮੇਰੇ ਦੀਨ ਦਇਆਲਾ
ਸੁਖ ਸਾਗਰ ਮੇਰੇ ਗੁਰ ਗੋਪਾਲ
ਸੁਖ ਸਾਗਰ ਮੇਰੇ ਗੁਰ ਗੋਪਾਲਾ
>>
>> ਸੰਤਾ ਸੇਤੀ ਰੰਗੁ ਨ ਲਾਇ
ਸਾਕਤਸੰਗਿ ਵਿਕਰਮ ਕਮਾਇ
>>
>> ਸੰਤਾ ਸੇਤੀ
ਰੰਗੁ ਨ ਲਾਇ ਸਾਕਤਸੰਗਿ
ਬਿਕਰਮ ਕਮਾਇ ਦੁਲਭਦੇ
ਹੋਈ ਅਗਿਆਨੀ
ਦੁਲਭ ਦੇ
ਕੋਈ ਅਗਿਆਨੀ ਜਨ ਅਪਣੀ ਆਪ
ਉਪਾੜੀ ਜੀਆ
ਲੜਿ ਅਪਨੀ ਆਪ ਉਪਾੜੀ ਦੀਆ
ਚੜਿ ਅਪਨੀ ਆਪ ਉਪਾੜੀ ਦੇਆ
ਚੜਿ ਅਪਨੀ ਆਪ ਉਪਾੜੀ ਦੇਆ
ਤੇਰੀ ਸਰਣ ਮੇਰੇ ਦੀਨ ਦਇਆਲਾ
ਸੁਖ ਸਾਗਰ ਮੇਰੇ ਗੁਰ ਗੋਪਾਲ
>> >> ਸੁਖ ਸਾਗਰ
ਮੇਰੇ ਗੁਨ ਗੋਪਾਲਾ ਕਰਿ ਕਿਰਪਾ
ਨਾਨਕ ਗੁਣ ਗਾਵੈ
ਕਰਿ ਕਿਰਪਾ ਨਾਨਕ ਗੁਣ ਗਾਵੈ
ਰਾਖ ਰਾਖਹੁ ਸਰਮ ਅਸਾੜੀ ਦੇਆ
ਰਾਖਹੁ ਸਰਮ ਅਸਾੜੀ ਜੀਆ
ਤੇਰੀ ਸਰਨ ਮੇਰੇ ਦੀਨ ਦਇਆਲਾ
ਸੁਖ ਸਾਗਰ ਮੇਰੇ ਗੁਰ ਗੋਪਾਲਾ
ਸੁਖ ਸਾਗਰ ਮੇਰੇ ਗੁਰ ਗੋਪਾਲਾ
ਤੇਰੀ ਸਰਣ ਮੇਰੇ ਦੀਨ ਦਇਆਲਾ
ਸੁਖ ਸਾਗਰ ਮੇਰੇ ਗੁਣ ਗੋਪਾਲਾ
ਸੁਖ ਸਾਗਰ ਮੇਰੇ ਗੁਣ ਗੋਪਾਲ ਸੁਖ ਸਾਗਰ
ਮੇਰੇ ਗੁਨ ਗੋਪਾਲਾ
ਪ੍ਰਭ ਜੀਉ ਪ੍ਰਭ ਜੀਉ
ਤੂ ਮੇਰੋ ਸਾਹਿਬ ਦਾਤਾ
ਪ੍ਰਭ ਜੀਉ ਪ੍ਰਭ ਜੀਉ
ਤੂ ਮੇਰੋ ਸਾਹਿਬ ਦਾਤਾ
ਕਰਿ ਕਿਰਪਾ ਪ੍ਰਭ ਦੀਨ ਦਇਆਲਾ
ਕਰਿ ਕਿਰਪਾ ਪ੍ਰਭ ਦੀਨ ਦਇਆਲਾ
ਗੁਣ ਗਾਵਹੁ ਰੰਗਿ ਰਾਤਾ ਪ੍ਰਭ ਜੀਉ
ਪ੍ਰਭ ਜੀਉ ਤੂ ਮੇਰੋ
ਸਾਹਿਬ ਦਾਤਾ ਪ੍ਰਭ ਜੀਉ
ਪ੍ਰਭ ਜੀਉ ਤੂ ਮੇਰੋ
ਸਾਹਿਬ ਦਾਤਾ >>
>> ਪ੍ਰਭ ਕੀ ਸਰਣ ਸਗਲ ਪੈਲਾਥੇ
ਦੁਖ ਬਿਨਸੇ ਸੁਖੁ ਪਾਇਆ
>> ਦੁਖ ਬਿਨਸੇ ਸੁਖੁ ਪਾਇਆ
>> ਦਇਆਲੁ ਹੋਆ ਪਾਰਬ੍ਰਹਮੁ ਸੁਆਮੀ
ਪੂਰਾ ਸਤਿਗੁਰੁ ਧਿਆਇਆ ਪੂਰਾ ਸਤਿਗੁਰੁ ਧਿਆਇਆ
ਪੂਰਾ ਸਤਿਗੁਰੁ ਧਿਆਇਆ
ਪੂਰਾ ਸਤਿਗੁਰੁ ਧਿਆਇਆ ਪ੍ਰਭ ਜੀਉ
ਪ੍ਰਭ ਜੀਉ ਤੂ ਮੇਰੋ
ਸਾਹਿਬ ਦਾਤਾ ਪ੍ਰਭ ਜੀਉ
ਪ੍ਰਭ ਜੀਉ ਤੂ ਮੇਰੋ
ਸਾਹਿਬ ਦਾਤਾ
ਸਤਿਗੁਰ ਨਾਮ ਨਿਧਾਨ ਦ੍ਰਿੜਾਇਆ
ਚਿੰਤਾ ਸਗਲ ਬਿਨਾਸੀ
>> ਚਿੰਤਾ ਸਗਲ ਬਿਨਾਸੀ ਸਤਿਗੁਰ ਨਾਮ
"""


def main():
    print("[load] SGGS...", file=sys.stderr)
    lines, global_tok_idx = load_sggs(DB_PATH)
    print(f"[load] {len(lines)} lines, {len(global_tok_idx)} skeletons", file=sys.stderr)
    idx = build_shabad_idx(lines)
    print(f"[load] {len(idx[1])} shabads", file=sys.stderr)

    rows_raw = [r.strip() for r in SAMPLE.strip().splitlines()]
    processable = [(i, r) for i, r in enumerate(rows_raw) if tokenize(r)]

    print(f"{'#':>3}  {'decision':10} {'sh':4} {'w':1} {'ops':<24}  caption  →  final_text  |  sggs_line")
    print("-" * 170)
    counts = Counter()
    video_hits: Counter[str] = Counter()  # Fix 1: simulate one video's shabad memory
    for k, (orig_i, row) in enumerate(processable):
        prev = processable[k - 1][1] if k > 0 else ""
        nxt = processable[k + 1][1] if k + 1 < len(processable) else ""
        prev2 = processable[k - 2][1] if k > 1 else ""
        nxt2 = processable[k + 2][1] if k + 2 < len(processable) else ""
        r = process_row(row, idx, (prev, nxt, prev2, nxt2), global_tok_idx, video_hits)
        dbg = r.pop("_debug", {})
        ops = dbg.get("ops", {})
        ops_str = " ".join(f"{k[0]}{v}" for k, v in ops.items() if v)
        sid = dbg.get("shabad_id", "")
        w = dbg.get("window", 0)
        counts[r["decision"]] += 1
        print(f"{orig_i+1:>3}  {r['decision']:10} {sid:>4} {w:>1} {ops_str:<24}  {row}  →  {r['final_text']}  |  {r['sggs_line']}")
    print()
    print(f"[summary] {dict(counts)}", file=sys.stderr)


if __name__ == "__main__":
    main()
