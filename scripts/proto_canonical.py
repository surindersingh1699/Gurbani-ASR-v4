#!/usr/bin/env python3
# DEPRECATED: kept for regression reference only.
# Production code lives at scripts/canonical/ and is tested in tests/canonical/.
"""Prototype: test canonical-text algorithm on a caption sample.

Not the final shipping pipeline — a minimal impl of the design in
docs/superpowers/specs/2026-04-18-kirtan-canonical-text-column-design.md
for validating alignment behavior on real data.

Pipeline per caption row:
  1. Strip '>>', tokenize, skeletonize
  2. Retrieve top SGGS shabad via 4-gram TF-IDF
  3. Within that shabad, greedy phrase align (match / fix / merge / split)
  4. Emit (sggs_line, final_text, decision)
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

# Gurmukhi matra / bindi / tippi / addak / halant / nukta
SKELETON_STRIP = re.compile(
    r"[\u0A3C\u0A3E-\u0A4D\u0A51\u0A70\u0A71\u0A75"  # matras, bindi, tippi, addak, halant, nukta, udaat
    r"\u0A66-\u0A6F"  # gurmukhi digits
    r"0-9\s\u200C\u200D।॥॰.,;:!?'\"()\[\]<>]+"
)
_NUKTA_PAIRS = [("ਸ਼", "ਸ"), ("ਖ਼", "ਖ"), ("ਗ਼", "ਗ"), ("ਜ਼", "ਜ"), ("ਫ਼", "ਫ"), ("ਲ਼", "ਲ")]


def _denukta(t: str) -> str:
    for a, b in _NUKTA_PAIRS:
        t = t.replace(a, b)
    return t


def skel(text: str) -> str:
    """Consonant-only skeleton of a Gurmukhi string."""
    if not text:
        return ""
    t = _denukta(text)
    return SKELETON_STRIP.sub("", t)


def tokenize(text: str) -> list[str]:
    """Split on whitespace, drop '>>' markers."""
    return [tok for tok in text.split() if tok and tok != ">>"]


def ngrams(s: str, n: int = 4) -> list[str]:
    return [s[i:i + n] for i in range(len(s) - n + 1)] if len(s) >= n else [s]


def lev(a: str, b: str) -> int:
    """Levenshtein distance."""
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


# ---------- index build ---------- #

@dataclass
class SggsLine:
    line_id: str
    shabad_id: str
    ang: int
    unicode: str          # vishraam-stripped Unicode
    skel: str             # consonant skeleton of whole line
    tokens: list[str]     # unicode tokens (whitespace split)
    tok_skels: list[str]  # skeleton per token


_VISHRAAM_TOKEN_RE = re.compile(r"^[॥।੦-੯0-9.,;:!?'\"()\[\]]+$")


def _clean_token(t: str) -> str:
    """Strip trailing vishraams/digits from an individual token."""
    return re.sub(r"[॥।੦-੯0-9.]+$", "", t).strip()


def load_sggs(db_path: Path) -> tuple[list[SggsLine], set[str]]:
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
    global_token_set: set[str] = set()
    for r in rows:
        uni = ascii_to_unicode(strip_vishraams(r["gurmukhi"])).strip()
        if not uni or len(uni) < 3:
            continue
        raw_tokens = uni.split()
        tokens = [_clean_token(t) for t in raw_tokens if not _VISHRAAM_TOKEN_RE.match(t)]
        tokens = [t for t in tokens if t]
        if not tokens:
            continue
        out.append(SggsLine(
            line_id=r["line_id"],
            shabad_id=r["shabad_id"],
            ang=r["ang"],
            unicode=" ".join(tokens),
            skel=skel(" ".join(tokens)),
            tokens=tokens,
            tok_skels=[skel(t) for t in tokens],
        ))
        global_token_set.update(tokens)
    return out, global_token_set


def build_shabad_idx(lines: list[SggsLine]) -> tuple[dict, dict, dict]:
    """Returns (shabad_ngrams, shabad_lines, doc_freq)."""
    shabad_ngrams: dict[str, Counter[str]] = defaultdict(Counter)
    shabad_lines: dict[str, list[SggsLine]] = defaultdict(list)
    for ln in lines:
        shabad_lines[ln.shabad_id].append(ln)
        for g in ngrams(ln.skel, 4):
            shabad_ngrams[ln.shabad_id][g] += 1
    # doc freq: how many shabads each 4-gram appears in
    df: Counter[str] = Counter()
    for sid, cnt in shabad_ngrams.items():
        for g in cnt:
            df[g] += 1
    return shabad_ngrams, shabad_lines, df


def retrieve_shabad(caption_skel: str, shabad_ngrams, df, n_shabads) -> tuple[str, float, float]:
    """TF-IDF 4-gram overlap. Returns (top_shabad_id, top_score, margin)."""
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


# ---------- phrase alignment ---------- #

MAX_SPAN = 3           # max caption tokens on either side of merge/split
MAX_OP_EDIT = 1        # max skeleton Levenshtein per op


def align(cap_tokens: list[str], shabad_lines: list[SggsLine],
          valid_caption_tokens: set[str] | None = None) -> list[dict]:
    """Greedy phrase alignment of caption tokens against shabad's lines in sequence.

    Returns list of ops: {op, cap_span: [toks], sggs_span: [toks], line_id}.
    Each op consumes a contiguous span of caption tokens.
    """
    # Flatten shabad to (token, tok_skel, line_id) triples
    sggs_stream = []
    for ln in shabad_lines:
        for tok, ts in zip(ln.tokens, ln.tok_skels):
            sggs_stream.append((tok, ts, ln.line_id))
    n_sggs = len(sggs_stream)

    cap_skels = [skel(t) for t in cap_tokens]
    n_cap = len(cap_tokens)
    ops: list[dict] = []
    valid_caption_tokens = valid_caption_tokens or set()

    WINDOW = 3  # tighter window — prefer forward advance
    last_pos = -WINDOW

    def pos_cost(j: int) -> int:
        # 0 cost inside window; small cost for backward jumps; large cost for far-forward
        if j < last_pos - WINDOW:
            return 2 + (last_pos - WINDOW - j)
        if j > last_pos + WINDOW:
            return max(0, j - last_pos - WINDOW)
        return 0

    i = 0
    while i < n_cap:
        best = None
        # 1:1 MATCH / FIX
        for j, (stok, sskel, lid) in enumerate(sggs_stream):
            d_skel = lev(cap_skels[i], sskel)
            if d_skel > MAX_OP_EDIT:
                continue
            d_char = lev(cap_tokens[i], stok)
            if cap_tokens[i] == stok:
                op_type = "match"
            elif d_char <= 3:
                op_type = "fix"
            else:
                continue
            # Tie-break: lower skel dist > match over fix > closer to last_pos > forward advance
            cand = (d_skel, 0 if op_type == "match" else 1, pos_cost(j), -j if j > last_pos else j,
                    op_type, [cap_tokens[i]], [stok], [lid], 1, 1, j)
            if best is None or cand < best:
                best = cand
        # 1:N SPLIT
        for j in range(n_sggs):
            for span in range(2, MAX_SPAN + 1):
                if j + span > n_sggs:
                    break
                s_join = "".join(sggs_stream[j + k][1] for k in range(span))
                d_skel = lev(cap_skels[i], s_join)
                if d_skel > MAX_OP_EDIT:
                    continue
                s_tokens = [sggs_stream[j + k][0] for k in range(span)]
                s_line_ids = [sggs_stream[j + k][2] for k in range(span)]
                cand = (d_skel, 2, pos_cost(j) + 1, -j if j > last_pos else j,
                        "split", [cap_tokens[i]], s_tokens, s_line_ids, 1, span, j)
                if best is None or cand < best:
                    best = cand
        # M:1 MERGE
        for m in range(2, MAX_SPAN + 1):
            if i + m > n_cap:
                break
            c_join = "".join(cap_skels[i + k] for k in range(m))
            for j, (stok, sskel, lid) in enumerate(sggs_stream):
                d_skel = lev(c_join, sskel)
                if d_skel > MAX_OP_EDIT:
                    continue
                cand = (d_skel, 2, pos_cost(j) + 1, -j if j > last_pos else j,
                        "merge", cap_tokens[i:i + m], [stok], [lid], m, 1, j)
                if best is None or cand < best:
                    best = cand

        if best is None:
            ops.append({"op": "delete", "cap": [cap_tokens[i]], "sggs": [cap_tokens[i]], "line_ids": []})
            i += 1
        else:
            # cand tuple: (d_skel, op_priority, pos_cost, tiebreak, op_type, cap_span, sggs_span, lids, m, span, j)
            op_type = best[4]
            cap_span = best[5]
            sggs_span = best[6]
            lids = best[7]
            m = best[8]
            span = best[9]
            j_pos = best[10]
            ops.append({"op": op_type, "cap": cap_span, "sggs": sggs_span, "line_ids": lids})
            last_pos = j_pos + span
            i += m
    return ops


# ---------- driver ---------- #

def process_row(caption: str, lines: list[SggsLine], idx,
                neighbors: tuple[str, str] = ("", "")) -> dict:
    shabad_ngrams, shabad_lines, df, global_tokens = idx
    cap_tokens = tokenize(caption)
    if not cap_tokens:
        return {"sggs_line": None, "final_text": "", "decision": "skip_marker"}
    # Sliding-window retrieval: use prev+cur+next tokens for shabad identification
    prev_tokens = tokenize(neighbors[0])
    next_tokens = tokenize(neighbors[1])
    retrieval_tokens = prev_tokens + cap_tokens + next_tokens
    retrieval_skel = "".join(skel(t) for t in retrieval_tokens)
    sid, score, margin = retrieve_shabad(retrieval_skel, shabad_ngrams, df, len(shabad_ngrams))
    if not sid or score < 2.0:
        return {
            "sggs_line": None,
            "final_text": " ".join(cap_tokens),
            "decision": "unchanged",
        }
    # Align ONLY the current row's tokens against the retrieved shabad
    ops = align(cap_tokens, shabad_lines[sid], valid_caption_tokens=global_tokens)
    # Build final_text + sggs_line
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
        else:  # delete
            final_parts.extend(op["cap"])
            n_del += 1
        line_ids.extend(op.get("line_ids", []))

    total_gur = n_match + n_fix + n_merge + n_split + n_del
    score_pct = (n_match + n_fix + n_merge + n_split) / total_gur if total_gur else 0.0
    if score_pct >= 0.92 and n_fix + n_merge + n_split == 0:
        decision = "matched"
    elif score_pct >= 0.92:
        decision = "replaced"
    elif score_pct >= 0.75:
        decision = "review"
    else:
        decision = "unchanged"

    return {
        "sggs_line": " ".join(dict.fromkeys(sggs_parts)) or None,  # dedupe while keeping order
        "final_text": " ".join(final_parts),
        "decision": decision,
        "_debug": {"shabad_id": sid, "margin": round(margin, 3),
                   "ops": {"match": n_match, "fix": n_fix, "merge": n_merge,
                           "split": n_split, "delete": n_del},
                   "line_ids": list(dict.fromkeys(line_ids))},
    }


SAMPLE = """\
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
    lines, global_tokens = load_sggs(DB_PATH)
    print(f"[load] {len(lines)} SGGS lines, {len(global_tokens)} unique tokens", file=sys.stderr)
    shabad_ngrams, shabad_lines, df = build_shabad_idx(lines)
    idx = (shabad_ngrams, shabad_lines, df, global_tokens)
    print(f"[load] {len(shabad_lines)} shabads", file=sys.stderr)

    rows_raw = [r.strip() for r in SAMPLE.strip().splitlines()]
    # Drop >>-only rows from processing (mark as skip_marker, don't match)
    processable = [(i, r) for i, r in enumerate(rows_raw) if tokenize(r)]

    print(f"{'#':>3}  {'decision':10}  {'shabad':4}  {'ops':<28}  caption  →  final_text  |  sggs_line")
    print("-" * 160)
    for idx_in_processable, (orig_i, row) in enumerate(processable):
        prev = processable[idx_in_processable - 1][1] if idx_in_processable > 0 else ""
        nxt = processable[idx_in_processable + 1][1] if idx_in_processable + 1 < len(processable) else ""
        r = process_row(row, lines, idx, neighbors=(prev, nxt))
        dbg = r.pop("_debug", {})
        ops = dbg.get("ops", {})
        ops_str = " ".join(f"{k[0]}{v}" for k, v in ops.items() if v)
        sid = dbg.get("shabad_id", "")
        print(f"{orig_i+1:>3}  {r['decision']:10}  {sid:>4}  {ops_str:<28}  {row}  →  {r['final_text']}  |  {r['sggs_line']}")


if __name__ == "__main__":
    main()
