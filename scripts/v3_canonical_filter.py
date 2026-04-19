#!/usr/bin/env python3
"""Enrich per-clip transcripts with canonical-corpus match columns.

Matches each Gemini transcription against a unified canonical corpus covering
SGGS + Dasam Granth + Bhai Gurdas Vaaran + Kabit Savaiye Bhai Gurdas. The
Gemini text is preserved as the PRIMARY column; canonical columns are
metadata for inspection and QA.

Input:  data/transcripts/<item_id>.jsonl
Canon:  canonical.json (from scripts/v3_build_canonical.py)
Output: data/transcripts_enriched/<item_id>.jsonl with extra columns:
    gemini_text            — original Gemini transcription (primary)
    canonical_text         — best-match canonical line ("" if no match)
    canonical_source       — "sggs" | "dasam" | "bhai_gurdas_vaaran"
                             | "bhai_gurdas_kabit" | ""
    canonical_line_id      — line_id from DB
    canonical_shabad_id    — shabad_id from DB
    canonical_raag         — raag name (SGGS only)
    canonical_match_score  — 0.0 … 1.0 (rapidfuzz token_set_ratio / 100)
    canonical_flag         — "accept" | "review" | "simran" | "none"

Thresholds:
    score >= 0.80  →  accept
    0.50 <= score < 0.80  →  review
    score < 0.50  AND  passes simran allowlist  →  simran
    score < 0.50  otherwise  →  none  (kept in dataset; flag is informational)

Note: "none" replaces the previous "drop" label — we do NOT drop these from
the training set. They are still valid Gemini transcriptions of kirtan audio,
just not from these four canonical sources (e.g. shaheedi poetry by Bhai Vir
Singh, modern compositions, non-listed writers).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from rapidfuzz import fuzz, process

DANDA_RE = re.compile(r"॥[੦-੯]+॥|॥|।|[\u200c\u200d]")
WS_RE = re.compile(r"\s+")

SIMRAN_PHRASES = {
    "ਵਾਹਿਗੁਰੂ", "ਵਾਹਗੁਰੂ",
    "ਸਤਿਨਾਮੁ ਵਾਹਿਗੁਰੂ", "ਸਤਿਨਾਮੁ", "ਸਤਿ ਨਾਮੁ",
    "ਰਾਮ", "ਹਰਿ ਹਰਿ", "ਗੁਰੂ ਗੁਰੂ", "ਧਨੁ ਧਨੁ",
}
_SIMRAN_NORMED = {
    WS_RE.sub(" ", DANDA_RE.sub(" ", p)).strip() for p in SIMRAN_PHRASES
}


def normalize(text: str) -> str:
    t = DANDA_RE.sub(" ", text)
    t = WS_RE.sub(" ", t).strip()
    return t


def is_simran(text: str) -> bool:
    norm = normalize(text)
    if not norm:
        return False
    # exact-phrase match
    if norm in _SIMRAN_NORMED:
        return True
    # dominant-token heuristic
    tokens = norm.split()
    if not tokens:
        return False
    top_tok, top_count = Counter(tokens).most_common(1)[0]
    if top_count / len(tokens) >= 0.6 and any(
        top_tok in p or p in top_tok for p in _SIMRAN_NORMED
    ):
        return True
    return False


def load_canonical(path: Path) -> tuple[list[dict], list[str]]:
    data = json.loads(path.read_text())
    rows = []
    choices = []
    for r in data:
        t = normalize(r.get("text", ""))
        if len(t) < 4:
            continue
        rows.append(r)
        choices.append(t)
    by_src = Counter(r["source"] for r in rows)
    print(f"[canon] {len(rows)} lines loaded: {dict(by_src)}", file=sys.stderr)
    return rows, choices


def classify(score: float, text: str) -> str:
    if score >= 0.80:
        return "accept"
    if score >= 0.50:
        return "review"
    if is_simran(text):
        return "simran"
    return "none"


def enrich_file(src: Path, dst: Path, canon_rows: list[dict],
                choices: list[str]) -> dict:
    lines = list(src.open())
    stats = Counter()

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w") as f:
        for line in lines:
            r = json.loads(line)
            gemini_text = r.get("text", "")
            r["gemini_text"] = gemini_text

            norm = normalize(gemini_text)
            blank_canon = {
                "canonical_text": "", "canonical_source": "",
                "canonical_line_id": "", "canonical_shabad_id": "",
                "canonical_raag": "",
            }

            if not norm:
                r.update(blank_canon)
                r["canonical_match_score"] = 0.0
                r["canonical_flag"] = "none"
                stats["empty"] += 1
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                continue

            best = process.extractOne(
                norm, choices,
                scorer=fuzz.token_set_ratio,
                score_cutoff=0,
            )
            if best is None:
                r.update(blank_canon)
                score = 0.0
            else:
                _match_text, raw_score, idx = best
                score = raw_score / 100.0
                c = canon_rows[idx]
                r["canonical_text"] = c.get("text", "")
                r["canonical_source"] = c.get("source", "")
                r["canonical_line_id"] = c.get("line_id", "")
                r["canonical_shabad_id"] = c.get("shabad_id", "")
                r["canonical_raag"] = c.get("raag", "")

            r["canonical_match_score"] = round(score, 3)
            flag = classify(score, gemini_text)
            r["canonical_flag"] = flag
            stats[flag] += 1

            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return dict(stats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/root/v3_data")
    ap.add_argument("--canonical", required=True,
                    help="Path to canonical.json (from v3_build_canonical.py)")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    src_dir = data_root / "transcripts"
    dst_dir = data_root / "transcripts_enriched"
    canon_rows, choices = load_canonical(Path(args.canonical))

    totals = Counter()
    files = sorted(src_dir.glob("*.jsonl"))
    for i, src in enumerate(files, 1):
        dst = dst_dir / src.name
        s = enrich_file(src, dst, canon_rows, choices)
        totals.update(s)
        if i % 10 == 0 or i == len(files):
            print(f"  [{i}/{len(files)}] {src.name}: {s}")

    total = sum(totals.values())
    print("\n[summary]")
    print(f"  total segments: {total}")
    for k in ("accept", "review", "simran", "none", "empty"):
        pct = 100 * totals.get(k, 0) / total if total else 0
        print(f"  {k:>8s}: {totals.get(k, 0):>5d}  ({pct:5.1f}%)")

    # Per-canonical-source breakdown for accept + review
    print("\n[canonical source breakdown] (accept + review only)")
    # re-scan output dir for source labels
    src_counter = Counter()
    for jsonl in dst_dir.glob("*.jsonl"):
        for line in jsonl.open():
            r = json.loads(line)
            if r.get("canonical_flag") in ("accept", "review"):
                src_counter[r.get("canonical_source", "")] += 1
    for k, v in src_counter.most_common():
        print(f"  {k or '(none)':>24s}: {v}")


if __name__ == "__main__":
    main()
