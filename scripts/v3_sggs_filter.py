#!/usr/bin/env python3
"""Enrich per-clip transcripts with SGGS canonical-match columns.

Reads data/transcripts/<item_id>.jsonl (Gemini output) and writes
data/transcripts_enriched/<item_id>.jsonl with added columns:
    gemini_text          — original Gemini transcription (same as old `text`)
    sggs_match_text      — best-matching canonical pangati text ("" if no match)
    sggs_shabad_id       — shabad_id from tuks.json
    sggs_tuk_id          — tuk_id from tuks.json
    sggs_raag            — raag name from tuks.json
    sggs_match_score     — 0.0 … 1.0 (rapidfuzz token_set_ratio/100)
    sggs_flag            — "accept" | "review" | "drop" | "simran"

Thresholds:
    score >= 0.80  →  accept
    0.50 <= score < 0.80  →  review
    score < 0.50   →  drop (unless matches a simran allowlist → "simran")

Simran allowlist: short repeated chants like "ਵਾਹਿਗੁਰੂ" / "ਸਤਿਨਾਮੁ" that are
legitimate kirtan content but not SGGS pangatis.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from rapidfuzz import fuzz, process

# Characters to strip before matching (dandas, digit-in-danda, zero-width)
DANDA_RE = re.compile(r"॥[੦-੯]+॥|॥|।|[\u200c\u200d]")
WS_RE = re.compile(r"\s+")

SIMRAN_PHRASES = {
    "ਵਾਹਿਗੁਰੂ",
    "ਵਾਹਗੁਰੂ",
    "ਸਤਿਨਾਮੁ ਵਾਹਿਗੁਰੂ",
    "ਸਤਿਨਾਮੁ",
    "ਰਾਮ",
    "ਹਰਿ ਹਰਿ",
    "ਗੁਰੂ ਗੁਰੂ",
    "ਧਨੁ ਧਨੁ",
}


def normalize(text: str) -> str:
    t = DANDA_RE.sub(" ", text)
    t = WS_RE.sub(" ", t).strip()
    return t


def is_simran(text: str) -> bool:
    """Heuristic: short, repetitive text made of known simran phrases."""
    norm = normalize(text)
    if not norm:
        return False
    tokens = norm.split()
    if not tokens:
        return False
    # dominant token frequency
    from collections import Counter
    counts = Counter(tokens)
    top_tok, top_count = counts.most_common(1)[0]
    # >=60% of tokens are the dominant one AND it's a known simran phrase
    if top_count / len(tokens) >= 0.6 and any(
        top_tok in p or p in top_tok for p in SIMRAN_PHRASES
    ):
        return True
    # phrase-level match
    if norm in {normalize(p) for p in SIMRAN_PHRASES}:
        return True
    return False


def load_tuks(tuks_path: Path) -> tuple[list[dict], list[str]]:
    """Returns (tuks_list, normalized_texts_list) in same order."""
    data = json.loads(tuks_path.read_text())
    tuks = []
    choices = []
    for t in data:
        norm = normalize(t.get("text") or "")
        if not norm:
            continue
        tuks.append(t)
        choices.append(norm)
    print(f"[sggs] {len(tuks)} canonical tuks loaded", file=sys.stderr)
    return tuks, choices


def classify(score: float, text: str) -> str:
    """Apply thresholds; simran allowlist wins over drop."""
    if score >= 0.80:
        return "accept"
    if score >= 0.50:
        return "review"
    if is_simran(text):
        return "simran"
    return "drop"


def enrich_file(src: Path, dst: Path, tuks: list[dict],
                choices: list[str]) -> dict:
    rows = [json.loads(l) for l in src.open()]
    stats = {"accept": 0, "review": 0, "drop": 0, "simran": 0, "empty": 0}

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w") as f:
        for r in rows:
            gemini_text = r.get("text", "")
            r["gemini_text"] = gemini_text
            norm = normalize(gemini_text)

            if not norm:
                r.update({
                    "sggs_match_text": "", "sggs_shabad_id": "",
                    "sggs_tuk_id": "", "sggs_raag": "",
                    "sggs_match_score": 0.0, "sggs_flag": "drop",
                })
                stats["empty"] += 1
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                continue

            best = process.extractOne(
                norm, choices,
                scorer=fuzz.token_set_ratio,
                score_cutoff=0,
            )
            if best is None:
                match_text, score, idx = "", 0.0, -1
            else:
                match_text, raw_score, idx = best
                score = raw_score / 100.0

            if idx >= 0:
                tuk = tuks[idx]
                r["sggs_match_text"] = tuk.get("text", "")
                r["sggs_shabad_id"] = tuk.get("shabad_id", "")
                r["sggs_tuk_id"] = tuk.get("tuk_id", "")
                r["sggs_raag"] = tuk.get("raag", "")
            else:
                r["sggs_match_text"] = ""
                r["sggs_shabad_id"] = ""
                r["sggs_tuk_id"] = ""
                r["sggs_raag"] = ""

            r["sggs_match_score"] = round(score, 3)
            flag = classify(score, gemini_text)
            r["sggs_flag"] = flag
            stats[flag] += 1

            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/root/v3_data")
    ap.add_argument("--tuks", required=True,
                    help="Path to tuks.json")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    src_dir = data_root / "transcripts"
    dst_dir = data_root / "transcripts_enriched"

    tuks, choices = load_tuks(Path(args.tuks))

    totals = {"accept": 0, "review": 0, "drop": 0, "simran": 0, "empty": 0}
    files = sorted(src_dir.glob("*.jsonl"))
    for i, src in enumerate(files, 1):
        dst = dst_dir / src.name
        s = enrich_file(src, dst, tuks, choices)
        for k in totals:
            totals[k] += s[k]
        if i % 10 == 0 or i == len(files):
            print(f"  [{i}/{len(files)}] {src.name}: {s}")

    total = sum(totals.values())
    print("\n[summary]")
    print(f"  total segments: {total}")
    for k in ("accept", "review", "simran", "drop", "empty"):
        pct = 100 * totals[k] / total if total else 0
        print(f"  {k:>8s}: {totals[k]:>5d}  ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
