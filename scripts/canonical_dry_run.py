#!/usr/bin/env python3
# scripts/canonical_dry_run.py
"""Dry-run the Stage 1 + Stage 2 pipeline on the first N rows of a dataset
and print a decision histogram + a few sample corrections for manual review."""
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from canonical.config import get_dataset_config
from canonical.pipeline import CanonicalPipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-parquet", required=True)
    ap.add_argument("--dataset", choices=["kirtan", "sehaj"], required=True)
    ap.add_argument("--db", default="database.sqlite")
    ap.add_argument("--n", type=int, default=1000, help="rows to process")
    ap.add_argument("--sample-per-decision", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = get_dataset_config(args.dataset)
    pipeline = CanonicalPipeline(cfg, db_path=args.db)

    df = pd.read_parquet(args.input_parquet)
    df = df.head(args.n)
    rows = df.to_dict(orient="records")
    print(f"[dry-run] {len(rows)} rows, dataset={args.dataset}", file=sys.stderr)

    results = pipeline.run(rows)
    hist = Counter(r["decision"] for r in results)
    print(f"\nDecision histogram:")
    for dec in ["matched", "replaced", "review", "unchanged", "simran"]:
        n = hist.get(dec, 0)
        pct = 100 * n / len(results) if results else 0
        print(f"  {dec:10} {n:5d}  ({pct:5.1f}%)")

    # Samples per decision
    rng = random.Random(args.seed)
    by_dec: dict[str, list[dict]] = {}
    for r in results:
        by_dec.setdefault(r["decision"], []).append(r)
    print(f"\nSamples (up to {args.sample_per_decision} per decision):")
    for dec, rr in sorted(by_dec.items()):
        rng.shuffle(rr)
        print(f"\n--- {dec} ---")
        for r in rr[:args.sample_per_decision]:
            print(f"  clip_id={r['clip_id']}")
            print(f"    caption: {r['text']}")
            print(f"    final:   {r['final_text']}")
            if r.get("sggs_line"):
                print(f"    sggs:    {r['sggs_line']}")


if __name__ == "__main__":
    main()
