#!/usr/bin/env python3
# scripts/add_canonical_column.py
"""Stage 1 CLI: run the DB-grounded canonical pipeline on a parquet input.

Outputs:
  --output-parquet: dataset with new columns (sggs_line, final_text, decision, is_simran)
  --audit-parquet : per-row audit sidecar (shabad_id, match_score, op_counts, line_ids, retrieval_margin)
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from canonical.config import get_dataset_config
from canonical.pipeline import CanonicalPipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-parquet", required=True)
    ap.add_argument("--output-parquet", required=True)
    ap.add_argument("--audit-parquet", required=True)
    ap.add_argument("--dataset", choices=["kirtan", "sehaj"], required=True)
    ap.add_argument("--db", default="database.sqlite")
    ap.add_argument("--limit", type=int, default=None,
                    help="process only first N rows (dry-run)")
    args = ap.parse_args()

    cfg = get_dataset_config(args.dataset)
    pipeline = CanonicalPipeline(cfg, db_path=Path(args.db))

    df = pd.read_parquet(args.input_parquet)
    if args.limit:
        df = df.head(args.limit)
    rows = df.to_dict(orient="records")
    print(f"[run] {len(rows)} rows from {args.input_parquet}", file=sys.stderr)

    results = pipeline.run(rows)
    print(f"[run] {len(results)} rows after pre-cleaning + simran quota", file=sys.stderr)

    # Decision histogram
    from collections import Counter
    dh = Counter(r["decision"] for r in results)
    print(f"[run] decisions: {dict(dh)}", file=sys.stderr)

    # Dataset columns (drop audit internals)
    dataset_cols = [
        "clip_id", "video_id", "text", "raw_text", "start_s", "end_s",
        "duration_s", "sggs_line", "final_text", "decision", "is_simran",
    ]
    audit_cols = [
        "clip_id", "shabad_id", "line_ids", "match_score",
        "op_counts", "retrieval_margin",
    ]
    existing = {c: df.columns.tolist() for c in ()}
    dataset_df = pd.DataFrame([
        {k: r.get(k) for k in dataset_cols if k in r} for r in results
    ])
    audit_df = pd.DataFrame([
        {k: r.get(k) for k in audit_cols} for r in results
    ])
    dataset_df.to_parquet(args.output_parquet)
    audit_df.to_parquet(args.audit_parquet)
    print(f"[run] wrote {args.output_parquet} + {args.audit_parquet}", file=sys.stderr)


if __name__ == "__main__":
    main()
