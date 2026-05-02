#!/usr/bin/env python3
# scripts/add_canonical_column.py
"""Stage 1 CLI: run the DB-grounded canonical pipeline on a parquet input.

Outputs:
  --output-parquet: dataset with new columns
      (sggs_line, final_text, decision, is_simran,
       canonical_shabad_id, canonical_match_score)
  --audit-parquet : per-row audit sidecar
      (shabad_id, match_score, op_counts, line_ids, retrieval_margin)

Input modes:
  --source-col text          (default) pipe row["text"] through the pipeline
  --source-col text_cleaned  pipe row["text_cleaned"] through the pipeline
                             (use when the parquet is a *-clean dataset whose
                              `>>`, `<unk>`, non-Gurmukhi, and Waheguru
                              normalization has already been done)

  --respect-drop-candidate   skip rows with row["drop_candidate"] == True
                             (safe default ON when source-col=text_cleaned)
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
    ap.add_argument(
        "--source-col", default="text",
        help="Column to feed the pipeline. Use 'text_cleaned' for *-clean repos.",
    )
    ap.add_argument(
        "--respect-drop-candidate",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Skip rows where drop_candidate==True. Default: True if "
             "source-col=text_cleaned, False otherwise.",
    )
    ap.add_argument("--limit", type=int, default=None,
                    help="process only first N rows (dry-run)")
    args = ap.parse_args()

    cfg = get_dataset_config(args.dataset)
    pipeline = CanonicalPipeline(cfg, db_path=Path(args.db))

    df = pd.read_parquet(args.input_parquet)
    if args.limit:
        df = df.head(args.limit)

    # Decide drop-candidate behavior
    respect_drop = args.respect_drop_candidate
    if respect_drop is None:
        respect_drop = args.source_col == "text_cleaned"

    # Pre-filter drop_candidate rows BEFORE running the pipeline.
    if respect_drop and "drop_candidate" in df.columns:
        before = len(df)
        df = df[~df["drop_candidate"].fillna(False).astype(bool)].reset_index(drop=True)
        print(
            f"[run] dropped {before - len(df)} rows with drop_candidate=True",
            file=sys.stderr,
        )

    # Materialize rows and remap source-col → text so the pipeline sees the
    # cleaned input on row["text"] without any internal changes.
    rows = df.to_dict(orient="records")
    if args.source_col != "text":
        if args.source_col not in df.columns:
            print(
                f"[run] ERROR: --source-col '{args.source_col}' not in parquet; "
                f"columns = {list(df.columns)}",
                file=sys.stderr,
            )
            sys.exit(2)
        for r in rows:
            # Preserve original text under raw_text for audit, overwrite text.
            r.setdefault("raw_text", r.get("text"))
            r["text"] = r[args.source_col]
    print(
        f"[run] {len(rows)} rows from {args.input_parquet} "
        f"(source_col={args.source_col}, respect_drop={respect_drop})",
        file=sys.stderr,
    )

    results = pipeline.run(rows)
    print(f"[run] {len(results)} rows after pre-cleaning + simran quota",
          file=sys.stderr)

    # Decision histogram
    from collections import Counter
    dh = Counter(r["decision"] for r in results)
    print(f"[run] decisions: {dict(dh)}", file=sys.stderr)

    # Dataset columns: original schema + canonical_* filter-friendly columns.
    # `shabad_id` and `match_score` are promoted from the audit sidecar to the
    # main dataset so downstream consumers can filter on decision + score
    # without loading the sidecar.
    base_cols = [
        "clip_id", "video_id", "text", "raw_text", "start_s", "end_s",
        "duration_s", "sggs_line", "final_text", "decision", "is_simran",
    ]
    canonical_cols = ["canonical_shabad_id", "canonical_match_score"]

    def _row_out(r: dict) -> dict:
        out = {k: r.get(k) for k in base_cols if k in r}
        # Promote shabad_id + match_score into canonical_* namespace for the
        # dataset. Keeps the audit sidecar column names intact.
        out["canonical_shabad_id"] = r.get("shabad_id")
        out["canonical_match_score"] = r.get("match_score")
        return out

    audit_cols = [
        "clip_id", "shabad_id", "line_ids", "match_score",
        "op_counts", "retrieval_margin",
    ]
    dataset_df = pd.DataFrame([_row_out(r) for r in results])
    audit_df = pd.DataFrame([
        {k: r.get(k) for k in audit_cols} for r in results
    ])
    dataset_df.to_parquet(args.output_parquet)
    audit_df.to_parquet(args.audit_parquet)
    print(f"[run] wrote {args.output_parquet} + {args.audit_parquet}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
