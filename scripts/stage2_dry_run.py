#!/usr/bin/env python3
"""Stage 2 dry-run harness: pulls rows labeled 'review' or rescue-path
'unchanged' from a parquet (or runs the full Stage 1 pipeline in-memory),
then calls corrector + reviewer on each. Prints caption / corrector /
reviewer / final for manual inspection.

Usage:
    python3 scripts/stage2_dry_run.py \
        --input-parquet /tmp/kirtan_eval_50.parquet \
        --dataset kirtan \
        --n 10 \
        [--only review|rescue_unchanged|both]
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from canonical.config import get_dataset_config
from canonical.corrector import CorrectorConfig
from canonical.pipeline import CanonicalPipeline
from canonical.reviewer import ReviewerConfig
from canonical.stage2 import Stage2Config, process_row, process_rows

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-parquet", required=True)
    ap.add_argument("--dataset", choices=["kirtan", "sehaj"], required=True)
    ap.add_argument("--db", default="database.sqlite")
    ap.add_argument("--n", type=int, default=10,
                    help="max stage-2 candidates to process")
    ap.add_argument("--only", choices=["review", "rescue_unchanged", "both"],
                    default="both")
    ap.add_argument("--corrector-model", default="gpt-5-nano")
    ap.add_argument("--reviewer-model", default="gemini-3-flash-lite")
    ap.add_argument("--batched", action="store_true",
                    help="Use batched + concurrent driver (process_rows).")
    ap.add_argument("--corrector-batch-size", type=int, default=15)
    ap.add_argument("--reviewer-batch-size", type=int, default=10)
    ap.add_argument("--max-workers", type=int, default=5,
                    help="Concurrent shabad-groups processed in parallel.")
    ap.add_argument("--source-col", default="text",
                    help="Column to feed the pipeline. Use 'text_cleaned' "
                         "for *-clean datasets.")
    ap.add_argument("--respect-drop-candidate",
                    action=argparse.BooleanOptionalAction, default=None,
                    help="Skip rows with drop_candidate=True. Default: "
                         "True if source-col=text_cleaned else False.")
    args = ap.parse_args()

    cfg = get_dataset_config(args.dataset)
    pipe = CanonicalPipeline(cfg, db_path=Path(args.db))
    df = pd.read_parquet(args.input_parquet)
    print(f"[stage2] loaded {len(df)} rows from {args.input_parquet} "
          f"(source_col={args.source_col})", file=sys.stderr)

    respect_drop = args.respect_drop_candidate
    if respect_drop is None:
        respect_drop = args.source_col == "text_cleaned"
    if respect_drop and "drop_candidate" in df.columns:
        before = len(df)
        df = df[~df["drop_candidate"].fillna(False).astype(bool)].reset_index(drop=True)
        print(f"[stage2] dropped {before - len(df)} drop_candidate rows",
              file=sys.stderr)

    records = df.to_dict(orient="records")
    if args.source_col != "text":
        if args.source_col not in df.columns:
            print(f"[stage2] ERROR: column '{args.source_col}' missing. "
                  f"Columns: {list(df.columns)}", file=sys.stderr)
            sys.exit(2)
        for r in records:
            r.setdefault("raw_text", r.get("text"))
            r["text"] = r[args.source_col]

    out = pipe.run(records)

    def is_candidate(r: dict) -> bool:
        dec = r["decision"]
        has_sid = bool(r.get("shabad_id"))
        if args.only == "review":
            return dec == "review"
        if args.only == "rescue_unchanged":
            return dec == "unchanged" and has_sid
        return (dec == "review") or (dec == "unchanged" and has_sid)

    candidates = [r for r in out if is_candidate(r)]
    print(f"[stage2] {len(candidates)} candidates (decision histogram so far: "
          f"{dict(Counter(r['decision'] for r in out))})", file=sys.stderr)
    candidates = candidates[: args.n]

    stage2_cfg = Stage2Config(
        corrector=CorrectorConfig(
            model=args.corrector_model,
            batch_size=args.corrector_batch_size,
        ),
        reviewer=ReviewerConfig(
            model=args.reviewer_model,
            batch_size=args.reviewer_batch_size,
        ),
    )

    # Rows that bypass Stage 2 because STTM was already confident.
    # Their final_text is STTM's output; training_usable = True by default.
    passthrough: list[dict] = []
    for r in out:
        dec = r["decision"]
        if dec in ("exact", "matched", "replaced", "simran"):
            r["stage2_decision"] = f"sttm_{dec}"
            r["stage2_text"] = r["final_text"]
            r["training_usable"] = True
            passthrough.append(r)
        elif dec == "unchanged" and not r.get("shabad_id"):
            # Low-confidence unchanged (no shabad retrieved) — flag as
            # caption-verbatim but NOT training-usable (no validation).
            r["stage2_decision"] = "sttm_unchanged_nocontext"
            r["stage2_text"] = r["final_text"]
            r["training_usable"] = False
            passthrough.append(r)

    # Build shabad lookup once
    shabad_lines = pipe.shabad_lines

    verdicts = Counter()
    final_rows = []

    import time
    t0 = time.monotonic()

    if args.batched:
        print(f"[stage2] batched mode: corrector_batch={args.corrector_batch_size} "
              f"reviewer_batch={args.reviewer_batch_size} workers={args.max_workers}",
              file=sys.stderr)
        stage2_results = process_rows(
            candidates, shabad_lines, stage2_cfg,
            max_workers=args.max_workers,
        )
        for row, result in zip(candidates, stage2_results):
            verdicts[result["stage2_decision"]] += 1
            final_rows.append({**row, **result})
            _print_row(row, result, len(final_rows), len(candidates))
    else:
        for i, row in enumerate(candidates, 1):
            sid = row["shabad_id"]
            lines = shabad_lines.get(sid, [])
            result = process_row(row, lines, stage2_cfg)
            verdicts[result["stage2_decision"]] += 1
            final_rows.append({**row, **result})
            _print_row(row, result, i, len(candidates))

    elapsed = time.monotonic() - t0
    print(f"\n=== Stage 2 verdict histogram ({len(candidates)} rows in {elapsed:.1f}s) ===")
    for k, v in sorted(verdicts.items()):
        print(f"  {k:25s} {v}")

    # Label training_usable on every row (Stage 2 candidates + passthroughs).
    USABLE_DECISIONS = {
        "sttm_exact", "sttm_matched", "sttm_replaced", "sttm_simran",
        "replaced_llm", "replaced_reviewed", "unchanged_noop",
    }
    for r in final_rows:
        r["training_usable"] = r["stage2_decision"] in USABLE_DECISIONS

    all_rows = final_rows + passthrough
    usable = sum(1 for r in all_rows if r.get("training_usable"))
    print(f"\n=== Training usability ({len(all_rows)} rows total) ===")
    print(f"  training_usable=True   {usable}  ({100*usable/max(len(all_rows),1):.1f}%)")
    print(f"  training_usable=False  {len(all_rows) - usable}")
    print("\n  by stage2_decision:")
    per_dec = Counter()
    for r in all_rows:
        per_dec[(r.get("stage2_decision") or f"sttm_{r['decision']}", r.get("training_usable", False))] += 1
    for (dec, use), n in sorted(per_dec.items()):
        tag = "USABLE" if use else "skip  "
        print(f"    [{tag}] {dec:25s} {n}")


def _print_row(row: dict, result: dict, i: int, n: int) -> None:
    caption = row.get("text", "")
    sttm_final = row.get("final_text", "")
    print(f"\n[{i}/{n}] clip={row['clip_id'][-20:]}  "
          f"decision={row['decision']}  shabad={row.get('shabad_id')}")
    print(f"  cap: {caption}")
    print(f"  sttm:{sttm_final}")
    print(f"  gpt: {result.get('stage2_corrector_output')}")
    print(f"  rev: {result.get('stage2_reviewer_verdict')}  "
          f"rev_final={result.get('stage2_reviewer_final')}")
    if result.get("stage2_changed_tokens"):
        print(f"  changes: {result['stage2_changed_tokens']}")
    print(f"  final_decision={result['stage2_decision']}")
    print(f"  OUT: {result['stage2_text']}")
    if result.get("stage2_reasons"):
        print(f"  reasons: {result['stage2_reasons']}")


if __name__ == "__main__":
    main()
