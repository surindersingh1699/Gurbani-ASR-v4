#!/usr/bin/env python3
"""Salvage pass for `error_fallback` rows — a failed-corrector row gets
a second chance via the reviewer (Gemini).

Flow:
  1. Load stage2 output parquet.
  2. Filter rows with stage2_decision == 'error_fallback' (rows where the
     OpenAI corrector cascade-failed, typically from rate-limit retries).
  3. Group by shabad_id, feed each group to `review_batch` with caption
     set as both the CAPTION and the CORRECTOR OUTPUT — asks the reviewer
     "audit this; it's either fine as-is or propose a fix yourself".
  4. On IMPROVED verdict + validator pass → upgrade row to
     `replaced_reviewed` with reviewer's final_text.
  5. Leave APPROVE / REJECT rows unchanged (still `error_fallback` but
     now we know caption-as-is is either acceptable to the reviewer or
     too risky to touch).
  6. Write updated parquet.

Cost: Gemini-only, roughly $0.0003 per row. Typical salvage batch of
500-2000 rows costs $0.15-0.60.

Usage:
    python scripts/salvage_error_fallback.py \
        --input-parquet /root/stage2_full/kirtan_stage2.parquet \
        --output-parquet /root/stage2_full/kirtan_stage2_salvaged.parquet \
        --dataset kirtan --db database.sqlite
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from canonical.config import get_dataset_config
from canonical.pipeline import CanonicalPipeline
from canonical.reviewer import ReviewerConfig, review_batch
from canonical.stage2 import _shabad_lines_text, _shabad_tokens
from canonical.validator import validate

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


USABLE_DECISIONS = {
    "sttm_exact", "sttm_matched", "sttm_replaced", "sttm_simran",
    "replaced_llm", "replaced_reviewed", "unchanged_noop",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-parquet", required=True)
    ap.add_argument("--output-parquet", required=True)
    ap.add_argument("--dataset", required=True, choices=["kirtan", "sehaj"])
    ap.add_argument("--db", default="database.sqlite")
    ap.add_argument("--reviewer-model", default="gemini-3-flash-lite")
    ap.add_argument("--reviewer-batch-size", type=int, default=5)
    ap.add_argument("--max-workers", type=int, default=10,
                    help="Concurrent shabad groups. Keep ≤ your Gemini RPM/20.")
    args = ap.parse_args()

    df = pd.read_parquet(args.input_parquet)
    print(f"[salvage] loaded {len(df)} rows from {args.input_parquet}",
          file=sys.stderr)

    err_mask = df["stage2_decision"] == "error_fallback"
    err_df = df[err_mask]
    print(f"[salvage] found {len(err_df)} error_fallback rows",
          file=sys.stderr)
    if len(err_df) == 0:
        print("[salvage] nothing to salvage — copying input as output",
              file=sys.stderr)
        df.to_parquet(args.output_parquet)
        return

    # Build review items grouped by shabad_id
    groups: dict[str, list[dict]] = defaultdict(list)
    skipped_no_shabad = 0
    for _, r in err_df.iterrows():
        sid = r.get("shabad_id") or ""
        if not sid:
            skipped_no_shabad += 1
            continue
        caption = (
            r.get("text_cleaned") or r.get("text") or r.get("stage2_text") or ""
        )
        if not caption:
            continue
        groups[sid].append({
            "clip_id": r["clip_id"],
            "caption": caption,
            # Pass caption as corrector_output too — asks reviewer to audit
            # the no-op assumption; if it sees a fix, it proposes IMPROVED.
            "corrector_output": caption,
        })
    if skipped_no_shabad:
        print(f"[salvage] skipped {skipped_no_shabad} rows with no shabad_id",
              file=sys.stderr)

    total_review = sum(len(v) for v in groups.values())
    print(f"[salvage] {total_review} rows across {len(groups)} shabads → reviewer",
          file=sys.stderr)

    # Load shabad context once
    pipe_cfg = get_dataset_config(args.dataset)
    pipe = CanonicalPipeline(pipe_cfg, db_path=Path(args.db))
    shabad_lines_by_sid = pipe.shabad_lines

    reviewer_cfg = ReviewerConfig(
        model=args.reviewer_model,
        batch_size=args.reviewer_batch_size,
    )

    def _process_group(sid: str, items: list[dict]):
        lines = _shabad_lines_text(shabad_lines_by_sid.get(sid, []))
        if not lines:
            return sid, {}
        return sid, review_batch(items, lines, reviewer_cfg)

    t0 = time.monotonic()
    reviewed_per_sid: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        fut_to_sid = {
            ex.submit(_process_group, sid, items): sid
            for sid, items in groups.items()
        }
        done = 0
        for fut in as_completed(fut_to_sid):
            sid, result = fut.result()
            reviewed_per_sid[sid] = result
            done += 1
            if done % 50 == 0 or done == len(fut_to_sid):
                print(f"[salvage] progress: {done}/{len(fut_to_sid)} shabads "
                      f"({(time.monotonic()-t0):.1f}s)", file=sys.stderr)

    print(f"[salvage] reviewer done in {time.monotonic()-t0:.1f}s",
          file=sys.stderr)

    # Apply results — update rows where reviewer IMPROVED and validator passes.
    # Iterate stored items (so we know each row's caption), not the reviewed
    # dict directly.
    clip_to_idx = {cid: i for i, cid in enumerate(df["clip_id"].tolist())}
    updated_rows: list[tuple[int, str, str, dict]] = []
    counts = Counter()
    for sid, items in groups.items():
        reviewed = reviewed_per_sid.get(sid, {})
        shabad_toks = _shabad_tokens(shabad_lines_by_sid.get(sid, []))
        for item in items:
            cid = item["clip_id"]
            rreq = reviewed.get(cid)
            if not rreq or not rreq.get("ok"):
                counts["reviewer_failed"] += 1
                continue
            verdict = rreq.get("verdict")
            if verdict == "APPROVE":
                counts["reviewer_approve_caption"] += 1
                continue
            if verdict == "REJECT":
                counts["reviewer_reject"] += 1
                continue
            if verdict != "IMPROVED":
                counts[f"unknown_verdict_{verdict}"] += 1
                continue
            revised = rreq.get("final_text") or ""
            if not revised or revised.strip() == item["caption"].strip():
                counts["improved_but_identical"] += 1
                continue
            ok, reasons = validate(item["caption"], revised, shabad_toks)
            if not ok:
                counts["improved_but_validator_rejected"] += 1
                continue
            counts["salvaged"] += 1
            idx = clip_to_idx.get(cid)
            if idx is None:
                continue
            updated_rows.append((idx, revised, "replaced_reviewed", rreq))

    print(f"[salvage] outcomes: {dict(counts)}", file=sys.stderr)

    # Apply updates using .at[] to preserve list-typed columns.
    for idx, revised, decision, rreq in updated_rows:
        df.at[idx, "stage2_decision"] = decision
        df.at[idx, "stage2_text"] = revised
        df.at[idx, "stage2_reviewer_verdict"] = rreq.get("verdict")
        df.at[idx, "stage2_reviewer_final"] = revised
        df.at[idx, "stage2_changed_tokens"] = rreq.get("changed_tokens") or []
        df.at[idx, "stage2_reasons"] = (
            (rreq.get("reasons") or []) + ["salvaged_from_error_fallback"]
        )
        df.at[idx, "stage2_reviewer_model"] = rreq.get("model")
        df.at[idx, "training_usable"] = True

    # Final histogram
    hist = Counter(df["stage2_decision"])
    usable = int(df["training_usable"].sum()) if "training_usable" in df.columns else 0
    print(f"\n=== post-salvage histogram ({len(df)} rows) ===", file=sys.stderr)
    for k, v in sorted(hist.items()):
        tag = "[USABLE]" if k in USABLE_DECISIONS else "[skip  ]"
        print(f"  {tag} {k:30s} {v}", file=sys.stderr)
    print(f"\ntraining_usable: {usable}/{len(df)} "
          f"({100*usable/max(len(df),1):.1f}%)", file=sys.stderr)

    df.to_parquet(args.output_parquet)
    print(f"[salvage] wrote {args.output_parquet}", file=sys.stderr)


if __name__ == "__main__":
    main()
