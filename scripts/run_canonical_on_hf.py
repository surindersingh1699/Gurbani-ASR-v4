#!/usr/bin/env python3
"""Run Stage-1 canonical (STTM) pipeline on a HuggingFace dataset and push
the augmented dataset back to the Hub under a `<source>-canonical` repo.

No LLM stage. Just preprocess + simran + retrieval + align + decide.

Adds these columns to every row that survives the pipeline:
    sggs_line, final_text, decision, is_simran,
    canonical_shabad_id, canonical_match_score,
    canonical_line_ids, canonical_op_counts, canonical_retrieval_margin

Rows that the pipeline drops (preprocess.should_drop_row + simran-quota cap)
are removed from the output dataset.

Parallelization: rows are grouped by video_id and dispatched across
multiprocessing workers. The simran quota is applied globally on the
master after gathering. Use --workers (default = nproc).

Usage:
    python scripts/run_canonical_on_hf.py \
        --src-repo surindersinghssj/gurbani-kirtan-yt-captions-eval \
        --dst-repo surindersinghssj/gurbani-kirtan-yt-captions-eval-canonical \
        --dataset kirtan \
        --db database.sqlite \
        --workers 16 \
        --private
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Sequence, Value, load_dataset

sys.path.insert(0, str(Path(__file__).parent))
from canonical.config import get_dataset_config
from canonical.pipeline import CanonicalPipeline
from canonical.simran import apply_simran_quota


META_COLS = (
    "clip_id", "video_id", "text", "raw_text",
    "start_s", "end_s", "duration_s",
)


_PIPELINE: CanonicalPipeline | None = None


def _worker_init(dataset_name: str, db_path: str):
    global _PIPELINE
    cfg = get_dataset_config(dataset_name)
    _PIPELINE = CanonicalPipeline(cfg, db_path=Path(db_path))


def _worker_process(rows: list[dict]) -> list[dict]:
    assert _PIPELINE is not None, "worker not initialized"
    return _PIPELINE.run(rows, defer_simran_quota=True)


def _process_split(
    split_name: str, ds: Dataset, dataset_name: str, db_path: str, workers: int,
    checkpoint_dir: Path | None = None, src_repo: str = "",
) -> Dataset:
    """Run canonical pipeline on one split (parallel per video_id) and return
    a new Dataset with canonical_* columns. Drops rows the pipeline removes."""
    cols = [c for c in META_COLS if c in ds.column_names]
    if "clip_id" not in cols or "text" not in cols or "video_id" not in cols:
        raise ValueError(
            f"split {split_name!r} missing required columns "
            f"(need clip_id+video_id+text, have {ds.column_names})"
        )
    meta = ds.remove_columns([c for c in ds.column_names if c not in cols])
    rows = meta.to_pandas().to_dict(orient="records")
    print(f"[{split_name}] {len(rows)} rows", file=sys.stderr)

    by_vid: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_vid[r["video_id"]].append(r)
    video_groups = list(by_vid.values())
    print(f"[{split_name}] {len(video_groups)} videos, {workers} workers",
          file=sys.stderr)

    ckpt_path = None
    if checkpoint_dir is not None:
        safe_repo = src_repo.replace("/", "__")
        ckpt_path = checkpoint_dir / f"canonical_{safe_repo}_{split_name}.parquet"

    if ckpt_path is not None and ckpt_path.exists():
        import pandas as pd
        print(f"[{split_name}] resuming from checkpoint {ckpt_path}", file=sys.stderr)
        combined = pd.read_parquet(ckpt_path).to_dict(orient="records")
        for r in combined:
            r["line_ids"] = json.loads(r.get("line_ids") or "[]")
            r["op_counts"] = json.loads(r.get("op_counts") or "{}")
    else:
        if workers <= 1:
            _worker_init(dataset_name, db_path)
            results_per_video = [_worker_process(g) for g in video_groups]
        else:
            ctx = mp.get_context("fork")
            with ctx.Pool(
                processes=workers,
                initializer=_worker_init,
                initargs=(dataset_name, db_path),
            ) as pool:
                results_per_video = pool.map(_worker_process, video_groups)
        combined: list[dict] = []
        for r in results_per_video:
            combined.extend(r)
        if ckpt_path is not None:
            import pandas as pd
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            ckpt_df = pd.DataFrame([
                {
                    "clip_id": r["clip_id"],
                    "video_id": r["video_id"],
                    "is_simran": bool(r.get("is_simran", False)),
                    "decision": r.get("decision"),
                    "sggs_line": r.get("sggs_line"),
                    "final_text": r.get("final_text"),
                    "shabad_id": r.get("shabad_id"),
                    "match_score": r.get("match_score"),
                    "line_ids": json.dumps(r.get("line_ids") or []),
                    "op_counts": json.dumps(r.get("op_counts") or {}),
                    "retrieval_margin": r.get("retrieval_margin"),
                }
                for r in combined
            ])
            ckpt_df.to_parquet(ckpt_path)
            print(f"[{split_name}] checkpoint saved → {ckpt_path}",
                  file=sys.stderr)

    cfg = get_dataset_config(dataset_name)
    if cfg.simran_detection:
        from canonical.simran import SimranConfig
        before = len(combined)
        combined = apply_simran_quota(combined, SimranConfig())
        n_simran = sum(1 for r in combined if r.get("is_simran"))
        print(
            f"[{split_name}] simran quota: dropped {before - len(combined)} "
            f"rows; {n_simran} simran kept",
            file=sys.stderr,
        )

    hist = Counter(r["decision"] for r in combined)
    print(f"[{split_name}] decisions: {dict(hist)}", file=sys.stderr)

    by_clip: dict[str, dict] = {}
    for r in combined:
        by_clip[r["clip_id"]] = {
            "sggs_line": r.get("sggs_line"),
            "final_text": r.get("final_text"),
            "decision": r.get("decision"),
            "is_simran": bool(r.get("is_simran", False)),
            "canonical_shabad_id": r.get("shabad_id"),
            "canonical_match_score": (
                float(r["match_score"]) if r.get("match_score") is not None else None
            ),
            "canonical_line_ids": r.get("line_ids") or [],
            "canonical_op_counts": json.dumps(r.get("op_counts") or {}),
            "canonical_retrieval_margin": (
                float(r["retrieval_margin"])
                if r.get("retrieval_margin") is not None else None
            ),
        }

    kept = ds.filter(lambda ex: ex["clip_id"] in by_clip, num_proc=workers)
    dropped = len(ds) - len(kept)
    print(f"[{split_name}] kept={len(kept)} dropped={dropped}", file=sys.stderr)

    # Attach via add_column (one column at a time) instead of Dataset.map.
    # Dataset.map rewrites the entire arrow table including the audio column,
    # which has hit `pyarrow.lib.ArrowTypeError: Expected bytes, got a 'float'
    # object` on rare malformed audio rows during re-encode at scale (sehaj
    # 63k crashed at ~11% attach). add_column doesn't touch existing columns
    # so audio passes through untouched.
    # Explicit feature types per new column. Without these, add_column runs
    # Features._from_column inference over the values list, which has hit
    # `pyarrow.lib.ArrowTypeError: Expected bytes, got a 'float' object` on
    # mixed-null columns at scale. Pinning the schema avoids inference.
    new_col_features = {
        "sggs_line": Value("string"),
        "final_text": Value("string"),
        "decision": Value("string"),
        "is_simran": Value("bool"),
        "canonical_shabad_id": Value("string"),
        "canonical_match_score": Value("float64"),
        "canonical_line_ids": Sequence(Value("string")),
        "canonical_op_counts": Value("string"),
        "canonical_retrieval_margin": Value("float64"),
    }
    # Pull clip_id column directly to avoid iterating the full dataset (which
    # would decode audio). kept["clip_id"] is O(n) memory but n=200k strings
    # is fine (~10MB).
    clip_ids = kept["clip_id"]
    out_ds = kept
    import math
    def _nan_to_none(v):
        if isinstance(v, float) and math.isnan(v):
            return None
        return v
    for col, feat in new_col_features.items():
        values = [_nan_to_none(by_clip[cid][col]) for cid in clip_ids]
        out_ds = out_ds.add_column(col, values, feature=feat)
        print(f"[{split_name}]   + {col}", file=sys.stderr)
    print(f"[{split_name}] attached {len(new_col_features)} canonical columns",
          file=sys.stderr)
    return out_ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-repo", required=True)
    ap.add_argument("--dst-repo", required=True)
    ap.add_argument("--dataset", choices=["kirtan", "sehaj"], required=True)
    ap.add_argument("--db", default="database.sqlite")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--checkpoint-dir", default=None,
                    help="if set, persist per-row pipeline output to "
                         "<dir>/canonical_<src>_<split>.parquet so that "
                         "future runs skip the pipeline phase on success.")
    ap.add_argument("--limit", type=int, default=None,
                    help="dry-run: process only first N rows of each split")
    args = ap.parse_args()

    print(f"[load] {args.src_repo}", file=sys.stderr)
    ds_in = load_dataset(args.src_repo)
    if not isinstance(ds_in, DatasetDict):
        ds_in = DatasetDict({"train": ds_in})

    out: dict[str, Dataset] = {}
    for split_name, split_ds in ds_in.items():
        if args.limit:
            split_ds = split_ds.select(range(min(args.limit, len(split_ds))))
        out[split_name] = _process_split(
            split_name, split_ds, args.dataset, args.db, args.workers,
            checkpoint_dir=Path(args.checkpoint_dir) if args.checkpoint_dir else None,
            src_repo=args.src_repo,
        )

    out_dd = DatasetDict(out)
    print(
        f"[push] {args.dst_repo} private={args.private} "
        f"splits={ {k: len(v) for k, v in out.items()} }",
        file=sys.stderr,
    )
    out_dd.push_to_hub(args.dst_repo, private=args.private)
    print("[done]", file=sys.stderr)


if __name__ == "__main__":
    main()
