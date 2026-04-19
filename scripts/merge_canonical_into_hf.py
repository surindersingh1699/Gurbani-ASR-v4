#!/usr/bin/env python3
# scripts/merge_canonical_into_hf.py
"""Merge Stage-1 + Stage-2 sidecars, then push a new revision of the HF dataset.

Safety: requires --confirm-push flag to actually push. Without it, just writes
the merged parquet locally for inspection.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from canonical.merge_hf import merge_sidecars

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1-parquet", required=True)
    ap.add_argument("--llm-sidecar", default=None)
    ap.add_argument("--output-parquet", required=True)
    ap.add_argument("--hf-repo", default=None,
                    help="e.g. surindersinghssj/gurbani-kirtan-yt-captions-300h")
    ap.add_argument("--confirm-push", action="store_true",
                    help="Actually push to HF Hub. Without this, only writes locally.")
    args = ap.parse_args()

    df = merge_sidecars(args.stage1_parquet, args.llm_sidecar)
    df.to_parquet(args.output_parquet)
    print(f"[merge] wrote {args.output_parquet} ({len(df)} rows)", file=sys.stderr)

    if not args.hf_repo:
        return

    if not args.confirm_push:
        print(f"[merge] --confirm-push not set; skipping HF push to {args.hf_repo}",
              file=sys.stderr)
        print(f"[merge] run again with --confirm-push to push", file=sys.stderr)
        return

    from datasets import Dataset, Audio
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("[merge] ERROR: HF_TOKEN missing", file=sys.stderr)
        sys.exit(2)

    # Cast audio column if present (column name "audio")
    ds = Dataset.from_pandas(df)
    if "audio" in ds.column_names:
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    ds.push_to_hub(args.hf_repo, token=token)
    print(f"[merge] pushed to {args.hf_repo}", file=sys.stderr)


if __name__ == "__main__":
    main()
