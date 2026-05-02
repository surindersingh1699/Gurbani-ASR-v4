"""Audit train/eval video_id overlap before fine-tuning IndicConformer.

The canonical eval slices and the YT-captions train slices are separate HF
repos but they were carved from the same source-video pool. If a video appears
in both, the model effectively sees its eval audio at training time and the
WER number lies.

Usage (RunPod or local):
    python scripts/audit_train_eval_leak.py
        --eval surindersinghssj/gurbani-sehajpath-yt-captions-eval-canonical \
               surindersinghssj/gurbani-kirtan-yt-captions-eval-canonical \
        --train surindersinghssj/gurbani-sehajpath-yt-captions-canonical \
                surindersinghssj/gurbani-kirtan-yt-captions-300h-canonical

Writes:
    bench_results/leak_audit.json   — per-eval-dataset list of video_ids that
                                       also appear in train; counts; safe-to-train flag
    bench_results/train_dropped_video_ids.txt  — newline list to feed to a
                                                 dataset.filter() at training time
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def collect_video_ids(name: str, split: str = "train") -> set[str]:
    from datasets import load_dataset
    ds = load_dataset(name, split=split)
    if "video_id" not in ds.column_names:
        raise ValueError(f"{name} has no `video_id` column")
    return set(ds["video_id"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", nargs="+", required=True)
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--out", default="bench_results/leak_audit.json")
    args = ap.parse_args()

    train_ids: set[str] = set()
    for name in args.train:
        ids = collect_video_ids(name)
        print(f"[train] {name}: {len(ids)} unique video_ids")
        train_ids |= ids

    audit = {"train_total_video_ids": len(train_ids), "eval_datasets": {}}
    leak_ids: set[str] = set()
    for name in args.eval:
        ids = collect_video_ids(name)
        overlap = ids & train_ids
        leak_ids |= overlap
        audit["eval_datasets"][name] = {
            "n_videos": len(ids),
            "n_videos_leaked_from_train": len(overlap),
            "leaked_video_ids": sorted(overlap),
            "safe_to_train": len(overlap) == 0,
        }
        print(f"[eval]  {name}: {len(ids)} videos, "
              f"{len(overlap)} appear in train  "
              f"({'⚠ LEAK' if overlap else '✅ clean'})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    drop_path = out.with_name("train_dropped_video_ids.txt")
    drop_path.write_text("\n".join(sorted(leak_ids)) + "\n")
    print(f"\nWrote audit → {out}")
    print(f"Wrote drop-list ({len(leak_ids)} video_ids) → {drop_path}")
    print("\nUse this in your NeMo manifest builder:")
    print(f"    leaked = set(open('{drop_path}').read().split())")
    print("    train_clips = [c for c in train_clips if c['video_id'] not in leaked]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
