#!/usr/bin/env python3
"""Build a pure-source kirtan eval+test dataset from
`surindersinghssj/gurbani-kirtan-dataset-v2`.

Filter: duration <= 30 s.
Diversity: round-robin across videos with per-video cap. Split eval vs test
on VIDEO boundary (no video appears in both splits) so per-clip eval/test
leakage is impossible.

Output schema (compatible with the canonical pipeline driver):
    clip_id, video_id, text, raw_text, start_s, end_s, duration_s, audio
plus all original v2 columns kept as-is.

Push to: surindersinghssj/gurbani-kirtan-eval-pure (public, splits=eval+test)
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict

from datasets import Dataset, DatasetDict, load_dataset


SRC_REPO = "surindersinghssj/gurbani-kirtan-dataset-v2"


def _build_indices(
    df, n_eval: int, n_test: int, per_video_cap: int, seed: int,
) -> tuple[list[int], list[int]]:
    """Return (eval_idx, test_idx) into df. Splits videos disjoint."""
    rng = random.Random(seed)
    by_vid: dict[str, list[int]] = defaultdict(list)
    for i, vid in enumerate(df["video_id"].tolist()):
        by_vid[vid].append(i)
    videos = sorted(by_vid.keys())
    rng.shuffle(videos)

    half = len(videos) // 2
    eval_vids = videos[:half]
    test_vids = videos[half:]

    def _sample(vid_list: list[str], n_target: int) -> list[int]:
        capped: dict[str, list[int]] = {}
        for v in vid_list:
            rows = by_vid[v]
            rng.shuffle(rows)
            capped[v] = rows[:per_video_cap]
        order = list(vid_list)
        rng.shuffle(order)
        out: list[int] = []
        idx = 0
        empty_streak = 0
        while len(out) < n_target and empty_streak < len(order):
            v = order[idx % len(order)]
            if capped[v]:
                out.append(capped[v].pop(0))
                empty_streak = 0
            else:
                empty_streak += 1
            idx += 1
        return out

    return _sample(eval_vids, n_eval), _sample(test_vids, n_test)


def _to_canonical_schema(ex: dict, idx: int) -> dict:
    """Add canonical-pipeline-compatible columns (clip_id, text, start_s, ...)
    while keeping all original columns."""
    return {
        **ex,
        "clip_id": f"{ex['video_id']}_slide{ex.get('slide_index', idx)}_{idx:05d}",
        "text": ex.get("gurmukhi_text") or ex.get("gurmukhi_ocr") or "",
        "raw_text": ex.get("gurmukhi_ocr") or ex.get("gurmukhi_text") or "",
        "start_s": float(ex["start_time"]) if ex.get("start_time") is not None else None,
        "end_s": float(ex["end_time"]) if ex.get("end_time") is not None else None,
        "duration_s": float(ex["duration"]) if ex.get("duration") is not None else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst-repo", default="surindersinghssj/gurbani-kirtan-eval-pure")
    ap.add_argument("--max-duration-s", type=float, default=30.0)
    ap.add_argument("--n-eval", type=int, default=600)
    ap.add_argument("--n-test", type=int, default=600)
    ap.add_argument("--per-video-cap", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    print(f"[load] {SRC_REPO}", file=sys.stderr)
    ds = load_dataset(SRC_REPO)
    sp = list(ds.keys())[0]
    full = ds[sp]
    print(f"[load] {len(full)} rows in split={sp!r}", file=sys.stderr)

    meta = full.remove_columns([c for c in full.column_names if c == "audio"])
    df = meta.to_pandas()
    df["duration_f"] = df["duration"].astype(float)
    keep_mask = (df["duration_f"] <= args.max_duration_s) & df["gurmukhi_text"].notna()
    sub = df[keep_mask].copy().reset_index().rename(columns={"index": "_orig_idx"})
    print(
        f"[filter] {len(sub)} rows after duration<={args.max_duration_s}s + "
        f"non-null text; {sub['video_id'].nunique()} videos",
        file=sys.stderr,
    )

    eval_local, test_local = _build_indices(
        sub, args.n_eval, args.n_test, args.per_video_cap, args.seed,
    )
    eval_orig = sub["_orig_idx"].iloc[eval_local].tolist()
    test_orig = sub["_orig_idx"].iloc[test_local].tolist()
    print(
        f"[sample] eval={len(eval_orig)} ({sub.iloc[eval_local]['video_id'].nunique()} vids) "
        f"test={len(test_orig)} ({sub.iloc[test_local]['video_id'].nunique()} vids); "
        f"per_video_cap={args.per_video_cap}",
        file=sys.stderr,
    )
    overlap = set(sub.iloc[eval_local]['video_id']) & set(sub.iloc[test_local]['video_id'])
    assert not overlap, f"video overlap between eval/test: {overlap}"

    eval_ds = full.select(eval_orig).map(
        _to_canonical_schema, with_indices=True, desc="schema[eval]",
    )
    test_ds = full.select(test_orig).map(
        _to_canonical_schema, with_indices=True, desc="schema[test]",
    )

    eval_h = sum(eval_ds["duration_s"]) / 3600
    test_h = sum(test_ds["duration_s"]) / 3600
    print(f"[build] eval={eval_h:.2f}h test={test_h:.2f}h", file=sys.stderr)

    out = DatasetDict({"eval": eval_ds, "test": test_ds})
    print(
        f"[push] {args.dst_repo} private={args.private}", file=sys.stderr,
    )
    out.push_to_hub(args.dst_repo, private=args.private)
    print("[done]", file=sys.stderr)


if __name__ == "__main__":
    main()
