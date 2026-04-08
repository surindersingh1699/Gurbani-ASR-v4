"""
Prepare kirtan dataset for v2 training.

Steps:
1. Load full kirtan dataset from HuggingFace
2. Filter to segments <=30s only
3. Rename gurmukhi_text -> transcription
4. Split by video_id: ~89% train / 363 val / 363 test
5. Push prepared dataset to a NEW HF repo (does not overwrite source)

Usage:
    python scripts/prepare_kirtan_v2.py [--dry-run]

Requires HF_TOKEN in environment.
"""

import argparse
from collections import defaultdict

from datasets import Audio, DatasetDict, concatenate_datasets, load_dataset

SRC_DATASET = "surindersinghssj/gurbani-kirtan-dataset-v2"  # source (read-only)
DST_DATASET = "surindersinghssj/gurbani-kirtan-v2-prepared"  # prepared output (new repo)
MAX_DURATION = 30.0  # seconds
VAL_SIZE = 363
TEST_SIZE = 363


def main():
    parser = argparse.ArgumentParser(description="Prepare kirtan dataset for v2 training")
    parser.add_argument("--dry-run", action="store_true", help="Print stats without pushing to HF")
    args = parser.parse_args()

    # Load all splits and combine
    print(f"[prep] Loading dataset: {SRC_DATASET}")
    ds = load_dataset(SRC_DATASET)

    all_splits = []
    for split_name in ds:
        print(f"[prep]   {split_name}: {len(ds[split_name])} rows")
        all_splits.append(ds[split_name])

    combined = concatenate_datasets(all_splits)
    print(f"[prep] Combined: {len(combined)} total rows")

    # Filter by duration
    duration_col = "duration" if "duration" in combined.column_names else "duration_sec"
    before = len(combined)
    filtered = combined.filter(lambda x: x[duration_col] <= MAX_DURATION)
    after = len(filtered)
    print(f"[prep] Filtered <={MAX_DURATION}s: {after} rows (dropped {before - after})")

    # Rename gurmukhi_text -> transcription
    if "gurmukhi_text" in filtered.column_names and "transcription" not in filtered.column_names:
        filtered = filtered.rename_column("gurmukhi_text", "transcription")
        print("[prep] Renamed gurmukhi_text -> transcription")

    # Group by video_id for leakage-free splitting
    video_ids = defaultdict(list)
    for i, vid in enumerate(filtered["video_id"]):
        video_ids[vid].append(i)

    videos = list(video_ids.keys())
    print(f"[prep] Unique video_ids: {len(videos)}")

    # Sort videos by number of segments (ascending) for stable splitting
    videos.sort(key=lambda v: len(video_ids[v]))

    # Assign videos to val, test, then train
    val_indices = []
    test_indices = []
    train_indices = []

    # Fill val first
    target = "val"
    for vid in videos:
        indices = video_ids[vid]
        if target == "val" and len(val_indices) < VAL_SIZE:
            val_indices.extend(indices)
            if len(val_indices) >= VAL_SIZE:
                target = "test"
        elif target == "test" and len(test_indices) < TEST_SIZE:
            test_indices.extend(indices)
            if len(test_indices) >= TEST_SIZE:
                target = "train"
        else:
            train_indices.extend(indices)

    # Trim val/test to exact sizes if slightly over
    val_indices = val_indices[:VAL_SIZE]
    test_indices = test_indices[:TEST_SIZE]

    train_ds = filtered.select(train_indices)
    val_ds = filtered.select(val_indices)
    test_ds = filtered.select(test_indices)

    print(f"[prep] Train: {len(train_ds)} rows")
    print(f"[prep] Val:   {len(val_ds)} rows")
    print(f"[prep] Test:  {len(test_ds)} rows")
    print(f"[prep] Total: {len(train_ds) + len(val_ds) + len(test_ds)} rows")

    # Print duration stats
    for name, split in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        durations = split[duration_col]
        total_h = sum(durations) / 3600
        avg_s = sum(durations) / len(durations) if durations else 0
        print(f"[prep]   {name}: {total_h:.1f}h total, {avg_s:.1f}s avg duration")

    if args.dry_run:
        print("[prep] Dry run -- not pushing to HF")
        print(f"[prep] Columns: {train_ds.column_names}")
        # Show a sample transcription
        if "transcription" in train_ds.column_names:
            print(f"[prep] Sample transcription: {train_ds[0]['transcription'][:100]}")
        return

    # Push to NEW HF repo (does not overwrite source dataset)
    dataset_dict = DatasetDict({
        "train": train_ds,
        "validation": val_ds,
        "test": test_ds,
    })

    print(f"[prep] Pushing to {DST_DATASET} (new repo, source {SRC_DATASET} unchanged)...")
    dataset_dict.push_to_hub(DST_DATASET)
    print("[prep] Done!")


if __name__ == "__main__":
    main()
