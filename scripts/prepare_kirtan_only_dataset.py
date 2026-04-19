#!/usr/bin/env python3
"""
Prepare kirtan-only dataset for training.

Train split:  All kirtan segments with duration <= 30s
Validation split: Kirtan validation split (from source dataset)

Output schema is normalized to:
  - audio
  - transcription
  - duration_sec
  - source_dataset
"""

from __future__ import annotations

import argparse
import os

from datasets import DatasetDict, load_dataset

NUM_PROC = max(1, os.cpu_count() - 1)

KIRTAN_DATASET = "surindersinghssj/gurbani-kirtan-dataset-v2"
OUT_DATASET = "surindersinghssj/gurbani-kirtan-only-pilot500-train"

KIRTAN_MAX_DURATION = 30.0
SEED = 42

KEEP_COLS = ["audio", "transcription", "duration_sec", "source_dataset"]


def _to_seconds(value: float | int | None) -> float:
    if value is None:
        return 0.0
    return float(value)


def _normalize_kirtan_row(x: dict) -> dict:
    return {
        "audio": x["audio"],
        "transcription": x["gurmukhi_text"],
        "duration_sec": _to_seconds(x.get("duration")),
        "source_dataset": "kirtan",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare kirtan-only dataset")
    parser.add_argument(
        "--out-dataset",
        default=OUT_DATASET,
        help=f"HF dataset repo to push (default: {OUT_DATASET})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run all checks/stats but do not push to HF",
    )
    args = parser.parse_args()

    print(f"[prep] Loading Kirtan dataset: {KIRTAN_DATASET}")
    kirtan_train = load_dataset(KIRTAN_DATASET, split="train")
    kirtan_val = load_dataset(KIRTAN_DATASET, split="validation")
    print(f"[prep] Kirtan train rows (raw): {len(kirtan_train)}")
    print(f"[prep] Kirtan val rows (raw): {len(kirtan_val)}")

    # Filter <=30s
    kirtan_train_30 = kirtan_train.filter(
        lambda x: _to_seconds(x.get("duration")) <= KIRTAN_MAX_DURATION,
        num_proc=NUM_PROC,
    )
    kirtan_val_30 = kirtan_val.filter(
        lambda x: _to_seconds(x.get("duration")) <= KIRTAN_MAX_DURATION,
        num_proc=NUM_PROC,
    )

    if len(kirtan_train_30) == 0:
        raise RuntimeError("Kirtan <=30s filter resulted in empty train split.")

    # Normalize columns
    train_norm = kirtan_train_30.map(
        _normalize_kirtan_row, num_proc=NUM_PROC
    ).select_columns(KEEP_COLS)
    val_norm = kirtan_val_30.map(
        _normalize_kirtan_row, num_proc=NUM_PROC
    ).select_columns(KEEP_COLS)

    # Shuffle
    train_final = train_norm.shuffle(seed=SEED)
    val_final = val_norm.shuffle(seed=SEED)

    # Stats
    train_hours = sum(train_final["duration_sec"]) / 3600.0
    val_hours = sum(val_final["duration_sec"]) / 3600.0

    print(f"[prep] Kirtan train <=30s: {len(train_final)} rows, {train_hours:.3f}h")
    print(f"[prep] Kirtan val <=30s: {len(val_final)} rows, {val_hours:.3f}h")

    # Validate
    if any((not t) for t in train_final["transcription"]):
        raise RuntimeError("Train split has empty transcription values")
    if any((not t) for t in val_final["transcription"]):
        raise RuntimeError("Validation split has empty transcription values")

    dataset_dict = DatasetDict({"train": train_final, "validation": val_final})

    if args.dry_run:
        print("[prep] Dry-run complete. Skipping push.")
        return

    print(f"[prep] Pushing dataset to {args.out_dataset} ...")
    dataset_dict.push_to_hub(args.out_dataset)
    print("[prep] Done.")


if __name__ == "__main__":
    main()
