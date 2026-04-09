#!/usr/bin/env python3
"""
Prepare fixed dataset for kirtan-first pilot training.

Train split:
  - Exactly 10h Sehaj Path (deterministic random seed)
  - Plus all Kirtan segments with duration <= 30s

Validation split:
  - Sehaj Path validation split (for side-by-side eval with kirtan aux eval)

Output schema is normalized to:
  - audio
  - transcription
  - duration_sec
  - source_dataset
"""

from __future__ import annotations

import argparse
import os
import random

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

NUM_PROC = max(1, os.cpu_count() - 1)  # leave 1 core for system

SEHAJ_DATASET = "surindersinghssj/gurbani-sehajpath"
KIRTAN_DATASET = "surindersinghssj/gurbani-kirtan-dataset-v2"
OUT_DATASET = "surindersinghssj/gurbani-kirtan-first-pilot500-train"

SEHAJ_TARGET_HOURS = 10.0
KIRTAN_MAX_DURATION = 30.0
SEED = 42


def _to_seconds(value: float | int | None) -> float:
    if value is None:
        return 0.0
    return float(value)


def _normalize_sehaj_row(x: dict) -> dict:
    return {
        "audio": x["audio"],
        "transcription": x["transcription"],
        "duration_sec": _to_seconds(x.get("duration_sec")),
        "source_dataset": "sehajpath",
    }


def _normalize_kirtan_row(x: dict) -> dict:
    return {
        "audio": x["audio"],
        "transcription": x["gurmukhi_text"],
        "duration_sec": _to_seconds(x.get("duration")),
        "source_dataset": "kirtan",
    }


def _check_columns(ds: Dataset, split_name: str) -> None:
    expected = {"audio", "transcription", "duration_sec", "source_dataset"}
    cols = set(ds.column_names)
    if cols != expected:
        raise RuntimeError(
            f"{split_name}: unexpected columns {sorted(cols)}; expected {sorted(expected)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare kirtan-first pilot dataset")
    parser.add_argument(
        "--out-dataset",
        default=OUT_DATASET,
        help=f"Hugging Face dataset repo to push (default: {OUT_DATASET})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run all checks/stats but do not push to Hugging Face",
    )
    args = parser.parse_args()

    target_seconds = SEHAJ_TARGET_HOURS * 3600.0

    print(f"[prep] Loading Sehaj dataset: {SEHAJ_DATASET} (train/validation)")
    sehaj_train = load_dataset(SEHAJ_DATASET, split="train")
    sehaj_val = load_dataset(SEHAJ_DATASET, split="validation")
    print(f"[prep] Sehaj train rows: {len(sehaj_train)}")
    print(f"[prep] Sehaj val rows: {len(sehaj_val)}")

    indices = list(range(len(sehaj_train)))
    rng = random.Random(SEED)
    rng.shuffle(indices)

    chosen = []
    total_seconds = 0.0
    for idx in indices:
        row = sehaj_train[idx]
        dur = _to_seconds(row.get("duration_sec"))
        if dur <= 0:
            continue
        chosen.append(idx)
        total_seconds += dur
        if total_seconds >= target_seconds:
            break

    if total_seconds < target_seconds:
        raise RuntimeError(
            f"Could not reach {SEHAJ_TARGET_HOURS}h Sehaj target. Got {total_seconds/3600:.2f}h."
        )

    KEEP_COLS = ["audio", "transcription", "duration_sec", "source_dataset"]

    sehaj_selected = sehaj_train.select(chosen)
    sehaj_subset = sehaj_selected.map(_normalize_sehaj_row, num_proc=NUM_PROC).select_columns(KEEP_COLS)
    sehaj_val_norm = sehaj_val.map(_normalize_sehaj_row, num_proc=NUM_PROC).select_columns(KEEP_COLS)
    sehaj_hours = sum(sehaj_subset["duration_sec"]) / 3600.0
    print(
        f"[prep] Sehaj subset: {len(sehaj_subset)} rows, {sehaj_hours:.3f}h "
        f"(seed={SEED}, target={SEHAJ_TARGET_HOURS}h)"
    )

    print(f"[prep] Loading Kirtan dataset: {KIRTAN_DATASET} (train only for mix)")
    kirtan_train = load_dataset(KIRTAN_DATASET, split="train")
    print(f"[prep] Kirtan train rows (raw): {len(kirtan_train)}")

    kirtan_train_30 = kirtan_train.filter(
        lambda x: _to_seconds(x.get("duration")) <= KIRTAN_MAX_DURATION,
        num_proc=NUM_PROC,
    )
    if len(kirtan_train_30) == 0:
        raise RuntimeError("Kirtan <=30s filter resulted in empty train split.")

    kirtan_train_norm = kirtan_train_30.map(_normalize_kirtan_row, num_proc=NUM_PROC).select_columns(KEEP_COLS)

    # Hard assertions requested by the plan.
    if any(d > KIRTAN_MAX_DURATION for d in kirtan_train_norm["duration_sec"]):
        raise RuntimeError("Train split contains kirtan durations > 30s")

    train_mix = concatenate_datasets([sehaj_subset, kirtan_train_norm]).shuffle(seed=SEED)
    val_mix = sehaj_val_norm.shuffle(seed=SEED)

    _check_columns(train_mix, "train")
    _check_columns(val_mix, "validation")

    if any((not t) for t in train_mix["transcription"]):
        raise RuntimeError("Train split has empty transcription values")
    if any((not t) for t in val_mix["transcription"]):
        raise RuntimeError("Validation split has empty transcription values")

    train_hours = sum(train_mix["duration_sec"]) / 3600.0
    val_hours = sum(val_mix["duration_sec"]) / 3600.0
    k_train_hours = sum(kirtan_train_norm["duration_sec"]) / 3600.0

    print(f"[prep] Kirtan train <=30s: {len(kirtan_train_norm)} rows, {k_train_hours:.3f}h")
    print(f"[prep] Final train mix: {len(train_mix)} rows, {train_hours:.3f}h")
    print(f"[prep] Final validation (sehaj): {len(val_mix)} rows, {val_hours:.3f}h")

    dataset_dict = DatasetDict({"train": train_mix, "validation": val_mix})

    if args.dry_run:
        print("[prep] Dry-run complete. Skipping push.")
        return

    print(f"[prep] Pushing dataset to {args.out_dataset} ...")
    dataset_dict.push_to_hub(args.out_dataset)
    print("[prep] Done.")


if __name__ == "__main__":
    main()
