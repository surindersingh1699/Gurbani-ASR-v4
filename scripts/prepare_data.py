#!/usr/bin/env python3
"""
Pre-cache training data using all CPU cores.

Run ONCE before training. HF datasets caches the result to disk so
subsequent training runs skip the map entirely (loads in seconds).

Why a separate script: the training process initialises CUDA before
loading data, making fork-based multiprocessing unsafe. This script
keeps CUDA uninitialised so all CPU cores are available for the map.

Usage:
    python scripts/prepare_data.py
    SURT_MAP_WORKERS=16 python scripts/prepare_data.py
"""
import os
import sys
import time

# Must be set before ANY surt/torch import — prevents CUDA init so fork is safe.
os.environ["SURT_NO_CUDA_INIT"] = "1"
os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import WhisperProcessor  # noqa: E402 — CPU-only, no CUDA
from surt.config import (  # noqa: E402
    AUX_TRAIN_DATASET_NAME,
    BASE_MODEL,
    DATASET_NAME,
)
from surt.data import (  # noqa: E402
    get_kirtan_val_dataset,
    get_train_dataset,
    get_val_dataset,
)

n_proc = int(os.environ.get("SURT_MAP_WORKERS", os.cpu_count() or 4))
print(f"[prepare_data] Workers: {n_proc}")
print(f"[prepare_data] Cache dir: {os.environ['HF_HOME']}/datasets/")

# Processor only (feature extractor + tokenizer) — CPU-only, no model weights loaded.
processor = WhisperProcessor.from_pretrained(BASE_MODEL, language="punjabi", task="transcribe")


def _cache(label: str, fn, *args, **kwargs):
    """Run a cache-building step, reporting duration. Non-fatal on per-step failure."""
    print(f"\n[prepare_data] ---- {label} ----")
    t0 = time.time()
    try:
        out = fn(*args, **kwargs)
        elapsed = time.time() - t0
        size = f"{len(out):,}" if hasattr(out, "__len__") else "streaming"
        print(f"[prepare_data] {label}: cached ({size} examples) in {elapsed / 60:.1f} min")
        return out
    except Exception as e:
        elapsed = time.time() - t0
        print(f"[prepare_data] {label}: FAILED after {elapsed / 60:.1f} min: {e}")
        return None


# 1. Primary training dataset (includes aux mixing if AUX_TRAIN_PROBABILITY > 0).
_cache(
    f"TRAIN primary: {DATASET_NAME}",
    get_train_dataset,
    DATASET_NAME,
    processor=processor,
    streaming=False,
)

# 2. Sehaj validation split (used every eval).
_cache(
    f"VAL sehaj: {DATASET_NAME}",
    get_val_dataset,
    DATASET_NAME,
    processor,
)

# 3. Aux/kirtan dataset — used standalone for kirtan eval AND sampled into training
#    when AUX_TRAIN_PROBABILITY > 0. Always worth caching so training starts instantly.
if AUX_TRAIN_DATASET_NAME:
    _cache(
        f"AUX train-mix: {AUX_TRAIN_DATASET_NAME}",
        get_train_dataset,
        AUX_TRAIN_DATASET_NAME,
        processor=processor,
        streaming=False,
    )
    _cache(
        f"VAL kirtan: {AUX_TRAIN_DATASET_NAME}",
        get_kirtan_val_dataset,
        AUX_TRAIN_DATASET_NAME,
        processor,
    )

print("\n[prepare_data] All caches built. Training starts from disk on next run.")
