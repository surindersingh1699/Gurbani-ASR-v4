#!/usr/bin/env python3
"""
Pre-cache training data using all CPU cores.

Run ONCE before training. HF datasets caches the result to disk, so
subsequent training runs skip the map entirely (loads in seconds).

Why a separate script: the training process initialises CUDA before
loading data, making fork-based multiprocessing unsafe. This script
runs with zero CUDA context, so all CPU cores are available.

Usage:
    python scripts/prepare_data.py
    SURT_MAP_WORKERS=16 python scripts/prepare_data.py
"""
import os
import sys
import time

# Ensure HF_HOME is set before any HF import
os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from surt.config import DATASET_NAME  # noqa: E402 – after sys.path fix

n_proc = int(os.environ.get("SURT_MAP_WORKERS", os.cpu_count() or 4))
print(f"[prepare_data] Caching {DATASET_NAME} with {n_proc} workers...")
print(f"[prepare_data] Cache dir: {os.environ['HF_HOME']}/datasets/")

t0 = time.time()

# Import here so torch (if transitively pulled in) is not yet CUDA-initialised
from surt.data import get_train_dataset  # noqa: E402

ds = get_train_dataset(DATASET_NAME)
elapsed = time.time() - t0

print(f"[prepare_data] Done: {len(ds)} examples in {elapsed/60:.1f} min")
print("[prepare_data] Training will load from cache instantly on next run.")
