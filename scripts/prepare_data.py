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
from surt.config import DATASET_NAME, BASE_MODEL  # noqa: E402
from surt.data import get_train_dataset  # noqa: E402

n_proc = int(os.environ.get("SURT_MAP_WORKERS", os.cpu_count() or 4))
print(f"[prepare_data] Caching {DATASET_NAME} with {n_proc} workers")
print(f"[prepare_data] Cache dir: {os.environ['HF_HOME']}/datasets/")

# Processor only (feature extractor + tokenizer) — CPU-only, no model weights loaded
processor = WhisperProcessor.from_pretrained(BASE_MODEL, language="punjabi", task="transcribe")

t0 = time.time()
ds = get_train_dataset(DATASET_NAME, processor=processor, streaming=False)
elapsed = time.time() - t0

print(f"[prepare_data] Done: {len(ds)} examples cached in {elapsed / 60:.1f} min")
print("[prepare_data] Training will load from cache instantly on next run.")
