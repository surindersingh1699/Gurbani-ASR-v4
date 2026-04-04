"""
Phase 0 — Script 03: Build n-gram FAISS index.

The longest-running script. Supports checkpoint/resume — safe to kill and restart.

Reads:  data/processed/ngrams.json
Writes: index/sggs_ngram.faiss, index/ngram_meta.pkl

Usage:
  python scripts/03_build_ngram_index.py [--batch-size 128] [--chunk-size 10000]
"""
import argparse
import json
import pickle
import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DIM = 768
CHECKPOINT_DIR = "index/ngram_checkpoints"


def find_resume_point(chunk_size: int) -> int:
    """Find where to resume from based on existing checkpoints."""
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    checkpoints = sorted(Path(CHECKPOINT_DIR).glob("chunk_*.npy"))
    if not checkpoints:
        return 0
    last = checkpoints[-1]
    chunk_id = int(last.stem.split("_")[1])
    return (chunk_id + 1) * chunk_size


def main(batch_size: int = 128, chunk_size: int = 10000):
    ngrams = json.load(open("data/processed/ngrams.json", encoding="utf-8"))
    texts = [ng["text"] for ng in ngrams]
    total = len(texts)
    print(f"Total n-grams: {total:,}")

    start_idx = find_resume_point(chunk_size)
    if start_idx > 0:
        print(f"Resuming from n-gram {start_idx:,} / {total:,}")
    if start_idx >= total:
        print("All chunks already computed. Merging...")
    else:
        model = SentenceTransformer("google/muril-base-cased")

        for chunk_start in range(start_idx, total, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total)
            chunk_id = chunk_start // chunk_size
            ckpt_file = f"{CHECKPOINT_DIR}/chunk_{chunk_id:04d}.npy"

            if Path(ckpt_file).exists():
                continue

            chunk_texts = texts[chunk_start:chunk_end]
            n = len(chunk_texts)
            t0 = time.time()

            vecs = model.encode(
                chunk_texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

            np.save(ckpt_file, vecs.astype("float32"))

            elapsed = time.time() - t0
            rate = n / elapsed
            remaining = (total - chunk_end) / rate / 3600 if rate > 0 else 0
            print(
                f"Chunk {chunk_id:4d}: {chunk_end:>8,}/{total:,} "
                f"| {rate:.1f} seq/s | {elapsed:.1f}s | ETA {remaining:.2f}h"
            )

    # Merge all checkpoints into FAISS index
    print("\nMerging checkpoints into FAISS index...")
    Path("index").mkdir(exist_ok=True)
    idx = faiss.IndexFlatIP(DIM)

    for ckpt_file in sorted(Path(CHECKPOINT_DIR).glob("chunk_*.npy")):
        vecs = np.load(ckpt_file)
        idx.add(vecs)

    faiss.write_index(idx, "index/sggs_ngram.faiss")
    print(f"N-gram index: {idx.ntotal:,} vectors -> index/sggs_ngram.faiss")

    # N-gram metadata
    meta = {}
    for i, ng in enumerate(ngrams):
        meta[i] = {
            "text": ng["text"],
            "source_tuk_ids": ng["source_tuk_ids"],
        }
    pickle.dump(meta, open("index/ngram_meta.pkl", "wb"), protocol=4)
    print(f"Metadata: {len(meta):,} entries -> index/ngram_meta.pkl")

    # Cleanup checkpoints
    for f in Path(CHECKPOINT_DIR).glob("chunk_*.npy"):
        f.unlink()
    try:
        Path(CHECKPOINT_DIR).rmdir()
    except OSError:
        pass
    print("Checkpoints cleaned up.")

    print("\nScript 03 complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=10000)
    args = parser.parse_args()
    main(args.batch_size, args.chunk_size)
