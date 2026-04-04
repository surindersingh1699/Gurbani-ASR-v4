"""
Phase 0 — Script 01: Build tuk-level FAISS index.

Reads:  data/processed/tuks.json
Writes: index/sggs_tuk.faiss, index/tuk_meta.pkl

Usage:
  python scripts/01_build_tuk_index.py [--batch-size 64]
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


def main(batch_size: int = 64):
    tuks = json.load(open("data/processed/tuks.json", encoding="utf-8"))
    texts = [t["text"] for t in tuks]
    print(f"Encoding {len(texts):,} tuks with MuRIL (batch_size={batch_size})...")

    model = SentenceTransformer("google/muril-base-cased")

    t0 = time.time()
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    elapsed = time.time() - t0
    print(f"  Encoded in {elapsed:.1f}s ({len(texts) / elapsed:.1f} seq/s)")

    Path("index").mkdir(exist_ok=True)

    idx = faiss.IndexFlatIP(DIM)
    idx.add(vecs.astype("float32"))
    faiss.write_index(idx, "index/sggs_tuk.faiss")
    print(f"  Tuk index: {idx.ntotal:,} vectors -> index/sggs_tuk.faiss")

    meta = {}
    for t in tuks:
        meta[t["tuk_id"]] = {
            "shabad_id": t["shabad_id"],
            "ang": t["ang"],
            "raag": t["raag"],
            "writer": t["writer"],
            "text": t["text"],
        }
    pickle.dump(meta, open("index/tuk_meta.pkl", "wb"), protocol=4)
    print(f"  Metadata: {len(meta):,} entries -> index/tuk_meta.pkl")

    # Quick sanity check
    q = model.encode(["ਸਤਿ ਨਾਮੁ"], normalize_embeddings=True)
    D, I = idx.search(q.astype("float32"), k=3)
    print(f"\n  Sanity check — query: 'ਸਤਿ ਨਾਮੁ'")
    for rank, (score, i) in enumerate(zip(D[0], I[0])):
        m = meta[int(i)]
        print(f"    #{rank + 1}: score={score:.4f} ang={m['ang']} text='{m['text'][:60]}'")

    print("\nScript 01 complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    main(args.batch_size)
