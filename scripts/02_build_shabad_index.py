"""
Phase 0 — Script 02: Build shabad-level FAISS index.

Mean-pools tuk vectors by shabad_id — no re-encoding needed.

Reads:  data/processed/tuks.json, index/sggs_tuk.faiss
Writes: index/sggs_shabad.faiss, index/shabad_meta.pkl

Usage:
  python scripts/02_build_shabad_index.py
"""
import json
import pickle
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np

DIM = 768


def main():
    tuks = json.load(open("data/processed/tuks.json", encoding="utf-8"))
    tuk_meta = pickle.load(open("index/tuk_meta.pkl", "rb"))

    # Reconstruct all tuk vectors from FAISS (no re-encoding)
    tuk_idx = faiss.read_index("index/sggs_tuk.faiss")
    all_vecs = tuk_idx.reconstruct_n(0, tuk_idx.ntotal)
    print(f"Reconstructed {all_vecs.shape[0]:,} tuk vectors from FAISS")

    # Group tuk vectors by shabad_id
    shabad_vecs: dict[str, list] = defaultdict(list)
    shabad_tuks: dict[str, list[dict]] = defaultdict(list)
    for tuk in tuks:
        shabad_vecs[tuk["shabad_id"]].append(all_vecs[tuk["tuk_id"]])
        shabad_tuks[tuk["shabad_id"]].append(tuk)

    # Mean-pool and re-normalize
    shabad_ids = sorted(shabad_vecs.keys())
    pooled = []
    for sid in shabad_ids:
        mean_vec = np.mean(shabad_vecs[sid], axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm > 1e-9:
            mean_vec = mean_vec / norm
        pooled.append(mean_vec)
    pooled = np.array(pooled, dtype="float32")

    # Build FAISS index
    idx = faiss.IndexFlatIP(DIM)
    idx.add(pooled)
    faiss.write_index(idx, "index/sggs_shabad.faiss")
    print(f"Shabad index: {idx.ntotal:,} vectors -> index/sggs_shabad.faiss")

    # Shabad metadata: faiss_row_id -> {shabad_id, ang, raag, first_tuk, num_tuks}
    meta = {}
    for i, sid in enumerate(shabad_ids):
        first_tuk = shabad_tuks[sid][0]
        meta[i] = {
            "shabad_id": sid,
            "ang": first_tuk["ang"],
            "raag": first_tuk["raag"],
            "writer": first_tuk["writer"],
            "first_tuk": first_tuk["text"][:80],
            "num_tuks": len(shabad_tuks[sid]),
        }
    pickle.dump(meta, open("index/shabad_meta.pkl", "wb"), protocol=4)
    print(f"Metadata: {len(meta):,} entries -> index/shabad_meta.pkl")

    print("\nScript 02 complete.")


if __name__ == "__main__":
    main()
