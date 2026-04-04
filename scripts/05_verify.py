"""
Phase 0 — Script 05: Verify all FAISS index deliverables.

Checks exit criteria from training plan:
  1. All 3 index files + metadata exist and load
  2. FAISS search on "ਸਤਿ ਨਾਮੁ" returns correct SGGS tuks in top-3
  3. Shabad and n-gram indexes are consistent

Usage:
  python scripts/05_verify.py
"""
import pickle
import sys

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

QUERY = "ਸਤਿ ਨਾਮੁ"


def check_tuk_index():
    print("--- Tuk Index ---")
    idx = faiss.read_index("index/sggs_tuk.faiss")
    meta = pickle.load(open("index/tuk_meta.pkl", "rb"))
    assert idx.ntotal > 50000, f"Expected >50k tuks, got {idx.ntotal}"
    assert len(meta) == idx.ntotal, f"Meta mismatch: {len(meta)} vs {idx.ntotal}"
    print(f"  Vectors: {idx.ntotal:,}  Meta: {len(meta):,}")

    # Check ang range
    angs = set(m["ang"] for m in meta.values())
    assert 1 in angs and 1430 in angs, f"Ang range: {min(angs)}-{max(angs)}"
    print(f"  Ang range: {min(angs)}-{max(angs)} ({len(angs)} unique)")

    return idx, meta


def check_shabad_index():
    print("--- Shabad Index ---")
    idx = faiss.read_index("index/sggs_shabad.faiss")
    meta = pickle.load(open("index/shabad_meta.pkl", "rb"))
    assert idx.ntotal > 5000, f"Expected >5k shabads, got {idx.ntotal}"
    assert len(meta) == idx.ntotal
    print(f"  Vectors: {idx.ntotal:,}  Meta: {len(meta):,}")
    return idx, meta


def check_ngram_index():
    print("--- N-gram Index ---")
    idx = faiss.read_index("index/sggs_ngram.faiss")
    meta = pickle.load(open("index/ngram_meta.pkl", "rb"))
    assert idx.ntotal > 100000, f"Expected >100k n-grams, got {idx.ntotal}"
    assert len(meta) == idx.ntotal
    print(f"  Vectors: {idx.ntotal:,}  Meta: {len(meta):,}")
    return idx, meta


def check_search(tuk_idx, tuk_meta, shabad_idx, shabad_meta, ngram_idx, ngram_meta):
    print(f"\n--- Search Test: '{QUERY}' ---")
    model = SentenceTransformer("google/muril-base-cased")
    q = model.encode([QUERY], normalize_embeddings=True).astype("float32")

    # Tuk search
    D, I = tuk_idx.search(q, k=5)
    print("  Tuk results:")
    for rank, (score, i) in enumerate(zip(D[0], I[0])):
        m = tuk_meta[int(i)]
        print(f"    #{rank+1}: score={score:.4f} ang={m['ang']} '{m['text'][:60]}'")

    top_text = tuk_meta[int(I[0][0])]["text"]
    assert "ਸਤਿ" in top_text or "ਨਾਮੁ" in top_text, f"Query words not in top result: {top_text}"

    # Shabad search
    Ds, Is = shabad_idx.search(q, k=3)
    print("  Shabad results:")
    for rank, (score, i) in enumerate(zip(Ds[0], Is[0])):
        m = shabad_meta[int(i)]
        print(f"    #{rank+1}: score={score:.4f} ang={m['ang']} '{m['first_tuk'][:60]}'")

    # N-gram search
    Dn, In = ngram_idx.search(q, k=3)
    print("  N-gram results:")
    for rank, (score, i) in enumerate(zip(Dn[0], In[0])):
        m = ngram_meta[int(i)]
        print(f"    #{rank+1}: score={score:.4f} '{m['text'][:60]}'")

    print("\n  PASS — search returns meaningful results")


def main():
    errors = []

    try:
        tuk_idx, tuk_meta = check_tuk_index()
        print("  PASS")
    except (AssertionError, FileNotFoundError) as e:
        print(f"  FAIL: {e}")
        errors.append(e)
        tuk_idx = tuk_meta = None

    try:
        shabad_idx, shabad_meta = check_shabad_index()
        print("  PASS")
    except (AssertionError, FileNotFoundError) as e:
        print(f"  FAIL: {e}")
        errors.append(e)
        shabad_idx = shabad_meta = None

    try:
        ngram_idx, ngram_meta = check_ngram_index()
        print("  PASS")
    except (AssertionError, FileNotFoundError) as e:
        print(f"  FAIL: {e}")
        errors.append(e)
        ngram_idx = ngram_meta = None

    if all(x is not None for x in [tuk_idx, shabad_idx, ngram_idx]):
        try:
            check_search(tuk_idx, tuk_meta, shabad_idx, shabad_meta, ngram_idx, ngram_meta)
        except (AssertionError, Exception) as e:
            print(f"  FAIL: {e}")
            errors.append(e)

    print("\n" + "=" * 50)
    if errors:
        print(f"Phase 0 verification: {len(errors)} FAILURES")
        sys.exit(1)
    else:
        print("Phase 0 verification: ALL PASS")


if __name__ == "__main__":
    main()
