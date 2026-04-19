"""
Approach A: Quick fix for SGPC live kirtan dataset.

Trims ~0.75s from the start of each audio clip (removes leaked previous-pangati word),
then fuzzy-matches the auto-caption text against canonical SGGS pangatis from tuks.json
and replaces with the correct text.

Uses streaming mode to avoid downloading the full dataset to disk.

Usage:
    python fix_sgpc_quick.py --dry-run          # inspect without pushing
    python fix_sgpc_quick.py                     # process and push to HF

Requires:
    pip install datasets soundfile huggingface_hub
"""

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import csv
import io
import tempfile

import numpy as np
import soundfile as sf
from datasets import Audio, Dataset, load_dataset

# --- Config ---
SRC_DATASET = "surindersinghssj/sgpc-amritsar-kirtan-live"
DST_DATASET = "surindersinghssj/sgpc-kirtan-quick-fix"
TUKS_PATH = Path(__file__).parent.parent / "data" / "processed" / "tuks.json"
MIN_DURATION = 1.5   # seconds
MAX_DURATION = 30.0  # seconds
TRIM_START = 0.75    # seconds to trim from the start of each clip
MATCH_THRESHOLD = 0.5
SAMPLE_RATE = 16000


def normalize_gurbani_text(text: str) -> str:
    """Strip non-spoken structural markers from Gurbani text."""
    text = re.sub(r'॥[੦-੯]+॥', '', text)
    text = re.sub(r'॥', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_canonical_tuks(tuks_path: str | Path) -> list[dict]:
    """Load tuks.json and normalize text for matching."""
    with open(tuks_path, encoding="utf-8") as f:
        tuks = json.load(f)

    for tuk in tuks:
        tuk["normalized"] = normalize_gurbani_text(tuk["text"])
        tuk["norm_len"] = len(tuk["normalized"])

    # Filter out empty after normalization
    tuks = [t for t in tuks if t["norm_len"] > 2]
    print(f"[tuks] Loaded {len(tuks)} canonical pangatis")
    return tuks


def build_trigram_index(tuks: list[dict]) -> dict[str, list[int]]:
    """Build character trigram → tuk indices for fast candidate retrieval."""
    index = defaultdict(list)
    for i, tuk in enumerate(tuks):
        text = tuk["normalized"]
        for j in range(len(text) - 2):
            trigram = text[j:j+3]
            # Only add each tuk once per trigram
            if not index[trigram] or index[trigram][-1] != i:
                index[trigram].append(i)
    print(f"[index] Built trigram index with {len(index)} unique trigrams")
    return index


def fuzzy_match_pangati(text: str, tuks: list[dict], trigram_idx: dict,
                        threshold: float = MATCH_THRESHOLD, top_k: int = 200):
    """Find the best matching canonical pangati using trigram pre-filtering.

    1. Extract trigrams from query text
    2. Score tuks by trigram overlap (fast)
    3. Run SequenceMatcher only on top-K candidates

    Returns (canonical_text, score, shabad_id, tuk_id) or (None, 0, None, None).
    """
    normalized = normalize_gurbani_text(text)
    if len(normalized) < 3:
        return None, 0.0, None, None

    query_len = len(normalized)

    # Extract query trigrams
    query_trigrams = set()
    for j in range(len(normalized) - 2):
        query_trigrams.add(normalized[j:j+3])

    # Score candidates by trigram overlap
    candidate_scores = defaultdict(int)
    for tri in query_trigrams:
        for idx in trigram_idx.get(tri, []):
            # Length filter: skip tuks whose length is way off
            tuk_len = tuks[idx]["norm_len"]
            if tuk_len < query_len * 0.3 or tuk_len > query_len * 3.0:
                continue
            candidate_scores[idx] += 1

    # Get top-K candidates by trigram overlap
    top_candidates = sorted(candidate_scores, key=candidate_scores.get, reverse=True)[:top_k]

    if not top_candidates:
        return None, 0.0, None, None

    best_score = 0.0
    best_text = None
    best_shabad = None
    best_tuk_id = None

    matcher = SequenceMatcher(None, normalized, "")

    # Check single pangatis
    for idx in top_candidates:
        tuk = tuks[idx]
        matcher.set_seq2(tuk["normalized"])
        # Early exit with quick_ratio
        if matcher.quick_ratio() < best_score:
            continue
        score = matcher.ratio()
        if score > best_score:
            best_score = score
            best_text = tuk["normalized"]
            best_shabad = tuk["shabad_id"]
            best_tuk_id = tuk["tuk_id"]

    # Also try consecutive pangati pairs from top candidates
    checked_pairs = set()
    for idx in top_candidates[:50]:  # Limit pair checks
        for delta in (0, -1):  # Check (idx, idx+1) and (idx-1, idx)
            pair_idx = idx + delta
            if pair_idx < 0 or pair_idx >= len(tuks) - 1:
                continue
            if pair_idx in checked_pairs:
                continue
            checked_pairs.add(pair_idx)

            t1, t2 = tuks[pair_idx], tuks[pair_idx + 1]
            if t1["shabad_id"] != t2["shabad_id"]:
                continue
            combined = t1["normalized"] + " " + t2["normalized"]
            # Length filter for pairs
            comb_len = len(combined)
            if comb_len < query_len * 0.3 or comb_len > query_len * 3.0:
                continue
            matcher.set_seq2(combined)
            if matcher.quick_ratio() < best_score:
                continue
            score = matcher.ratio()
            if score > best_score:
                best_score = score
                best_text = combined
                best_shabad = t1["shabad_id"]
                best_tuk_id = t1["tuk_id"]

    if best_score >= threshold:
        return best_text, best_score, best_shabad, best_tuk_id
    return None, best_score, None, None


def trim_audio_start(audio_array: np.ndarray, sr: int, trim_seconds: float) -> np.ndarray:
    """Trim the first `trim_seconds` from an audio array."""
    trim_samples = int(trim_seconds * sr)
    if trim_samples >= len(audio_array):
        return audio_array
    return audio_array[trim_samples:]


def main():
    parser = argparse.ArgumentParser(description="Quick fix SGPC kirtan dataset")
    parser.add_argument("--dry-run", action="store_true", help="Print stats without pushing")
    parser.add_argument("--tuks", default=str(TUKS_PATH), help="Path to tuks.json")
    parser.add_argument("--src", default=SRC_DATASET, help="Source HF dataset")
    parser.add_argument("--dst", default=DST_DATASET, help="Destination HF dataset")
    parser.add_argument("--trim", type=float, default=TRIM_START, help="Seconds to trim from start")
    parser.add_argument("--threshold", type=float, default=MATCH_THRESHOLD, help="Min match score")
    parser.add_argument("--audio-dir", default="/root/sgpc_fix/audio_out",
                        help="Directory to save trimmed audio FLAC files")
    args = parser.parse_args()

    # Create audio output directory
    audio_dir = Path(args.audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Load canonical text
    tuks = load_canonical_tuks(args.tuks)
    trigram_idx = build_trigram_index(tuks)

    # Load source dataset in STREAMING mode (no full download)
    # Use decode=False to avoid torchcodec dependency — decode audio manually
    print(f"\n[dataset] Streaming {args.src}...")
    ds = load_dataset(args.src, split="train", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    # Process each segment via streaming
    results = []
    drop_reasons = defaultdict(int)
    sample_matches = []
    total_seen = 0
    t0 = time.time()

    for example in ds:
        total_seen += 1
        duration = example.get("duration", 0)

        # Duration filter
        if duration < MIN_DURATION or duration > MAX_DURATION:
            drop_reasons["duration_filter"] += 1
            if total_seen % 1000 == 0:
                elapsed = time.time() - t0
                print(f"  Seen {total_seen}, kept {len(results)}, "
                      f"elapsed {elapsed:.0f}s...")
            continue

        # Decode audio manually with soundfile (avoids torchcodec)
        audio_bytes = example["audio"]["bytes"]
        try:
            audio_data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        except Exception as e:
            drop_reasons["audio_decode_error"] += 1
            continue

        # Resample to target if needed
        if sr != SAMPLE_RATE:
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=SAMPLE_RATE)
            sr = SAMPLE_RATE

        # Trim start
        audio_trimmed = trim_audio_start(audio_data, sr, args.trim)
        new_duration = len(audio_trimmed) / sr

        # Skip if too short after trimming
        if new_duration < MIN_DURATION:
            drop_reasons["too_short_after_trim"] += 1
            continue

        # Fuzzy match against canonical text
        canonical, score, shabad_id, tuk_id = fuzzy_match_pangati(
            example["gurmukhi_text"], tuks, trigram_idx, threshold=args.threshold
        )

        if canonical is None:
            drop_reasons["low_match_score"] += 1
            if len(sample_matches) < 5:
                sample_matches.append({
                    "auto": example["gurmukhi_text"],
                    "best_score": score,
                    "status": "DROPPED",
                })
            continue

        # Record sample matches for inspection
        if len(sample_matches) < 20:
            sample_matches.append({
                "auto": example["gurmukhi_text"],
                "canonical": canonical,
                "score": score,
                "status": "KEPT",
            })

        # Save audio to disk as FLAC (avoids OOM from holding 22K arrays in RAM)
        seg_idx = len(results)
        audio_path = audio_dir / f"seg_{seg_idx:06d}.flac"
        sf.write(str(audio_path), audio_trimmed, sr, format="FLAC")

        results.append({
            "audio_path": str(audio_path),
            "transcription": canonical,
            "gurmukhi_text_original": example["gurmukhi_text"],
            "duration": new_duration,
            "start_time": example["start_time"] + args.trim,
            "end_time": example["end_time"],
            "match_score": round(score, 3),
            "shabad_id": shabad_id,
            "video_id": example.get("video_id", ""),
        })

        if total_seen % 200 == 0:
            elapsed = time.time() - t0
            print(f"  Seen {total_seen}, kept {len(results)}, "
                  f"elapsed {elapsed:.0f}s...")

    elapsed_total = time.time() - t0

    # Print results
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"  Total segments seen:  {total_seen}")
    print(f"  After text matching:  {len(results)}")
    print(f"  Dropped total:        {total_seen - len(results)}")
    print(f"  Processing time:      {elapsed_total:.0f}s")
    print(f"\n  Drop reasons:")
    for reason, count in sorted(drop_reasons.items()):
        print(f"    {reason}: {count}")

    # Print score distribution
    if results:
        scores = [r["match_score"] for r in results]
        print(f"\n  Match score distribution:")
        print(f"    Mean:   {np.mean(scores):.3f}")
        print(f"    Median: {np.median(scores):.3f}")
        print(f"    Min:    {np.min(scores):.3f}")
        print(f"    Max:    {np.max(scores):.3f}")
        for bucket in [0.5, 0.6, 0.7, 0.8, 0.9]:
            count = sum(1 for s in scores if s >= bucket)
            print(f"    >= {bucket}: {count} ({count/len(scores)*100:.0f}%)")

    # Print sample matches
    print(f"\n  Sample matches:")
    for m in sample_matches[:20]:
        status = m["status"]
        auto = m["auto"][:60]
        if status == "KEPT":
            canonical = m["canonical"][:60]
            print(f"    [{m['score']:.2f}] {auto}")
            print(f"         → {canonical}")
        else:
            print(f"    [DROPPED {m['best_score']:.2f}] {auto}")
        print()

    if args.dry_run:
        print("[dry-run] Skipping push. Use without --dry-run to push to HF.")
        return

    if not results:
        print("[error] No segments survived filtering. Nothing to push.")
        return

    # Build and push dataset from disk files (avoids OOM)
    print(f"\n[push] Building dataset from {len(results)} audio files on disk...")

    # Build dataset with audio paths — datasets will read files lazily
    audio_paths = [r.pop("audio_path") for r in results]
    out_ds = Dataset.from_dict({
        **{k: [r[k] for r in results] for k in results[0].keys()},
        "audio": audio_paths,
    })
    out_ds = out_ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))

    print(f"[push] Pushing to {args.dst}...")
    out_ds.push_to_hub(args.dst, split="train")
    print(f"[done] Dataset pushed to https://huggingface.co/datasets/{args.dst}")

    # Clean up temp audio files
    print("[cleanup] Removing temp audio files...")
    import shutil
    shutil.rmtree(str(audio_dir), ignore_errors=True)
    print("[cleanup] Done.")


if __name__ == "__main__":
    main()
