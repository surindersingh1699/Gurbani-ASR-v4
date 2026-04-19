"""
Fix SGPC live kirtan dataset using Groq Whisper API.

Downloads full video audio, sends chunks to Groq's whisper-large-v3-turbo
with Gurbani context prompt (Mool Mantar etc.), gets word-level timestamps,
then matches against canonical SGGS text from tuks.json.

No GPU needed — runs entirely via API.

Usage:
    # Set API key
    export GROQ_API_KEY="gsk_..."

    # Test on a single video (dry run)
    python scripts/fix_sgpc_groq.py --dry-run --max-videos 1

    # Process all videos and push
    python scripts/fix_sgpc_groq.py

Requires:
    pip install groq datasets soundfile yt-dlp
"""

import argparse
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import soundfile as sf

SRC_DATASET = "surindersinghssj/sgpc-amritsar-kirtan-live"
DST_DATASET = "surindersinghssj/sgpc-kirtan-groq-aligned"
TUKS_PATH = Path(__file__).parent.parent / "data" / "processed" / "tuks.json"

# Groq limits: 25MB free tier. 16kHz mono FLAC ≈ 1MB/min, so 25min chunks max.
# We'll use 10-minute chunks to stay safe.
CHUNK_DURATION = 600  # seconds (10 min)
CHUNK_OVERLAP = 5     # seconds overlap between chunks for continuity
SAMPLE_RATE = 16000

# Segment output constraints
MIN_DURATION = 1.5
MAX_DURATION = 30.0
BOUNDARY_PAD = 0.15   # padding at segment boundaries
MATCH_THRESHOLD = 0.50
CONFIDENCE_THRESHOLD = 0.3

# Gurbani context prompt — Mool Mantar + common phrases.
# This tells Whisper what language/domain to expect.
GURBANI_PROMPT = (
    "ੴ ਸਤਿ ਨਾਮੁ ਕਰਤਾ ਪੁਰਖੁ ਨਿਰਭਉ ਨਿਰਵੈਰੁ ਅਕਾਲ ਮੂਰਤਿ ਅਜੂਨੀ ਸੈਭੰ ਗੁਰ ਪ੍ਰਸਾਦਿ ॥ "
    "ਵਾਹਿਗੁਰੂ ਜੀ ਕਾ ਖਾਲਸਾ ਵਾਹਿਗੁਰੂ ਜੀ ਕੀ ਫਤਹਿ ॥ "
    "ਸ੍ਰੀ ਭਗੌਤੀ ਜੀ ਸਹਾਇ ਵਾਰ ਸ੍ਰੀ ਭਗੌਤੀ ਜੀ ਕੀ ਪਾਤਸ਼ਾਹੀ ੧੦ ॥ "
    "ਪ੍ਰਿਥਮ ਭਗੌਤੀ ਸਿਮਰਿ ਕੈ ਗੁਰ ਨਾਨਕ ਲਈ ਧਿਆਇ ॥ "
    "ਰਾਗੁ ਆਸਾ ਮਹਲਾ ਗਉੜੀ ਸਿਰੀਰਾਗੁ ਭੈਰਉ ਬਿਲਾਵਲੁ ਧਨਾਸਰੀ "
    "ਗੁਰੂ ਗ੍ਰੰਥ ਸਾਹਿਬ ਜੀ ਸ਼ਬਦ ਕੀਰਤਨ ਹੁਕਮਨਾਮਾ ਅਰਦਾਸ"
)

# Groq model
GROQ_MODEL = "whisper-large-v3-turbo"


def get_groq_client():
    """Initialize Groq client."""
    try:
        from groq import Groq
    except ImportError:
        print("ERROR: pip install groq")
        sys.exit(1)

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERROR: Set GROQ_API_KEY environment variable")
        print("  Get a free key at https://console.groq.com/keys")
        sys.exit(1)

    return Groq(api_key=api_key)


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
    tuks = [t for t in tuks if t["norm_len"] > 2]
    print(f"[tuks] Loaded {len(tuks)} canonical pangatis")
    return tuks


def build_trigram_index(tuks: list[dict]) -> dict[str, list[int]]:
    """Build character trigram index for fast candidate retrieval."""
    index = defaultdict(list)
    for i, tuk in enumerate(tuks):
        text = tuk["normalized"]
        for j in range(len(text) - 2):
            trigram = text[j:j+3]
            if not index[trigram] or index[trigram][-1] != i:
                index[trigram].append(i)
    return index


def fuzzy_match_pangati(text: str, tuks: list[dict], trigram_idx: dict,
                        threshold: float = MATCH_THRESHOLD, top_k: int = 200):
    """Find the best matching canonical pangati using trigram pre-filtering."""
    normalized = normalize_gurbani_text(text)
    if len(normalized) < 3:
        return None, 0.0, None, None

    query_len = len(normalized)
    query_trigrams = set()
    for j in range(len(normalized) - 2):
        query_trigrams.add(normalized[j:j+3])

    candidate_scores = defaultdict(int)
    for tri in query_trigrams:
        for idx in trigram_idx.get(tri, []):
            tuk_len = tuks[idx]["norm_len"]
            if tuk_len < query_len * 0.3 or tuk_len > query_len * 3.0:
                continue
            candidate_scores[idx] += 1

    top_candidates = sorted(candidate_scores, key=candidate_scores.get, reverse=True)[:top_k]
    if not top_candidates:
        return None, 0.0, None, None

    best_score = 0.0
    best_text = None
    best_shabad = None
    best_tuk_id = None

    matcher = SequenceMatcher(None, normalized, "")

    for idx in top_candidates:
        tuk = tuks[idx]
        matcher.set_seq2(tuk["normalized"])
        if matcher.quick_ratio() < best_score:
            continue
        score = matcher.ratio()
        if score > best_score:
            best_score = score
            best_text = tuk["normalized"]
            best_shabad = tuk.get("shabad_id")
            best_tuk_id = tuk.get("tuk_id")

    # Also try consecutive pangati pairs
    checked_pairs = set()
    for idx in top_candidates[:50]:
        for delta in (0, -1):
            pair_idx = idx + delta
            if pair_idx < 0 or pair_idx >= len(tuks) - 1:
                continue
            if pair_idx in checked_pairs:
                continue
            checked_pairs.add(pair_idx)
            t1, t2 = tuks[pair_idx], tuks[pair_idx + 1]
            if t1.get("shabad_id") != t2.get("shabad_id"):
                continue
            combined = t1["normalized"] + " " + t2["normalized"]
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
                best_shabad = t1.get("shabad_id")
                best_tuk_id = t1.get("tuk_id")

    if best_score >= threshold:
        return best_text, best_score, best_shabad, best_tuk_id
    return None, best_score, None, None


def get_video_ids_from_dataset(max_videos: int = None) -> list[str]:
    """Get unique video IDs from the HF dataset."""
    from datasets import load_dataset

    print(f"[dataset] Getting video IDs from {SRC_DATASET}...")
    ds = load_dataset(SRC_DATASET, split="train", streaming=True)

    video_ids = []
    seen = set()
    for example in ds:
        vid = example.get("video_id", "")
        if vid and vid not in seen:
            seen.add(vid)
            video_ids.append(vid)
            print(f"  Found: {vid}")
            if max_videos and len(video_ids) >= max_videos:
                break

    print(f"[dataset] Found {len(video_ids)} unique videos")
    return video_ids


def download_video_audio(video_id: str, output_dir: Path,
                         cookies_path: str = None) -> Path:
    """Download audio from YouTube as 16kHz mono WAV."""
    output_path = output_dir / f"{video_id}.wav"
    if output_path.exists():
        print(f"[download] Audio exists: {output_path}")
        return output_path

    output_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://www.youtube.com/watch?v={video_id}"
    print(f"[download] Downloading {url}...")

    cmd = [
        "yt-dlp", "-x",
        "--audio-format", "wav",
        "--postprocessor-args", "ffmpeg:-ar 16000 -ac 1",
        "-o", str(output_dir / f"{video_id}.%(ext)s"),
        url,
    ]
    if cookies_path and os.path.exists(cookies_path):
        cmd.extend(["--cookies", cookies_path])

    subprocess.run(cmd, check=True, capture_output=True)

    # yt-dlp may produce different extension
    if not output_path.exists():
        for f in output_dir.glob(f"{video_id}.*"):
            if f.suffix != ".wav":
                subprocess.run([
                    "ffmpeg", "-y", "-i", str(f),
                    "-ar", "16000", "-ac", "1", str(output_path),
                ], check=True, capture_output=True)
                f.unlink()
                break

    info = sf.info(str(output_path))
    print(f"[download] {info.duration:.0f}s, {info.samplerate}Hz")
    return output_path


def transcribe_chunk_groq(client, audio_chunk: np.ndarray, sr: int,
                          chunk_offset: float = 0.0,
                          retries: int = 3) -> list[dict]:
    """Send audio chunk to Groq Whisper API, get word-level timestamps.

    Returns list of {word, start, end} dicts with absolute timestamps.
    """
    # Write chunk to temp FLAC file
    with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as tmp:
        sf.write(tmp.name, audio_chunk, sr, format="FLAC")
        tmp_path = tmp.name

    try:
        for attempt in range(retries):
            try:
                with open(tmp_path, "rb") as f:
                    response = client.audio.transcriptions.create(
                        file=f,
                        model=GROQ_MODEL,
                        response_format="verbose_json",
                        timestamp_granularities=["word", "segment"],
                        language="pa",
                        temperature=0.0,
                        prompt=GURBANI_PROMPT,
                    )

                words = []
                for w in (response.words or []):
                    words.append({
                        "word": w.word.strip(),
                        "start": round(w.start + chunk_offset, 3),
                        "end": round(w.end + chunk_offset, 3),
                    })
                return words

            except Exception as e:
                err_str = str(e)
                if "rate_limit" in err_str.lower() or "429" in err_str:
                    wait = 2 ** (attempt + 1)
                    print(f"    Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"    Groq API error (attempt {attempt+1}): {e}")
                    if attempt == retries - 1:
                        return []
                    time.sleep(1)
    finally:
        os.unlink(tmp_path)

    return []


def transcribe_full_audio(client, audio_path: Path) -> list[dict]:
    """Transcribe full audio file in chunks via Groq API.

    Returns all words with absolute timestamps.
    """
    audio_data, sr = sf.read(str(audio_path))
    total_duration = len(audio_data) / sr
    print(f"[transcribe] Full audio: {total_duration:.0f}s ({total_duration/3600:.1f}h)")

    all_words = []
    chunk_samples = CHUNK_DURATION * sr
    overlap_samples = CHUNK_OVERLAP * sr
    step_samples = chunk_samples - overlap_samples

    n_chunks = int(np.ceil(len(audio_data) / step_samples))
    print(f"[transcribe] Processing {n_chunks} chunks of {CHUNK_DURATION}s...")

    for i in range(n_chunks):
        start_sample = i * step_samples
        end_sample = min(start_sample + chunk_samples, len(audio_data))
        chunk = audio_data[start_sample:end_sample]
        chunk_offset = start_sample / sr

        chunk_dur = len(chunk) / sr
        print(f"  Chunk {i+1}/{n_chunks}: {chunk_offset:.0f}s - "
              f"{chunk_offset + chunk_dur:.0f}s ...", end=" ", flush=True)

        t0 = time.time()
        words = transcribe_chunk_groq(client, chunk, sr, chunk_offset)
        elapsed = time.time() - t0

        # Deduplicate words from overlapping regions
        if all_words and words:
            last_end = all_words[-1]["end"]
            words = [w for w in words if w["start"] > last_end - 0.1]

        all_words.extend(words)
        print(f"{len(words)} words, {elapsed:.1f}s")

        # Small delay to avoid rate limits
        time.sleep(0.3)

    print(f"[transcribe] Total: {len(all_words)} words")
    return all_words


def match_words_to_pangatis(words: list[dict], tuks: list[dict],
                            trigram_idx: dict) -> list[dict]:
    """Match word stream against canonical pangatis.

    Groups consecutive words into segments and fuzzy-matches each against tuks.
    """
    if not words:
        return []

    # Build segments from word stream: group by natural pauses
    segments = []
    current_words = []

    for w in words:
        if current_words:
            gap = w["start"] - current_words[-1]["end"]
            # Split on pauses > 0.8s or if segment > 15 words
            if gap > 0.8 or len(current_words) >= 15:
                text = " ".join(cw["word"] for cw in current_words)
                segments.append({
                    "text": text,
                    "start": current_words[0]["start"],
                    "end": current_words[-1]["end"],
                    "words": current_words,
                })
                current_words = []
        current_words.append(w)

    if current_words:
        text = " ".join(cw["word"] for cw in current_words)
        segments.append({
            "text": text,
            "start": current_words[0]["start"],
            "end": current_words[-1]["end"],
            "words": current_words,
        })

    print(f"[match] Built {len(segments)} natural segments from word stream")

    # Match each segment against canonical text
    matched = []
    for seg in segments:
        canonical, score, shabad_id, tuk_id = fuzzy_match_pangati(
            seg["text"], tuks, trigram_idx, threshold=MATCH_THRESHOLD
        )

        duration = seg["end"] - seg["start"]
        if canonical and MIN_DURATION <= duration <= MAX_DURATION:
            matched.append({
                "transcription": canonical,
                "raw_text": seg["text"],
                "start_time": round(seg["start"] - BOUNDARY_PAD, 3),
                "end_time": round(seg["end"] + BOUNDARY_PAD, 3),
                "duration": round(duration + 2 * BOUNDARY_PAD, 3),
                "match_score": round(score, 3),
                "shabad_id": shabad_id,
                "word_count": len(seg["words"]),
            })

    matched.sort(key=lambda x: x["start_time"])
    print(f"[match] Matched {len(matched)} segments against canonical text")
    return matched


def extract_audio_segments(audio_path: Path, segments: list[dict],
                           output_dir: Path) -> list[dict]:
    """Extract audio clips for matched segments."""
    audio_data, sr = sf.read(str(audio_path))
    total_duration = len(audio_data) / sr
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, seg in enumerate(segments):
        start = max(0, seg["start_time"])
        end = min(total_duration, seg["end_time"])

        start_sample = int(start * sr)
        end_sample = int(end * sr)
        audio_slice = audio_data[start_sample:end_sample]

        audio_path_out = output_dir / f"seg_{i:06d}.flac"
        sf.write(str(audio_path_out), audio_slice.astype(np.float32), sr, format="FLAC")

        seg["audio_path"] = str(audio_path_out)
        seg["duration"] = round(end - start, 3)
        seg["start_time"] = round(start, 3)
        seg["end_time"] = round(end, 3)
        results.append(seg)

    return results


def process_video(client, video_id: str, tuks: list[dict], trigram_idx: dict,
                  audio_dir: Path, output_dir: Path,
                  cookies_path: str = None) -> list[dict]:
    """Full pipeline for a single video."""
    print(f"\n{'='*60}")
    print(f"Processing video: {video_id}")
    print(f"{'='*60}")

    # Step 1: Download audio
    try:
        audio_path = download_video_audio(video_id, audio_dir, cookies_path)
    except Exception as e:
        print(f"[error] Download failed: {e}")
        return []

    # Step 2: Transcribe via Groq
    all_words = transcribe_full_audio(client, audio_path)
    if not all_words:
        print("[error] No words transcribed")
        return []

    # Step 3: Match against canonical text
    matched = match_words_to_pangatis(all_words, tuks, trigram_idx)
    if not matched:
        print("[error] No segments matched canonical text")
        return []

    # Step 4: Extract audio clips
    seg_dir = output_dir / video_id
    segments = extract_audio_segments(audio_path, matched, seg_dir)

    # Add video_id
    for seg in segments:
        seg["video_id"] = video_id

    # Print summary
    if segments:
        scores = [s["match_score"] for s in segments]
        durations = [s["duration"] for s in segments]
        print(f"\n  Video results:")
        print(f"    Segments: {len(segments)}")
        print(f"    Total audio: {sum(durations)/3600:.2f}h")
        print(f"    Match score: mean={np.mean(scores):.3f}, min={np.min(scores):.3f}")
        print(f"    Duration: mean={np.mean(durations):.1f}s, "
              f"median={np.median(durations):.1f}s")

        print(f"\n  Sample segments:")
        for seg in segments[:5]:
            print(f"    [{seg['start_time']:.1f}-{seg['end_time']:.1f}s] "
                  f"score={seg['match_score']:.2f}")
            print(f"      Canon: {seg['transcription'][:70]}")
            print(f"      Raw:   {seg['raw_text'][:70]}")

    return segments


def main():
    parser = argparse.ArgumentParser(description="Fix SGPC kirtan with Groq Whisper")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tuks", default=str(TUKS_PATH))
    parser.add_argument("--dst", default=DST_DATASET)
    parser.add_argument("--max-videos", type=int, default=None,
                        help="Limit number of videos to process")
    parser.add_argument("--video-ids", nargs="+", default=None,
                        help="Specific video IDs to process")
    parser.add_argument("--audio-dir", default="/tmp/sgpc_audio")
    parser.add_argument("--output-dir", default="/tmp/sgpc_segments")
    parser.add_argument("--cookies", default=None,
                        help="Path to cookies.txt for yt-dlp")
    args = parser.parse_args()

    # Init
    client = get_groq_client()
    tuks = load_canonical_tuks(args.tuks)
    trigram_idx = build_trigram_index(tuks)

    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)

    # Get video IDs
    if args.video_ids:
        video_ids = args.video_ids
    else:
        video_ids = get_video_ids_from_dataset(args.max_videos)

    # Process each video
    all_segments = []
    for vid in video_ids:
        segments = process_video(
            client, vid, tuks, trigram_idx,
            audio_dir, output_dir, args.cookies
        )
        all_segments.extend(segments)

    # Final summary
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"  Videos processed: {len(video_ids)}")
    print(f"  Total segments:   {len(all_segments)}")

    if all_segments:
        scores = [s["match_score"] for s in all_segments]
        durations = [s["duration"] for s in all_segments]
        print(f"  Total audio:      {sum(durations)/3600:.2f}h")
        print(f"  Match score:      mean={np.mean(scores):.3f}")
        for bucket in [0.5, 0.6, 0.7, 0.8, 0.9]:
            count = sum(1 for s in scores if s >= bucket)
            print(f"    >= {bucket}: {count} ({count/len(scores)*100:.0f}%)")

    if args.dry_run:
        print("\n[dry-run] Skipping push.")
        return

    if not all_segments:
        print("[error] No segments to push.")
        return

    # Build and push dataset
    from datasets import Audio, Dataset

    print(f"\n[push] Building dataset from {len(all_segments)} segments...")

    audio_paths = [s.pop("audio_path") for s in all_segments]
    for s in all_segments:
        s.pop("raw_text", None)
        s.pop("word_count", None)

    out_ds = Dataset.from_dict({
        **{k: [s[k] for s in all_segments] for k in all_segments[0].keys()},
        "audio": audio_paths,
    })
    out_ds = out_ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))

    print(f"[push] Pushing to {args.dst}...")
    out_ds.push_to_hub(args.dst, split="train")
    print(f"[done] https://huggingface.co/datasets/{args.dst}")

    # Cleanup
    import shutil
    shutil.rmtree(str(output_dir), ignore_errors=True)


if __name__ == "__main__":
    main()
