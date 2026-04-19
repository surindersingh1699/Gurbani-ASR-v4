"""
Approach B: Forced alignment rebuild of SGPC live kirtan dataset.

Downloads the full video audio, identifies shabads via fuzzy matching
against auto-captions, runs faster-whisper (int8, CPU) for word-level
timestamps, segments at canonical pangati boundaries with padding, and
pushes a clean dataset to HuggingFace.

Usage:
    # Step 1: Download audio
    python fix_sgpc_forced_align.py --download-only

    # Step 2: Full pipeline (dry run)
    python fix_sgpc_forced_align.py --dry-run

    # Step 3: Process and push
    python fix_sgpc_forced_align.py

Requires:
    pip install datasets soundfile faster-whisper yt-dlp
"""

import argparse
import io
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import soundfile as sf

# --- Config ---
SRC_DATASET = "surindersinghssj/sgpc-amritsar-kirtan-live"
DST_DATASET = "surindersinghssj/sgpc-kirtan-forced-align"
TUKS_PATH = Path(__file__).parent.parent / "data" / "processed" / "tuks.json"
VIDEO_ID = "-MbIEo-8dCE"
AUDIO_DIR = Path("/root/sgpc_fix/audio")
FULL_AUDIO_PATH = AUDIO_DIR / "full_audio.wav"

MIN_DURATION = 1.5
MAX_DURATION = 30.0
BOUNDARY_PAD = 0.3     # seconds of padding at pangati boundaries
CONFIDENCE_THRESHOLD = 0.5
MATCH_THRESHOLD = 0.45
SAMPLE_RATE = 16000

# Model for forced alignment (faster-whisper format)
DEFAULT_MODEL = "small"  # faster-whisper built-in model


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
    tuks = [t for t in tuks if tuk["norm_len"] > 2]
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


def download_audio(video_id: str, output_path: Path, cookies_path: str = None):
    """Download audio from YouTube as 16kHz mono WAV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        print(f"[download] Audio already exists at {output_path}")
        return

    url = f"https://www.youtube.com/watch?v={video_id}"
    print(f"[download] Downloading audio from {url}...")

    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format", "wav",
        "--postprocessor-args", "ffmpeg:-ar 16000 -ac 1",
        "-o", str(output_path.with_suffix(".%(ext)s")),
        url,
    ]
    if cookies_path and os.path.exists(cookies_path):
        cmd.extend(["--cookies", cookies_path])

    subprocess.run(cmd, check=True)

    # yt-dlp may produce the file with original extension then convert
    if not output_path.exists():
        candidates = list(output_path.parent.glob("full_audio.*"))
        if candidates:
            actual = candidates[0]
            if actual.suffix != ".wav":
                subprocess.run([
                    "ffmpeg", "-i", str(actual),
                    "-ar", "16000", "-ac", "1",
                    str(output_path),
                ], check=True)
                actual.unlink()
            else:
                actual.rename(output_path)

    print(f"[download] Audio saved to {output_path}")
    info = sf.info(str(output_path))
    print(f"[download] Duration: {info.duration:.0f}s, SR: {info.samplerate}, "
          f"Channels: {info.channels}")


def identify_shabads_from_vtt(vtt_path: str, tuks: list[dict],
                               trigram_idx: dict) -> list[dict]:
    """Identify shabads from a VTT subtitle file (avoids HF download).

    Parses VTT, extracts caption text, fuzzy-matches against tuks to
    identify which shabads appear in the video.
    """
    print(f"\n[identify] Parsing VTT: {vtt_path}")
    captions = parse_vtt(vtt_path)
    print(f"[identify] Parsed {len(captions)} captions")

    # Concatenate all caption text
    full_text = " ".join(c["text"] for c in captions)
    full_normalized = normalize_gurbani_text(full_text)

    # Group tuks by shabad
    shabad_index = defaultdict(list)
    for tuk in tuks:
        shabad_index[tuk["shabad_id"]].append(tuk)

    # Score each shabad by how many of its pangatis appear in the caption text
    shabad_scores = {}
    full_words = full_normalized.split()

    for shabad_id, shabad_tuks in shabad_index.items():
        if len(shabad_tuks) < 2:
            continue

        pangati_hits = 0
        for tuk in shabad_tuks:
            if len(tuk["normalized"]) < 5:
                continue
            tuk_words = tuk["normalized"].split()
            tuk_len = len(tuk_words)

            # Sliding window match
            best_sub_score = 0
            for j in range(max(1, len(full_words) - tuk_len + 1)):
                window = " ".join(full_words[j:j + tuk_len])
                sub_score = SequenceMatcher(None, window, tuk["normalized"]).ratio()
                if sub_score > best_sub_score:
                    best_sub_score = sub_score
                if best_sub_score > 0.7:
                    break  # Good enough, stop searching

            if best_sub_score > 0.55:
                pangati_hits += 1

        if pangati_hits >= 2:
            shabad_scores[shabad_id] = {
                "pangati_hits": pangati_hits,
                "total_pangatis": len(shabad_tuks),
                "tuks": shabad_tuks,
                "first_line": shabad_tuks[0]["normalized"][:60],
            }

    ranked = sorted(shabad_scores.items(), key=lambda x: -x[1]["pangati_hits"])

    print(f"\n[identify] Found {len(ranked)} candidate shabads:")
    for shabad_id, info in ranked[:10]:
        print(f"  {shabad_id}: {info['pangati_hits']}/{info['total_pangatis']} pangatis")
        print(f"    First line: {info['first_line']}")

    confirmed = []
    for shabad_id, info in ranked:
        hit_rate = info["pangati_hits"] / max(info["total_pangatis"], 1)
        if info["pangati_hits"] >= 3 or (info["pangati_hits"] >= 2 and hit_rate > 0.3):
            confirmed.append({
                "shabad_id": shabad_id,
                "pangatis": [t["normalized"] for t in info["tuks"]],
                "tuks": info["tuks"],
                "pangati_hits": info["pangati_hits"],
            })

    print(f"[identify] Confirmed {len(confirmed)} shabads for alignment")
    return confirmed


def parse_vtt(vtt_path: str) -> list[dict]:
    """Parse a VTT file into list of {start, end, text} dicts."""
    import html

    with open(vtt_path, encoding="utf-8") as f:
        content = f.read()

    captions = []
    blocks = content.split("\n\n")

    for block in blocks:
        lines = block.strip().split("\n")
        timestamp_line = None
        text_lines = []

        for line in lines:
            if "-->" in line:
                timestamp_line = line
            elif timestamp_line and line.strip():
                text_lines.append(line.strip())

        if not timestamp_line or not text_lines:
            continue

        parts = timestamp_line.split("-->")
        start = _parse_ts(parts[0].strip())
        end = _parse_ts(parts[1].strip().split()[0])

        text = " ".join(text_lines)
        text = html.unescape(text)
        text = re.sub(r'<[^>]+>', '', text)  # Strip HTML tags
        text = re.sub(r'\s+', ' ', text).strip()

        if text:
            captions.append({"start": start, "end": end, "text": text})

    return captions


def _parse_ts(ts: str) -> float:
    """Parse VTT timestamp HH:MM:SS.mmm to seconds."""
    parts = ts.split(":")
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h = "0"
        m, s = parts
    else:
        return 0.0
    return int(h) * 3600 + int(m) * 60 + float(s)


def run_transcription(audio_path: Path, shabads: list[dict], model_size: str):
    """Run faster-whisper on the full audio to get word-level timestamps.

    Uses int8 quantization on CPU for memory efficiency.
    """
    from faster_whisper import WhisperModel

    print(f"\n[align] Loading faster-whisper model '{model_size}' (int8, CPU)...")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    # Build initial prompt from shabad first lines
    prompt_lines = [s["pangatis"][0] for s in shabads[:5]]
    initial_prompt = " ".join(prompt_lines)

    print(f"[align] Transcribing {audio_path}...")
    print(f"[align] Initial prompt: {initial_prompt[:100]}...")

    t0 = time.time()
    segments, info = model.transcribe(
        str(audio_path),
        language="pa",
        initial_prompt=initial_prompt,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=500,
        ),
    )

    # Collect all words with timestamps
    all_words = []
    seg_count = 0
    for seg in segments:
        seg_count += 1
        for word in (seg.words or []):
            all_words.append({
                "text": word.word.strip(),
                "start": word.start,
                "end": word.end,
                "confidence": word.probability,
            })
        if seg_count % 50 == 0:
            elapsed = time.time() - t0
            print(f"  Processed {seg_count} segments, {len(all_words)} words, "
                  f"{elapsed:.0f}s elapsed...")

    elapsed = time.time() - t0
    print(f"[align] Got {len(all_words)} timestamped words from {seg_count} segments "
          f"in {elapsed:.0f}s")
    return all_words


def match_words_to_pangatis(words: list[dict], shabads: list[dict]):
    """Match the timestamped word stream to canonical pangatis.

    Greedy sliding-window: for each position in the word stream,
    try to match the best canonical pangati.
    """
    if not words:
        return []

    # Build flat list of all canonical pangatis
    all_pangatis = []
    for shabad in shabads:
        for i, pangati in enumerate(shabad["pangatis"]):
            all_pangatis.append({
                "text": pangati,
                "shabad_id": shabad["shabad_id"],
                "pangati_idx": i,
                "word_count": len(pangati.split()),
            })

    word_texts = [w["text"] for w in words]
    matched_segments = []
    used_word_ranges = set()

    print(f"[match] Matching {len(words)} words against {len(all_pangatis)} pangatis...")
    t0 = time.time()

    i = 0
    while i < len(words):
        best_match = None
        best_score = 0.0
        best_span = (0, 0)

        for pangati in all_pangatis:
            n = pangati["word_count"]
            for window_size in [n, n + 1, n - 1, n + 2]:
                if window_size < 1 or i + window_size > len(words):
                    continue

                window_text = " ".join(word_texts[i:i + window_size])
                score = SequenceMatcher(
                    None,
                    normalize_gurbani_text(window_text),
                    pangati["text"]
                ).ratio()

                if score > best_score and score > MATCH_THRESHOLD:
                    best_score = score
                    best_match = pangati
                    best_span = (i, i + window_size)

        if best_match and best_score > MATCH_THRESHOLD:
            start_idx, end_idx = best_span
            span_range = set(range(start_idx, end_idx))
            if not span_range & used_word_ranges:
                used_word_ranges.update(span_range)

                seg_start = words[start_idx]["start"]
                seg_end = words[end_idx - 1]["end"]
                avg_confidence = np.mean([
                    words[j]["confidence"] for j in range(start_idx, end_idx)
                ])

                matched_segments.append({
                    "transcription": best_match["text"],
                    "shabad_id": best_match["shabad_id"],
                    "start_time": seg_start,
                    "end_time": seg_end,
                    "duration": seg_end - seg_start,
                    "match_score": round(best_score, 3),
                    "confidence": round(float(avg_confidence), 3),
                    "word_count": end_idx - start_idx,
                    "raw_text": " ".join(word_texts[start_idx:end_idx]),
                })
                i = end_idx
                continue

        i += 1

    elapsed = time.time() - t0
    matched_segments.sort(key=lambda x: x["start_time"])
    print(f"[match] Matched {len(matched_segments)} segments in {elapsed:.0f}s")
    return matched_segments


def extract_audio_segments(full_audio_path: Path, segments: list[dict],
                           pad: float = BOUNDARY_PAD):
    """Extract audio slices from the full video audio."""
    audio_data, sr = sf.read(str(full_audio_path))
    total_duration = len(audio_data) / sr
    print(f"[extract] Full audio: {total_duration:.0f}s at {sr}Hz")

    results = []
    drop_reasons = defaultdict(int)

    for seg in segments:
        start = max(0, seg["start_time"] - pad)
        end = min(total_duration, seg["end_time"] + pad)
        duration = end - start

        if duration < MIN_DURATION:
            drop_reasons["too_short"] += 1
            continue
        if duration > MAX_DURATION:
            drop_reasons["too_long"] += 1
            continue
        if seg["confidence"] < CONFIDENCE_THRESHOLD:
            drop_reasons["low_confidence"] += 1
            continue

        start_sample = int(start * sr)
        end_sample = int(end * sr)
        audio_slice = audio_data[start_sample:end_sample]

        seg["audio"] = {"array": audio_slice.astype(np.float32), "sampling_rate": sr}
        seg["duration"] = round(duration, 3)
        seg["start_time"] = round(start, 3)
        seg["end_time"] = round(end, 3)
        results.append(seg)

    print(f"[extract] Kept {len(results)}, dropped: {dict(drop_reasons)}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Forced alignment rebuild of SGPC kirtan")
    parser.add_argument("--dry-run", action="store_true", help="Print stats without pushing")
    parser.add_argument("--download-only", action="store_true", help="Only download audio")
    parser.add_argument("--tuks", default=str(TUKS_PATH), help="Path to tuks.json")
    parser.add_argument("--src", default=SRC_DATASET, help="Source HF dataset")
    parser.add_argument("--dst", default=DST_DATASET, help="Destination HF dataset")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Whisper model size")
    parser.add_argument("--audio", default=str(FULL_AUDIO_PATH), help="Path to full audio WAV")
    parser.add_argument("--video-id", default=VIDEO_ID, help="YouTube video ID")
    parser.add_argument("--vtt", default=None, help="Path to VTT subtitle file (skips HF download)")
    parser.add_argument("--cookies", default="/root/gurbani_kirtan_v3/cookies.txt",
                        help="Path to cookies.txt for yt-dlp")
    args = parser.parse_args()

    audio_path = Path(args.audio)

    # Step 1: Download audio
    download_audio(args.video_id, audio_path, args.cookies)
    if args.download_only:
        print("[done] Audio downloaded. Run again without --download-only for full pipeline.")
        return

    # Step 2: Load canonical text
    tuks = load_canonical_tuks(args.tuks)
    trigram_idx = build_trigram_index(tuks)

    # Step 3: Identify shabads from VTT (avoid HF dataset download)
    vtt_path = args.vtt
    if not vtt_path:
        # Try known location
        vtt_path = f"/root/gurbani_kirtan_v3/test_subs/{args.video_id}.pa.vtt"
    if not os.path.exists(vtt_path):
        print(f"[error] VTT file not found: {vtt_path}")
        print("  Provide --vtt path or place VTT file in expected location.")
        return

    shabads = identify_shabads_from_vtt(vtt_path, tuks, trigram_idx)
    if not shabads:
        print("[error] No shabads identified. Check VTT/tuks data.")
        return

    # Step 4: Transcribe with faster-whisper
    all_words = run_transcription(audio_path, shabads, args.model)

    # Step 5: Match words to pangatis
    matched = match_words_to_pangatis(all_words, shabads)

    # Step 6: Extract audio segments
    print(f"\n[extract] Extracting audio segments with {BOUNDARY_PAD}s padding...")
    final_segments = extract_audio_segments(audio_path, matched)

    # Print results
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"  Shabads identified:   {len(shabads)}")
    print(f"  Words transcribed:    {len(all_words)}")
    print(f"  Segments matched:     {len(matched)}")
    print(f"  After quality filter: {len(final_segments)}")

    if final_segments:
        durations = [s["duration"] for s in final_segments]
        scores = [s["match_score"] for s in final_segments]
        confidences = [s["confidence"] for s in final_segments]

        print(f"\n  Duration: mean={np.mean(durations):.1f}s, "
              f"median={np.median(durations):.1f}s, "
              f"total={sum(durations)/3600:.2f}h")
        print(f"  Match score: mean={np.mean(scores):.3f}, min={np.min(scores):.3f}")
        print(f"  Confidence:  mean={np.mean(confidences):.3f}, min={np.min(confidences):.3f}")

        unique_shabads = set(s["shabad_id"] for s in final_segments)
        print(f"  Unique shabads: {len(unique_shabads)}")

        print(f"\n  Sample segments:")
        for seg in final_segments[:10]:
            print(f"    [{seg['start_time']:.1f}-{seg['end_time']:.1f}s] "
                  f"score={seg['match_score']:.2f} conf={seg['confidence']:.2f}")
            print(f"      Text: {seg['transcription'][:70]}")
            print(f"      Raw:  {seg['raw_text'][:70]}")
            print()

    if args.dry_run:
        print("[dry-run] Skipping push. Use without --dry-run to push to HF.")
        return

    if not final_segments:
        print("[error] No segments survived filtering. Nothing to push.")
        return

    # Build and push dataset
    from datasets import Audio, Dataset

    print(f"\n[push] Building dataset with {len(final_segments)} segments...")

    audio_arrays = [s.pop("audio") for s in final_segments]
    for s in final_segments:
        s.pop("raw_text", None)
        s.pop("word_count", None)
        s["video_id"] = VIDEO_ID

    out_ds = Dataset.from_dict({
        **{k: [s[k] for s in final_segments] for k in final_segments[0].keys()},
        "audio": audio_arrays,
    })
    out_ds = out_ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))

    print(f"[push] Pushing to {args.dst}...")
    out_ds.push_to_hub(args.dst, split="train")
    print(f"[done] Dataset pushed to https://huggingface.co/datasets/{args.dst}")


if __name__ == "__main__":
    main()
