#!/usr/bin/env python3
"""
Compare Gemini Flash vs Groq Whisper for kirtan transcription quality.

Loads audio segments from an existing HuggingFace dataset, sends each to
both APIs, and pushes results to two separate HF repos for side-by-side
comparison (FLAC audio + transcription).

Usage:
    # Dry run — test 10 rows, print results
    python scripts/fix_kirtan_captions.py \
        --hf-dataset surindersinghssj/sgpc-amritsar-kirtan-live \
        --test-rows 10 --dry-run

    # Process ~1 hour and push to HF for comparison
    python scripts/fix_kirtan_captions.py \
        --hf-dataset surindersinghssj/sgpc-amritsar-kirtan-live \
        --test-rows 800 \
        --push-gemini surindersinghssj/kirtan-gemini-test \
        --push-groq surindersinghssj/kirtan-groq-test

    # Filter to one video
    python scripts/fix_kirtan_captions.py \
        --hf-dataset surindersinghssj/sgpc-amritsar-kirtan-live \
        --video-id "p0jn2BgORAw" \
        --test-rows 500 --dry-run

Requires:
    pip install google-generativeai soundfile numpy datasets requests
    Environment: GEMINI_API_KEY, GROQ_API_KEY
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000

# ---------------------------------------------------------------------------
# HF dataset loading
# ---------------------------------------------------------------------------


def load_local_audio(audio_path: str, segment_duration: float = 20.0) -> list[dict]:
    """Load a local audio file and split into segments for transcription."""
    audio_array, sr = sf.read(audio_path)
    total_dur = len(audio_array) / sr
    print(f"[local] Loaded {audio_path}: {total_dur:.1f}s ({total_dur / 60:.1f}min), {sr}Hz")

    segments = []
    offset_samples = 0
    seg_samples = int(segment_duration * sr)

    while offset_samples < len(audio_array):
        end_samples = min(offset_samples + seg_samples, len(audio_array))
        chunk = audio_array[offset_samples:end_samples]
        dur = len(chunk) / sr
        if dur < 1.5:
            break
        segments.append({
            "audio": {"array": chunk, "sampling_rate": sr},
            "text": "",
            "video_id": Path(audio_path).stem,
            "duration": dur,
        })
        offset_samples = end_samples

    print(f"[local] Split into {len(segments)} segments of ~{segment_duration:.0f}s")
    return segments


def load_hf_segments(dataset_id: str, video_id: str | None, n_rows: int, offset: int = 0) -> list[dict]:
    """Load segments from HF dataset using rows API (fast, no full download)."""
    import urllib.request

    print(f"[hf] Fetching {n_rows} rows from {dataset_id} (offset={offset}) via API...")

    # Use HF rows API — downloads only the rows we need
    fetch = n_rows * 3 if video_id else n_rows
    fetch = min(fetch, 300)
    url = f"https://datasets-server.huggingface.co/rows?dataset={dataset_id}&config=default&split=train&offset={offset}&length={fetch}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())

    rows = data.get("rows", [])
    print(f"[hf] Got {len(rows)} rows from API")

    segments = []
    for item in rows:
        row = item["row"]
        if video_id and row.get("video_id") != video_id:
            continue

        # Audio comes as [{"src": url, "type": ...}] from API — need to download
        audio_info = row.get("audio", [])
        if isinstance(audio_info, list) and audio_info:
            audio_url = audio_info[0].get("src", "")
        elif isinstance(audio_info, dict):
            audio_url = audio_info.get("src", "")
        else:
            audio_url = ""

        if not audio_url:
            continue

        try:
            with urllib.request.urlopen(audio_url, timeout=30) as audio_resp:
                audio_bytes = audio_resp.read()
            audio_array, sr = sf.read(io.BytesIO(audio_bytes))
        except Exception as e:
            print(f"  [skip] Audio download failed: {e}")
            continue

        segments.append({
            "audio": {"array": audio_array, "sampling_rate": sr},
            "text": row.get("text", row.get("transcription", "")),
            "video_id": row.get("video_id", "unknown"),
            "duration": len(audio_array) / sr,
        })

        if len(segments) >= n_rows:
            break

        if len(segments) % 20 == 0:
            print(f"  Downloaded {len(segments)}/{n_rows} audio segments...")

    total_dur = sum(s["duration"] for s in segments)
    print(f"[hf] Loaded {len(segments)} segments ({total_dur / 3600:.2f}h)"
          + (f" for video_id={video_id}" if video_id else ""))
    return segments


# ---------------------------------------------------------------------------
# Gemini Flash: audio → transcription
# ---------------------------------------------------------------------------


def init_gemini(api_key: str | None = None):
    """Initialize Gemini client using new google.genai SDK."""
    from google import genai

    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY env variable")
    return genai.Client(api_key=key)


def transcribe_with_gemini(client, audio_array: np.ndarray, sr: int, raw_text: str = "") -> dict:
    """Send audio to Gemini Flash, get Gurmukhi transcription."""
    from google.genai import types

    buf = io.BytesIO()
    audio_np = np.array(audio_array, dtype=np.float32)
    sf.write(buf, audio_np, sr, format="WAV", subtype="PCM_16")
    audio_bytes = buf.getvalue()

    hint = f"\nExisting (possibly incorrect) caption for reference: {raw_text}" if raw_text else ""

    prompt = f"""You are a Gurbani/Punjabi text expert. This audio is from a Sikh kirtan recording
at Sri Harmandir Sahib (Golden Temple).

Listen carefully and transcribe ONLY the Gurmukhi text being sung.
- Fix any Gurmukhi spelling (correct matras: ੀ/ਿ/ੁ/ੂ/ੇ/ੈ/ੋ/ੌ, consonants, nukta)
- If this is Gurbani (shabad text being sung), set type to "gurbani"
- If this is Punjabi meaning/translation, set type to "meaning"
- If this is an announcement or non-singing audio, set type to "other"
- Only transcribe what is actually being SUNG, not background sounds
{hint}

Return ONLY a JSON object, no other text:
{{"text": "gurmukhi transcription", "type": "gurbani"}}"""

    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=[
                    types.Part.from_text(text=prompt),
                    types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
                ],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                ),
            )
            parsed = json.loads(response.text.strip())
            return {
                "text": parsed.get("text", ""),
                "type": parsed.get("type", "gurbani"),
                "error": None,
            }
        except Exception as e:
            err = str(e)
            if "429" in err and attempt < 4:
                wait = 2 ** (attempt + 1)
                print(f"    [gemini] Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            return {"text": "", "type": "error", "error": err}


# ---------------------------------------------------------------------------
# Groq Whisper: audio → transcription
# ---------------------------------------------------------------------------


def transcribe_with_groq(audio_array: np.ndarray, sr: int, groq_api_key: str) -> dict:
    """Send audio to Groq Whisper API, get transcription."""
    # Encode as WAV in memory
    buf = io.BytesIO()
    audio_np = np.array(audio_array, dtype=np.float32)
    sf.write(buf, audio_np, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {groq_api_key}"},
            files={"file": ("audio.wav", buf, "audio/wav")},
            data={
                "model": "whisper-large-v3-turbo",
                "language": "pa",
                "response_format": "verbose_json",
            },
            timeout=60,
        )
        response.raise_for_status()
        result = response.json()
        return {
            "text": result.get("text", ""),
            "type": "gurbani",  # Groq doesn't classify, assume gurbani
            "error": None,
            "segments": result.get("segments", []),
        }
    except Exception as e:
        return {"text": "", "type": "error", "error": str(e)}


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------


def process_segments(
    segments: list[dict],
    gemini_model=None,
    groq_api_key: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """Process segments through both APIs. Returns (gemini_results, groq_results)."""
    gemini_results = []
    groq_results = []

    for i, seg in enumerate(segments):
        audio_array = seg["audio"]["array"]
        sr = seg["audio"]["sampling_rate"]
        raw_text = seg.get("text", seg.get("transcription", ""))
        video_id = seg.get("video_id", "unknown")
        duration = seg.get("duration", len(audio_array) / sr)

        base = {
            "original_text": raw_text,
            "duration_sec": round(float(duration), 3),
            "video_id": video_id,
        }

        # Gemini
        if gemini_model:
            g = transcribe_with_gemini(gemini_model, audio_array, sr, raw_text)
            gemini_results.append({
                **base,
                "transcription": g["text"],
                "type": g["type"],
                "error": g["error"],
            })
            if g["error"]:
                print(f"  [{i + 1}] Gemini error: {g['error'][:60]}")

        # Groq
        if groq_api_key:
            gr = transcribe_with_groq(audio_array, sr, groq_api_key)
            groq_results.append({
                **base,
                "transcription": gr["text"],
                "type": gr["type"],
                "error": gr["error"],
            })
            if gr["error"]:
                print(f"  [{i + 1}] Groq error: {gr['error'][:60]}")

        # Progress
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Processed {i + 1}/{len(segments)} segments...")

        # Rate limit — paid tier 2000 RPM, minimal delay
        time.sleep(0.5)

    return gemini_results, groq_results


# ---------------------------------------------------------------------------
# Push to HF
# ---------------------------------------------------------------------------


def push_to_hf(
    results: list[dict],
    audio_arrays: list[dict],
    repo_id: str,
    source_label: str,
):
    """Push results to HuggingFace as playable FLAC dataset."""
    from datasets import Audio, Dataset

    # Filter out errors and non-gurbani
    valid_idx = [
        i for i, r in enumerate(results)
        if r.get("error") is None and r.get("transcription", "").strip()
    ]

    if not valid_idx:
        print(f"[push] No valid segments for {repo_id}, skipping.")
        return

    print(f"[push] Building {repo_id}: {len(valid_idx)} segments "
          f"(dropped {len(results) - len(valid_idx)} errors/empty)...")

    ds = Dataset.from_dict({
        "audio": [audio_arrays[i] for i in valid_idx],
        "transcription": [results[i]["transcription"] for i in valid_idx],
        "original_text": [results[i]["original_text"] for i in valid_idx],
        "duration_sec": [results[i]["duration_sec"] for i in valid_idx],
        "video_id": [results[i]["video_id"] for i in valid_idx],
        "source": [source_label] * len(valid_idx),
    })
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))

    print(f"[push] Pushing to {repo_id}...")
    ds.push_to_hub(repo_id, split="train")
    print(f"[push] Done: https://huggingface.co/datasets/{repo_id}")


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def print_results(
    gemini_results: list[dict],
    groq_results: list[dict],
    n: int = 20,
):
    """Print side-by-side comparison."""
    print(f"\n{'=' * 90}")
    print(f"{'COMPARISON':^90}")
    print(f"{'=' * 90}")

    count = min(n, len(gemini_results), len(groq_results)) if groq_results else min(n, len(gemini_results))

    for i in range(count):
        g = gemini_results[i] if i < len(gemini_results) else None
        gr = groq_results[i] if groq_results and i < len(groq_results) else None

        print(f"\n--- Segment {i + 1} ({gemini_results[i]['duration_sec']:.1f}s) ---")
        print(f"  Original:  {gemini_results[i]['original_text'][:80]}")
        if g:
            marker = "✓" if g["type"] == "gurbani" else "✗"
            print(f"  Gemini:    {g['transcription'][:80]}  [{g['type']}] {marker}")
        if gr:
            print(f"  Groq:      {gr['transcription'][:80]}")

        # Show if they agree
        if g and gr and g["transcription"] and gr["transcription"]:
            match = g["transcription"].strip() == gr["transcription"].strip()
            print(f"  Match:     {'SAME' if match else 'DIFFERENT'}")

    # Summary stats
    print(f"\n{'=' * 90}")
    print(f"{'SUMMARY':^90}")
    print(f"{'=' * 90}")

    if gemini_results:
        g_ok = sum(1 for r in gemini_results if not r["error"])
        g_gurbani = sum(1 for r in gemini_results if r["type"] == "gurbani")
        g_meaning = sum(1 for r in gemini_results if r["type"] == "meaning")
        g_changed = sum(1 for r in gemini_results
                        if r["transcription"].strip() != r["original_text"].strip()
                        and r["transcription"])
        print(f"  Gemini:  {g_ok}/{len(gemini_results)} ok | "
              f"{g_gurbani} gurbani | {g_meaning} meaning | {g_changed} text changed")

    if groq_results:
        gr_ok = sum(1 for r in groq_results if not r["error"])
        gr_nonempty = sum(1 for r in groq_results if r["transcription"].strip())
        gr_changed = sum(1 for r in groq_results
                         if r["transcription"].strip() != r["original_text"].strip()
                         and r["transcription"])
        print(f"  Groq:    {gr_ok}/{len(groq_results)} ok | "
              f"{gr_nonempty} non-empty | {gr_changed} text changed")

    total_dur = sum(r["duration_sec"] for r in gemini_results) if gemini_results else 0
    print(f"  Total audio: {total_dur / 3600:.2f}h ({total_dur:.0f}s)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Compare Gemini Flash vs Groq Whisper for kirtan transcription"
    )
    parser.add_argument("--hf-dataset",
                        help="Source HuggingFace dataset ID")
    parser.add_argument("--audio-file",
                        help="Local audio file (split into segments automatically)")
    parser.add_argument("--segment-duration", type=float, default=20.0,
                        help="Segment duration in seconds for local audio (default: 20)")
    parser.add_argument("--video-id", help="Filter to specific video_id")
    parser.add_argument("--test-rows", type=int, default=10,
                        help="Number of segments to process (default: 10)")
    parser.add_argument("--offset", type=int, default=0,
                        help="Row offset in dataset (skip first N rows)")

    parser.add_argument("--gemini-api-key", help="Gemini API key (or GEMINI_API_KEY env)")
    parser.add_argument("--groq-api-key", help="Groq API key (or GROQ_API_KEY env)")
    parser.add_argument("--skip-gemini", action="store_true", help="Skip Gemini, only run Groq")
    parser.add_argument("--skip-groq", action="store_true", help="Skip Groq, only run Gemini")

    parser.add_argument("--push-gemini", help="HF repo for Gemini results")
    parser.add_argument("--push-groq", help="HF repo for Groq results")
    parser.add_argument("--dry-run", action="store_true", help="Print results, don't push to HF")

    args = parser.parse_args()

    # Resolve API keys
    gemini_key = args.gemini_api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    groq_key = args.groq_api_key or os.environ.get("GROQ_API_KEY")

    run_gemini = not args.skip_gemini and bool(gemini_key)
    run_groq = not args.skip_groq and bool(groq_key)

    if not run_gemini and not run_groq:
        parser.error("Need at least one API key. Set GEMINI_API_KEY and/or GROQ_API_KEY")

    print(f"[config] Gemini: {'ON' if run_gemini else 'OFF'}")
    print(f"[config] Groq:   {'ON' if run_groq else 'OFF'}")
    print(f"[config] Rows:   {args.test_rows}")

    if not args.hf_dataset and not args.audio_file:
        parser.error("Need --hf-dataset or --audio-file")

    # Load segments
    if args.audio_file:
        segments = load_local_audio(args.audio_file, args.segment_duration)
    else:
        segments = load_hf_segments(args.hf_dataset, args.video_id, args.test_rows, args.offset)
    if not segments:
        print("[error] No segments loaded.")
        sys.exit(1)

    # Init APIs
    gemini_model = None
    if run_gemini:
        print("[gemini] Initializing Gemini 2.5 Flash Lite...")
        gemini_model = init_gemini(gemini_key)

    # Process
    print(f"\n[process] Running transcription on {len(segments)} segments...")
    gemini_results, groq_results = process_segments(
        segments,
        gemini_model=gemini_model,
        groq_api_key=groq_key if run_groq else None,
    )

    # Display
    print_results(gemini_results, groq_results)

    # Push to HF
    if not args.dry_run:
        audio_arrays = [seg["audio"] for seg in segments]

        if args.push_gemini and gemini_results:
            push_to_hf(gemini_results, audio_arrays, args.push_gemini, "gemini-flash")

        if args.push_groq and groq_results:
            push_to_hf(groq_results, audio_arrays, args.push_groq, "groq-whisper")

    print("\n[done]")


if __name__ == "__main__":
    main()
