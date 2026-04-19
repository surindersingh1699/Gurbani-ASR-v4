#!/usr/bin/env python3
"""
SikhNet Kirtan → Whisper Training Data Pipeline.

Downloads kirtan audio from SikhNet, uses the known shabad text from our SGGS
database (tuks.json), sends ONE Gemini API call per audio file to get timestamps
per tuk, then extracts aligned audio segments (<30s each).

Best practices for cost/speed:
- One Gemini call per full audio file (not per segment)
- Known text provided = alignment task (easier, more accurate than transcription)
- Segments extracted locally with soundfile (no extra API calls)
- Batch multiple shabads per API call when audio is short

Usage:
    python scripts/sikhnet_kirtan_align.py \
        --track-url "https://www.sikhnet.com/gurbani/audio/play/46489" \
        --shabad-search "ਸਤਿਗੁਰੁ ਪਿਆਰਾ" \
        --push-hf surindersinghssj/kirtan-sikhnet-test \
        --dry-run

    python scripts/sikhnet_kirtan_align.py \
        --audio-file path/to/kirtan.mp3 \
        --shabad-id 63E \
        --push-hf surindersinghssj/kirtan-sikhnet-test
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import tempfile
import time
import urllib.request

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
TUKS_PATH = "data/processed/tuks.json"


# ---------------------------------------------------------------------------
# SGGS text lookup
# ---------------------------------------------------------------------------

def load_tuks(path: str = TUKS_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_shabad(tuks: list[dict], shabad_id: str | None = None,
                search_text: str | None = None) -> list[dict]:
    """Find shabad tuks by ID or text search."""
    if shabad_id:
        matches = [t for t in tuks if t["shabad_id"] == shabad_id]
        if matches:
            return matches

    if search_text:
        # Search for tuks containing the text
        matches = [t for t in tuks if search_text in t["text"]]
        if matches:
            # Get full shabad from first match
            shabad_id = matches[0]["shabad_id"]
            return [t for t in tuks if t["shabad_id"] == shabad_id]

    return []


# ---------------------------------------------------------------------------
# Audio download
# ---------------------------------------------------------------------------

def download_audio(url: str, output_path: str | None = None) -> str:
    """Download audio from URL, convert to 16kHz mono WAV."""
    print(f"[download] Fetching audio from {url[:80]}...")

    # Follow redirects manually for sikhnet
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        audio_bytes = resp.read()
        final_url = resp.url
    print(f"[download] Got {len(audio_bytes) / 1024 / 1024:.1f} MB from {final_url[:60]}...")

    # Save to temp file
    suffix = ".mp3" if ".mp3" in final_url else ".wav"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(audio_bytes)
    tmp.close()

    # Convert to 16kHz mono WAV using ffmpeg
    out_path = output_path or tmp.name.replace(suffix, "_16k.wav")
    import subprocess
    subprocess.run([
        "ffmpeg", "-y", "-i", tmp.name,
        "-ar", "16000", "-ac", "1", out_path
    ], capture_output=True, check=True)

    os.unlink(tmp.name)
    print(f"[download] Converted to 16kHz WAV: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Gemini alignment (ONE call per full audio)
# ---------------------------------------------------------------------------

def init_gemini(api_key: str | None = None):
    """Initialize Gemini client using new google.genai SDK."""
    from google import genai
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY")
    return genai.Client(api_key=key)


def align_audio_with_text(client, audio_path: str, shabad_tuks: list[dict]) -> list[dict]:
    """Send full audio + known text → Gemini returns timestamps per tuk.

    ONE API call for the entire audio file. Cost-efficient.
    """
    from google.genai import types

    # Read audio
    audio_array, sr = sf.read(audio_path)
    duration = len(audio_array) / sr
    print(f"[align] Audio: {duration:.1f}s ({duration / 60:.1f}min), {len(shabad_tuks)} tuks")

    # Encode as WAV bytes
    buf = io.BytesIO()
    sf.write(buf, audio_array.astype(np.float32), sr, format="WAV", subtype="PCM_16")
    audio_bytes = buf.getvalue()

    # Build tuk list for prompt
    tuk_lines = []
    for i, t in enumerate(shabad_tuks):
        tuk_lines.append(f'{i}: "{t["text"]}"')
    tuk_text = "\n".join(tuk_lines)

    prompt = f"""You are a Gurbani expert. This audio is a Sikh kirtan recording.
The exact shabad text (tuks) is provided below for SPELLING REFERENCE ONLY.

Known shabad tuks (for spelling accuracy):
{tuk_text}

YOUR TASK: Listen to the audio and identify each singing phrase — the EXACT
Gurmukhi words being sung in each time window. Kirtan singers do NOT sing a
full tuk at once. They sing a few words, pause, repeat them, then move to the
next few words. Each segment you return must contain ONLY the words actually
sung in that time window.

RULES:
- Each segment = one continuous singing phrase (a few words, NOT a full tuk)
- "text" = ONLY the words sung in that segment (use the tuks above for correct spelling)
- Singers repeat phrases — mark EACH repetition as a separate segment
- Skip instrumental-only sections (tabla, harmonium interludes)
- Each segment duration: 3–25 seconds (split longer phrases, merge very short ones)
- Timestamps MUST be in total seconds from audio start (e.g. 125.0 = 2 min 5 sec)
- Return AT MOST 80 segments total. Prefer longer, meaningful phrases over tiny fragments.
- Do NOT split a single continuous repetition into sub-segments. One phrase = one segment.

Return a JSON array of objects:
- "tuk_index": which tuk (0-indexed) these words come from
- "start_sec": when this phrase starts being sung
- "end_sec": when this phrase ends
- "text": the EXACT Gurmukhi words sung (subset of the tuk, correct spelling)

Return ONLY the JSON array, nothing else."""

    for attempt in range(3):
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
                    max_output_tokens=16384,
                ),
            )
            # Parse response
            raw = response.text.strip()
            print(f"[align] Raw response ({len(raw)} chars): {raw[:300]}...")
            segments = json.loads(raw)
            if isinstance(segments, dict):
                segments = segments.get("segments", segments.get("results", []))
            if not isinstance(segments, list):
                print(f"[align] Unexpected response type: {type(segments)}")
                return []
            print(f"[align] Gemini returned {len(segments)} segments")
            return segments
        except Exception as e:
            err = str(e)
            if "429" in err and attempt < 2:
                wait = 2 ** (attempt + 1)
                print(f"[align] Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            print(f"[align] Error: {err[:200]}")
            return []


# ---------------------------------------------------------------------------
# Extract audio segments
# ---------------------------------------------------------------------------

def fix_timestamp(t: float, total_dur: float) -> float:
    """Fix timestamps that might be in min.sec format instead of seconds."""
    # If timestamp looks like min.sec (e.g., 2.30 meaning 2min 30sec)
    # Heuristic: if all timestamps are < total_dur/60, they're probably in minutes
    return t


def fix_all_timestamps(alignments: list[dict], total_dur: float) -> list[dict]:
    """Detect and fix min.sec vs seconds format."""
    if not alignments:
        return alignments

    max_end = max(float(a.get("end_sec", 0)) for a in alignments)

    # If max timestamp is way less than total duration, probably in min.sec format
    if max_end < total_dur / 10:
        print(f"[fix] Timestamps look like min.sec format (max={max_end:.1f}, dur={total_dur:.0f}s). Converting...")
        for a in alignments:
            for key in ("start_sec", "end_sec"):
                val = float(a.get(key, 0))
                minutes = int(val)
                seconds = (val - minutes) * 100
                a[key] = minutes * 60 + seconds
    return alignments


def extract_segments(audio_path: str, alignments: list[dict],
                     max_duration: float = 30.0,
                     min_duration: float = 1.5) -> list[dict]:
    """Extract audio segments from alignment timestamps."""
    audio_array, sr = sf.read(audio_path)
    total_dur = len(audio_array) / sr

    # Fix timestamps if needed
    alignments = fix_all_timestamps(alignments, total_dur)

    segments = []
    for seg in alignments:
        start = float(seg.get("start_sec", 0))
        end = float(seg.get("end_sec", 0))
        text = seg.get("text", "")
        tuk_idx = seg.get("tuk_index", -1)
        duration = end - start

        # Validate
        if duration < min_duration or duration > max_duration:
            print(f"  [skip] tuk {tuk_idx}: {duration:.1f}s (out of range)")
            continue
        if start < 0 or end > total_dur + 0.5:
            print(f"  [skip] tuk {tuk_idx}: timestamps out of bounds ({start:.1f}-{end:.1f})")
            continue
        if not text.strip():
            continue

        # Extract audio slice
        start_sample = max(0, int(start * sr))
        end_sample = min(len(audio_array), int(end * sr))
        audio_slice = audio_array[start_sample:end_sample]

        segments.append({
            "audio": {"array": audio_slice, "sampling_rate": sr},
            "transcription": text.strip(),
            "duration_sec": round(duration, 3),
            "tuk_index": tuk_idx,
            "start_sec": round(start, 3),
            "end_sec": round(end, 3),
            "source": "sikhnet-aligned",
        })

    print(f"[extract] {len(segments)} valid segments (dropped {len(alignments) - len(segments)})")
    return segments


# ---------------------------------------------------------------------------
# Push to HF
# ---------------------------------------------------------------------------

def push_to_hf(segments: list[dict], repo_id: str):
    """Push aligned segments to HuggingFace."""
    from datasets import Audio, Dataset

    if not segments:
        print("[push] No segments to push.")
        return

    ds = Dataset.from_dict({
        "audio": [s["audio"] for s in segments],
        "transcription": [s["transcription"] for s in segments],
        "duration_sec": [s["duration_sec"] for s in segments],
        "tuk_index": [s["tuk_index"] for s in segments],
        "source": [s["source"] for s in segments],
    })
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))

    print(f"[push] Pushing {len(segments)} segments to {repo_id}...")
    ds.push_to_hub(repo_id, split="train")
    print(f"[push] Done: https://huggingface.co/datasets/{repo_id}")


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_results(segments: list[dict], shabad_tuks: list[dict]):
    """Print alignment results."""
    print(f"\n{'=' * 80}")
    print(f"{'ALIGNMENT RESULTS':^80}")
    print(f"{'=' * 80}")

    for s in segments:
        idx = s["tuk_index"]
        print(f"\n  [{s['start_sec']:6.1f}s - {s['end_sec']:6.1f}s] ({s['duration_sec']:.1f}s) tuk {idx}")
        print(f"    {s['transcription'][:80]}")

    total_dur = sum(s["duration_sec"] for s in segments)
    print(f"\n{'=' * 80}")
    print(f"  Total: {len(segments)} segments, {total_dur:.1f}s ({total_dur / 60:.1f}min)")
    print(f"  Tuks covered: {len(set(s['tuk_index'] for s in segments))}/{len(shabad_tuks)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SikhNet kirtan → aligned training data via Gemini"
    )
    # Audio source
    parser.add_argument("--track-url", help="SikhNet audio/play URL (e.g. .../play/46489)")
    parser.add_argument("--audio-file", help="Local audio file path")

    # Shabad identification
    parser.add_argument("--shabad-id", help="SikhiToTheMax shabad ID (e.g. 63E)")
    parser.add_argument("--shabad-search", help="Search text to find shabad (Gurmukhi)")

    # Output
    parser.add_argument("--push-hf", help="HuggingFace repo to push results")
    parser.add_argument("--dry-run", action="store_true", help="Print results only")

    # Config
    parser.add_argument("--tuks", default=TUKS_PATH, help="Path to tuks.json")
    parser.add_argument("--gemini-api-key", help="Gemini API key")

    args = parser.parse_args()

    if not args.track_url and not args.audio_file:
        parser.error("Need --track-url or --audio-file")
    if not args.shabad_id and not args.shabad_search:
        parser.error("Need --shabad-id or --shabad-search")

    # Find shabad text
    print("[tuks] Loading SGGS text...")
    tuks = load_tuks(args.tuks)
    shabad_tuks = find_shabad(tuks, args.shabad_id, args.shabad_search)
    if not shabad_tuks:
        print("[error] Shabad not found.")
        sys.exit(1)
    print(f"[tuks] Found shabad: {shabad_tuks[0]['shabad_id']}, "
          f"{len(shabad_tuks)} tuks, Ang {shabad_tuks[0]['ang']}, "
          f"{shabad_tuks[0]['raag']}")
    for t in shabad_tuks:
        print(f"  {t['text'][:70]}")

    # Get audio
    if args.audio_file:
        audio_path = args.audio_file
    else:
        audio_path = download_audio(args.track_url)

    # Init Gemini
    model = init_gemini(args.gemini_api_key)

    # Align — ONE API call
    print(f"\n[align] Sending full audio to Gemini (1 API call)...")
    alignments = align_audio_with_text(model, audio_path, shabad_tuks)

    if not alignments:
        print("[error] No alignments returned.")
        sys.exit(1)

    # Extract segments
    segments = extract_segments(audio_path, alignments)

    # Display
    print_results(segments, shabad_tuks)

    # Push
    if not args.dry_run and args.push_hf:
        push_to_hf(segments, args.push_hf)
    elif args.dry_run:
        print("\n[dry-run] Skipping HF push.")

    print("\n[done]")


if __name__ == "__main__":
    main()
