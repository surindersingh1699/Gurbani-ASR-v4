#!/usr/bin/env python3
"""Upload existing Gemini transcription results to HuggingFace.

Parses the dry-run output from fix_kirtan_captions.py, downloads matching
audio from HF rows API, and pushes as a playable dataset.
"""
import io
import json
import re
import sys
import urllib.request

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000


def parse_gemini_output(output_path: str) -> list[dict]:
    """Parse Gemini results from dry-run output file."""
    with open(output_path) as f:
        lines = f.readlines()

    results = []
    current = {}
    for line in lines:
        line = line.rstrip()

        # --- Segment N (Xs) ---
        m = re.match(r"--- Segment (\d+) \((\d+\.\d+)s\) ---", line)
        if m:
            if current:
                results.append(current)
            current = {
                "seg_idx": int(m.group(1)) - 1,
                "duration": float(m.group(2)),
            }
            continue

        # Gemini line
        m = re.match(r"\s+Gemini:\s+(.+?)\s+\[(gurbani|other|meaning|error)\]", line)
        if m and current:
            current["transcription"] = m.group(1).strip()
            current["type"] = m.group(2)

    if current:
        results.append(current)

    return results


def download_audio_segments(dataset_id: str, n_rows: int) -> list[dict]:
    """Download audio from HF rows API."""
    print(f"[hf] Fetching {n_rows} rows from {dataset_id}...")
    url = f"https://datasets-server.huggingface.co/rows?dataset={dataset_id}&config=default&split=train&offset=0&length={n_rows}"
    with urllib.request.urlopen(url, timeout=120) as resp:
        data = json.loads(resp.read())

    rows = data.get("rows", [])
    segments = []
    for i, item in enumerate(rows):
        row = item["row"]
        audio_info = row.get("audio", [])
        if isinstance(audio_info, list) and audio_info:
            audio_url = audio_info[0].get("src", "")
        elif isinstance(audio_info, dict):
            audio_url = audio_info.get("src", "")
        else:
            audio_url = ""

        if not audio_url:
            segments.append(None)
            continue

        try:
            with urllib.request.urlopen(audio_url, timeout=30) as audio_resp:
                audio_bytes = audio_resp.read()
            audio_array, sr = sf.read(io.BytesIO(audio_bytes))
            segments.append({"array": audio_array, "sampling_rate": sr})
        except Exception as e:
            print(f"  [skip] Row {i} audio failed: {e}")
            segments.append(None)

        if (i + 1) % 20 == 0:
            print(f"  Downloaded {i + 1}/{n_rows}...")

    print(f"[hf] Downloaded {sum(1 for s in segments if s)} audio segments")
    return segments


def main():
    output_path = sys.argv[1] if len(sys.argv) > 1 else "/private/tmp/claude-501/-Users-surindersingh-Developer-Gurbani-ASR-v4/tasks/bc032da.output"
    repo_id = sys.argv[2] if len(sys.argv) > 2 else "surindersinghssj/kirtan-gemini-test"
    dataset_id = "surindersinghssj/sgpc-amritsar-kirtan-live"

    # Parse existing results
    print(f"[parse] Reading results from {output_path}...")
    results = parse_gemini_output(output_path)
    print(f"[parse] Found {len(results)} Gemini transcriptions")

    # Filter to gurbani only (skip other/error)
    gurbani = [r for r in results if r["type"] == "gurbani" and r["transcription"]]
    print(f"[parse] {len(gurbani)} gurbani segments (filtered)")

    # Download audio
    n_audio = 100  # same number we tested
    audio_segments = download_audio_segments(dataset_id, n_audio)

    # Build dataset
    from datasets import Audio, Dataset

    valid_audio = []
    valid_text = []
    valid_dur = []
    valid_type = []

    for r in gurbani:
        idx = r["seg_idx"]
        if idx < len(audio_segments) and audio_segments[idx] is not None:
            valid_audio.append(audio_segments[idx])
            valid_text.append(r["transcription"])
            valid_dur.append(r["duration"])
            valid_type.append(r["type"])

    print(f"[build] {len(valid_audio)} segments with audio + transcription")

    ds = Dataset.from_dict({
        "audio": valid_audio,
        "transcription": valid_text,
        "duration_sec": valid_dur,
        "source": ["gemini-flash"] * len(valid_audio),
    })
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))

    print(f"[push] Pushing to {repo_id}...")
    ds.push_to_hub(repo_id, split="train")
    print(f"[done] https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
