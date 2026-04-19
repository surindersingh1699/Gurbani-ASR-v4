#!/usr/bin/env python3
"""
Kirtan transcription: pre-split audio into fixed 20s clips, batch 5 minutes
of clips per Gemini API call, get back text only.

We control timestamps (fixed 20s boundaries). Gemini only transcribes.
~3 API calls for 12 min audio.

Usage:
    python scripts/kirtan_bulk_transcribe.py \
        --audio-file /tmp/kirtan_test/kx-UWANhX2Y.wav \
        --push-hf surindersinghssj/kirtan-bulk-test \
        --dry-run
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000


def init_gemini(api_key: str | None = None):
    from google import genai
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY")
    return genai.Client(api_key=key)


def split_audio(audio_path: str, segment_secs: float = 20.0) -> list[dict]:
    """Pre-split audio into fixed-length clips. Returns list of clip metadata."""
    audio_array, sr = sf.read(audio_path)
    total_dur = len(audio_array) / sr
    segment_samples = int(segment_secs * sr)

    clips = []
    offset = 0
    while offset < len(audio_array):
        end = min(offset + segment_samples, len(audio_array))
        clip = audio_array[offset:end]
        clip_dur = len(clip) / sr

        if clip_dur < 2.0:
            break

        # Encode as WAV bytes
        buf = io.BytesIO()
        sf.write(buf, clip.astype(np.float32), sr, format="WAV", subtype="PCM_16")

        clips.append({
            "audio_bytes": buf.getvalue(),
            "audio_array": clip,
            "start_sec": round(offset / sr, 3),
            "end_sec": round(end / sr, 3),
            "duration_sec": round(clip_dur, 3),
        })
        offset = end

    print(f"[split] {len(clips)} clips of ~{segment_secs:.0f}s "
          f"from {total_dur:.1f}s ({total_dur / 60:.1f}min) audio")
    return clips


def transcribe_batch(client, clips: list[dict], batch_idx: int,
                     n_batches: int) -> list[str]:
    """Send a batch of audio clips in one API call, get back text for each."""
    from google.genai import types

    n = len(clips)
    prompt = f"""You are a Gurbani/Punjabi transcription expert. This is Sikh kirtan.

I am sending {n} audio clips (each ~20 seconds). For each clip:
- Transcribe ONLY the Gurmukhi text being sung (correct matras and spelling)
- If a clip is instrumental only (no singing), return empty string ""
- If the clip contains Punjabi meaning/explanation (not Gurbani), return empty string ""

Return a JSON array of exactly {n} strings, one per clip in order.
Example: ["ਗੁਰਮੁਖੀ ਟੈਕਸਟ", "ਹੋਰ ਟੈਕਸਟ", "", "ਗੁਰਮੁਖੀ ਟੈਕਸਟ"]
Return ONLY the JSON array."""

    # Build contents: prompt + all audio clips
    contents = [types.Part.from_text(text=prompt)]
    for i, clip in enumerate(clips):
        contents.append(types.Part.from_text(text=f"Clip {i + 1}:"))
        contents.append(types.Part.from_bytes(
            data=clip["audio_bytes"], mime_type="audio/wav"))

    time_range = f"{clips[0]['start_sec']:.0f}-{clips[-1]['end_sec']:.0f}s"
    print(f"[batch {batch_idx + 1}/{n_batches}] {n} clips ({time_range})")

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                ),
            )
            raw = response.text.strip()
            texts = json.loads(raw)

            if isinstance(texts, dict):
                texts = list(texts.values())
            if not isinstance(texts, list):
                print(f"    [error] Expected list, got {type(texts).__name__}")
                return [""] * n

            # Pad or truncate to match clip count
            if len(texts) < n:
                print(f"    [warn] Got {len(texts)} texts for {n} clips, padding")
                texts.extend([""] * (n - len(texts)))
            elif len(texts) > n:
                print(f"    [warn] Got {len(texts)} texts for {n} clips, truncating")
                texts = texts[:n]

            filled = sum(1 for t in texts if t.strip())
            print(f"  → {filled}/{n} clips have text")
            return texts

        except json.JSONDecodeError:
            raw = response.text.strip()
            print(f"    [error] JSON parse failed ({len(raw)} chars)")
            if attempt < 2:
                print(f"    [retry] attempt {attempt + 2}/3")
                time.sleep(1)
                continue
            return [""] * n
        except Exception as e:
            err = str(e)
            if "429" in err and attempt < 2:
                wait = 2 ** (attempt + 1)
                print(f"    [rate-limit] waiting {wait}s...")
                time.sleep(wait)
                continue
            print(f"    [error] {err[:200]}")
            return [""] * n

    return [""] * n


def transcribe_all(client, clips: list[dict],
                   batch_size: int = 15) -> list[dict]:
    """Batch clips and transcribe. Returns segments with text + audio."""
    n_batches = max(1, (len(clips) + batch_size - 1) // batch_size)

    segments = []
    for b in range(n_batches):
        batch_clips = clips[b * batch_size:(b + 1) * batch_size]
        texts = transcribe_batch(client, batch_clips, b, n_batches)

        for clip, text in zip(batch_clips, texts):
            text = text.strip() if isinstance(text, str) else ""
            if not text:
                continue
            segments.append({
                "audio": {"array": clip["audio_array"],
                          "sampling_rate": SAMPLE_RATE},
                "transcription": text,
                "duration_sec": clip["duration_sec"],
                "start_sec": clip["start_sec"],
                "end_sec": clip["end_sec"],
                "source": "gemini-batch",
            })

        if b < n_batches - 1:
            time.sleep(0.5)

    print(f"[total] {len(segments)} segments with text "
          f"(from {len(clips)} clips, {n_batches} API calls)")
    return segments


def push_to_hf(segments: list[dict], repo_id: str):
    from datasets import Audio, Dataset

    if not segments:
        print("[push] No segments.")
        return

    ds = Dataset.from_dict({
        "audio": [s["audio"] for s in segments],
        "transcription": [s["transcription"] for s in segments],
        "duration_sec": [s["duration_sec"] for s in segments],
        "source": [s["source"] for s in segments],
    })
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))

    print(f"[push] Pushing {len(segments)} segments to {repo_id}...")
    ds.push_to_hub(repo_id, split="train")
    print(f"[push] Done: https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Kirtan transcription: pre-split + batch Gemini calls")
    parser.add_argument("--audio-file", required=True, help="Audio file path")
    parser.add_argument("--segment-duration", type=float, default=20.0,
                        help="Clip duration in seconds (default: 20)")
    parser.add_argument("--batch-size", type=int, default=15,
                        help="Clips per API call (default: 15 = ~5min)")
    parser.add_argument("--push-hf", help="HF repo to push results")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print results, don't push")
    args = parser.parse_args()

    client = init_gemini()
    t0 = time.time()

    # 1. Pre-split audio into fixed clips
    clips = split_audio(args.audio_file, args.segment_duration)
    if not clips:
        print("[error] No clips generated.")
        sys.exit(1)

    # 2. Batch transcribe (text only, no timestamps needed)
    segments = transcribe_all(client, clips, args.batch_size)
    if not segments:
        print("[error] No transcriptions returned.")
        sys.exit(1)

    elapsed = time.time() - t0
    audio_dur = sum(s["duration_sec"] for s in segments)
    audio_total = sf.info(args.audio_file).duration

    # 3. Print results
    print(f"\n{'=' * 80}")
    for s in segments:
        print(f"  [{s['start_sec']:6.1f}s - {s['end_sec']:6.1f}s] "
              f"({s['duration_sec']:.1f}s) {s['transcription'][:70]}")
    print(f"{'=' * 80}")
    print(f"  Total: {len(segments)} segments, {audio_dur:.1f}s "
          f"({audio_dur / 60:.1f}min)")
    print(f"  Processing: {elapsed:.1f}s for {audio_total:.1f}s audio "
          f"({audio_total / elapsed:.1f}x realtime)")

    # 4. Push
    if not args.dry_run and args.push_hf:
        push_to_hf(segments, args.push_hf)
    elif args.dry_run:
        print("\n[dry-run] Skipping HF push.")

    print("\n[done]")


if __name__ == "__main__":
    main()
