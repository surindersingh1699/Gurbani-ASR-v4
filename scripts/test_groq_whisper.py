"""
Quick test: Send a few SGPC kirtan samples to Groq Whisper API
and compare the word-level timestamps against original caption timestamps.

Usage:
    export GROQ_API_KEY="gsk_..."
    python scripts/test_groq_whisper.py
    python scripts/test_groq_whisper.py --n-samples 20 --offset 5000
"""

import argparse
import io
import json
import os
import sys
import tempfile
import time

import numpy as np
import soundfile as sf
from groq import Groq

SRC_DATASET = "surindersinghssj/sgpc-amritsar-kirtan-live"
SAMPLE_RATE = 16000

# Gurbani context prompt
GURBANI_PROMPT = (
    "ੴ ਸਤਿ ਨਾਮੁ ਕਰਤਾ ਪੁਰਖੁ ਨਿਰਭਉ ਨਿਰਵੈਰੁ ਅਕਾਲ ਮੂਰਤਿ ਅਜੂਨੀ ਸੈਭੰ ਗੁਰ ਪ੍ਰਸਾਦਿ ॥ "
    "ਵਾਹਿਗੁਰੂ ਜੀ ਕਾ ਖਾਲਸਾ ਵਾਹਿਗੁਰੂ ਜੀ ਕੀ ਫਤਹਿ ॥ "
    "ਸ੍ਰੀ ਭਗੌਤੀ ਜੀ ਸਹਾਇ ਵਾਰ ਸ੍ਰੀ ਭਗੌਤੀ ਜੀ ਕੀ ਪਾਤਸ਼ਾਹੀ ੧੦ ॥ "
    "ਗੁਰੂ ਗ੍ਰੰਥ ਸਾਹਿਬ ਜੀ ਸ਼ਬਦ ਕੀਰਤਨ ਹੁਕਮਨਾਮਾ ਅਰਦਾਸ"
)


def load_samples(n_samples: int, offset: int = 0):
    from datasets import Audio, load_dataset
    print(f"[data] Loading {n_samples} samples from offset {offset}...")
    ds = load_dataset(SRC_DATASET, split=f"train[{offset}:{offset + n_samples}]")
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE, decode=True))
    print(f"[data] Loaded {len(ds)} samples")
    return ds


def transcribe_sample(client, audio_array: np.ndarray, sr: int):
    """Send a single audio sample to Groq and get word timestamps."""
    with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as tmp:
        sf.write(tmp.name, audio_array, sr, format="FLAC")
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            response = client.audio.transcriptions.create(
                file=f,
                model="whisper-large-v3-turbo",
                response_format="verbose_json",
                timestamp_granularities=["word", "segment"],
                language="pa",
                temperature=0.0,
                prompt=GURBANI_PROMPT,
            )
        return response
    finally:
        os.unlink(tmp_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERROR: Set GROQ_API_KEY")
        sys.exit(1)

    client = Groq(api_key=api_key)
    ds = load_samples(args.n_samples, args.offset)

    print(f"\n{'='*70}")
    print(f"GROQ WHISPER RESULTS (with Mool Mantar prompt)")
    print(f"{'='*70}")

    for i, sample in enumerate(ds):
        caption_text = sample["gurmukhi_text"]
        audio = sample["audio"]
        orig_start = sample.get("start_time", 0)
        orig_end = sample.get("end_time", 0)
        orig_dur = sample.get("duration", 0)

        print(f"\n--- Sample {i} ({orig_start:.1f}s - {orig_end:.1f}s, "
              f"dur={orig_dur:.1f}s) ---")
        print(f"  Caption:  {caption_text}")

        audio_array = np.array(audio["array"], dtype=np.float32)

        try:
            t0 = time.time()
            resp = transcribe_sample(client, audio_array, audio["sampling_rate"])
            elapsed = time.time() - t0
        except Exception as e:
            print(f"  ERROR: {e}")
            time.sleep(2)
            continue

        whisper_text = resp.text if hasattr(resp, 'text') else str(resp.get('text', ''))
        print(f"  Whisper:  {whisper_text}")
        print(f"  API time: {elapsed:.2f}s")

        # Word-level timestamps
        words = resp.words if hasattr(resp, 'words') else resp.get('words', [])
        if words:
            print(f"  Words ({len(words)}):")
            for w in words:
                word = w.word if hasattr(w, 'word') else w.get('word', '')
                start = w.start if hasattr(w, 'start') else w.get('start', 0)
                end = w.end if hasattr(w, 'end') else w.get('end', 0)
                print(f"    {start:6.2f}s - {end:6.2f}s  {word}")

            # Timing analysis
            first_start = words[0].start if hasattr(words[0], 'start') else words[0]['start']
            last_end = words[-1].end if hasattr(words[-1], 'end') else words[-1]['end']
            audio_dur = len(audio_array) / audio["sampling_rate"]

            print(f"\n  Timing:")
            print(f"    Audio:     {audio_dur:.2f}s")
            print(f"    Speech:    {first_start:.2f}s - {last_end:.2f}s")
            print(f"    Lead gap:  {first_start:.2f}s")
            print(f"    Trail gap: {audio_dur - last_end:.2f}s")
        else:
            print(f"  No word timestamps returned")

        # Rate limit courtesy
        time.sleep(0.5)

    print(f"\n{'='*70}")
    print(f"DONE — check Whisper text vs Caption text above")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
