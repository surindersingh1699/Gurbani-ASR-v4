#!/usr/bin/env python3
"""A/B compare gemini-2.5-flash-lite vs gemini-3.1-flash-lite-preview on
Gurmukhi kirtan transcription.

Keeps the prompt, clip length, batch size, config, and retry logic identical to
scripts/kirtan_bulk_transcribe.py — the only variable is the model ID. Sends
the first 5 minutes of one audio file per source type (ragi / akj / sgpc),
producing 15 × 20 s clips per source = 45 clips total, 1 batch per
(source, model) = 6 API calls.

Outputs a side-by-side HF dataset at surindersinghssj/gurbani-gemini-ab-test
with columns:
  source_type, clip_idx, start_sec, end_sec, duration_sec,
  text_25_flash_lite, text_31_flash_lite_preview, audio
"""
from __future__ import annotations

import argparse
import io
import json
import os
import time

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
CLIP_SECS = 20.0
TAKE_SECS = 300.0
BATCH_SIZE = 15
MODEL_25 = "gemini-2.5-flash-lite"
MODEL_31 = "gemini-3.1-flash-lite-preview"

SOURCES = [
    ("ragi", "/root/v3_data/raw_audio/ragi/sikhnet-10802.wav"),
    ("akj",  "/root/v3_data/raw_audio/akj/"
             "akj-010_Bareilly_24Jan2026_SatMor_DSK_BhaiInderjeetSinghJeeMoga.wav"),
    ("sgpc", "/root/v3_data/raw_audio/sgpc/sgpc_ab_test.wav"),
]

PROMPT_TMPL = """You are a Gurbani/Punjabi transcription expert. This is Sikh kirtan.

I am sending {n} audio clips (each ~20 seconds). For each clip:
- Transcribe ONLY the Gurmukhi text being sung (correct matras and spelling)
- If a clip is instrumental only (no singing), return empty string ""
- If the clip contains Punjabi meaning/explanation (not Gurbani), return empty string ""

Return a JSON array of exactly {n} strings, one per clip in order.
Example: ["ਗੁਰਮੁਖੀ ਟੈਕਸਟ", "ਹੋਰ ਟੈਕਸਟ", "", "ਗੁਰਮੁਖੀ ਟੈਕਸਟ"]
Return ONLY the JSON array."""


def load_clips(path: str) -> list[dict]:
    arr, sr = sf.read(path)
    if sr != SAMPLE_RATE:
        raise RuntimeError(f"{path}: sr={sr}, expected {SAMPLE_RATE}")
    n_samples = int(min(TAKE_SECS, len(arr) / sr) * sr)
    arr = arr[:n_samples]
    seg_samples = int(CLIP_SECS * sr)
    clips: list[dict] = []
    for i, off in enumerate(range(0, len(arr), seg_samples)):
        chunk = arr[off:off + seg_samples]
        if len(chunk) < sr * 2:
            break
        buf = io.BytesIO()
        sf.write(buf, chunk.astype(np.float32), sr, format="WAV", subtype="PCM_16")
        clips.append({
            "idx": i,
            "audio_array": chunk,
            "audio_bytes": buf.getvalue(),
            "start_sec": round(off / sr, 3),
            "end_sec": round(min(off + seg_samples, len(arr)) / sr, 3),
        })
    return clips


def transcribe_batch(client, model: str, clips: list[dict]) -> tuple[list[str], float]:
    from google.genai import types

    n = len(clips)
    prompt = PROMPT_TMPL.format(n=n)
    contents = [types.Part.from_text(text=prompt)]
    for i, c in enumerate(clips):
        contents.append(types.Part.from_text(text=f"Clip {i + 1}:"))
        contents.append(types.Part.from_bytes(
            data=c["audio_bytes"], mime_type="audio/wav"))

    t0 = time.time()
    for attempt in range(3):
        try:
            r = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                ),
            )
            texts = json.loads(r.text.strip())
            if isinstance(texts, dict):
                texts = list(texts.values())
            if not isinstance(texts, list):
                return [""] * n, time.time() - t0
            if len(texts) < n:
                texts.extend([""] * (n - len(texts)))
            elif len(texts) > n:
                texts = texts[:n]
            return texts, time.time() - t0
        except json.JSONDecodeError:
            if attempt < 2:
                time.sleep(1)
                continue
            return [""] * n, time.time() - t0
        except Exception as e:
            err = str(e)
            if "429" in err and attempt < 2:
                time.sleep(2 ** (attempt + 1))
                continue
            print(f"[{model}] error: {err[:200]}")
            return [""] * n, time.time() - t0
    return [""] * n, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--push-hf", default="surindersinghssj/gurbani-gemini-ab-test")
    ap.add_argument("--private", action="store_true",
                    help="Push HF dataset as private (default: public)")
    args = ap.parse_args()

    from google import genai
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    all_rows: list[dict] = []

    for src_type, path in SOURCES:
        print(f"\n=== {src_type}: {os.path.basename(path)} ===")
        clips = load_clips(path)
        print(f"  {len(clips)} clips × {CLIP_SECS:.0f}s = "
              f"{clips[-1]['end_sec']:.0f}s audio")

        texts_by_model: dict[str, list[str]] = {}
        for model in (MODEL_25, MODEL_31):
            print(f"  [{model}] sending {len(clips)} clips...")
            texts, elapsed = transcribe_batch(client, model, clips)
            non_empty = sum(1 for t in texts if t.strip())
            avg_chars = (sum(len(t) for t in texts if t.strip())
                         / max(1, non_empty))
            print(f"    ← {non_empty}/{len(clips)} filled, "
                  f"avg {avg_chars:.0f} chars, {elapsed:.1f}s")
            texts_by_model[model] = texts
            time.sleep(0.5)

        for c, t25, t31 in zip(clips, texts_by_model[MODEL_25],
                               texts_by_model[MODEL_31]):
            all_rows.append({
                "source_type": src_type,
                "clip_idx": c["idx"],
                "start_sec": c["start_sec"],
                "end_sec": c["end_sec"],
                "duration_sec": round(c["end_sec"] - c["start_sec"], 3),
                "text_25_flash_lite": t25.strip() if isinstance(t25, str) else "",
                "text_31_flash_lite_preview": (
                    t31.strip() if isinstance(t31, str) else ""),
                "audio": {"array": c["audio_array"],
                          "sampling_rate": SAMPLE_RATE},
            })

    print("\n=== SUMMARY ===")
    for src_type, _ in SOURCES:
        rows = [r for r in all_rows if r["source_type"] == src_type]
        a = sum(1 for r in rows if r["text_25_flash_lite"])
        b = sum(1 for r in rows if r["text_31_flash_lite_preview"])
        avg_a = (sum(len(r["text_25_flash_lite"]) for r in rows)
                 / max(1, a))
        avg_b = (sum(len(r["text_31_flash_lite_preview"]) for r in rows)
                 / max(1, b))
        print(f"  {src_type}: 2.5 filled {a}/{len(rows)} (avg {avg_a:.0f} ch)  |  "
              f"3.1 filled {b}/{len(rows)} (avg {avg_b:.0f} ch)")
    identical = sum(1 for r in all_rows
                    if r["text_25_flash_lite"] == r["text_31_flash_lite_preview"])
    print(f"  identical strings: {identical}/{len(all_rows)}")

    if args.push_hf:
        from datasets import Audio, Dataset
        ds = Dataset.from_dict({
            "source_type": [r["source_type"] for r in all_rows],
            "clip_idx": [r["clip_idx"] for r in all_rows],
            "start_sec": [r["start_sec"] for r in all_rows],
            "end_sec": [r["end_sec"] for r in all_rows],
            "duration_sec": [r["duration_sec"] for r in all_rows],
            "text_25_flash_lite":
                [r["text_25_flash_lite"] for r in all_rows],
            "text_31_flash_lite_preview":
                [r["text_31_flash_lite_preview"] for r in all_rows],
            "audio": [r["audio"] for r in all_rows],
        })
        ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
        visibility = "private" if args.private else "public"
        print(f"\n[push] -> {args.push_hf} ({visibility})")
        ds.push_to_hub(args.push_hf, split="train", private=args.private)
        print("[push] done.")


if __name__ == "__main__":
    main()
