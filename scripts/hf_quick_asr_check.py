#!/usr/bin/env python3
"""
Quick 2-sample ASR sanity check against HF-hosted model endpoint.

Runs one Mool-like sample and one random sample from the dataset, prints REF/PRED.
"""

from __future__ import annotations

import os
import tempfile

import soundfile as sf
from datasets import load_dataset
from huggingface_hub import InferenceClient
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch

MODEL_ID = os.environ.get("ASR_MODEL_ID", "surindersinghssj/surt-small-v1")
DATASET_ID = os.environ.get("ASR_DATASET_ID", "surindersinghssj/gurbani-asr")


def pick_two_samples():
    ds = load_dataset(DATASET_ID, split="train", streaming=True)
    mool_like = None
    random_sample = None

    for ex in ds:
        txt = (ex.get("transcription") or "").strip()
        if mool_like is None and ("ੴ" in txt or "ਸਤਿ ਨਾਮੁ" in txt):
            mool_like = ex
        elif random_sample is None:
            random_sample = ex
        if mool_like is not None and random_sample is not None:
            break

    return [("Mool-like", mool_like), ("Random", random_sample)]


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set. Export it first.")

    client = InferenceClient(model=MODEL_ID, token=token)
    local_processor = None
    local_model = None
    samples = pick_two_samples()

    print(f"Model: {MODEL_ID}")
    print(f"Dataset: {DATASET_ID}")

    for name, ex in samples:
        if ex is None:
            continue

        audio = ex["audio"]
        arr = audio["array"]
        sr = audio["sampling_rate"]
        ref = ex.get("transcription", "")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name
        sf.write(wav_path, arr, sr)

        try:
            out = client.automatic_speech_recognition(wav_path)
            pred = out.get("text", str(out)) if isinstance(out, dict) else str(out)
            backend = "hf-inference-api"
        except Exception:
            # Fallback when model has no hosted ASR provider.
            if local_processor is None or local_model is None:
                device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
                local_processor = WhisperProcessor.from_pretrained(
                    MODEL_ID, language="punjabi", task="transcribe"
                )
                local_model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID).to(device)
                local_model.eval()

            feats = local_processor(arr, sampling_rate=sr, return_tensors="pt").input_features.to(local_model.device)
            with torch.inference_mode():
                pred_ids = local_model.generate(feats, max_length=448)
            pred = local_processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)[0]
            backend = "local-transformers"

        print(f"\n=== {name} sample ===")
        print("BACKEND:", backend)
        print("REF :", ref)
        print("PRED:", pred)


if __name__ == "__main__":
    main()
