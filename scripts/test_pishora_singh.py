"""Test Surt model on Bhai Pishora Singh's Sehaj Path (different speaker).

Downloads a few minutes of audio from Internet Archive,
transcribes with the model, and displays results alongside
known SGGS text for manual comparison.

Usage:
    pip install transformers torch librosa requests
    python scripts/test_pishora_singh.py
"""

import os
import tempfile
import subprocess
import torch
import numpy as np
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration

MODEL_ID = "surindersinghssj/surt-small-v1"

AUDIO_URL = (
    "https://archive.org/download/aad-sri-guru-granth-sahib-ji-da-sahaj-paath/"
    "Aad%20Sri%20Guru%20Granth%20Sahib%20Ji%20Da%20Sahaj%20Paath%20%28Vol%20-%201%29"
    "%20%20%20Page%20No.%201%20to%2023%20%20%20Bhai%20Pishora%20Singh%20Ji.mp3"
)

EXPECTED_OPENING = [
    "ੴ ਸਤਿ ਨਾਮੁ ਕਰਤਾ ਪੁਰਖੁ ਨਿਰਭਉ ਨਿਰਵੈਰੁ ਅਕਾਲ ਮੂਰਤਿ ਅਜੂਨੀ ਸੈਭੰ ਗੁਰ ਪ੍ਰਸਾਦਿ",
    "ਜਪੁ ॥ ਆਦਿ ਸਚੁ ਜੁਗਾਦਿ ਸਚੁ ॥ ਹੈ ਭੀ ਸਚੁ ਨਾਨਕ ਹੋਸੀ ਭੀ ਸਚੁ",
    "ਸੋਚੈ ਸੋਚਿ ਨ ਹੋਵਈ ਜੇ ਸੋਚੀ ਲਖ ਵਾਰ",
    "ਚੁਪੈ ਚੁਪ ਨ ਹੋਵਈ ਜੇ ਲਾਇ ਰਹਾ ਲਿਵ ਤਾਰ",
]


def download_first_minutes(url, output_path, max_mb=5):
    """Download first few MB of audio (about 3 minutes of MP3)."""
    import requests
    print(f"Downloading first {max_mb} MB of audio...")
    mp3_path = output_path.replace(".wav", ".mp3")
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    downloaded = 0
    max_bytes = max_mb * 1024 * 1024
    with open(mp3_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if downloaded >= max_bytes:
                break
    print(f"Downloaded {downloaded / 1024 / 1024:.1f} MB")

    subprocess.run(
        ["ffmpeg", "-y", "-i", mp3_path, "-ar", "16000", "-ac", "1", output_path],
        capture_output=True,
    )
    if os.path.exists(mp3_path):
        os.remove(mp3_path)
    print(f"Converted to WAV: {output_path}")


def chunk_audio(audio, sr=16000, chunk_seconds=15):
    """Split audio into chunks of N seconds."""
    chunk_samples = chunk_seconds * sr
    chunks = []
    for start in range(0, len(audio), chunk_samples):
        chunk = audio[start : start + chunk_samples]
        if len(chunk) > sr:
            chunks.append(chunk)
    return chunks


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    print(f"Loading model {MODEL_ID}...")
    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID).to(device)
    model.eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, "pishora_vol1.wav")
        download_first_minutes(AUDIO_URL, wav_path, max_mb=5)

        audio, sr = librosa.load(wav_path, sr=16000)
        duration = len(audio) / sr
        print(f"Loaded {duration:.1f}s of audio at {sr}Hz")

        chunks = chunk_audio(audio, sr=16000, chunk_seconds=15)
        print(f"Split into {len(chunks)} chunks of ~15s each\n")

        print("=" * 60)
        print("TRANSCRIPTIONS (Bhai Pishora Singh - Ang 1-23)")
        print("=" * 60)

        all_transcriptions = []
        for i, chunk in enumerate(chunks):
            input_features = processor(
                chunk, sampling_rate=16000, return_tensors="pt"
            ).input_features.to(device)

            with torch.no_grad():
                predicted_ids = model.generate(input_features)

            text = processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0]

            all_transcriptions.append(text)
            mins = i * 15 // 60
            secs = i * 15 % 60
            print(f"[{mins}:{secs:02d}] {text}")

        print("\n" + "=" * 60)
        print("EXPECTED (opening lines - Mool Mantar / Japji Sahib)")
        print("=" * 60)
        for line in EXPECTED_OPENING:
            print(f"  {line}")

        print("\n" + "=" * 60)
        print("FULL TRANSCRIPTION (concatenated)")
        print("=" * 60)
        print(" ".join(all_transcriptions))


if __name__ == "__main__":
    main()
