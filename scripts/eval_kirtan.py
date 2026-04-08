"""Evaluate Surt model on kirtan and Sehaj Path datasets.

Computes WER and CER on:
  1. gurbani-asr validation split (Sehaj Path, same speaker)
  2. gurbani-asr-v2-test (kirtan, different speakers)
  3. Optionally: baseline whisper-small for comparison

Usage:
    pip install transformers datasets jiwer librosa
    python scripts/eval_kirtan.py
    python scripts/eval_kirtan.py --baseline  # also eval stock whisper-small
"""

import argparse
import torch
import numpy as np
from datasets import load_dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from jiwer import wer, cer

from surt.data import normalize_gurbani_text


def load_model(model_id, device):
    print(f"Loading {model_id}...")
    # Always load processor from baseline (our fine-tuned tokenizer config
    # can be incompatible with newer transformers versions)
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)
    return processor, model


def transcribe_batch(audio_arrays, sample_rates, processor, model, device):
    """Transcribe a list of audio arrays."""
    resampled = []
    for audio, sr in zip(audio_arrays, sample_rates):
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        resampled.append(audio)

    inputs = processor(
        resampled, sampling_rate=16000, return_tensors="pt", padding=True
    )
    input_features = inputs.input_features.to(device)

    with torch.no_grad():
        predicted_ids = model.generate(input_features)

    transcriptions = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcriptions


def evaluate_dataset(dataset, text_column, processor, model, device, max_samples=None, batch_size=8):
    """Evaluate model on a dataset, return WER and CER."""
    all_references = []
    all_predictions = []

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    total = len(dataset)
    print(f"  Evaluating {total} samples (batch_size={batch_size})...")

    for i in range(0, total, batch_size):
        batch = dataset[i : i + batch_size]
        audio_arrays = [a["array"] for a in batch["audio"]]
        sample_rates = [a["sampling_rate"] for a in batch["audio"]]
        references = batch[text_column]

        valid_indices = [j for j, r in enumerate(references) if r and r.strip()]
        if not valid_indices:
            continue

        audio_arrays = [audio_arrays[j] for j in valid_indices]
        sample_rates = [sample_rates[j] for j in valid_indices]
        references = [references[j] for j in valid_indices]

        predictions = transcribe_batch(audio_arrays, sample_rates, processor, model, device)

        # Normalize both references and predictions for consistent metrics
        references = [normalize_gurbani_text(r) for r in references]
        predictions = [normalize_gurbani_text(p) for p in predictions]

        all_references.extend(references)
        all_predictions.extend(predictions)

        if (i // batch_size) % 10 == 0:
            print(f"    {i + len(valid_indices)}/{total} done...")

    if not all_references:
        print("  No valid samples found!")
        return None, None

    w = wer(all_references, all_predictions) * 100
    c = cer(all_references, all_predictions) * 100

    print(f"\n  Sample predictions:")
    for idx in range(min(3, len(all_references))):
        print(f"    REF: {all_references[idx][:80]}")
        print(f"    HYP: {all_predictions[idx][:80]}")
        print()

    return w, c


def main():
    parser = argparse.ArgumentParser(description="Evaluate Surt ASR model")
    parser.add_argument(
        "--model", default="surindersinghssj/surt-small-v2-training",
        help="HuggingFace model ID to evaluate",
    )
    parser.add_argument(
        "--baseline", action="store_true",
        help="Also evaluate stock openai/whisper-small for comparison",
    )
    parser.add_argument(
        "--max-samples", type=int, default=300,
        help="Max samples per dataset (default: 300)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Batch size for inference (default: 8)",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    print("Loading datasets...")

    print("  1. Sehaj Path validation set...")
    sehaj_ds = load_dataset(
        "surindersinghssj/gurbani-asr", split="validation"
    )
    sehaj_ds = sehaj_ds.cast_column("audio", Audio(sampling_rate=16000))
    sehaj_text_col = "transcription"

    print("  2. Kirtan v2 test set...")
    kirtan_ds = None
    kirtan_text_col = None
    try:
        kirtan_ds = load_dataset(
            "surindersinghssj/gurbani-kirtan-v2-prepared", split="test"
        )
        kirtan_ds = kirtan_ds.cast_column("audio", Audio(sampling_rate=16000))
        kirtan_text_col = "transcription"
        print(f"    Kirtan text column: {kirtan_text_col}")
        print(f"    Columns: {kirtan_ds.column_names}")
    except Exception as e:
        print(f"    Could not load kirtan v2 dataset: {e}")

    print()

    models_to_eval = [(args.model, "Surt (fine-tuned)")]
    if args.baseline:
        models_to_eval.append(("openai/whisper-small", "Whisper Small (baseline)"))

    results = []

    for model_id, model_name in models_to_eval:
        processor, model = load_model(model_id, device)
        model.eval()

        print(f"\n{'='*60}")
        print(f"Model: {model_name} ({model_id})")
        print(f"{'='*60}")

        print(f"\n[Sehaj Path] ({len(sehaj_ds)} samples)")
        sehaj_wer, sehaj_cer = evaluate_dataset(
            sehaj_ds, sehaj_text_col, processor, model, device,
            max_samples=args.max_samples, batch_size=args.batch_size,
        )
        if sehaj_wer is not None:
            print(f"  >>> Sehaj Path WER: {sehaj_wer:.2f}%  CER: {sehaj_cer:.2f}%")

        if kirtan_ds is not None:
            print(f"\n[Kirtan] ({len(kirtan_ds)} samples)")
            kirtan_wer, kirtan_cer = evaluate_dataset(
                kirtan_ds, kirtan_text_col, processor, model, device,
                max_samples=args.max_samples, batch_size=args.batch_size,
            )
            if kirtan_wer is not None:
                print(f"  >>> Kirtan WER: {kirtan_wer:.2f}%  CER: {kirtan_cer:.2f}%")
        else:
            kirtan_wer, kirtan_cer = None, None

        results.append({
            "model": model_name,
            "sehaj_wer": sehaj_wer,
            "sehaj_cer": sehaj_cer,
            "kirtan_wer": kirtan_wer,
            "kirtan_cer": kirtan_cer,
        })

        del model, processor
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'Sehaj WER':>10} {'Sehaj CER':>10} {'Kirtan WER':>11} {'Kirtan CER':>11}")
    print("-" * 70)
    for r in results:
        sw = f"{r['sehaj_wer']:.2f}%" if r['sehaj_wer'] else "N/A"
        sc = f"{r['sehaj_cer']:.2f}%" if r['sehaj_cer'] else "N/A"
        kw = f"{r['kirtan_wer']:.2f}%" if r['kirtan_wer'] else "N/A"
        kc = f"{r['kirtan_cer']:.2f}%" if r['kirtan_cer'] else "N/A"
        print(f"{r['model']:<25} {sw:>10} {sc:>10} {kw:>11} {kc:>11}")


if __name__ == "__main__":
    main()
