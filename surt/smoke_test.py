"""
Phase 4 pre-flight checks for the Surt training pipeline.

Implements:
1. TEST-01: Generate one sample and verify the output is Gurmukhi (not English)
2. TEST-02: Validate one batch for 16kHz audio, -100 label masking, and BOS sanity
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import torch
from datasets import Audio, load_dataset

from surt.config import (
    DATASET_NAME,
    GENERATION_MAX_LENGTH,
    TEXT_COLUMN,
    VAL_SPLIT,
)
from surt.data import DataCollatorSpeechSeq2SeqWithPadding, get_train_dataset
from surt.model import get_mool_mantar_prompt_ids

AUDIO_COLUMN = "audio"
_GURMUKHI_RE = re.compile(r"[\u0A00-\u0A7F]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def _contains_gurmukhi(text: str) -> bool:
    return bool(_GURMUKHI_RE.search(text))


def _contains_latin(text: str) -> bool:
    return bool(_LATIN_RE.search(text))


def _load_first_raw_example(dataset_name: str, split: str) -> dict:
    """Load one raw example with audio cast to 16kHz."""
    ds = load_dataset(dataset_name, split=split, streaming=True)
    ds = ds.cast_column(AUDIO_COLUMN, Audio(sampling_rate=16000))
    return next(iter(ds.take(1)))


def _take_n_examples(stream: Iterable[dict], n: int) -> list[dict]:
    out = []
    for ex in stream:
        out.append(ex)
        if len(out) >= n:
            break
    if len(out) < n:
        raise RuntimeError(f"Expected at least {n} examples, got {len(out)}")
    return out


def run_generation_preflight(
    model,
    processor,
    dataset_name: str = DATASET_NAME,
    split: str = VAL_SPLIT,
) -> str:
    """TEST-01: generate one sample and enforce Gurmukhi output sanity."""
    example = _load_first_raw_example(dataset_name, split)
    audio = example[AUDIO_COLUMN]
    text_ref = example.get(TEXT_COLUMN, "")
    sample_rate = audio["sampling_rate"]
    waveform = audio["array"]

    device = next(model.parameters()).device
    model_inputs = processor.feature_extractor(
        waveform,
        sampling_rate=sample_rate,
        return_tensors="pt",
    ).input_features.to(device)

    prompt_ids = torch.tensor(
        get_mool_mantar_prompt_ids(processor),
        dtype=torch.long,
        device=device,
    )
    with torch.inference_mode():
        pred_ids = model.generate(
            model_inputs,
            prompt_ids=prompt_ids,
            max_length=GENERATION_MAX_LENGTH,
        )
    pred_text = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()

    if not _contains_gurmukhi(pred_text):
        raise AssertionError(
            f"TEST-01 failed: prediction contains no Gurmukhi characters: {pred_text!r}"
        )
    if _contains_latin(pred_text):
        raise AssertionError(
            f"TEST-01 failed: prediction contains Latin script: {pred_text!r}"
        )

    print("[smoke] TEST-01 PASS")
    print(f"[smoke] Reference:   {str(text_ref)[:120]}")
    print(f"[smoke] Prediction:  {pred_text[:120]}")
    return pred_text


def run_batch_preflight(
    model,
    processor,
    dataset_name: str = DATASET_NAME,
) -> None:
    """TEST-02: validate 16kHz, -100 padding mask, and BOS behavior."""
    raw_example = _load_first_raw_example(dataset_name, VAL_SPLIT)
    raw_sr = raw_example[AUDIO_COLUMN]["sampling_rate"]
    if raw_sr != 16000:
        raise AssertionError(f"TEST-02 failed: expected 16kHz, got {raw_sr}")

    train_dataset = get_train_dataset(dataset_name, processor)
    features = _take_n_examples(iter(train_dataset), 3)

    # Force variable label lengths to guarantee pad positions in the collated batch.
    # This keeps TEST-02 deterministic instead of relying on sampled sequence lengths.
    features[1] = {
        "input_features": features[1]["input_features"],
        "labels": features[1]["labels"][:1],
    }
    if features[2]["labels"]:
        features[2] = {
            "input_features": features[2]["input_features"],
            "labels": features[2]["labels"] + [features[2]["labels"][-1]],
        }

    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    batch = collator(features)
    labels = batch["labels"]

    if not (labels == -100).any():
        raise AssertionError("TEST-02 failed: no -100 pad masking found in labels")

    decoder_start = model.config.decoder_start_token_id
    for row in labels:
        tokens = row[row != -100].tolist()
        if len(tokens) >= 2 and tokens[0] == decoder_start and tokens[1] == decoder_start:
            raise AssertionError("TEST-02 failed: found double BOS token in labels")

    print("[smoke] TEST-02 PASS")
    print("[smoke] Batch check: sr=16000, -100 masking present, BOS sanity OK")


def run_preflight_checks(model, processor, dataset_name: str = DATASET_NAME) -> None:
    """Run TEST-01 and TEST-02 before any non-trivial training run."""
    print("[smoke] === Pre-flight Checks ===")
    run_generation_preflight(model, processor, dataset_name=dataset_name)
    run_batch_preflight(model, processor, dataset_name=dataset_name)
    print("[smoke] Pre-flight checks complete")
