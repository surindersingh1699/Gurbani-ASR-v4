"""
Streaming data pipeline for Gurbani ASR fine-tuning.

This module provides:
1. Streaming training dataset with waveform augmentation (noise, reverb, stretch, pitch)
2. Fixed-size validation dataset materialized in memory (no augmentation)
3. Data collator that pads input features and masks label padding to -100

Critical design decisions:
- Augmentation is applied to raw waveform BEFORE log-Mel feature extraction
- Training and validation use separate prepare functions to prevent augmentation leakage
- The data collator strips double BOS tokens that Whisper's tokenizer sometimes adds
- Processor is injected as a parameter (not imported) for clean dependency direction
- HuggingFace dataset loading uses retry with exponential backoff for rate limit handling
"""

import os
import re
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


def normalize_gurbani_text(text: str) -> str:
    """Strip non-spoken structural markers from Gurbani text.

    Removes double danda (॥), verse numbers (॥੧॥), and collapses whitespace.
    These are visual markers never spoken aloud — removing them gives the model
    a cleaner signal and prevents WER/CER inflation from misplaced markers.
    """
    # Remove verse numbers: ॥੧॥ ॥੨॥ ॥੧੨॥ etc.
    text = re.sub(r'॥[੦-੯]+॥', '', text)
    # Remove standalone double danda ॥
    text = re.sub(r'॥', '', text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text
try:
    from audiomentations import (
        AddGaussianNoise,
        Compose,
        PitchShift,
        RoomSimulator,
        TimeStretch,
    )
    _HAS_AUGMENT = True
except ModuleNotFoundError:
    _HAS_AUGMENT = False
from datasets import Audio, Dataset, IterableDataset, interleave_datasets, load_dataset

from surt.config import SHUFFLE_BUFFER, VAL_SIZE, VAL_SPLIT

# --- Column names ---
AUDIO_COLUMN = "audio"
TEXT_COLUMN = "transcription"

# --- Augmentation pipeline (training only) ---
# Applied to raw waveform BEFORE feature extraction.
# Probabilities tuned for Gurbani audio (single speaker, clean recordings):
#   - Gaussian noise at p=0.4: simulate microphone/environment noise
#   - Room reverb at p=0.3: simulate different recording spaces
#   - Time stretch at p=0.2: tempo variation without pitch change
#   - Pitch shift at p=0.2: simulate slight speaker pitch variation
if _HAS_AUGMENT:
    augment = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.4),
        RoomSimulator(p=0.3),
        TimeStretch(min_rate=0.8, max_rate=1.25, p=0.2),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.2),
    ])
else:
    def augment(*, samples, sample_rate):  # type: ignore[override]
        return samples

# --- Retry configuration for HuggingFace Hub rate limits ---
_HF_MAX_RETRIES = 5
_HF_INITIAL_BACKOFF = 2.0  # seconds
_HF_BACKOFF_FACTOR = 2.0   # exponential multiplier


def _load_dataset_with_retry(
    dataset_name: str,
    split: str,
    streaming: bool = True,
) -> IterableDataset | Dataset:
    """Load dataset from HuggingFace Hub with retry and exponential backoff.

    Streaming datasets from HuggingFace Hub can hit rate limits (HTTP 429).
    This wrapper retries with exponential backoff to handle transient failures.

    Args:
        dataset_name: HuggingFace dataset identifier.
        split: Dataset split to load.
        streaming: Whether to load as streaming IterableDataset.

    Returns:
        Loaded dataset (IterableDataset if streaming=True, Dataset otherwise).

    Raises:
        Exception: If all retries are exhausted.
    """
    last_exception = None
    backoff = _HF_INITIAL_BACKOFF

    for attempt in range(1, _HF_MAX_RETRIES + 1):
        try:
            ds = load_dataset(dataset_name, split=split, streaming=streaming)
            return ds
        except Exception as e:
            last_exception = e
            error_str = str(e).lower()
            # Retry on rate limits, connection errors, and server errors
            is_retryable = any(
                keyword in error_str
                for keyword in ["429", "rate limit", "too many requests",
                                "connection", "timeout", "503", "502"]
            )
            if not is_retryable or attempt == _HF_MAX_RETRIES:
                raise
            print(
                f"[data] HF load attempt {attempt}/{_HF_MAX_RETRIES} failed: {e}. "
                f"Retrying in {backoff:.1f}s..."
            )
            time.sleep(backoff)
            backoff *= _HF_BACKOFF_FACTOR

    # Should not reach here, but just in case
    raise last_exception  # type: ignore[misc]


def get_train_dataset(
    dataset_name: str,
    processor: Any,
    split: str = "train",
    text_column: str = TEXT_COLUMN,
    aux_dataset_name: str | None = None,
    aux_probability: float = 0.15,
    streaming: bool = False,
) -> IterableDataset | Dataset:
    """Load training dataset with waveform augmentation.

    Supports two modes:
    - streaming=True: IterableDataset, low memory but slow resume
    - streaming=False: In-memory Dataset, uses more RAM but instant resume

    Audio is resampled to 16kHz, augmented with noise/reverb/stretch/pitch,
    then converted to 80-bin log-Mel spectrograms. Text is tokenized to
    Gurmukhi BPE token IDs.

    CRITICAL: Augmentation is applied to raw waveform BEFORE feature extraction.
    Never apply augmentation to log-Mel features.

    Args:
        dataset_name: HuggingFace dataset identifier.
        processor: WhisperProcessor with Punjabi tokenizer.
        split: Dataset split to load (default: "train").
        text_column: Name of the transcription text column.
        streaming: If True, use streaming IterableDataset. If False, load in-memory.

    Returns:
        Dataset or IterableDataset yielding {"input_features", "labels"} dicts.
    """
    def prepare_train(example: dict) -> dict:
        audio = example[AUDIO_COLUMN]
        samples = np.array(audio["array"], dtype=np.float32)

        # Apply augmentation to raw waveform BEFORE feature extraction
        samples = augment(samples=samples, sample_rate=16000)

        # Extract 80-bin log-Mel spectrogram
        input_features = processor.feature_extractor(
            samples, sampling_rate=16000
        ).input_features[0]

        # Tokenize Gurmukhi text (normalize to strip ॥ markers)
        text = normalize_gurbani_text(example[text_column])
        labels = processor.tokenizer(text).input_ids

        return {"input_features": input_features, "labels": labels}

    if streaming:
        def map_train_stream(ds: IterableDataset) -> IterableDataset:
            ds = ds.cast_column(AUDIO_COLUMN, Audio(sampling_rate=16000))
            ds = ds.shuffle(seed=42, buffer_size=SHUFFLE_BUFFER)
            try:
                columns = ds.column_names
            except (AttributeError, TypeError):
                columns = None
            if columns:
                return ds.map(prepare_train, remove_columns=columns)
            return ds.map(prepare_train)

        ds_primary = _load_dataset_with_retry(dataset_name, split=split, streaming=True)
        ds_primary = map_train_stream(ds_primary)

        ds = ds_primary
        if aux_dataset_name and aux_probability > 0:
            ds_aux = _load_dataset_with_retry(aux_dataset_name, split=split, streaming=True)
            ds_aux = map_train_stream(ds_aux)
            aux_p = min(max(aux_probability, 0.01), 0.99)
            ds = interleave_datasets(
                [ds_primary, ds_aux],
                probabilities=[1.0 - aux_p, aux_p],
                seed=42,
                stopping_strategy="all_exhausted",
            )
            print(
                f"[data] Training dataset stream mixed: primary={dataset_name}, "
                f"aux={aux_dataset_name}, aux_probability={aux_p:.2f}"
            )
        else:
            print(f"[data] Training dataset streaming from {dataset_name} (split={split})")

        return ds

    # In-memory mode: download data once, apply augmentation on-the-fly each access.
    # set_transform() runs prepare_train lazily every time the DataLoader fetches a
    # batch, so each epoch sees freshly augmented waveforms — no frozen artifacts.
    print(f"[data] Loading training dataset in-memory from {dataset_name} (split={split})...")
    ds = _load_dataset_with_retry(dataset_name, split=split, streaming=False)
    ds = ds.cast_column(AUDIO_COLUMN, Audio(sampling_rate=16000))

    def prepare_train_batch(examples: dict) -> dict:
        result = {"input_features": [], "labels": []}
        for audio, text in zip(examples[AUDIO_COLUMN], examples[text_column]):
            samples = np.array(audio["array"], dtype=np.float32)
            samples = augment(samples=samples, sample_rate=16000)
            feats = processor.feature_extractor(
                samples, sampling_rate=16000
            ).input_features[0]
            text = normalize_gurbani_text(text)
            labels = processor.tokenizer(text).input_ids
            result["input_features"].append(feats)
            result["labels"].append(labels)
        return result

    ds.set_transform(prepare_train_batch)
    print(f"[data] Training dataset: {len(ds)} examples with on-the-fly augmentation")
    return ds


def get_val_dataset(
    dataset_name: str,
    processor: Any,
    split: str = VAL_SPLIT,
    text_column: str = TEXT_COLUMN,
    val_size: int = VAL_SIZE,
) -> Dataset:
    """Load and materialize a fixed-size validation dataset (no augmentation).

    Takes the first `val_size` examples from the streaming dataset and
    materializes them into a regular in-memory Dataset. No augmentation is
    applied -- validation data must reflect real distribution.

    Args:
        dataset_name: HuggingFace dataset identifier.
        processor: WhisperProcessor with Punjabi tokenizer.
        split: Dataset split to load (default: from config VAL_SPLIT).
        text_column: Name of the transcription text column.
        val_size: Number of examples to materialize (default: from config VAL_SIZE).

    Returns:
        Regular Dataset with exactly `val_size` examples, each containing
        {"input_features", "labels"}.
    """
    val_size = int(os.environ.get("SURT_VAL_SIZE", val_size))
    stream = _load_dataset_with_retry(dataset_name, split=split, streaming=True)
    stream = stream.cast_column(AUDIO_COLUMN, Audio(sampling_rate=16000))

    val_examples = list(stream.take(val_size))

    # Materialize into regular Dataset
    try:
        val_dataset = Dataset.from_list(val_examples)
    except (AttributeError, TypeError):
        # Fallback for older datasets versions without from_list
        val_dataset = Dataset.from_dict(
            {k: [ex[k] for ex in val_examples] for k in val_examples[0]}
        )

    def prepare_val(example: dict) -> dict:
        audio = example[AUDIO_COLUMN]
        # NO augmentation on validation data
        input_features = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]

        # Tokenize Gurmukhi text (normalize to strip ॥ markers)
        text = normalize_gurbani_text(example[text_column])
        labels = processor.tokenizer(text).input_ids

        return {"input_features": input_features, "labels": labels}

    val_dataset = val_dataset.map(prepare_val, remove_columns=val_dataset.column_names)

    print(f"[data] Validation dataset: {len(val_dataset)} examples materialized in memory")
    return val_dataset


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Data collator that pads speech features and label sequences.

    Handles two critical Whisper-specific concerns:
    1. Masks padding tokens to -100 so they are ignored by the loss function
    2. Strips double BOS tokens that the tokenizer sometimes prepends

    Args:
        processor: WhisperProcessor for feature and label padding.
        decoder_start_token_id: Token ID that marks the start of decoding
            (used for double-BOS detection).
    """

    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: list[dict]) -> dict:
        # Separate input features and labels
        input_features = [
            {"input_features": f["input_features"]} for f in features
        ]
        label_features = [
            {"input_ids": f["labels"]} for f in features
        ]

        # Pad input features (log-Mel spectrograms)
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # Pad label sequences
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        # Mask padding tokens to -100 (ignored by cross-entropy loss)
        labels = labels_batch["input_ids"]
        attention_mask = labels_batch["attention_mask"]
        labels = labels.masked_fill(attention_mask.ne(1), -100)

        # Double-BOS guard: if ALL labels in batch start with decoder_start_token_id,
        # strip the first column to prevent double BOS which causes garbled output.
        if (labels[:, 0] == self.decoder_start_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch
