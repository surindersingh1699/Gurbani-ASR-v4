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

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from audiomentations import (
    AddGaussianNoise,
    Compose,
    PitchShift,
    RoomSimulator,
    TimeStretch,
)
from datasets import Audio, Dataset, IterableDataset, load_dataset

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
augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.4),
    RoomSimulator(p=0.3),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.2),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.2),
])

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
) -> IterableDataset:
    """Load streaming training dataset with waveform augmentation.

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

    Returns:
        Streaming IterableDataset yielding {"input_features", "labels"} dicts.
    """
    ds = _load_dataset_with_retry(dataset_name, split=split, streaming=True)
    ds = ds.cast_column(AUDIO_COLUMN, Audio(sampling_rate=16000))
    ds = ds.shuffle(seed=42, buffer_size=SHUFFLE_BUFFER)

    def prepare_train(example: dict) -> dict:
        audio = example[AUDIO_COLUMN]
        samples = np.array(audio["array"], dtype=np.float32)

        # Apply augmentation to raw waveform BEFORE feature extraction
        samples = augment(samples=samples, sample_rate=16000)

        # Extract 80-bin log-Mel spectrogram
        input_features = processor.feature_extractor(
            samples, sampling_rate=16000
        ).input_features[0]

        # Tokenize Gurmukhi text
        labels = processor.tokenizer(example[text_column]).input_ids

        return {"input_features": input_features, "labels": labels}

    # Streaming IterableDataset may not have .column_names -- handle gracefully
    try:
        columns = ds.column_names
    except (AttributeError, TypeError):
        columns = None

    if columns:
        ds = ds.map(prepare_train, remove_columns=columns)
    else:
        ds = ds.map(prepare_train)

    print(f"[data] Training dataset streaming from {dataset_name} (split={split})")
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
    stream = _load_dataset_with_retry(dataset_name, split=split, streaming=True)
    stream = stream.cast_column(AUDIO_COLUMN, Audio(sampling_rate=16000))

    # Take first val_size examples and materialize into list
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

        # Tokenize Gurmukhi text
        labels = processor.tokenizer(example[text_column]).input_ids

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
