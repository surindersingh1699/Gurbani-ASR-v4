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
        RoomSimulator,
        TimeStretch,
    )
    _HAS_AUGMENT = True
except ModuleNotFoundError:
    _HAS_AUGMENT = False
from datasets import Audio, Dataset, IterableDataset, concatenate_datasets, interleave_datasets, load_dataset

from surt.config import (
    GENERATION_MAX_LENGTH,
    SHUFFLE_BUFFER,
    TEXT_COLUMN,
    VAL_SIZE,
    VAL_SPLIT,
)

# --- Column names ---
AUDIO_COLUMN = "audio"

# --- Augmentation pipeline (training only) ---
# Applied to raw waveform BEFORE feature extraction.
# v3 tuning — kirtan is tonal/rhythmic content where pitch/tempo distortion
# corrupts raga tonal center and tabla cadence. Dropped PitchShift entirely
# and narrowed TimeStretch. Noise + reverb remain — those are environmental,
# not musical.
if _HAS_AUGMENT:
    augment = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.4),
        RoomSimulator(p=0.3),
        TimeStretch(min_rate=0.9, max_rate=1.1, p=0.1),
    ])
else:
    def augment(*, samples, sample_rate):  # type: ignore[override]
        return samples

# --- Retry configuration for HuggingFace Hub rate limits ---
_HF_MAX_RETRIES = 5
_HF_INITIAL_BACKOFF = 2.0  # seconds
_HF_BACKOFF_FACTOR = 2.0   # exponential multiplier


def _make_label_fits_filter(processor: Any, text_column: str, max_labels: int):
    """Return a predicate that drops rows whose tokenized label exceeds max_labels.

    Gurmukhi BPE expands 3–5× vs English, so a dense 30 s kirtan segment can
    exceed GENERATION_MAX_LENGTH=448 tokens. Those rows would be silently
    truncated during collation and train the model on incomplete targets.
    Cheap tokenizer-only check (no audio work) — ~a few minutes for 190k rows.
    """
    tokenizer = processor.tokenizer

    def _fits(example: dict) -> bool:
        raw = example.get(text_column)
        if raw is None:
            return False
        text = normalize_gurbani_text(raw)
        if not text:
            return False
        return len(tokenizer(text, add_special_tokens=True).input_ids) <= max_labels

    return _fits


def _load_dataset_with_retry(
    dataset_name: str,
    split: str,
    streaming: bool = True,
) -> IterableDataset | Dataset:
    """Load dataset from HuggingFace Hub with retry and exponential backoff.

    Streaming datasets from HuggingFace Hub can hit rate limits (HTTP 429).
    This wrapper retries with exponential backoff to handle transient failures.

    For non-streaming loads, parquet shards are downloaded in parallel
    (controlled by $SURT_DOWNLOAD_WORKERS, default 8) — HF caps single-stream
    at ~8 MB/s, so parallelism is the only way to saturate the link.

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
    # Parallel shard download — only meaningful when streaming=False.
    download_workers = int(os.environ.get("SURT_DOWNLOAD_WORKERS", "8"))

    for attempt in range(1, _HF_MAX_RETRIES + 1):
        try:
            load_kwargs: dict[str, Any] = {"split": split, "streaming": streaming}
            if not streaming and download_workers > 1:
                load_kwargs["num_proc"] = download_workers
            ds = load_dataset(dataset_name, **load_kwargs)
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
    extra_sehaj_dataset_name: str | None = None,
    extra_sehaj_text_column: str | None = None,
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

    label_fits_primary = _make_label_fits_filter(
        processor, text_column, GENERATION_MAX_LENGTH
    )

    if streaming:
        if extra_sehaj_dataset_name:
            # Streaming path is only used for smoke tests (~10 steps). Adding a
            # second sehaj interleave here isn't worth the complexity — pilot
            # and full runs both use non-streaming, where extra sehaj IS wired.
            print(
                "[data] NOTE: extra_sehaj_dataset_name is set but streaming=True. "
                "Extra sehaj is only concatenated in non-streaming mode; ignoring here."
            )

        def _process_stream(ds: IterableDataset, fits_fn) -> IterableDataset:
            """Cast audio, drop over-length labels, shuffle, prepare."""
            ds = ds.cast_column(AUDIO_COLUMN, Audio(sampling_rate=16000))
            ds = ds.filter(fits_fn)
            ds = ds.shuffle(seed=42, buffer_size=SHUFFLE_BUFFER)
            return ds.map(prepare_train)

        ds_primary = _load_dataset_with_retry(dataset_name, split=split, streaming=True)
        ds_primary = _process_stream(ds_primary, label_fits_primary)

        ds = ds_primary
        if aux_dataset_name and aux_probability > 0:
            aux_p = min(max(aux_probability, 0.01), 0.99)
            try:
                ds_aux = _load_dataset_with_retry(aux_dataset_name, split=split, streaming=True)
                ds_aux = ds_aux.cast_column(AUDIO_COLUMN, Audio(sampling_rate=16000))
                # Filter to ≤30s — longer segments degrade training quality
                ds_aux = ds_aux.filter(lambda x: x.get("duration", 0) <= 30)
                # Canonical kirtan exposes `final_text` (same name as primary sehaj
                # via TEXT_COLUMN), so no column rename is needed. If a future aux
                # dataset uses a different column, harmonize here with a .map().
                ds_aux = ds_aux.filter(
                    _make_label_fits_filter(processor, text_column, GENERATION_MAX_LENGTH)
                )
                ds_aux = ds_aux.shuffle(seed=42, buffer_size=SHUFFLE_BUFFER)
                ds_aux = ds_aux.map(prepare_train)

                # Manual interleave — interleave_datasets can't handle
                # processed numpy arrays (PyArrow serialization fails)
                import random as _rng_mod
                _rng = _rng_mod.Random(42)

                def _interleaved_gen():
                    it1, it2 = iter(ds_primary), iter(ds_aux)
                    while True:
                        use_aux = _rng.random() < aux_p
                        src_iter = it2 if use_aux else it1
                        try:
                            yield next(src_iter)
                        except StopIteration:
                            # Restart exhausted stream
                            if use_aux:
                                it2 = iter(ds_aux)
                                yield next(it2)
                            else:
                                it1 = iter(ds_primary)
                                yield next(it1)

                ds = IterableDataset.from_generator(_interleaved_gen)
                print(
                    f"[data] Training dataset stream mixed: primary={dataset_name}, "
                    f"aux={aux_dataset_name}, aux_probability={aux_p:.2f}"
                )
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(
                    f"[data] WARNING: aux dataset unavailable ({aux_dataset_name}): {e}. "
                    "Continuing with primary dataset only."
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
    _pre_filter_n = len(ds)
    ds = ds.filter(label_fits_primary)
    _dropped = _pre_filter_n - len(ds)
    if _dropped:
        print(
            f"[data] Dropped {_dropped}/{_pre_filter_n} primary rows with "
            f"labels > {GENERATION_MAX_LENGTH} tokens"
        )

    # Optional: concatenate an additional sehaj source with the primary sehaj
    # stream BEFORE aux (kirtan) mixing. Column name is harmonized to
    # `text_column` (the primary canonical name). Rows whose labels exceed
    # GENERATION_MAX_LENGTH are filtered using the same tokenizer check as primary.
    if extra_sehaj_dataset_name:
        try:
            print(
                f"[data] Loading extra sehaj in-memory from "
                f"{extra_sehaj_dataset_name} (col={extra_sehaj_text_column})..."
            )
            ds_extra = _load_dataset_with_retry(
                extra_sehaj_dataset_name, split=split, streaming=False
            )
            ds_extra = ds_extra.cast_column(AUDIO_COLUMN, Audio(sampling_rate=16000))

            # Rename source text column → canonical text_column if they differ.
            if extra_sehaj_text_column and extra_sehaj_text_column != text_column:
                _src_col = extra_sehaj_text_column
                _dst_col = text_column
                ds_extra = ds_extra.map(lambda x: {_dst_col: x[_src_col]})

            _extra_pre = len(ds_extra)
            ds_extra = ds_extra.filter(
                _make_label_fits_filter(processor, text_column, GENERATION_MAX_LENGTH)
            )
            _extra_dropped = _extra_pre - len(ds_extra)
            if _extra_dropped:
                print(
                    f"[data] Dropped {_extra_dropped}/{_extra_pre} extra sehaj rows "
                    f"with labels > {GENERATION_MAX_LENGTH} tokens"
                )

            # Harmonize columns across both sehaj streams before concatenating.
            keep_cols = {AUDIO_COLUMN, text_column}
            ds = ds.remove_columns([c for c in ds.column_names if c not in keep_cols])
            ds_extra = ds_extra.remove_columns(
                [c for c in ds_extra.column_names if c not in keep_cols]
            )

            _primary_n = len(ds)
            ds = concatenate_datasets([ds, ds_extra])
            print(
                f"[data] Sehaj streams concatenated: primary={_primary_n} + "
                f"extra={len(ds_extra)} = {len(ds)}"
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(
                f"[data] WARNING: extra sehaj unavailable ({extra_sehaj_dataset_name}): {e}. "
                "Continuing with primary sehaj only."
            )

    # Mix in auxiliary (kirtan) dataset if provided
    if aux_dataset_name and aux_probability > 0:
        aux_p = min(max(aux_probability, 0.01), 0.99)
        try:
            print(f"[data] Loading aux dataset in-memory from {aux_dataset_name}...")
            ds_aux = _load_dataset_with_retry(aux_dataset_name, split=split, streaming=False)
            ds_aux = ds_aux.cast_column(AUDIO_COLUMN, Audio(sampling_rate=16000))
            # Filter to ≤30s — longer segments degrade training quality
            ds_aux = ds_aux.filter(lambda x: x.get("duration", 0) <= 30)
            # Canonical kirtan exposes TEXT_COLUMN directly — no column rename.
            _aux_pre_n = len(ds_aux)
            ds_aux = ds_aux.filter(
                _make_label_fits_filter(processor, text_column, GENERATION_MAX_LENGTH)
            )
            _aux_dropped = _aux_pre_n - len(ds_aux)
            if _aux_dropped:
                print(
                    f"[data] Dropped {_aux_dropped}/{_aux_pre_n} aux rows with "
                    f"labels > {GENERATION_MAX_LENGTH} tokens"
                )

            # Harmonize columns: keep only audio + text_column
            keep_cols = {AUDIO_COLUMN, text_column}
            ds = ds.remove_columns([c for c in ds.column_names if c not in keep_cols])
            ds_aux = ds_aux.remove_columns([c for c in ds_aux.column_names if c not in keep_cols])

            # Oversample kirtan to hit target ratio
            # target_aux_count = primary_n * p / (1-p)
            primary_n = len(ds)
            target_aux_count = int(primary_n * aux_p / (1.0 - aux_p))
            repeats = target_aux_count // len(ds_aux)
            remainder = target_aux_count % len(ds_aux)

            aux_parts = [ds_aux] * repeats
            if remainder > 0:
                aux_parts.append(ds_aux.select(range(remainder)))
            ds_aux_oversampled = concatenate_datasets(aux_parts)

            ds = concatenate_datasets([ds, ds_aux_oversampled])
            ds = ds.shuffle(seed=42)

            print(
                f"[data] Mixed training: primary={primary_n}, "
                f"aux={len(ds_aux)}x{repeats}+{remainder}={len(ds_aux_oversampled)}, "
                f"total={len(ds)}, target_aux_ratio={aux_p:.0%}"
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(
                f"[data] WARNING: aux dataset unavailable ({aux_dataset_name}): {e}. "
                "Continuing with primary dataset only."
            )

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


def get_kirtan_val_dataset(
    dataset_name: str,
    processor: Any,
    split: str = "validation",
    text_column: str = TEXT_COLUMN,
    val_size: int = 400,
    max_duration: float = 30.0,
) -> Dataset:
    """Load a kirtan validation set for separate eval alongside sehaj path.

    Uses the dataset's own validation split. Falls back to sampling from train
    if no validation split exists.

    Args:
        dataset_name: HuggingFace kirtan dataset identifier.
        processor: WhisperProcessor with Punjabi tokenizer.
        split: Dataset split to use (default: "validation").
        text_column: Name of the transcription column (default: gurmukhi_text).
        val_size: Number of examples to materialize.
        max_duration: Maximum audio duration in seconds.

    Returns:
        Regular Dataset with {"input_features", "labels"} dicts.
    """
    stream = _load_dataset_with_retry(dataset_name, split=split, streaming=True)
    stream = stream.cast_column(AUDIO_COLUMN, Audio(sampling_rate=16000))
    stream = stream.filter(lambda x: x.get("duration", 0) <= max_duration)
    # Skip eval rows whose labels would be truncated — WER on those is meaningless.
    stream = stream.filter(
        _make_label_fits_filter(processor, text_column, GENERATION_MAX_LENGTH)
    )

    val_examples = list(stream.take(val_size))

    try:
        val_dataset = Dataset.from_list(val_examples)
    except (AttributeError, TypeError):
        val_dataset = Dataset.from_dict(
            {k: [ex[k] for ex in val_examples] for k in val_examples[0]}
        )

    def prepare_val(example: dict) -> dict:
        audio = example[AUDIO_COLUMN]
        input_features = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]

        text = normalize_gurbani_text(example[text_column])
        labels = processor.tokenizer(text).input_ids

        return {"input_features": input_features, "labels": labels}

    val_dataset = val_dataset.map(prepare_val, remove_columns=val_dataset.column_names)

    print(f"[data] Kirtan validation dataset: {len(val_dataset)} examples materialized in memory")
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
