---
phase: 02-data-pipeline-and-model-initialization
verified: 2026-04-04T14:00:00Z
status: human_needed
score: 9/9 must-haves verified
re_verification: false
human_verification:
  - test: "Run load_model_and_processor() on RunPod A40 and assert model.generation_config.language == 'punjabi'"
    expected: "Prints '[model] Loaded openai/whisper-small with language=punjabi, task=transcribe, max_length=448'. All generation_config asserts pass."
    why_human: "RunPod was stopped during both plans. Local CPU verification was done by the executor but not captured as a test artifact. Confirming on the actual A40 GPU removes any doubt about transformers version compatibility."
  - test: "Run get_train_dataset('surindersinghssj/gurbani-asr', processor) and consume one sample; inspect input_features shape and labels"
    expected: "Sample has input_features of shape (80, 3000), labels is a non-empty list of ints containing Gurmukhi BPE tokens, no English text in decoded output."
    why_human: "Streaming pipeline against the real HF dataset was never run (SSH unavailable both times). Structural checks pass; actual streaming + augmentation path is unverified at runtime."
  - test: "Run get_val_dataset('surindersinghssj/gurbani-asr', processor, val_size=5) and check len(val_ds) == 5 and hasattr(val_ds, '__len__')"
    expected: "Returns a regular Dataset (not IterableDataset) with exactly 5 examples, each with input_features and labels keys."
    why_human: "Same SSH unavailability. Materialisation from streaming via Dataset.from_list must be confirmed against the actual gurbani-asr parquet files."
  - test: "Run DataCollatorSpeechSeq2SeqWithPadding on a batch of 2 val examples and inspect batch['labels']"
    expected: "labels tensor has shape (2, N), contains -100 values for padding positions, and does NOT start with a duplicate decoder_start_token_id."
    why_human: "Collator -100 masking and double-BOS strip require a real tokenized batch to confirm; cannot assert tensor values from static analysis alone."
---

# Phase 2: Data Pipeline and Model Initialization — Verification Report

**Phase Goal:** The streaming data pipeline produces correctly preprocessed Gurmukhi training examples, and the model is initialized with the right language, task, and generation settings to prevent catastrophic misconfiguration.
**Verified:** 2026-04-04T14:00:00Z
**Status:** human_needed (all automated checks passed; runtime against real GPU/dataset deferred)
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| #  | Truth | Status | Evidence |
|----|-------|--------|----------|
| 1  | Processor tokenizes text with Punjabi language prefix tokens | VERIFIED | `WhisperProcessor.from_pretrained(BASE_MODEL, language="punjabi", task="transcribe")` — line 36-40 of model.py |
| 2  | model.generate() on real audio will produce Gurmukhi tokens (not English) | VERIFIED (static) / ? RUNTIME | generation_config has language="punjabi", task="transcribe", forced_decoder_ids=None set on model.generation_config (lines 47-53 model.py); runtime against real audio needs human check |
| 3  | generation_config has language=punjabi, task=transcribe, forced_decoder_ids=None, max_length=448 | VERIFIED | All four assignments present in model.py lines 47-56; uses GENERATION_MAX_LENGTH constant (448) not hardcoded |
| 4  | Mool Mantar prompt_ids are available for generation/eval anchoring | VERIFIED | get_mool_mantar_prompt_ids() calls processor.tokenizer(MOOL_MANTAR, add_special_tokens=False).input_ids — lines 80-83 model.py; executor confirmed 137 tokens returned locally |
| 5  | Training dataset streams from HuggingFace Hub without downloading to disk | VERIFIED | load_dataset(..., streaming=True) in _load_dataset_with_retry; get_train_dataset returns IterableDataset |
| 6  | Audio is resampled to 16kHz before feature extraction | VERIFIED | .cast_column(AUDIO_COLUMN, Audio(sampling_rate=16000)) in both get_train_dataset (line 133) and get_val_dataset (line 193) |
| 7  | Augmentation (noise, reverb, time stretch, pitch shift) is applied to training examples only | VERIFIED | augment() called in prepare_train before feature_extractor (line 141); prepare_val has explicit comment and NO augment() call confirmed by source inspection |
| 8  | Validation subset is exactly 300 examples loaded eagerly into memory | VERIFIED | list(stream.take(val_size)) + Dataset.from_list() pattern; VAL_SIZE=300 in config.py; val_size parameter defaults to VAL_SIZE |
| 9  | Labels are tokenized as Gurmukhi with pad tokens masked to -100 and no double BOS | VERIFIED | labels.masked_fill(attention_mask.ne(1), -100) at line 264; double-BOS guard strips first column when all rows start with decoder_start_token_id (lines 268-269) |

**Score:** 9/9 truths verified (static analysis; 4 items flagged for runtime human check)

---

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `surt/model.py` | Model and processor initialization with Gurmukhi configuration | VERIFIED | 85 lines; exports load_model_and_processor and get_mool_mantar_prompt_ids; no stubs, no TODOs |
| `surt/data.py` | Streaming data pipeline, augmentation, data collator | VERIFIED | 273 lines; exports get_train_dataset, get_val_dataset, DataCollatorSpeechSeq2SeqWithPadding; fully implemented |
| `surt/config.py` | Dataset and model constants (modified) | VERIFIED | All 9 expected constants present: DATASET_NAME, TRAIN_SPLIT, VAL_SPLIT, TEXT_COLUMN, VAL_SIZE, SHUFFLE_BUFFER, BASE_MODEL, GENERATION_MAX_LENGTH, MOOL_MANTAR |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `surt/model.py` | `surt/config.py` | `from surt.config import BASE_MODEL, GENERATION_MAX_LENGTH, MOOL_MANTAR` | WIRED | Exact import present; all three constants used (BASE_MODEL in from_pretrained calls, GENERATION_MAX_LENGTH in generation_config.max_length, MOOL_MANTAR in tokenizer call) |
| `surt/model.py` | `transformers` | `from transformers import WhisperForConditionalGeneration, WhisperProcessor` | WIRED | Import present; both classes used in load_model_and_processor |
| `surt/data.py` | `surt/config.py` | `from surt.config import SHUFFLE_BUFFER, VAL_SIZE, VAL_SPLIT` | WIRED | Import present; SHUFFLE_BUFFER used in shuffle(), VAL_SIZE and VAL_SPLIT used as defaults in get_val_dataset signature |
| `surt/data.py` | `datasets` | `from datasets import Audio, Dataset, IterableDataset, load_dataset` | WIRED | All four used: load_dataset in _load_dataset_with_retry, Audio in cast_column, Dataset.from_list in get_val_dataset, IterableDataset in type hints |
| `surt/data.py` | `audiomentations` | `from audiomentations import Compose, AddGaussianNoise, RoomSimulator, TimeStretch, PitchShift` | WIRED | All five used in augment Compose definition (module-level); augment() called in prepare_train |
| `surt/data.py` | `surt/model.py` | processor injected as parameter — no import | WIRED (dependency injection) | data.py does NOT import from surt.model; processor typed as Any and accepted as parameter to all public functions |

---

## Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| MODEL-01 | 02-01-PLAN.md | Base model is openai/whisper-small with full fine-tuning (no adapters) | SATISFIED | WhisperForConditionalGeneration.from_pretrained("openai/whisper-small") — no PEFT, no LoRA, no adapter imports anywhere in model.py |
| MODEL-02 | 02-01-PLAN.md | language=punjabi and task=transcribe set on both processor AND model.generation_config | SATISFIED | Set in processor.from_pretrained args (lines 37-40) AND model.generation_config.language/task (lines 47-48) |
| MODEL-03 | 02-01-PLAN.md | forced_decoder_ids explicitly set to None | SATISFIED | model.generation_config.forced_decoder_ids = None (line 53); comment explains model.config variant is wrong |
| MODEL-04 | 02-01-PLAN.md | Mool Mantar tokenized into prompt_ids for generation anchoring | SATISFIED | get_mool_mantar_prompt_ids returns processor.tokenizer(MOOL_MANTAR, add_special_tokens=False).input_ids |
| MODEL-05 | 02-01-PLAN.md | generation_max_length set to 448 on model.generation_config | SATISFIED | model.generation_config.max_length = GENERATION_MAX_LENGTH; GENERATION_MAX_LENGTH = 448 in config.py |
| DATA-01 | 02-02-PLAN.md | Training dataset loads with streaming=True, returns IterableDataset | SATISFIED | _load_dataset_with_retry calls load_dataset(..., streaming=streaming); get_train_dataset returns IterableDataset |
| DATA-02 | 02-02-PLAN.md | Audio cast to 16kHz via cast_column("audio", Audio(sampling_rate=16000)) | SATISFIED | cast_column present in both get_train_dataset and get_val_dataset with Audio(sampling_rate=16000) |
| DATA-03 | 02-02-PLAN.md | Augmentation pipeline with correct probabilities applied to training only (noise p=0.4, reverb p=0.3, stretch p=0.2, pitch p=0.2) | SATISFIED | All four probabilities confirmed verbatim in augment Compose; augment() only in prepare_train; prepare_val confirmed clean |
| DATA-04 | 02-02-PLAN.md | Fixed 300-example validation subset materialized as regular Dataset | SATISFIED | stream.take(val_size) + Dataset.from_list; val_size defaults to VAL_SIZE=300; fallback to Dataset.from_dict for older versions |
| DATA-05 | 02-02-PLAN.md | Labels tokenized as Gurmukhi, pad tokens masked to -100, double-BOS stripped in collator | SATISFIED | processor.tokenizer(...).input_ids; masked_fill(attention_mask.ne(1), -100); labels[:, 1:] when all rows start with decoder_start_token_id |

**Orphaned requirements check:** REQUIREMENTS.md traceability table maps DATA-01 through DATA-05 and MODEL-01 through MODEL-05 to Phase 2. Both plans claim exactly these 10 IDs. No orphaned requirements.

---

## Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `surt/model.py` | 14, 52 | `model.config.forced_decoder_ids` mentioned | INFO | Appears only in docstring/comment as the WRONG approach. Actual code correctly uses model.generation_config. False positive — not an anti-pattern. |

No blocker or warning anti-patterns found across surt/model.py, surt/data.py, or surt/config.py.

---

## Human Verification Required

All automated structural checks passed. The following four runtime checks require a live RunPod A40 session because SSH was unavailable during both plan executions.

### 1. Model generation_config on GPU

**Test:** SSH into RunPod, install deps, run:
```python
from surt.model import load_model_and_processor
model, processor = load_model_and_processor()
gc = model.generation_config
assert gc.language == "punjabi"
assert gc.task == "transcribe"
assert gc.forced_decoder_ids is None
assert gc.max_length == 448
prompt_ids = __import__('surt.model', fromlist=['get_mool_mantar_prompt_ids']).get_mool_mantar_prompt_ids(processor)
assert len(prompt_ids) > 0
print("PASS")
```
**Expected:** Prints "[model] Loaded openai/whisper-small..." then PASS.
**Why human:** Executor ran this locally on CPU (transformers 5.4.0); needs confirmation on actual A40 with production transformers version.

### 2. Streaming pipeline against real dataset

**Test:** Run:
```python
from surt.model import load_model_and_processor
from surt.data import get_train_dataset
_, processor = load_model_and_processor()
ds = get_train_dataset("surindersinghssj/gurbani-asr", processor)
sample = next(iter(ds))
assert "input_features" in sample
assert len(sample["input_features"]) == 80
assert len(sample["labels"]) > 0
print("PASS — labels:", sample["labels"][:5])
```
**Expected:** Returns sample with 80-bin mel features and non-empty Gurmukhi token IDs.
**Why human:** Real HuggingFace Hub streaming and actual gurbani-asr parquet files were never touched at runtime; structural code is correct but end-to-end requires network access and the actual dataset.

### 3. Validation dataset materialisation

**Test:** Run:
```python
from surt.data import get_val_dataset
val_ds = get_val_dataset("surindersinghssj/gurbani-asr", processor, val_size=5)
assert hasattr(val_ds, '__len__')
assert len(val_ds) == 5
print("PASS — val type:", type(val_ds).__name__)
```
**Expected:** Regular in-memory Dataset with exactly 5 examples.
**Why human:** Dataset.from_list path must be confirmed against real parquet audio data.

### 4. DataCollator -100 masking and BOS strip

**Test:** Run:
```python
from surt.data import get_val_dataset, DataCollatorSpeechSeq2SeqWithPadding
val_ds = get_val_dataset("surindersinghssj/gurbani-asr", processor, val_size=3)
collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, decoder_start_token_id=model.config.decoder_start_token_id)
batch = collator([val_ds[0], val_ds[1]])
assert (batch["labels"] == -100).any(), "No -100 masking"
print("PASS — labels shape:", batch["labels"].shape)
```
**Expected:** labels tensor has shape (2, N) with -100 values for padding.
**Why human:** Tensor values cannot be asserted from static analysis; requires tokenized real examples.

---

## Gaps Summary

No gaps found. All 10 requirements (MODEL-01 through MODEL-05, DATA-01 through DATA-05) are fully implemented in the codebase with substantive, wired code. Both artifacts exist, are non-trivial, and are connected to their dependencies.

The only outstanding work is runtime verification of the streaming pipeline against the real `surindersinghssj/gurbani-asr` dataset on a live RunPod pod. This is recommended before beginning Phase 3 (training loop), as both summaries note SSH was unavailable during execution.

---

_Verified: 2026-04-04T14:00:00Z_
_Verifier: Claude (gsd-verifier)_
