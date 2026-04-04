# Phase 2: Data Pipeline and Model Initialization - Research

**Researched:** 2026-04-04
**Domain:** HuggingFace streaming datasets, Whisper model configuration, waveform augmentation, Gurmukhi tokenization
**Confidence:** HIGH

## Summary

Phase 2 is the highest-risk phase in the project because it concentrates every critical configuration decision that can silently cause catastrophic failure: wrong language output, double BOS tokens, incorrect padding, or augmentation leaking into validation. The standard approach is well-documented via HuggingFace's official Whisper fine-tuning guide and community event notebooks, but several sharp edges require careful attention.

The data pipeline has two distinct halves: (1) a streaming `IterableDataset` for training that applies waveform augmentation via audiomentations on-the-fly inside the `.map()` call, and (2) a fixed 300-example validation `Dataset` materialized eagerly into memory from the streaming source using `.take(300)` + `list()` + `Dataset.from_dict()`. These two halves use different dataset types and must be handled separately.

The model initialization must explicitly set `language="pa"`, `task="transcribe"` on both the processor/tokenizer AND `model.generation_config`, set `forced_decoder_ids=None`, and set `generation_max_length=448`. The `DataCollatorSpeechSeq2SeqWithPadding` must mask pad tokens to `-100` and strip the leading BOS token to prevent double-BOS corruption.

**Primary recommendation:** Implement the data pipeline and model initialization as two separate modules (`surt/data.py` and `surt/model.py`) following the canonical HuggingFace Whisper fine-tuning pattern, with explicit verification assertions for every configuration pitfall.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| DATA-01 | Training dataset streams from HuggingFace Hub with no local disk copy | Use `load_dataset(..., streaming=True)` returning `IterableDataset`. Verified in Context7 and official docs. |
| DATA-02 | Audio resampled to 16kHz via `cast_column("audio", Audio(sampling_rate=16000))` | Works on both `Dataset` and `IterableDataset`. Verified in Context7. |
| DATA-03 | Waveform augmentation on training data only (noise p=0.4, reverb p=0.3, time stretch p=0.2, pitch shift p=0.2) | Use `audiomentations.Compose` applied inside `.map()` on training split only. Verified in Context7. |
| DATA-04 | Fixed 300-example validation subset loaded eagerly into memory | Use `.take(300)` on streaming dataset, materialize via `list()` + `Dataset.from_dict()`. Reproducible with deterministic ordering before shuffle. |
| DATA-05 | Labels tokenized as Gurmukhi with pad tokens masked to -100 | Use `WhisperProcessor(language="punjabi", task="transcribe")` for tokenization, `DataCollatorSpeechSeq2SeqWithPadding` for -100 masking. Strip double BOS. |
| MODEL-01 | Base model is `openai/whisper-small` with full fine-tuning | Load with `WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")`. No LoRA/adapter. |
| MODEL-02 | Language set to `pa` and task set to `transcribe` on processor AND `model.generation_config` | Set on processor via `from_pretrained(language="punjabi", task="transcribe")` AND on `model.generation_config.language = "punjabi"`, `model.generation_config.task = "transcribe"`. |
| MODEL-03 | `forced_decoder_ids` set to `None` | Set `model.generation_config.forced_decoder_ids = None`. Prevents legacy forced token behavior that conflicts with language/task settings. |
| MODEL-04 | Mool Mantar used as `initial_prompt` during generation/eval | Pass as `prompt_ids` to `model.generate()` via the processor's `get_prompt_ids()` method. Already defined in `surt/config.py`. |
| MODEL-05 | `generation_max_length` set to 448 | Set `model.generation_config.max_length = 448`. Already defined in `surt/config.py` as `GENERATION_MAX_LENGTH = 448`. |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| transformers | >=4.46 (v5 compatible) | WhisperForConditionalGeneration, WhisperProcessor, Seq2SeqTrainer | Official HF implementation; only maintained Whisper fine-tuning path |
| datasets | >=2.16 | Streaming IterableDataset, Audio feature, cast_column | Official HF data loading; streaming mode eliminates disk requirement |
| audiomentations | >=0.42 | Waveform augmentation (noise, reverb, time stretch, pitch shift) | De facto standard for waveform-level audio augmentation; numpy-based, no torch dependency |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| evaluate | >=0.4 | WER metric loading via `evaluate.load("wer")` | During compute_metrics in eval loop |
| jiwer | >=4.0 | Backend for WER computation | Loaded by evaluate; also usable directly |
| torch | >=2.0 | Tensor operations in data collator, model inference | Core dependency throughout |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| audiomentations | torch-audiomentations | GPU-accelerated but adds CUDA dependency in data pipeline; CPU augmentation is fine for streaming |
| evaluate.load("wer") | jiwer directly | evaluate provides consistent API; jiwer is lighter but same backend |

**Installation:** Already in `requirements.txt`:
```bash
pip install transformers>=4.46 datasets>=2.16 audiomentations>=0.42 jiwer>=4.0 evaluate>=0.4 accelerate>=0.26 huggingface_hub>=0.20
```

## Architecture Patterns

### Recommended Module Structure
```
surt/
├── __init__.py       # existing
├── config.py         # existing -- GPU detection, constants, MOOL_MANTAR
├── data.py           # NEW -- streaming dataset, augmentation, data collator, prepare_dataset
└── model.py          # NEW -- model init, processor init, generation_config setup
```

### Pattern 1: Streaming Training + Eager Validation Split
**What:** Training data streams from HuggingFace Hub via `IterableDataset`; validation data is materialized into a fixed in-memory `Dataset` of exactly 300 examples.
**When to use:** Always for this project -- training set is too large for disk, validation must be reproducible.
**Example:**
```python
# Source: HuggingFace datasets docs (streaming guide) + HuggingFace blog (fine-tune-whisper)
from datasets import load_dataset, Audio, Dataset

# Training: streaming IterableDataset
train_stream = load_dataset("DATASET_NAME", split="train", streaming=True)
train_stream = train_stream.cast_column("audio", Audio(sampling_rate=16000))
# Apply augmentation + feature extraction via .map()
train_dataset = train_stream.map(prepare_dataset_train, remove_columns=...)

# Validation: materialize fixed 300 examples into regular Dataset
val_stream = load_dataset("DATASET_NAME", split="validation", streaming=True)
val_stream = val_stream.cast_column("audio", Audio(sampling_rate=16000))
val_examples = list(val_stream.take(300))  # eagerly load 300 examples
val_dataset = Dataset.from_list(val_examples)
val_dataset = val_dataset.map(prepare_dataset_val)  # NO augmentation
```

### Pattern 2: Separate prepare_dataset for Train vs Val
**What:** Two `prepare_dataset` functions -- one that applies augmentation (training) and one that does not (validation).
**When to use:** Always when augmentation is applied to training only.
**Example:**
```python
# Source: audiomentations docs + HuggingFace Whisper fine-tuning blog
from audiomentations import Compose, AddGaussianNoise, RoomSimulator, TimeStretch, PitchShift

augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.4),
    RoomSimulator(p=0.3),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.2),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.2),
])

def prepare_dataset_train(example):
    audio = example["audio"]
    # Apply augmentation to waveform BEFORE feature extraction
    samples = audio["array"].astype(np.float32)
    samples = augment(samples=samples, sample_rate=16000)
    # Compute log-Mel features from augmented waveform
    example["input_features"] = processor.feature_extractor(
        samples, sampling_rate=16000
    ).input_features[0]
    # Tokenize labels
    example["labels"] = processor.tokenizer(example["sentence"]).input_ids
    return example

def prepare_dataset_val(example):
    audio = example["audio"]
    # NO augmentation for validation
    example["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    example["labels"] = processor.tokenizer(example["sentence"]).input_ids
    return example
```

### Pattern 3: DataCollatorSpeechSeq2SeqWithPadding with Double-BOS Guard
**What:** Custom data collator that pads inputs/labels separately, masks pad tokens to -100, and strips the leading BOS token if present.
**When to use:** Always with Whisper Seq2SeqTrainer.
**Example:**
```python
# Source: HuggingFace blog (fine-tune-whisper)
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Mask padding tokens to -100 (ignored by CrossEntropyLoss)
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Strip double BOS token
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch
```

### Pattern 4: Model Initialization with Explicit Language/Task Config
**What:** Load Whisper model and set language, task, forced_decoder_ids, and generation_max_length on both processor and generation_config.
**When to use:** Always for non-English Whisper fine-tuning.
**Example:**
```python
# Source: HuggingFace blog (fine-tune-whisper) + transformers v5 docs
from transformers import WhisperForConditionalGeneration, WhisperProcessor

processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small", language="punjabi", task="transcribe"
)

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model.generation_config.language = "punjabi"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None
model.generation_config.max_length = 448
```

### Anti-Patterns to Avoid
- **Setting language/task only on processor but not on model.generation_config:** Model will generate English during eval even if processor encodes Punjabi labels correctly.
- **Using `forced_decoder_ids` instead of `language`/`task`:** Legacy approach that is 5x slower (one forward pass per forced token) and conflicts with modern `generation_config` settings.
- **Applying augmentation inside the data collator:** Augmentation must happen on the raw waveform BEFORE log-Mel extraction, not after.
- **Using `num_proc > 0` with streaming IterableDataset:** `IterableDataset.map()` does not support `num_proc`; processing is on-the-fly.
- **Using `evaluation_strategy` instead of `eval_strategy`:** Deprecated in transformers v4.46, removed in v5.
- **Using `tokenizer=` instead of `processing_class=` in Seq2SeqTrainer:** Deprecated in transformers v5. Use `processing_class=processor.feature_extractor` (or `processing_class=processor`).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Audio resampling | Custom scipy/librosa resampling | `datasets.Audio(sampling_rate=16000)` + `cast_column` | Handles codec detection, mono conversion, sample rate conversion automatically |
| Log-Mel feature extraction | Custom mel spectrogram | `WhisperProcessor.feature_extractor()` | Must match Whisper's exact 80-bin mel filterbank, normalization, and padding |
| Label tokenization for Gurmukhi | Custom Unicode tokenization | `WhisperProcessor.tokenizer(text)` with `language="punjabi"` | Whisper's tokenizer handles Gurmukhi Unicode, BPE encoding, and special tokens |
| Padding + -100 masking | Custom batch padding logic | `DataCollatorSpeechSeq2SeqWithPadding` | Must handle variable-length audio and labels with different padding strategies |
| WER computation | Custom string matching | `evaluate.load("wer")` / `jiwer` | Handles Unicode normalization, tokenization edge cases |
| Waveform augmentation | Custom numpy noise injection | `audiomentations.Compose(...)` | Parameterized, reproducible, handles sample rate, edge cases (clipping, silence) |

**Key insight:** Every component in the Whisper data pipeline must produce outputs that exactly match what Whisper expects. The feature extractor produces 80-bin log-Mel spectrograms with specific normalization; the tokenizer produces BPE token IDs with language/task prefix tokens. Hand-rolling any of these will produce silently wrong training data.

## Common Pitfalls

### Pitfall 1: Double BOS Token in Labels
**What goes wrong:** The tokenizer prepends a BOS (`<|startoftranscript|>`) token to labels. The model's decoder also prepends `decoder_start_token_id` during generation. If labels contain BOS, the model trains on double-BOS sequences, causing garbled output.
**Why it happens:** `processor.tokenizer(text).input_ids` includes special tokens by default.
**How to avoid:** The `DataCollatorSpeechSeq2SeqWithPadding` must check `if (labels[:, 0] == decoder_start_token_id).all()` and strip the first column.
**Warning signs:** Generated text starts with repeated special tokens or empty prefix before actual content.

### Pitfall 2: Language Misconfiguration -- Model Outputs English Instead of Gurmukhi
**What goes wrong:** Whisper generates English text despite being fine-tuned on Punjabi data.
**Why it happens:** Language is set on the processor/tokenizer (which affects label encoding) but NOT on `model.generation_config` (which controls `model.generate()`). The model defaults to English during eval.
**How to avoid:** Explicitly set `model.generation_config.language = "punjabi"` AND `model.generation_config.task = "transcribe"`. Verify with a single-sample `model.generate()` call before training.
**Warning signs:** WER is extremely high (>90%); decoded predictions are English text.

### Pitfall 3: Whisper Detects Punjabi as Urdu
**What goes wrong:** Whisper's language detection classifies Punjabi audio as Urdu ~99% of the time.
**Why it happens:** Punjabi and Urdu share significant phonetic similarity; Whisper has limited Punjabi training data (~0.06% of training set).
**How to avoid:** Set `forced_decoder_ids=None` and explicitly set `language="punjabi"` on generation_config. Never rely on Whisper's auto-language-detection for Punjabi.
**Warning signs:** Model outputs Urdu/Shahmukhi script instead of Gurmukhi.

### Pitfall 4: Augmentation Applied to Validation Data
**What goes wrong:** Validation metrics are noisy and non-reproducible across runs.
**Why it happens:** Using a single `prepare_dataset` function with augmentation for both train and val splits.
**How to avoid:** Use separate `prepare_dataset_train` (with augmentation) and `prepare_dataset_val` (without augmentation) functions. The validation `Dataset` is a regular (non-streaming) Dataset so `.map()` runs eagerly.
**Warning signs:** Validation WER fluctuates significantly between identical runs.

### Pitfall 5: Streaming Dataset Has No Length -- `num_train_epochs` Fails
**What goes wrong:** Trainer crashes or runs indefinitely because `IterableDataset` has no `__len__`.
**Why it happens:** Streaming datasets don't know their total size upfront.
**How to avoid:** Use `max_steps` instead of `num_train_epochs` in `Seq2SeqTrainingArguments`. Set `dataloader_num_workers=0` (multiprocessing does not work with streaming). Already noted in project requirements (TRAIN-05).
**Warning signs:** RuntimeError about dataset length; trainer hangs at epoch boundary.

### Pitfall 6: Augmentation After Feature Extraction (Wrong Order)
**What goes wrong:** Augmentation is applied to log-Mel features instead of raw waveform, producing nonsensical spectrograms.
**Why it happens:** Calling augmentation inside the data collator or after `feature_extractor()`.
**How to avoid:** Apply `audiomentations.Compose(...)` to the raw `audio["array"]` numpy waveform BEFORE passing to `processor.feature_extractor()`.
**Warning signs:** Training loss doesn't decrease; model produces random output.

### Pitfall 7: `processing_class` vs `tokenizer` in Seq2SeqTrainer (transformers v5)
**What goes wrong:** Deprecation warning or error when passing `tokenizer=` to Seq2SeqTrainer.
**Why it happens:** transformers v5 renamed `tokenizer` to `processing_class` in Trainer API.
**How to avoid:** Use `processing_class=processor.feature_extractor` (for padding in data collator). The HF blog example passes `processor.feature_extractor` as the tokenizer argument to Seq2SeqTrainer.
**Warning signs:** FutureWarning about deprecated `tokenizer` parameter.

## Code Examples

### Complete Model Initialization
```python
# Source: HuggingFace blog + transformers v5 docs
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Processor: sets language prefix tokens for label encoding
processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small",
    language="punjabi",
    task="transcribe",
)

# Model: full fine-tune (encoder + decoder)
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

# Generation config: controls model.generate() behavior during eval
model.generation_config.language = "punjabi"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None
model.generation_config.max_length = 448  # Gurmukhi tokenizer expansion

# Verification: generate on a dummy input to confirm Gurmukhi output
import torch
dummy_input = torch.randn(1, 80, 3000)  # 30s of silence
with torch.no_grad():
    generated_ids = model.generate(dummy_input.to(model.device))
decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
print(f"Verification output: {decoded}")
# Should contain Gurmukhi characters or empty string, NOT English
```

### Complete Streaming Data Pipeline
```python
# Source: HuggingFace datasets streaming docs + Whisper fine-tuning blog
from datasets import load_dataset, Audio, Dataset
import numpy as np
from audiomentations import Compose, AddGaussianNoise, RoomSimulator, TimeStretch, PitchShift

# Augmentation pipeline (training only)
augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.4),
    RoomSimulator(p=0.3),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.2),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.2),
])

def prepare_dataset_train(example):
    audio = example["audio"]
    samples = audio["array"].astype(np.float32)
    # Augment waveform BEFORE feature extraction
    samples = augment(samples=samples, sample_rate=16000)
    example["input_features"] = processor.feature_extractor(
        samples, sampling_rate=16000
    ).input_features[0]
    example["labels"] = processor.tokenizer(example["sentence"]).input_ids
    return example

def prepare_dataset_val(example):
    audio = example["audio"]
    example["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    example["labels"] = processor.tokenizer(example["sentence"]).input_ids
    return example

# --- Training dataset: streaming ---
train_stream = load_dataset("DATASET_NAME", split="train", streaming=True)
train_stream = train_stream.cast_column("audio", Audio(sampling_rate=16000))
train_dataset = train_stream.shuffle(seed=42, buffer_size=500).map(
    prepare_dataset_train,
    remove_columns=train_stream.column_names if hasattr(train_stream, 'column_names') else None,
)

# --- Validation dataset: fixed 300 examples, eagerly loaded ---
val_stream = load_dataset("DATASET_NAME", split="validation", streaming=True)
val_stream = val_stream.cast_column("audio", Audio(sampling_rate=16000))
val_examples = list(val_stream.take(300))
val_dataset = Dataset.from_list(val_examples)
val_dataset = val_dataset.map(prepare_dataset_val, remove_columns=val_dataset.column_names)
```

### Mool Mantar as Initial Prompt for Generation
```python
# Source: Whisper transformers docs (generation with prompt_ids)
from surt.config import MOOL_MANTAR

# Get prompt token IDs from the Mool Mantar text
prompt_ids = processor.get_decoder_prompt_ids(language="punjabi", task="transcribe")
# For initial_prompt, use tokenizer directly:
mool_mantar_ids = processor.tokenizer(MOOL_MANTAR, add_special_tokens=False).input_ids

# During generation/eval, pass as prompt_ids
generated_ids = model.generate(
    input_features,
    language="punjabi",
    task="transcribe",
    prompt_ids=torch.tensor(mool_mantar_ids, dtype=torch.long),
)
```

### Compute Metrics with WER
```python
# Source: HuggingFace blog (fine-tune-whisper)
import evaluate
metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    # Replace -100 with pad_token_id for decoding
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `forced_decoder_ids = [(1, lang_id), (2, task_id)]` | `generation_config.language = "punjabi"`, `generation_config.task = "transcribe"` | transformers ~4.36 (2023) | Simpler, 5x faster generation, no token-by-token forcing |
| `evaluation_strategy="steps"` | `eval_strategy="steps"` | transformers 4.46 (2024) | Old name removed in v5; must use new name |
| `Seq2SeqTrainer(tokenizer=...)` | `Seq2SeqTrainer(processing_class=...)` | transformers v5 (2025) | `tokenizer` param deprecated in favor of `processing_class` |
| `model.config.forced_decoder_ids` | `model.generation_config.forced_decoder_ids = None` | transformers ~4.36 | generation_config is the canonical location for all generation params |

**Deprecated/outdated:**
- `evaluation_strategy`: Removed in transformers v5. Use `eval_strategy`.
- `Trainer(tokenizer=...)`: Deprecated. Use `Trainer(processing_class=...)`.
- `forced_decoder_ids` for language/task: Legacy. Use `generation_config.language` and `generation_config.task`.
- `model.config.forced_decoder_ids`: Moved to `model.generation_config.forced_decoder_ids`.

## Open Questions

1. **Dataset path on HuggingFace Hub**
   - What we know: The project needs a Gurbani sehaj path audio dataset with `audio` and `sentence` columns.
   - What's unclear: The exact dataset name/path on HuggingFace Hub has not been provided by the user.
   - Recommendation: Block execution of Phase 2 plans until the dataset path is provided. The code can be written with a placeholder that is filled in at runtime from config.

2. **Column names in the dataset**
   - What we know: Standard ASR datasets use `audio` and `sentence` (Common Voice) or `audio` and `text`.
   - What's unclear: The actual column names in the user's dataset.
   - Recommendation: Code should accept configurable column names via `config.py`, defaulting to `audio` and `sentence`.

3. **Validation split availability**
   - What we know: The requirement says "300-example validation subset."
   - What's unclear: Whether the dataset has a pre-defined `validation` split, or if we need to take 300 from `train`.
   - Recommendation: If no `validation` split exists, take the first 300 examples from the stream before shuffling. Document this in config.

4. **`Dataset.from_list()` availability**
   - What we know: `Dataset.from_list()` is the cleanest way to materialize streaming examples. It was added in datasets ~2.14.
   - What's unclear: Whether the exact method name is `from_list` or if we need `from_dict` with manual key aggregation.
   - Recommendation: Use `Dataset.from_list()` (available in datasets >=2.14, we require >=2.16). Falls back to `Dataset.from_dict({k: [ex[k] for ex in examples] for k in examples[0]})`.

5. **`processing_class` exact value for Whisper**
   - What we know: The HuggingFace blog passes `processor.feature_extractor` as the old `tokenizer` arg. In v5, this becomes `processing_class`.
   - What's unclear: Whether `processing_class=processor` (the full WhisperProcessor) or `processing_class=processor.feature_extractor` is correct.
   - Recommendation: Pass `processing_class=processor` first; fall back to `processor.feature_extractor` if issues arise. The Trainer uses this for padding during prediction.

6. **Mool Mantar prompt_ids integration with Trainer eval**
   - What we know: `model.generate()` accepts `prompt_ids` for vocabulary anchoring. `MOOL_MANTAR` is defined in config.
   - What's unclear: How to pass `prompt_ids` through `Seq2SeqTrainer.predict_with_generate` -- it may require a custom `generate_kwargs` in training args or a callback.
   - Recommendation: Research `Seq2SeqTrainingArguments.generation_config` or `generate_kwargs` parameter to pass `prompt_ids` at eval time.

## Sources

### Primary (HIGH confidence)
- [Context7: /huggingface/transformers] - WhisperForConditionalGeneration, Seq2SeqTrainer, GenerationConfig, Whisper model docs
- [Context7: /llmstxt/huggingface_co_datasets_main_en_llms_txt] - Streaming IterableDataset, cast_column, Audio feature, map
- [Context7: /iver56/audiomentations] - Compose, AddGaussianNoise, RoomSimulator, TimeStretch, PitchShift
- [HuggingFace blog: fine-tune-whisper](https://huggingface.co/blog/fine-tune-whisper) - Canonical Whisper fine-tuning guide with DataCollator, compute_metrics, training args
- [HuggingFace datasets streaming docs](https://huggingface.co/docs/datasets/main/en/stream) - IterableDataset API: shuffle, take, skip, map, state_dict
- [HuggingFace datasets: Dataset vs IterableDataset](https://huggingface.co/docs/datasets/about_mapstyle_vs_iterable) - Differences, conversion patterns

### Secondary (MEDIUM confidence)
- [HuggingFace community events: Whisper streaming notebook](https://github.com/huggingface/community-events/blob/main/whisper-fine-tuning-event/fine_tune_whisper_streaming_colab.ipynb) - Verified streaming pattern with prepare_dataset
- [transformers issue #35446](https://github.com/huggingface/transformers/issues/35446) - `processing_class` replaces `tokenizer` in Seq2SeqTrainer (merged Jan 2025)
- [transformers issue #23845](https://github.com/huggingface/transformers/issues/23845) - `forced_decoder_ids` performance issues, recommendation to use `decoder_input_ids`
- [OpenAI Whisper tokenizer.py](https://github.com/openai/whisper/blob/main/whisper/tokenizer.py) - Punjabi language code `"pa"`, language name `"punjabi"`, alias `"panjabi"`
- [Whisper Punjabi discussion #2033](https://github.com/openai/whisper/discussions/2033) - Punjabi misidentified as Urdu; explicit language setting required

### Tertiary (LOW confidence)
- [Benchmarking Whisper for Punjabi (ACL 2025)](https://aclanthology.org/2025.chipsal-1.20/) - Few-shot fine-tuning reduces WER for Punjabi; Whisper Small is a viable base
- Web search results for streaming + augmentation patterns - Multiple community sources agree on waveform-first augmentation order

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All libraries verified via Context7 and official docs; versions match requirements.txt
- Architecture: HIGH - Patterns directly from HuggingFace's canonical Whisper fine-tuning guide and streaming docs
- Pitfalls: HIGH - Double BOS, forced_decoder_ids, and language misconfiguration are well-documented in official issues and blog posts
- Punjabi-specific: MEDIUM - Language code "pa"/"punjabi" verified in OpenAI source; Urdu misidentification confirmed by community reports but limited official documentation

**Research date:** 2026-04-04
**Valid until:** 2026-05-04 (30 days -- stable libraries, well-established patterns)
