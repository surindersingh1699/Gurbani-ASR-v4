# Architecture Research

**Domain:** Whisper fine-tuning pipeline for Gurbani ASR (RunPod A40 via SSH)
**Researched:** 2026-04-04
**Confidence:** HIGH

## Standard Architecture

### System Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           LOCAL (Developer Machine)                          │
│  ┌──────────────┐                                          ┌─────────────┐  │
│  │ Claude Code  │─── SSH ──────────────────────────────────▶│  RunPod A40 │  │
│  │ (orchestrator│                                          │  (GPU node) │  │
│  └──────────────┘                                          └──────┬──────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
                                                                    │
                    ┌───────────────────────────────────────────────┐│
                    │              REMOTE (RunPod A40)               ││
                    │                                               ▼│
┌───────────────┐   │  ┌─────────┐   ┌──────────┐   ┌───────────┐  │
│  HuggingFace  │◀──┼──│Checkpoint│◀──│ Training │◀──│   Data    │──┼──┐
│     Hub       │   │  │  Pusher  │   │   Loop   │   │  Pipeline │  │  │
│ (model store) │   │  └─────────┘   └──────────┘   └───────────┘  │  │
└───────┬───────┘   │       ▲              ▲              ▲         │  │
        │           │       │              │              │         │  │
        │           │  ┌────┴────┐   ┌─────┴─────┐  ┌────┴──────┐ │  │
        │           │  │  HfApi  │   │Seq2Seq    │  │  Whisper  │ │  │
        │           │  │ upload  │   │Trainer +  │  │ Processor │ │  │
        │           │  │ folder  │   │Callbacks  │  │ + Augment │ │  │
        │           │  └─────────┘   └───────────┘  └───────────┘ │  │
        │           └───────────────────────────────────────────────┘  │
        │                                                              │
        └──────────── streams FLAC audio on demand ────────────────────┘
                                                          ▲
                                                          │
                                                ┌─────────┴─────────┐
                                                │  HuggingFace      │
                                                │  Datasets         │
                                                │  (100h sehaj path)│
                                                └───────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Communicates With |
|-----------|---------------|-------------------|
| **Config Module** | Central hyperparameters, paths, tokens, GPU-adaptive batch sizing | All components read from it |
| **Data Pipeline** | Stream audio from HF, resample to 16kHz, extract log-Mel spectrograms, tokenize Gurmukhi labels | HuggingFace Datasets (input), Training Loop (output) |
| **Audio Augmentation** | Apply noise, room sim, time stretch, pitch shift to training audio | Data Pipeline (inline transform) |
| **WhisperProcessor** | Wraps WhisperFeatureExtractor + WhisperTokenizer; converts raw audio to `input_features` and text to `labels` | Data Pipeline (called per example) |
| **Data Collator** | Pads variable-length label sequences per batch, replaces padding with -100 for loss masking | Training Loop (batch assembly) |
| **Model Setup** | Load `openai/whisper-small`, configure generation config (`language="pa"`, `task="transcribe"`), set discriminative LR optimizer | Training Loop (model + optimizer) |
| **Training Loop (Seq2SeqTrainer)** | Orchestrates forward/backward, gradient accumulation, FP16, gradient checkpointing, evaluation, checkpoint saving | All other components |
| **Checkpoint Pusher (Callback)** | On each eval step, pushes best checkpoint folder to HuggingFace Hub via `HfApi.upload_folder` | HuggingFace Hub (output), Trainer (triggered by eval events) |
| **Resume Logic** | Detects existing checkpoints (local or HF Hub), passes path to `trainer.train(resume_from_checkpoint=...)` | Trainer (startup), HF Hub (download if needed) |
| **WER Evaluation** | Decodes predictions + labels, computes Word Error Rate via `evaluate` library | Training Loop (called at eval steps) |

## Recommended Project Structure

```
src/
├── config.py              # All hyperparameters, paths, HF token loading, GPU detection
├── data.py                # Dataset loading (streaming), preprocessing, augmentation
├── collator.py            # DataCollatorSpeechSeq2SeqWithPadding
├── model.py               # Model loading, generation config, discriminative LR optimizer
├── callbacks.py           # PushBestToHub callback, optional logging callbacks
├── metrics.py             # WER compute_metrics function
├── train.py               # Main entrypoint: wires everything, calls trainer.train()
└── requirements.txt       # Pinned dependencies
```

### Structure Rationale

- **`config.py`:** Single source of truth for all tunable values. GPU auto-detection lives here because batch size depends on it. HF token loaded from environment variable (never hardcoded). This is the only file that needs editing between runs.
- **`data.py`:** Isolates all HuggingFace Datasets and audiomentations logic. Streaming mode means no disk usage on RunPod. Augmentation is applied inline during `.map()` on the IterableDataset.
- **`collator.py`:** Separated because the data collator has specific padding logic (labels padded with -100, input features already fixed-size from Whisper's 30s window). Easy to test independently.
- **`model.py`:** Model loading, generation config setup, and optimizer construction are tightly coupled (discriminative LR requires knowing model parameter groups). Kept together.
- **`callbacks.py`:** The HuggingFace Hub push callback is the critical checkpoint safety mechanism. Isolated for clarity and testability.
- **`train.py`:** Thin orchestration script. Imports from all other modules, constructs the Seq2SeqTrainer, handles resume logic, calls `.train()`. This is what gets executed via SSH.

## Architectural Patterns

### Pattern 1: Streaming Data Pipeline (No Local Disk)

**What:** Load audio dataset from HuggingFace Hub in streaming mode (`streaming=True`), apply preprocessing via `.map()`, and shuffle via an in-memory buffer. Audio never written to RunPod disk.

**When to use:** Always for this project. RunPod disk is ephemeral and limited; the dataset is 100h of FLAC (~9GB) hosted on HuggingFace.

**Trade-offs:**
- Pro: Zero disk usage, immediate start, no upload step
- Pro: Shuffle buffer provides adequate randomization for training
- Con: Cannot do random access (no `ds[i]`); must use IterableDataset patterns
- Con: Cannot compute exact dataset length upfront (use `max_steps` instead of `num_train_epochs`)
- Con: `set_epoch()` must be called for proper reshuffling across epochs

**Example:**
```python
from datasets import load_dataset, Audio

ds = load_dataset(HF_DATASET, split="train", streaming=True, token=HF_TOKEN)
ds = ds.cast_column("audio", Audio(sampling_rate=16000))
ds = ds.shuffle(seed=42, buffer_size=1000)
ds = ds.map(prepare_example)
```

**Confidence:** HIGH -- verified via Context7 (HuggingFace Datasets docs) and official HuggingFace Whisper fine-tuning blog.

### Pattern 2: Discriminative Learning Rates via Custom Optimizer

**What:** Assign different learning rates to encoder (5e-5) and decoder (1e-4) parameter groups. The encoder has useful pretrained audio representations; the decoder must learn Gurmukhi nearly from scratch.

**When to use:** When fine-tuning Whisper for a low-resource language where the decoder vocabulary is drastically different from pretraining data.

**Trade-offs:**
- Pro: Encoder retains general audio knowledge while decoder adapts faster
- Pro: Prevents catastrophic forgetting of encoder representations
- Con: Requires passing custom optimizer to Seq2SeqTrainer via `optimizers=(optimizer, None)`, bypassing the built-in LR scheduler
- Con: Must manually construct AdamW with parameter groups

**Example:**
```python
from torch.optim import AdamW

optimizer = AdamW([
    {"params": model.model.encoder.parameters(), "lr": 5e-5},
    {"params": model.model.decoder.parameters(), "lr": 1e-4},
    {"params": model.proj_out.parameters(),      "lr": 1e-4},
])
trainer = Seq2SeqTrainer(..., optimizers=(optimizer, None))
```

**Confidence:** HIGH -- standard pattern documented in HuggingFace Trainer API; the specific LR split is a domain decision validated by the training plan.

### Pattern 3: Callback-Driven Checkpoint Safety

**What:** A `TrainerCallback` fires `on_evaluate`, checks if a new best model checkpoint exists, and pushes the entire checkpoint folder to HuggingFace Hub via `HfApi.upload_folder`.

**When to use:** When training on ephemeral compute (RunPod, Colab, spot instances) where the machine can disappear at any time.

**Trade-offs:**
- Pro: At most 300 steps (one eval interval) of work lost if session dies
- Pro: Checkpoints accessible from any machine for resume
- Pro: HuggingFace Hub provides versioned storage with commit messages (step + WER logged)
- Con: Upload adds ~30-60s per eval step (network-bound, runs after eval completes)
- Con: Must handle HF authentication on RunPod (token passed via environment variable)

**Example:**
```python
from transformers import TrainerCallback
from huggingface_hub import HfApi

class PushBestToHub(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if state.best_model_checkpoint:
            HfApi().upload_folder(
                folder_path=state.best_model_checkpoint,
                repo_id=HF_MODEL_REPO,
                repo_type="model",
                commit_message=f"step {state.global_step} wer={metrics.get('eval_wer', '?')}",
                token=HF_TOKEN,
            )
```

**Confidence:** HIGH -- `HfApi.upload_folder` is the official Hub upload method; `TrainerCallback.on_evaluate` is documented in Transformers API.

## Data Flow

### Training Data Flow (per batch)

```
HuggingFace Hub (FLAC audio, streaming)
    │
    ▼
IterableDataset.map(prepare_example)
    │
    ├── 1. Audio array extracted (16kHz float32 numpy)
    │
    ├── 2. [Train only] audiomentations.Compose applied:
    │       AddGaussianNoise → RoomSimulator → TimeStretch → PitchShift
    │
    ├── 3. WhisperFeatureExtractor:
    │       float32 array → 80-bin log-Mel spectrogram (30s padded)
    │       Output: input_features [1, 80, 3000]
    │
    └── 4. WhisperTokenizer:
            Gurmukhi text → token IDs, pad_token_id replaced with -100
            Output: labels [variable length]
    │
    ▼
DataCollatorSpeechSeq2SeqWithPadding
    │
    ├── Pads labels to max length in batch (with -100)
    ├── Stacks input_features into batch tensor
    └── Strips BOS token from labels (Trainer prepends it)
    │
    ▼
WhisperForConditionalGeneration.forward()
    │
    ├── Encoder: input_features → encoder_hidden_states [B, 1500, 768]
    ├── Decoder: labels (shifted) + cross-attention to encoder → logits
    └── Loss: cross-entropy on non-masked (-100) label positions
    │
    ▼
Backward pass → gradient accumulation → optimizer step
    │
    ▼
Every 300 steps: eval on validation set → compute WER
    │
    ├── If best WER: save checkpoint locally
    └── PushBestToHub callback: upload_folder to HuggingFace Hub
```

### Checkpoint Resume Flow

```
Session start
    │
    ├── Check local checkpoint dir for checkpoint-XXXX folders
    │   └── If found: resume_from_checkpoint = latest local
    │
    ├── If no local checkpoint: check HuggingFace Hub repo
    │   └── If exists: download model, resume from step 0 with pretrained weights
    │       (Note: HF Hub stores best model, not optimizer state.
    │        Full resume requires local checkpoint with optimizer + scheduler state)
    │
    └── If nothing found: start fresh from openai/whisper-small
```

### Key Data Flows

1. **Audio streaming flow:** HuggingFace Hub FLAC files are fetched on demand over HTTPS, decoded into 16kHz float32 numpy arrays, never written to RunPod disk. The shuffle buffer holds ~1000 examples in memory (~250MB for 30s clips).

2. **Checkpoint safety flow:** Local checkpoint saved every 300 steps. Callback uploads best checkpoint folder (model weights + config + tokenizer, ~950MB for whisper-small) to HuggingFace Hub. If session dies, local checkpoint is lost but Hub checkpoint survives.

3. **Validation flow:** Fixed 300-example subset loaded eagerly (not streaming) into memory at startup. Used for WER evaluation every 300 steps. No augmentation applied to validation data.

## Scaling Considerations

| Concern | Phase 1 (100h, ~12-18h) | Phase 3 (800h, ~36-42h) | Notes |
|---------|-------------------------|-------------------------|-------|
| Dataset size in memory | ~250MB shuffle buffer | ~250MB shuffle buffer (streaming scales) | Streaming mode means dataset size is irrelevant to memory |
| GPU memory (A40 48GB) | batch=8, accum=4 (effective 32) | Same batch config | A40 has plenty of headroom for whisper-small |
| Checkpoint upload time | ~60s per eval (every 300 steps) | Same, more total uploads | Network is bottleneck, not compute |
| Training duration risk | 12-18h, one RunPod session | 36-42h, may need session restart | Checkpoint resume is essential for Phase 3 |
| Optimizer state on resume | Full resume from local checkpoint | Must persist optimizer state to HF Hub or NFS for multi-session | Phase 3 may need enhanced resume logic |

### Scaling Priorities

1. **First bottleneck: Session survival.** RunPod sessions can be interrupted. The callback-based push to HuggingFace Hub is the primary defense. For Phase 3 (36-42h), consider also pushing optimizer state to survive full restarts without LR schedule discontinuity.

2. **Second bottleneck: Data loading speed.** Streaming from HuggingFace over network could become a bottleneck if network is slow. Mitigation: increase `dataloader_num_workers` (2 baseline, up to 4) and increase shuffle `buffer_size` to prefetch more aggressively. The A40 GPU should never be idle waiting for data.

## Anti-Patterns

### Anti-Pattern 1: Downloading Full Dataset to RunPod Disk

**What people do:** `load_dataset(HF_DATASET, split="train")` without `streaming=True`, which downloads 9GB of FLAC to local disk.
**Why it's wrong:** RunPod disk is ephemeral and potentially small. Download takes time. If session restarts, must download again.
**Do this instead:** Always use `streaming=True`. The IterableDataset fetches and decodes audio on the fly with no disk footprint.

### Anti-Pattern 2: Using `push_to_hub=True` in TrainingArguments Instead of Custom Callback

**What people do:** Set `push_to_hub=True` in `Seq2SeqTrainingArguments`, expecting automatic Hub uploads.
**Why it's wrong:** The built-in `push_to_hub` pushes at training end and periodically, but does not guarantee push of the *best* checkpoint specifically. On ephemeral compute, training may never reach the end. The built-in mechanism also pushes the full repo which can be slow.
**Do this instead:** Use a custom `TrainerCallback.on_evaluate` that explicitly pushes `state.best_model_checkpoint` via `HfApi.upload_folder`. This guarantees the best model is always on the Hub.

### Anti-Pattern 3: Using `num_train_epochs` with Streaming Datasets

**What people do:** Set `num_train_epochs=3` when training on an IterableDataset.
**Why it's wrong:** IterableDatasets do not report their length to the Trainer. The Trainer cannot calculate total steps from epochs without knowing dataset size. This causes either an error or incorrect scheduling.
**Do this instead:** Use `max_steps` explicitly. Calculate from your known dataset size: `max_steps = (num_examples * num_epochs) / (batch_size * gradient_accumulation_steps)`. For 100h sehaj path (~36,000 examples at 10s average), 3 epochs with effective batch 32 = ~3,375 steps.

### Anti-Pattern 4: Single Learning Rate for Encoder and Decoder

**What people do:** Use a single `learning_rate=1e-5` in TrainingArguments for all parameters.
**Why it's wrong:** For low-resource languages like Punjabi (0.06% of Whisper pretraining), the decoder needs much more adaptation than the encoder. A single low LR under-trains the decoder; a single high LR damages the encoder.
**Do this instead:** Construct a custom AdamW optimizer with parameter groups, pass via `optimizers=(optimizer, None)`.

### Anti-Pattern 5: Hardcoding HuggingFace Token in Script

**What people do:** `HF_TOKEN = "hf_abc123..."` directly in Python source.
**Why it's wrong:** Token gets committed to git, pushed to GitHub, exposed in logs.
**Do this instead:** Load from environment variable: `HF_TOKEN = os.environ["HF_TOKEN"]`. Set it on RunPod via SSH: `export HF_TOKEN=...` before running the training script.

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| HuggingFace Hub (dataset) | `load_dataset(name, streaming=True, token=HF_TOKEN)` | Read-only, streamed over HTTPS |
| HuggingFace Hub (model) | `HfApi().upload_folder(folder_path, repo_id, token)` | Write, called from callback every 300 steps |
| HuggingFace Hub (resume) | `WhisperForConditionalGeneration.from_pretrained(repo_id)` | Read, at session startup only |
| RunPod A40 | SSH from local machine, Claude Code runs commands | Ephemeral GPU compute, no persistent storage guarantee |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| Config -> all modules | Python import, module-level constants | Config is read-only at runtime; everything else imports from it |
| Data Pipeline -> Trainer | IterableDataset passed as `train_dataset` arg | Trainer calls `iter()` on it; no random access |
| Collator -> Trainer | Passed as `data_collator` arg, called per batch | Returns dict with `input_features` and `labels` tensors |
| Callback -> HF Hub | `HfApi.upload_folder()` over HTTPS | Blocking call during `on_evaluate`; adds ~60s per eval |
| Model -> Trainer | Passed as `model` arg | Trainer handles `.train()`, `.eval()`, device placement |
| Optimizer -> Trainer | Passed as `optimizers=(optimizer, None)` tuple | `None` for scheduler means no built-in LR scheduling (fixed LR per group) |

## Build Order (Dependency Graph)

The following order reflects actual dependencies between components. Each layer depends on the one above it being functional.

```
Layer 1 (no dependencies):
  config.py ─────────────────── Must exist first; everything reads from it

Layer 2 (depends on config):
  data.py ──────────────────── Needs HF_DATASET, HF_TOKEN, AUDIO_COL, TEXT_COL from config
  model.py ─────────────────── Needs base model name, generation config values from config
  metrics.py ───────────────── Standalone; needs only the `evaluate` library

Layer 3 (depends on data + model):
  collator.py ──────────────── Needs processor (from model.py) and padding logic
  callbacks.py ─────────────── Needs HF_MODEL_REPO, HF_TOKEN from config

Layer 4 (depends on everything):
  train.py ─────────────────── Wires all components into Seq2SeqTrainer
```

### Suggested Build Sequence

1. **config.py first** -- Define all constants. Test by importing and printing values. Verify GPU detection works.
2. **data.py second** -- Build the streaming pipeline. Test by iterating 5 examples and printing shapes. This validates HF dataset access, audio decoding, augmentation, and feature extraction.
3. **model.py third** -- Load the model, construct the optimizer. Test by running a single forward pass on dummy input.
4. **collator.py + metrics.py in parallel** -- Both are small, self-contained. Test collator with a list of 4 dummy examples. Test metrics with dummy predictions.
5. **callbacks.py fifth** -- Requires a valid HF repo to test. Can mock the upload for unit testing.
6. **train.py last** -- Assembles everything. First test with `max_steps=10` to verify the full pipeline runs end-to-end before committing to a real training run.

### Build Order Implications for Roadmap

- **Phase 1 (scaffold):** config.py + data.py + model.py. These three modules are the foundation. Get streaming data flowing and the model loading on GPU before anything else.
- **Phase 2 (training core):** collator.py + metrics.py + train.py. Once data and model work, wire the training loop. Verify with a short 10-step run.
- **Phase 3 (safety):** callbacks.py + resume logic in train.py. Add checkpoint pushing and resume capability. Test by killing a 50-step run and resuming.
- **Phase 4 (production run):** Launch the full training run with `max_steps=3375` (or equivalent). Monitor via SSH.

## Sources

- HuggingFace Transformers official docs -- Whisper model architecture, WhisperProcessor, WhisperFeatureExtractor, WhisperForConditionalGeneration API (https://huggingface.co/docs/transformers/model_doc/whisper) [HIGH confidence]
- HuggingFace Transformers -- Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback, resume_from_checkpoint (Context7: /llmstxt/huggingface_co_transformers_v5_2_0_llms_txt) [HIGH confidence]
- HuggingFace Datasets official docs -- IterableDataset streaming, shuffle with buffer_size, .map() on streaming datasets, set_epoch() (Context7: /llmstxt/huggingface_co_datasets_main_en_llms_txt) [HIGH confidence]
- HuggingFace blog: "Fine-Tune Whisper" by Sanchit Gandhi -- Complete end-to-end fine-tuning pipeline, data collator pattern, evaluation setup (https://huggingface.co/blog/fine-tune-whisper) [HIGH confidence]
- audiomentations library docs -- Compose, AddGaussianNoise, RoomSimulator, TimeStretch, PitchShift API (Context7: /iver56/audiomentations) [HIGH confidence]
- Project training plan (surt_training_plan.md) -- Domain-specific decisions: discriminative LR values, augmentation parameters, Mool Mantar prompt, checkpoint interval [project-internal]

---
*Architecture research for: Whisper fine-tuning pipeline (Gurbani ASR Phase 1)*
*Researched: 2026-04-04*
