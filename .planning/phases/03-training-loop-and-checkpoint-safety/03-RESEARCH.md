# Phase 3: Training Loop and Checkpoint Safety - Research

**Researched:** 2026-04-04
**Domain:** HuggingFace Seq2SeqTrainer training pipeline with discriminative LR, WER evaluation, Hub checkpoint safety
**Confidence:** HIGH

## Summary

Phase 3 wires together the Seq2SeqTrainer with all training configuration needed for Whisper fine-tuning on Gurbani audio. The core technical challenges are: (1) implementing discriminative learning rates across three parameter groups (encoder, decoder, proj_out) by subclassing Seq2SeqTrainer and overriding `create_optimizer`, (2) building a custom `TrainerCallback` for Hub push logic that tracks best WER across resume cycles via a `best_wer.json` file, and (3) ensuring seamless checkpoint resume for spot instance preemption using `get_last_checkpoint()` auto-detection.

All pieces are well-supported by the HuggingFace ecosystem. The Seq2SeqTrainer with `predict_with_generate=True` handles autoregressive WER evaluation natively. Gradient checkpointing, bf16, cosine LR scheduling, and checkpoint saving are all first-class TrainingArguments. The main custom work is the discriminative LR optimizer (requires a Trainer subclass) and the Hub push callback (requires a TrainerCallback subclass). The data pipeline (Phase 2) and model initialization (Phase 2) are complete and provide clean integration points.

**Primary recommendation:** Subclass `Seq2SeqTrainer` to override `create_optimizer()` for discriminative LR groups, and implement a `HubPushCallback(TrainerCallback)` with `on_evaluate` hook that pushes to a separate training repo with WER-gated best-model pushes plus periodic safety pushes.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Hardcode `MAX_STEPS = 5000` in config.py (not calculated from epochs)
- ~5 effective epochs over 64k rows with effective batch 64
- No early stopping logic in this phase -- run all 5,000 steps, pick best checkpoint afterward
- Push on WER improvement ("best") PLUS periodic safety push (e.g., every 3rd eval regardless of WER)
- Push full model folder: model weights + processor + tokenizer + generation_config -- Hub repo is instantly loadable with `from_pretrained()`
- Push to a separate training repo (e.g., `surindersinghssj/surt-small-v1-training`), NOT the final `surindersingh/surt-small-v1` repo
- Auto-detect: if `OUTPUT_DIR` contains checkpoints, automatically resume from the latest -- no flag needed
- Trust the checkpoint -- no compatibility verification before resume
- Track best WER in a file (`best_wer.json` alongside checkpoints) so the Hub push callback knows the previous best WER after resume
- Cosine decay to 0 after 400-step warmup (`lr_scheduler_type="cosine"`)
- Discriminative LRs: encoder 5e-5, decoder 1e-4, proj_out 1e-4
- Evaluation must consume less than 5% of total training wall time
- Eval every 300 steps with eval set of 300 examples (already configured in config.py)

### Claude's Discretion
- Whether to use early stopping (patience-based) or run all max_steps -- lean toward simplest approach for v1
- Epoch estimation logging at training start (informational)
- Hub commit message format (include step number, WER, and push reason)
- LR schedule shape per parameter group (same cosine for all groups vs encoder-slower warmup)
- LR config style (explicit constants vs base+ratio)
- Weight decay value (standard 0.01 default is fine)
- Resume status logging format

### Deferred Ideas (OUT OF SCOPE)
- ETRAIN-01 (custom LR scheduler beyond cosine) -- v2 requirement
- ETRAIN-02 (TensorBoard logging) -- v2, console logging sufficient
- ETRAIN-03 (optimizer state to Hub) -- v2, local optimizer state + checkpoint resume is sufficient
- EVAL-01 (Gurmukhi text normalization for WER) -- v2, raw jiwer WER for v1
- Final model push to `surt-small-v1` -- Phase 4 (CKPT-04)
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| TRAIN-01 | Discriminative learning rates -- encoder at 5e-5, decoder at 1e-4, proj_out at 1e-4 | Subclass Seq2SeqTrainer, override `create_optimizer()` with three AdamW parameter groups using `model.model.encoder`, `model.model.decoder`, `model.proj_out` attribute paths |
| TRAIN-02 | Gradient checkpointing enabled to fit in VRAM | `gradient_checkpointing=True` in Seq2SeqTrainingArguments, plus `gradient_checkpointing_kwargs={"use_reentrant": False}` for PyTorch 2.x compatibility |
| TRAIN-03 | bf16 mixed precision on Ampere GPUs | `bf16=True` in Seq2SeqTrainingArguments (not fp16) |
| TRAIN-04 | 400 warmup steps to prevent early divergence | `warmup_steps=400` in Seq2SeqTrainingArguments, combined with `lr_scheduler_type="cosine"` |
| TRAIN-05 | Training uses max_steps (not num_train_epochs) for streaming | `max_steps=5000` in Seq2SeqTrainingArguments, `dataloader_num_workers=0` for streaming IterableDataset safety |
| TRAIN-06 | Seq2SeqTrainer with predict_with_generate=True | `predict_with_generate=True` in Seq2SeqTrainingArguments; Seq2SeqTrainer handles autoregressive generation during eval natively |
| TRAIN-07 | WER computed via jiwer at each eval step (every 300 steps) | `compute_metrics` function using `jiwer.wer()` directly (not `evaluate.load("wer")`); decode predictions and labels with `processor.tokenizer.batch_decode()`, replace -100 with pad_token_id |
| TRAIN-08 | GPU auto-detection adjusts batch size and gradient accumulation | Already implemented in `surt/config.py`; training args read `BATCH_SIZE` and `GRAD_ACCUM` from config |
| CKPT-01 | Checkpoints saved locally every 300 steps with save_total_limit=3 | `save_steps=300`, `save_total_limit=3` in Seq2SeqTrainingArguments; already defined as constants in config.py |
| CKPT-02 | Best checkpoint pushed to HuggingFace Hub after every evaluation via custom callback | Custom `HubPushCallback(TrainerCallback)` with `on_evaluate` hook; uses `HfApi.upload_folder()` to push model/processor/generation_config to training repo |
| CKPT-03 | Training can resume from last checkpoint if spot instance preempted | `get_last_checkpoint(OUTPUT_DIR)` auto-detection at startup; pass result to `trainer.train(resume_from_checkpoint=...)` |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| transformers | >=5.0.0 | Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback, get_last_checkpoint | Official HuggingFace training framework; project already uses v5 API conventions (`eval_strategy` not `evaluation_strategy`) |
| jiwer | >=3.0.0 | WER computation | Requirement TRAIN-07 specifies jiwer directly; lightweight, no evaluate library dependency needed |
| huggingface_hub | >=0.20.0 | HfApi for upload_folder to Hub | Official Hub client; `upload_folder()` provides atomic commit of full model directory |
| torch | >=2.0.0 | AdamW optimizer, gradient checkpointing | PyTorch 2.x required for `use_reentrant=False` gradient checkpointing |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numpy | any | Array manipulation in compute_metrics | Replacing -100 padding in label arrays |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| jiwer directly | evaluate.load("wer") | evaluate wraps jiwer but adds dependency; had jiwer 4.0 compat issues (now fixed); jiwer direct is simpler and explicit |
| Custom HubPushCallback | Trainer built-in push_to_hub=True | Built-in pushes on every save, not WER-gated; no best-model tracking across resumes; no periodic safety push logic |
| Subclass create_optimizer | Pass optimizer via optimizers=(optimizer, scheduler) tuple | Passing tuple requires model to be on device first and scheduler to be pre-created; subclass approach is cleaner and lets Trainer manage scheduler creation |

**Installation:**
```bash
pip install transformers>=5.0.0 jiwer>=3.0.0 huggingface_hub>=0.20.0
```

## Architecture Patterns

### Recommended File Structure
```
surt/
├── config.py          # Add: MAX_STEPS, LR constants, TRAINING_HUB_REPO
├── model.py           # Existing: load_model_and_processor()
├── data.py            # Existing: get_train_dataset(), get_val_dataset(), DataCollator
├── train.py           # NEW: main training entry point
│   ├── SurtTrainer(Seq2SeqTrainer)     # Override create_optimizer
│   ├── HubPushCallback(TrainerCallback) # on_evaluate hub push logic
│   ├── compute_metrics()               # WER via jiwer
│   └── main()                          # Entry point with auto-resume
└── __init__.py
```

### Pattern 1: Discriminative Learning Rates via Trainer Subclass
**What:** Override `create_optimizer()` in a Seq2SeqTrainer subclass to create AdamW with three parameter groups at different learning rates.
**When to use:** When different model components need different learning rates (encoder slower than decoder for transfer learning).
**Example:**
```python
# Source: HuggingFace Trainer docs (create_optimizer method) + GitHub issue #10140
from transformers import Seq2SeqTrainer
from torch.optim import AdamW

class SurtTrainer(Seq2SeqTrainer):
    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        model = self.model
        # WhisperForConditionalGeneration structure:
        #   model.model.encoder.* -- encoder parameters
        #   model.model.decoder.* -- decoder parameters
        #   model.proj_out.*      -- output projection
        encoder_params = [p for n, p in model.named_parameters()
                         if "model.encoder" in n and p.requires_grad]
        decoder_params = [p for n, p in model.named_parameters()
                         if "model.decoder" in n and p.requires_grad]
        proj_params    = [p for n, p in model.named_parameters()
                         if "proj_out" in n and p.requires_grad]

        optimizer_grouped_parameters = [
            {"params": encoder_params, "lr": 5e-5,  "weight_decay": 0.01},
            {"params": decoder_params, "lr": 1e-4,  "weight_decay": 0.01},
            {"params": proj_params,    "lr": 1e-4,  "weight_decay": 0.01},
        ]

        self.optimizer = AdamW(optimizer_grouped_parameters)
        return self.optimizer
```

### Pattern 2: Custom Hub Push Callback with Best WER Tracking
**What:** A `TrainerCallback` that pushes to Hub on WER improvement (best) and periodically (safety), tracking best WER in a JSON file that survives resume cycles.
**When to use:** Spot instance training where you need both best-model Hub safety and periodic Hub backup.
**Example:**
```python
# Source: HuggingFace TrainerCallback docs (on_evaluate event)
import json
from pathlib import Path
from transformers import TrainerCallback
from huggingface_hub import HfApi

class HubPushCallback(TrainerCallback):
    def __init__(self, hub_repo, processor, best_wer_path, push_every_n_evals=3):
        self.hub_repo = hub_repo
        self.processor = processor
        self.best_wer_path = Path(best_wer_path)
        self.push_every_n_evals = push_every_n_evals
        self.eval_count = 0
        self.api = HfApi()
        # Load best WER from file (survives resume)
        self.best_wer = self._load_best_wer()

    def _load_best_wer(self):
        if self.best_wer_path.exists():
            data = json.loads(self.best_wer_path.read_text())
            return data.get("best_wer", float("inf"))
        return float("inf")

    def _save_best_wer(self, wer, step):
        self.best_wer_path.write_text(json.dumps({
            "best_wer": wer, "step": step
        }))

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        self.eval_count += 1
        current_wer = metrics.get("eval_wer", float("inf"))
        step = state.global_step

        is_best = current_wer < self.best_wer
        is_periodic = (self.eval_count % self.push_every_n_evals) == 0

        if is_best:
            self.best_wer = current_wer
            self._save_best_wer(current_wer, step)

        if is_best or is_periodic:
            reason = "best" if is_best else "periodic"
            commit_msg = f"step {step} | WER {current_wer:.2f} | {reason}"
            # Push model + processor to Hub
            model = kwargs.get("model")
            if model:
                model.save_pretrained(args.output_dir + "/hub_staging")
                self.processor.save_pretrained(args.output_dir + "/hub_staging")
                self.api.upload_folder(
                    repo_id=self.hub_repo,
                    folder_path=args.output_dir + "/hub_staging",
                    commit_message=commit_msg,
                )
            print(f"[train] Hub push: {commit_msg}")
```

### Pattern 3: Auto-Resume from Latest Checkpoint
**What:** At training startup, detect existing checkpoints in OUTPUT_DIR and automatically resume without user flags.
**When to use:** Spot instance training where the script re-runs after preemption.
**Example:**
```python
# Source: HuggingFace trainer_utils (get_last_checkpoint)
from transformers.trainer_utils import get_last_checkpoint

last_ckpt = get_last_checkpoint(OUTPUT_DIR)
if last_ckpt:
    print(f"[train] Resuming from checkpoint: {last_ckpt}")
else:
    print("[train] Starting fresh training run")

trainer.train(resume_from_checkpoint=last_ckpt)
```

### Pattern 4: WER compute_metrics with jiwer
**What:** Decode model predictions and compute WER using jiwer directly.
**When to use:** Every eval step (300 steps).
**Example:**
```python
# Source: HuggingFace Whisper fine-tuning blog + jiwer docs
import jiwer
import numpy as np

def make_compute_metrics(processor):
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 padding with pad_token_id for decoding
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # Decode predictions and labels to text
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # Compute WER as percentage
        wer = 100 * jiwer.wer(label_str, pred_str)
        return {"wer": wer}

    return compute_metrics
```

### Anti-Patterns to Avoid
- **Using `evaluation_strategy` instead of `eval_strategy`:** Renamed in transformers v5. Using the old name will raise a warning or error.
- **Setting `fp16=True` on Ampere GPUs:** Use `bf16=True` instead for better numerical stability on A40/A100. fp16 can cause NaN losses with Whisper.
- **Setting `num_train_epochs` with streaming datasets:** Streaming `IterableDataset` has no length; use `max_steps` instead. Setting epochs will error or produce incorrect behavior.
- **Using `push_to_hub=True` in TrainingArguments for custom push logic:** The built-in push strategy conflicts with custom HubPushCallback. Set `push_to_hub=False` and handle pushes entirely in the callback.
- **Forgetting `use_reentrant=False` for gradient checkpointing:** PyTorch 2.x requires explicit `use_reentrant` parameter. Without it, a deprecation warning appears and future versions will error. Pass via `gradient_checkpointing_kwargs={"use_reentrant": False}`.
- **Using `tokenizer=` parameter in Seq2SeqTrainer:** Deprecated in transformers v5. Use `processing_class=` instead.
- **Putting `dataloader_num_workers > 0` with streaming IterableDataset:** Can cause data duplication. Use `dataloader_num_workers=0` for safety.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Checkpoint detection | Manual glob/sort of checkpoint dirs | `get_last_checkpoint(output_dir)` from `transformers.trainer_utils` | Handles incomplete checkpoints, numeric sorting, edge cases |
| LR scheduling | Custom cosine scheduler | `lr_scheduler_type="cosine"` in TrainingArguments | Trainer creates scheduler automatically with warmup_steps; handles per-group rates correctly |
| Training loop | Custom forward/backward/optimizer step | Seq2SeqTrainer.train() | Handles gradient accumulation, mixed precision, logging, eval loop, checkpoint saving |
| Prediction generation during eval | Manual model.generate() loop | `predict_with_generate=True` | Seq2SeqTrainer handles generation, padding, and metric computation integration |
| Model saving for Hub | Manual torch.save + config dump | `model.save_pretrained()` + `processor.save_pretrained()` | Saves all config files, generation_config, tokenizer -- instantly loadable with `from_pretrained()` |
| Checkpoint resume state | Manual optimizer/scheduler state restoration | `trainer.train(resume_from_checkpoint=path)` | Restores model weights, optimizer state, scheduler state, RNG states, trainer state (global_step) |

**Key insight:** The Trainer framework handles 90% of the training complexity. The only custom pieces needed are: (1) discriminative LR optimizer creation, (2) Hub push callback with WER gating, and (3) auto-resume detection at script startup.

## Common Pitfalls

### Pitfall 1: Discriminative LR Parameter Group Overlap
**What goes wrong:** Parameters appear in multiple groups or some parameters are missed, causing either duplicate gradient updates or untrained parameters.
**Why it happens:** Using string matching on parameter names (e.g., `"encoder" in name`) can match unexpected parameters or miss edge cases. WhisperForConditionalGeneration has `model.model.encoder` and `model.model.decoder` -- note the double `model.` prefix.
**How to avoid:** After creating parameter groups, verify the total parameter count matches `sum(p.numel() for p in model.parameters() if p.requires_grad)`. Log group sizes at startup.
**Warning signs:** Total parameters in groups != total trainable parameters. Training loss behaves unexpectedly (stuck or diverging).

### Pitfall 2: WER = 100% Due to Language Misconfiguration
**What goes wrong:** Model generates English text instead of Gurmukhi during eval, producing 100% WER.
**Why it happens:** `generation_config` settings (language, task, forced_decoder_ids) not properly configured on the model. Already addressed in Phase 2's model.py, but can regress if model is reloaded without these settings.
**How to avoid:** model.py already sets `model.generation_config.language = "punjabi"` and `forced_decoder_ids = None`. Verify at training startup by generating one sample and checking output script.
**Warning signs:** WER at 100% on first eval step. Generated text is ASCII/Latin instead of Gurmukhi.

### Pitfall 3: best_wer.json Not Loaded After Resume
**What goes wrong:** After spot instance preemption and resume, the Hub push callback doesn't know the previous best WER and re-pushes a worse model as "best."
**Why it happens:** `best_wer.json` stored only in memory (callback instance variable) instead of persisted to disk.
**How to avoid:** Write `best_wer.json` to `OUTPUT_DIR` (alongside checkpoints) on every best-WER update. Load it in callback `__init__`. The file persists on local disk across resumes.
**Warning signs:** After resume, first eval push says "best" even though WER is worse than pre-preemption.

### Pitfall 4: Hub Push Blocks Training
**What goes wrong:** Uploading model weights to Hub takes significant time, blocking the training loop during eval.
**Why it happens:** `upload_folder()` is synchronous by default. Model weights for Whisper Small are ~1 GB.
**How to avoid:** Use `run_as_future=True` parameter on `upload_folder()` to make Hub push non-blocking. Or accept the blocking push since it only happens every 300+ steps.
**Warning signs:** Long pauses after eval steps. Training wall time significantly exceeds expected compute time.

### Pitfall 5: Gradient Checkpointing + Frozen Layers Incompatibility
**What goes wrong:** `requires_grad` is silently set to False on some layers when gradient checkpointing uses reentrant mode.
**Why it happens:** PyTorch's reentrant checkpointing implementation has a known interaction with frozen parameters.
**How to avoid:** Always use `gradient_checkpointing_kwargs={"use_reentrant": False}`. This project does full fine-tuning (no frozen layers), but this setting is still recommended for future safety.
**Warning signs:** Some parameter gradients are None despite not being frozen. Loss doesn't decrease.

### Pitfall 6: Eval Takes Too Long (>5% Wall Time Budget)
**What goes wrong:** With `predict_with_generate=True`, evaluation generates full sequences for all 300 val examples, consuming too much time.
**Why it happens:** Autoregressive generation is slow; each token requires a full decoder forward pass. With `generation_max_length=448`, worst case is 448 decode steps per example.
**How to avoid:** Keep eval set small (300 examples -- already configured). Consider reducing `generation_max_length` during eval if still too slow. Monitor eval time vs train time.
**Warning signs:** Eval steps take >30 seconds. Total eval time exceeds 5% of wall time.

### Pitfall 7: Streaming Dataset Epoch Boundary with max_steps
**What goes wrong:** When max_steps is reached mid-stream, training stops correctly. But if the streaming dataset wraps around (new epoch), augmentation randomness and data order may differ.
**Why it happens:** Streaming IterableDataset shuffles with a buffer, and the shuffle state isn't checkpointed.
**How to avoid:** This is acceptable for v1 -- the data is shuffled differently per epoch anyway (feature, not bug). The fixed seed=42 in shuffle provides reproducibility within an epoch. After resume, data order may differ but this is fine for training.
**Warning signs:** None -- this is expected behavior.

## Code Examples

Verified patterns from official sources:

### Seq2SeqTrainingArguments Configuration
```python
# Source: HuggingFace docs (transformers v5) + project CONTEXT.md decisions
from transformers import Seq2SeqTrainingArguments
from surt.config import (
    OUTPUT_DIR, BATCH_SIZE, GRAD_ACCUM, EVAL_STEPS, SAVE_STEPS,
    SAVE_TOTAL_LIMIT, WARMUP_STEPS, GENERATION_MAX_LENGTH, MAX_STEPS,
)

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    max_steps=MAX_STEPS,
    warmup_steps=WARMUP_STEPS,
    learning_rate=1e-4,               # Base LR (decoder/proj_out); encoder uses custom group
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=SAVE_TOTAL_LIMIT,
    predict_with_generate=True,
    generation_max_length=GENERATION_MAX_LENGTH,
    logging_steps=25,
    dataloader_num_workers=0,          # Required for streaming IterableDataset
    load_best_model_at_end=False,      # We handle best model via custom callback
    push_to_hub=False,                 # We handle Hub push via custom callback
    report_to="none",                  # No TensorBoard/W&B for v1
    remove_unused_columns=False,       # Streaming datasets may not have column_names
)
```

### Trainer Initialization and Training
```python
# Source: HuggingFace Whisper fine-tuning blog + transformers v5 docs
from surt.model import load_model_and_processor
from surt.data import (
    get_train_dataset, get_val_dataset,
    DataCollatorSpeechSeq2SeqWithPadding,
)
from surt.config import DATASET_NAME

model, processor = load_model_and_processor()

train_dataset = get_train_dataset(DATASET_NAME, processor)
val_dataset = get_val_dataset(DATASET_NAME, processor)

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

trainer = SurtTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=make_compute_metrics(processor),
    processing_class=processor.feature_extractor,  # v5: processing_class, not tokenizer
    callbacks=[hub_push_callback],
)

# Auto-resume
last_ckpt = get_last_checkpoint(OUTPUT_DIR)
trainer.train(resume_from_checkpoint=last_ckpt)
```

### WhisperForConditionalGeneration Parameter Structure
```python
# Source: transformers/models/whisper/modeling_whisper.py
# WhisperForConditionalGeneration.__init__:
#   self.model = WhisperModel(config)      # contains encoder + decoder
#   self.proj_out = nn.Linear(d_model, vocab_size, bias=False)
#
# Named parameter prefixes when iterating model.named_parameters():
#   "model.encoder.*"  -- all encoder layers
#   "model.decoder.*"  -- all decoder layers (includes embed_tokens)
#   "proj_out.*"       -- output projection (tied with decoder.embed_tokens)
#
# IMPORTANT: proj_out.weight is tied with model.decoder.embed_tokens.weight
# When creating parameter groups, filter by name to avoid double-counting.
# The tied weight will appear under BOTH names in named_parameters().
# Use a set() to deduplicate by parameter id().

def build_param_groups(model, encoder_lr, decoder_lr, proj_lr, weight_decay):
    """Build parameter groups avoiding tied weight duplication."""
    seen_ids = set()
    encoder_params, decoder_params, proj_params = [], [], []

    for name, param in model.named_parameters():
        if not param.requires_grad or id(param) in seen_ids:
            continue
        seen_ids.add(id(param))

        if "model.encoder" in name:
            encoder_params.append(param)
        elif "proj_out" in name:
            proj_params.append(param)
        elif "model.decoder" in name:
            decoder_params.append(param)

    return [
        {"params": encoder_params, "lr": encoder_lr, "weight_decay": weight_decay},
        {"params": decoder_params, "lr": decoder_lr, "weight_decay": weight_decay},
        {"params": proj_params,    "lr": proj_lr,    "weight_decay": weight_decay},
    ]
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `evaluation_strategy` | `eval_strategy` | transformers v5 | Old name raises deprecation warning then error |
| `tokenizer=` in Trainer | `processing_class=` | transformers v5 (PR #35452) | Old name deprecated; use processing_class |
| `fp16=True` on Ampere | `bf16=True` | PyTorch 1.10+ / Ampere GPUs | bf16 has better dynamic range, avoids NaN issues with Whisper |
| `use_reentrant=True` (default) | `use_reentrant=False` (recommended) | PyTorch 2.x | Old default causes issues with frozen layers; new default will change in future PyTorch |
| `evaluate.load("wer")` | `jiwer.wer()` directly | jiwer 4.0 (Feb 2025) | evaluate had compat issues with jiwer 4.0; now fixed but direct jiwer is simpler |

**Deprecated/outdated:**
- `evaluation_strategy`: replaced by `eval_strategy` in transformers v5
- `tokenizer` parameter in Trainer: replaced by `processing_class`
- `jiwer.compute_measures()`: removed in jiwer 4.0; use `jiwer.wer()` or `jiwer.process_words()` instead

## Open Questions

1. **Tied weight handling in parameter groups**
   - What we know: `proj_out.weight` is tied with `model.decoder.embed_tokens.weight`. Both names appear in `named_parameters()` but point to the same tensor.
   - What's unclear: Whether PyTorch's AdamW handles the tied weight correctly if it appears in two groups, or if it causes double gradient accumulation.
   - Recommendation: Deduplicate by `id(param)` when building groups. Put the tied weight in `proj_out` group (since it's the output projection that directly affects loss). Verify at startup that `sum(group params) == total trainable params`.

2. **Eval time budget (5% wall time)**
   - What we know: 300 examples with `generation_max_length=448` could be slow. Whisper Small generates ~10-20 tokens/second per example on A40.
   - What's unclear: Exact wall time for full eval pass. This depends on average Gurmukhi sequence length.
   - Recommendation: Measure first eval step's wall time and log it. If it exceeds budget, reduce `generation_max_length` for eval or batch_size adjustments. The eval set is already small (300 examples) which helps.

3. **Cosine scheduler behavior with per-group learning rates**
   - What we know: The Trainer creates one cosine scheduler for the optimizer. PyTorch's cosine scheduler applies the same decay ratio to all parameter groups.
   - What's unclear: Whether the cosine scheduler correctly decays each group from its own initial LR to 0, or uses the global `learning_rate` arg.
   - Recommendation: HIGH confidence this works correctly. PyTorch LR schedulers operate on `optimizer.param_groups[i]["lr"]` which respects per-group initial LRs. The `learning_rate` in TrainingArguments is used only as the default for `create_optimizer` (which we override). Verify by logging per-group LRs at step 0, step 400 (post-warmup peak), and step 2500 (mid-decay).

## Sources

### Primary (HIGH confidence)
- Context7 `/llmstxt/huggingface_co_transformers_v5_2_0_llms_txt` - Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback, create_optimizer API
- Context7 `/jitsi/jiwer` - jiwer.wer() API and usage patterns
- Context7 `/huggingface/huggingface_hub` - HfApi.upload_folder() for Hub push
- [HuggingFace Trainer docs v5.5.0](https://huggingface.co/docs/transformers/main_classes/trainer) - Full Trainer API reference, create_optimizer, resume_from_checkpoint
- [HuggingFace Whisper fine-tuning blog](https://huggingface.co/blog/fine-tune-whisper) - compute_metrics pattern, Seq2SeqTrainingArguments example, data collator

### Secondary (MEDIUM confidence)
- [GitHub Issue #10140](https://github.com/huggingface/transformers/issues/10140) - Discriminative LR via Trainer subclass (recommended approach)
- [GitHub Issue #35446](https://github.com/huggingface/transformers/issues/35446) - processing_class replaces tokenizer in Seq2SeqTrainer
- [GitHub Issue #28536](https://github.com/huggingface/transformers/issues/28536) - gradient_checkpointing_kwargs use_reentrant=False
- [transformers/models/whisper/modeling_whisper.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py) - WhisperForConditionalGeneration structure (model.encoder, model.decoder, proj_out)
- [GitHub Issue #35782](https://github.com/huggingface/transformers/issues/35782) - Auto-resume from incomplete checkpoints handling

### Tertiary (LOW confidence)
- None -- all findings verified with primary or secondary sources

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All libraries are official HuggingFace ecosystem, well-documented, verified via Context7
- Architecture: HIGH - Patterns verified from official blog, Trainer docs, and source code
- Pitfalls: HIGH - Known issues documented in GitHub issues with fixes; tied weight handling verified via source code
- Discriminative LR: MEDIUM - Subclass pattern is officially recommended but not extensively documented; tied weight edge case needs runtime verification

**Research date:** 2026-04-04
**Valid until:** 2026-05-04 (stable ecosystem, 30-day validity)
