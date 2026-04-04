# Stack Research

**Domain:** Low-resource Indic ASR — Fine-tuning Whisper Small on Gurmukhi audio (RunPod A40 via SSH)
**Researched:** 2026-04-04
**Confidence:** HIGH

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| **PyTorch** | 2.11.0 (CUDA 12.x) | Deep learning framework | Only framework Whisper training supports. Use RunPod's `runpod/pytorch:2.11.0-py3.11-cuda12.8.1-devel-ubuntu22.04` template image. A40 is Ampere (sm_86), fully supported. **Confidence: HIGH** (PyPI verified) |
| **transformers** | 5.5.0 | Whisper model, Seq2SeqTrainer, tokenizers | The canonical library for Whisper fine-tuning. v5.x is current stable (v5.5.0 released 2026-04-02). `Seq2SeqTrainer` + `Seq2SeqTrainingArguments` remain the standard training loop. v5 renamed `evaluation_strategy` to `eval_strategy` — use the new name. `fp16=True` and `bf16=True` both still valid. **Confidence: HIGH** (Context7 v5.2.0 docs verified + PyPI) |
| **datasets** | 4.8.4 | Streaming audio data from HuggingFace Hub | Streaming mode (`streaming=True`) avoids downloading 100h FLAC to RunPod disk. `Audio` feature auto-decodes and resamples. `cast_column("audio", Audio(sampling_rate=16000))` works in both streaming and non-streaming. **Confidence: HIGH** (Context7 verified + PyPI) |
| **accelerate** | 1.13.0 | Distributed training utilities, mixed precision | Required by `transformers.Trainer`. Handles device placement, gradient accumulation, mixed precision under the hood. Single-GPU A40 setup still benefits from its memory management. **Confidence: HIGH** (PyPI verified) |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| **audiomentations** | 0.43.1 | Audio augmentation (noise, reverb, stretch, pitch) | During training preprocessing. Apply `Compose([AddGaussianNoise, RoomSimulator, TimeStretch, PitchShift])` to training audio before feature extraction. All four transforms confirmed present in current API. Do NOT augment validation data. **Confidence: HIGH** (Context7 verified + PyPI) |
| **huggingface_hub** | 1.9.0 | Push checkpoints to HuggingFace, resume from Hub | Use `HfApi().upload_folder()` in a `TrainerCallback.on_evaluate()` to push best checkpoint every 300 steps. Critical for RunPod session safety. `create_repo(exist_ok=True)` for idempotent repo creation. **Confidence: HIGH** (Context7 verified + PyPI) |
| **jiwer** | 4.0.0 | Word Error Rate (WER) computation | Use directly instead of `evaluate` for WER. Simpler, actively maintained, no network calls at runtime. `jiwer.wer(reference, hypothesis)` is all you need. **Confidence: HIGH** (PyPI verified) |
| **safetensors** | 0.7.0 | Safe, fast model serialization | Default format in transformers v5. No action needed — transformers uses it automatically. Mentioned for awareness: checkpoints saved as `.safetensors` not `.bin`. **Confidence: HIGH** (PyPI verified) |
| **soundfile** | 0.13.1 | FLAC/WAV audio I/O backend | Required by `datasets` for audio decoding. Must be installed explicitly on RunPod — not always in base images. `pip install soundfile` or it fails silently on audio load. **Confidence: HIGH** (PyPI verified) |
| **trackio** | 0.20.2 | Experiment tracking (metrics, loss curves) | Optional but recommended. HuggingFace's new lightweight tracker, replaces wandb integration. Set `report_to="trackio"` in TrainingArguments. Local SQLite storage — no external service needed. LLM-friendly CLI for querying experiment state. **Confidence: MEDIUM** (PyPI verified, relatively new library — transformers v5 docs reference it) |

### Development / Infrastructure Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| **RunPod A40 Pod** | GPU compute (48GB VRAM) | Use SSH pod, not serverless. Select `runpod/pytorch` template with CUDA 12.x. A40 = Ampere architecture, supports bf16 natively. **Use bf16=True instead of fp16=True** — A40 has hardware bf16 support, avoids fp16 overflow issues with loss scaling. |
| **tmux / screen** | Session persistence on RunPod | SSH connections drop. Always run training inside `tmux` so the process survives disconnection. `tmux new -s train` before launching. |
| **HuggingFace Hub** | Model registry + checkpoint store | Push checkpoints incrementally. Resume by pulling latest checkpoint from Hub if RunPod session dies. Free private repos for model storage. |
| **nvidia-smi / nvitop** | GPU monitoring | `nvitop` is a better `nvidia-smi` — shows memory, utilization, process info. `pip install nvitop` on RunPod for monitoring. |

## Installation

```bash
# Core stack — install in this order on RunPod
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128

pip install transformers==5.5.0 \
            datasets==4.8.4 \
            accelerate==1.13.0 \
            huggingface_hub==1.9.0

# Audio processing
pip install audiomentations==0.43.1 \
            soundfile==0.13.1

# Metrics
pip install jiwer==4.0.0

# Serialization (usually already installed as transformers dep)
pip install safetensors==0.7.0

# Optional: experiment tracking
pip install trackio==0.20.2

# Optional: GPU monitoring
pip install nvitop
```

> **Note on RunPod:** If using the `runpod/pytorch:2.11.0` template, PyTorch + CUDA are pre-installed. Skip the torch install line and just install the rest. Verify with `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`.

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| **transformers Seq2SeqTrainer** | Custom PyTorch training loop | Never for this project. Seq2SeqTrainer handles gradient accumulation, mixed precision, checkpointing, `predict_with_generate`, and eval metric computation. A custom loop would replicate all of this for zero benefit. |
| **transformers Seq2SeqTrainer** | HuggingFace `trl` (SFTTrainer) | Never for ASR. `trl` is for LLM alignment (RLHF/DPO). It wraps the same Trainer but adds text-generation-specific logic irrelevant to audio. |
| **bf16** | fp16 | Use fp16 only if running on pre-Ampere GPUs (T4, V100). A40 is Ampere — bf16 has larger dynamic range, no loss scaling needed, slightly faster on A40. |
| **jiwer** (direct) | `evaluate` library | Use `evaluate` only if you need many different metrics beyond WER. `evaluate` (v0.4.6) adds complexity (downloads metric scripts from Hub at runtime) for a single WER computation. `jiwer` is the WER backend anyway — cut the middleman. |
| **trackio** | Weights & Biases (wandb) | Use wandb if you already have a wandb account and want cloud-synced dashboards. trackio is simpler, local-first, no account needed, and is the new default in transformers v5. For unattended RunPod training with Claude Code, trackio's LLM-friendly CLI is a better fit. |
| **HfApi.upload_folder** | `push_to_hub=True` in TrainingArguments | Use `push_to_hub=True` for simple cases. For this project, explicit `HfApi.upload_folder()` in a callback gives more control: push only best checkpoint, custom commit messages with WER, and no coupling between save_steps and push frequency. |
| **streaming datasets** | Download full dataset to disk | Use disk download only if RunPod has >50GB free disk and you need random access for multi-epoch training. Streaming avoids disk usage entirely but shuffles within a buffer (not globally). For 100h FLAC (~9GB), streaming is the right call on RunPod. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| **LoRA / QLoRA / PEFT adapters** | Whisper saw ~0.06% Punjabi in pretraining. The encoder needs deep adaptation to Gurbani acoustics — adapter layers alone cannot reshape the audio representation sufficiently. Full fine-tuning of encoder + decoder is necessary for this level of domain shift. | Full fine-tuning with discriminative learning rates (encoder 5e-5, decoder 1e-4) |
| **whisper (openai/whisper)** | OpenAI's original `whisper` package is frozen, uses its own tokenizer/model format, and cannot be fine-tuned with HuggingFace Trainer. The HuggingFace `transformers` implementation (`WhisperForConditionalGeneration`) is the standard for training. | `transformers.WhisperForConditionalGeneration` |
| **faster-whisper** | CTranslate2-based inference-only library. Cannot train or fine-tune. Useful for Phase 5 inference optimization but not for Phase 1 training. | N/A for training. Consider for inference pipeline later. |
| **DeepSpeed** | Overkill for single-GPU A40 training of Whisper Small (244M params). Adds configuration complexity. DeepSpeed ZeRO stages are for multi-GPU memory partitioning — meaningless on one GPU. Gradient checkpointing via `transformers` is sufficient. | `gradient_checkpointing=True` in TrainingArguments |
| **FSDP (Fully Sharded Data Parallel)** | Same reasoning as DeepSpeed — multi-GPU distribution strategy. Whisper Small fits comfortably in A40's 48GB VRAM even with gradient checkpointing disabled. | Single-GPU training with `accelerate` |
| **Colab / Kaggle notebooks** | Session limits, unreliable GPU allocation, cannot be managed via SSH by Claude Code. RunPod A40 provides stable 48GB VRAM SSH sessions. The training plan originally called for Colab Pro — RunPod is strictly better for unattended training. | RunPod A40 via SSH |
| **evaluate** library (for WER only) | Adds unnecessary indirection. Downloads metric computation scripts from Hub at import time (requires network). `jiwer` is the actual WER computation backend. For a single metric, use `jiwer` directly. | `jiwer.wer(reference, hypothesis)` |
| **librosa** (for training pipeline) | Heavy dependency with numba/llvmlite compilation overhead. `datasets` + `soundfile` handle all audio loading and resampling natively via the `Audio` feature type. librosa is useful for offline analysis but adds install complexity on RunPod for no benefit during training. | `datasets.Audio(sampling_rate=16000)` for resampling |
| **wandb** (unless already committed) | Requires account, API key, network connectivity. trackio is local-first, no account needed, and is the new HuggingFace default. For unattended RunPod training, fewer external dependencies = fewer failure modes. | `trackio` with `report_to="trackio"` |

## Stack Patterns by Variant

**If RunPod A40 (48GB VRAM) — primary target:**
- Use `bf16=True` (Ampere hardware support)
- `per_device_train_batch_size=8`, `gradient_accumulation_steps=4` (effective batch 32)
- `gradient_checkpointing=True` (saves ~40% VRAM, enables larger batch)
- Streaming dataset (avoids disk)
- Push to Hub every 300 eval steps

**If RunPod A100 (40GB or 80GB) — upgraded option:**
- Same stack, increase batch to 16 with accumulation 2
- bf16=True (also Ampere)
- Consider `torch.compile(model)` for ~15% speedup (PyTorch 2.x feature)

**If fallback to V100 (16GB) or T4 (16GB):**
- Switch to `fp16=True` (no bf16 on Volta/Turing)
- Reduce `per_device_train_batch_size=4`, `gradient_accumulation_steps=8`
- `gradient_checkpointing=True` is mandatory
- May need to reduce `generation_max_length` for eval to fit in VRAM

## Version Compatibility

| Package | Compatible With | Notes |
|---------|-----------------|-------|
| transformers 5.5.0 | PyTorch >= 2.6.0 | v5 dropped support for PyTorch < 2.6. PyTorch 2.11.0 is well within range. |
| transformers 5.5.0 | accelerate >= 1.3.0 | v5 requires recent accelerate. 1.13.0 is current and compatible. |
| transformers 5.5.0 | datasets >= 3.0.0 | datasets 4.8.4 is compatible. Note: datasets 4.x uses `torchcodec` for audio decoding (replaces `torchaudio`/`soundfile` as primary backend in some cases — but `soundfile` still works as fallback). |
| transformers 5.5.0 | huggingface_hub >= 0.27.0 | hub 1.9.0 is far above minimum. Full compatibility. |
| audiomentations 0.43.1 | numpy >= 1.20 | Works with any modern numpy. No torch dependency — operates on raw numpy arrays before feature extraction. |
| datasets 4.8.4 | soundfile >= 0.12.0 | soundfile 0.13.1 for FLAC decoding. Install explicitly — not always in RunPod base images. |
| jiwer 4.0.0 | standalone | No framework dependencies. Pure Python WER/CER computation. |

## Key API Patterns for This Project

### transformers v5 changes from training plan code

The training plan (`surt_training_plan.md`) was written against transformers v4.x. Key adjustments for v5.5.0:

1. **`evaluation_strategy` renamed to `eval_strategy`** — The old name was deprecated in v4.46 and removed in v5. Use `eval_strategy="steps"`.

2. **`DataCollatorForSeq2Seq` tokenizer parameter** — In v5, pass `tokenizer=processor` (the WhisperProcessor), not `tokenizer=processor.feature_extractor`. The collator extracts what it needs.

3. **`report_to`** — Default changed. Use `report_to="trackio"` (new HuggingFace tracker) or `report_to="none"` to disable. The `"tensorboard"` option still works but trackio is preferred.

4. **`bf16` over `fp16`** — On A40, switch from `fp16=True` to `bf16=True`. Better numerical stability, no loss scaler needed.

5. **safetensors is default** — Checkpoints save as `.safetensors` automatically. No `.bin` files. This is fine — `from_pretrained` loads both formats.

6. **`attn_implementation="sdpa"`** — Optional but free speedup. Enables PyTorch's native Scaled Dot Product Attention for Whisper. Add to `from_pretrained()` call.

### Checkpoint safety pattern (RunPod)

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

### Resume from Hub (if RunPod session dies)

```python
from huggingface_hub import snapshot_download

# Pull latest checkpoint from Hub
local_dir = snapshot_download(
    repo_id=HF_MODEL_REPO,
    local_dir="./checkpoints/resumed",
    token=HF_TOKEN,
)
# Then pass to trainer.train(resume_from_checkpoint=local_dir)
```

## Sources

- **Context7** `/llmstxt/huggingface_co_transformers_v5_2_0_llms_txt` — Whisper model docs, Seq2SeqTrainingArguments, fp16/bf16 training args, eval_strategy API (HIGH confidence)
- **Context7** `/llmstxt/huggingface_co_datasets_main_en_llms_txt` — Audio feature, streaming, cast_column, dataset loading (HIGH confidence)
- **Context7** `/iver56/audiomentations` — AddGaussianNoise, RoomSimulator, TimeStretch, PitchShift API (HIGH confidence)
- **Context7** `/huggingface/huggingface_hub` — upload_folder, push_to_hub patterns, checkpoint management (HIGH confidence)
- **PyPI JSON API** — Version numbers verified for all packages (2026-04-04): transformers 5.5.0, datasets 4.8.4, accelerate 1.13.0, audiomentations 0.43.1, huggingface_hub 1.9.0, jiwer 4.0.0, soundfile 0.13.1, safetensors 0.7.0, torch 2.11.0, trackio 0.20.2 (HIGH confidence)
- **GitHub Releases** (huggingface/transformers) — v5.5.0 release notes, v5 breaking changes summary (HIGH confidence)

---
*Stack research for: Whisper Small fine-tuning on Gurmukhi audio (Surt Phase 1)*
*Researched: 2026-04-04*
