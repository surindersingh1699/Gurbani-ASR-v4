import os
import torch

# --- GPU detection ---
# SURT_NO_CUDA_INIT=1 skips all CUDA calls at import time so that data-prep
# scripts (prepare_data.py) can safely use fork-based multiprocessing.
# Training scripts never need to set this — GPU is auto-detected as usual.
_skip_cuda = os.environ.get("SURT_NO_CUDA_INIT", "0") == "1"

if _skip_cuda:
    GPU_NAME = os.environ.get("SURT_GPU_TYPE", "A40")
    NUM_GPUS = int(os.environ.get("SURT_NUM_GPUS", "1"))
else:
    GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    NUM_GPUS = max(1, torch.cuda.device_count()) if torch.cuda.is_available() else 1

# Per-device batch size (VRAM-tuned per GPU type). Override with $SURT_BATCH_SIZE.
if "A100" in GPU_NAME:
    BATCH_SIZE = 64  # 80GB VRAM, no accum needed → 2x faster
elif "A40" in GPU_NAME:
    BATCH_SIZE = 64  # 48GB VRAM, Whisper Small + bf16 + grad_checkpointing fits easily
elif "4090" in GPU_NAME:
    BATCH_SIZE = 8
elif "3090" in GPU_NAME:
    BATCH_SIZE = 8
elif "A5000" in GPU_NAME:
    BATCH_SIZE = 8
elif "L4" in GPU_NAME:
    BATCH_SIZE = 8
elif "T4" in GPU_NAME:
    BATCH_SIZE = 4
elif "V100" in GPU_NAME:
    BATCH_SIZE = 4
else:
    BATCH_SIZE = 4

_batch_override = os.environ.get("SURT_BATCH_SIZE", "").strip()
if _batch_override:
    BATCH_SIZE = int(_batch_override)

# Gradient accumulation: target ~64 effective batch, auto-adjusted for multi-GPU
# effective_batch = BATCH_SIZE * GRAD_ACCUM * NUM_GPUS
GRAD_ACCUM = max(1, 64 // (BATCH_SIZE * NUM_GPUS))
EFFECTIVE_BATCH = BATCH_SIZE * GRAD_ACCUM * NUM_GPUS

# --- Paths ---
OUTPUT_DIR = "/workspace/surt/checkpoints"
LOG_DIR = "/workspace/surt/logs"

# --- Model ---
# v3 starts from fresh openai/whisper-small — v1 sehaj WER was inflated by a
# data leak in its eval set, so its "prior knowledge" is untrustable;
# v2 has baked-in kirtan hallucination priors from the 28h noisy v2 kirtan
# dataset. With 700h of clean v3 data we prefer a clean base.
BASE_MODEL = "openai/whisper-small"
HF_MODEL_REPO = "surindersinghssj/surt-small-v3"

# --- Training schedule (auto-scaled for effective batch and GPU count) ---
# v3 sizing: 450h kirtan + 250h sehaj ≈ 700h ≈ ~190k 20s clips.
# 3 epochs is the sweet spot at this scale — 5 overfits, 2 under-trains.
_APPROX_TRAIN_SIZE = 190_000
_TARGET_EPOCHS = 3
MAX_STEPS = (_APPROX_TRAIN_SIZE * _TARGET_EPOCHS) // EFFECTIVE_BATCH
WARMUP_STEPS = 900  # ~10% of MAX_STEPS — cold start from openai/whisper-small needs slower ramp
EVAL_STEPS = 500
SAVE_STEPS = 500
SAVE_TOTAL_LIMIT = 3
GENERATION_MAX_LENGTH = 448  # Gurmukhi tokenizer expansion: 3-5x longer than English
LEARNING_RATE = 3e-5       # Base LR (matches decoder) — cold-start from whisper-small, more to learn
ENCODER_LR = 5e-5          # Encoder adapts acoustics for kirtan
DECODER_LR = 3e-5          # Decoder adapts Gurmukhi + kirtan from whisper-small's multilingual base
WEIGHT_DECAY = 0.01        # Standard AdamW weight decay
EARLY_STOP_PATIENCE = 3    # Stop training if neither WER nor CER improves for N consecutive evals
EARLY_STOP_METRIC = "kirtan"  # Which eval split drives early stopping: "kirtan" | "sehaj_path" | "none"

# --- Hub (training checkpoints) ---
TRAINING_HUB_REPO = "surindersinghssj/surt-small-v3-training"

# --- Mool Mantar (Gurmukhi vocabulary anchor for initial_prompt) ---
MOOL_MANTAR = "ੴ ਸਤਿ ਨਾਮੁ ਕਰਤਾ ਪੁਰਖੁ ਨਿਰਭਉ ਨਿਰਵੈਰੁ ਅਕਾਲ ਮੂਰਤਿ ਅਜੂਨੀ ਸੈਭੰ ਗੁਰ ਪ੍ਰਸਾਦਿ"

# --- Dataset ---
# v3 training mix: sehaj primary + kirtan aux at natural hour-ratio (~64% kirtan).
# Sehaj primary ≈ 160h, kirtan aux ≈ 450h → aux_p = 450/(450+160+66) = 0.67,
# using 0.64 as a round stable ratio target.
# Canonical datasets all expose the `final_text` column. The older
# `gurbani-sehajpath` studio sehaj is not canonicalized yet and exposes
# `gurmukhi_text` instead — harmonized during load via EXTRA_SEHAJ_TEXT_COLUMN.
DATASET_NAME = "surindersinghssj/gurbani-sehajpath-yt-captions-canonical"
AUX_TRAIN_DATASET_NAME = "surindersinghssj/gurbani-kirtan-yt-captions-300h-canonical"
AUX_TRAIN_PROBABILITY = 0.64

# Extra sehaj: older studio-recorded sehaj dataset, concatenated with primary
# sehaj BEFORE kirtan aux mixing. Uses `gurmukhi_text` column (non-canonical
# schema) — the data loader renames it to TEXT_COLUMN on load.
EXTRA_SEHAJ_DATASET_NAME = "surindersinghssj/gurbani-sehajpath"
EXTRA_SEHAJ_TEXT_COLUMN = "gurmukhi_text"

# v3 held-out evals live in separate repos (not `validation` splits of train repos).
# Both canonical eval datasets use the `final_text` column.
SEHAJ_EVAL_DATASET_NAME = "surindersinghssj/gurbani-sehajpath-yt-captions-eval-canonical"
KIRTAN_EVAL_DATASET_NAME = "surindersinghssj/gurbani-kirtan-eval-pure-canonical"

TRAIN_SPLIT = "train"
VAL_SPLIT = "validation"   # Legacy default — used only if dataset-specific override below is None

# v3 eval splits: sehaj eval repo uses "train" (the eval corpus is its only split);
# kirtan eval repo has "eval" + "test". TRAINING ONLY EVER USES THE `eval` SPLIT.
# The `test` split is reserved for final post-training reporting and MUST NOT be
# read during training — doing so would leak the holdout and inflate reported WER.
SEHAJ_EVAL_SPLIT = "train"
KIRTAN_EVAL_SPLIT = "eval"

TEXT_COLUMN = "final_text"  # Canonical v3 text column (primary sehaj, aux kirtan, both evals)
VAL_SIZE = 400             # ≈1h of sehaj eval (~10s avg clips)
SHUFFLE_BUFFER = 8000      # Needs to span multiple shards at 190k-clip scale

# --- Weights & Biases ---
# Auto-enabled when WANDB_API_KEY is set in environment.
# On RunPod: export WANDB_API_KEY="your-key" in ~/.bashrc
WANDB_PROJECT = "surt"
WANDB_ENTITY = "sabysurinder-surinder"

print(f"[config] GPU: {GPU_NAME} x{NUM_GPUS}")
print(f"[config] Batch: {BATCH_SIZE} x Accum: {GRAD_ACCUM} x GPUs: {NUM_GPUS} = Effective: {EFFECTIVE_BATCH}")
print(f"[config] Max steps: {MAX_STEPS}, Warmup: {WARMUP_STEPS}, LR: encoder={ENCODER_LR}, decoder={DECODER_LR}")
print(f"[config] Early stop: patience={EARLY_STOP_PATIENCE} evals on split={EARLY_STOP_METRIC}")
print(f"[config] Train repo: {TRAINING_HUB_REPO}")
print(f"[config] Final repo: {HF_MODEL_REPO}")
