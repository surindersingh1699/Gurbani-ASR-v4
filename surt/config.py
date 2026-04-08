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

# Per-device batch size (VRAM-tuned per GPU type)
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

# Gradient accumulation: target ~64 effective batch, auto-adjusted for multi-GPU
# effective_batch = BATCH_SIZE * GRAD_ACCUM * NUM_GPUS
GRAD_ACCUM = max(1, 64 // (BATCH_SIZE * NUM_GPUS))
EFFECTIVE_BATCH = BATCH_SIZE * GRAD_ACCUM * NUM_GPUS

# --- Paths ---
OUTPUT_DIR = "/workspace/surt/checkpoints"
LOG_DIR = "/workspace/surt/logs"

# --- Model ---
BASE_MODEL = "openai/whisper-small"
HF_MODEL_REPO = "surindersinghssj/surt-small-v2"

# --- Training schedule (auto-scaled for effective batch and GPU count) ---
# Target: ~5 epochs over ~64k training examples
_APPROX_TRAIN_SIZE = 64000
_TARGET_EPOCHS = 5
MAX_STEPS = (_APPROX_TRAIN_SIZE * _TARGET_EPOCHS) // EFFECTIVE_BATCH
WARMUP_STEPS = 150  # ~3% — warm start from v1, no cold ramp needed
EVAL_STEPS = 200
SAVE_STEPS = 200
SAVE_TOTAL_LIMIT = 3
GENERATION_MAX_LENGTH = 448  # Gurmukhi tokenizer expansion: 3-5x longer than English
LEARNING_RATE = 1e-5       # Base LR (matches decoder) — v2 warm start from v1
ENCODER_LR = 4e-5          # Encoder adapts acoustics for kirtan (4x decoder LR)
DECODER_LR = 1e-5          # Decoder preserves Gurbani vocab from v1
WEIGHT_DECAY = 0.01        # Standard AdamW weight decay

# --- Hub (training checkpoints) ---
TRAINING_HUB_REPO = "surindersinghssj/surt-small-v2-training"

# --- Mool Mantar (Gurmukhi vocabulary anchor for initial_prompt) ---
MOOL_MANTAR = "ੴ ਸਤਿ ਨਾਮੁ ਕਰਤਾ ਪੁਰਖੁ ਨਿਰਭਉ ਨਿਰਵੈਰੁ ਅਕਾਲ ਮੂਰਤਿ ਅਜੂਨੀ ਸੈਭੰ ਗੁਰ ਪ੍ਰਸਾਦਿ"

# --- Dataset ---
DATASET_NAME = "surindersinghssj/gurbani-asr"
# Auxiliary kirtan dataset interleaved into training stream (~4x oversample).
AUX_TRAIN_DATASET_NAME = "surindersinghssj/gurbani-kirtan-dataset-v2"
# Fraction of training data from kirtan dataset (supplement: 30% kirtan, 70% sehaj path).
AUX_TRAIN_PROBABILITY = 0.30
TRAIN_SPLIT = "train"
VAL_SPLIT = "validation"   # Proper split created by scripts/create_val_split.py
TEXT_COLUMN = "transcription"  # Column name for transcription text in Gurbani ASR dataset
VAL_SIZE = 300             # Number of validation examples to materialize eagerly
SHUFFLE_BUFFER = 500       # Buffer size for streaming shuffle

# --- Weights & Biases ---
# Auto-enabled when WANDB_API_KEY is set in environment.
# On RunPod: export WANDB_API_KEY="your-key" in ~/.bashrc
WANDB_PROJECT = "surt"
WANDB_ENTITY = "sabysurinder-surinder"

print(f"[config] GPU: {GPU_NAME} x{NUM_GPUS}")
print(f"[config] Batch: {BATCH_SIZE} x Accum: {GRAD_ACCUM} x GPUs: {NUM_GPUS} = Effective: {EFFECTIVE_BATCH}")
print(f"[config] Max steps: {MAX_STEPS}, Warmup: {WARMUP_STEPS}, LR: encoder={ENCODER_LR}, decoder={DECODER_LR}")
print(f"[config] Train repo: {TRAINING_HUB_REPO}")
print(f"[config] Final repo: {HF_MODEL_REPO}")
