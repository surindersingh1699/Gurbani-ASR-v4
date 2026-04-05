import torch

# --- GPU detection ---
GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

# Target effective batch size = 64
# Adjust per_device_train_batch_size and gradient_accumulation_steps
# to maintain this while fitting in VRAM
if "A100" in GPU_NAME:
    BATCH_SIZE = 32
    GRAD_ACCUM = 2
elif "A40" in GPU_NAME:
    BATCH_SIZE = 32
    GRAD_ACCUM = 2
elif "4090" in GPU_NAME:
    BATCH_SIZE = 8
    GRAD_ACCUM = 4
elif "3090" in GPU_NAME:
    BATCH_SIZE = 8
    GRAD_ACCUM = 4
elif "A5000" in GPU_NAME:
    BATCH_SIZE = 8
    GRAD_ACCUM = 4
elif "L4" in GPU_NAME:
    BATCH_SIZE = 8
    GRAD_ACCUM = 4
elif "T4" in GPU_NAME:
    BATCH_SIZE = 4
    GRAD_ACCUM = 8
elif "V100" in GPU_NAME:
    BATCH_SIZE = 4
    GRAD_ACCUM = 8
else:
    # Conservative fallback for unknown GPUs
    BATCH_SIZE = 4
    GRAD_ACCUM = 8

EFFECTIVE_BATCH = BATCH_SIZE * GRAD_ACCUM

# --- Paths ---
OUTPUT_DIR = "/workspace/surt/checkpoints"
LOG_DIR = "/workspace/surt/logs"

# --- Model ---
BASE_MODEL = "openai/whisper-small"
HF_MODEL_REPO = "surindersinghssj/surt-small-v1"

# --- Training constants ---
EVAL_STEPS = 200
SAVE_STEPS = 200
SAVE_TOTAL_LIMIT = 3
WARMUP_STEPS = 400
GENERATION_MAX_LENGTH = 448  # Gurmukhi tokenizer expansion: 3-5x longer than English

# --- Training hyperparameters ---
MAX_STEPS = 5000           # ~5 effective epochs over 64k rows with effective batch 64
LEARNING_RATE = 1e-4       # Base LR for decoder and proj_out
ENCODER_LR = 5e-5          # Encoder learns slower (transfer learning)
DECODER_LR = 1e-4          # Decoder + proj_out at base LR
WEIGHT_DECAY = 0.01        # Standard AdamW weight decay

# --- Hub (training checkpoints) ---
TRAINING_HUB_REPO = "surindersinghssj/surt-small-v1-training"

# --- Mool Mantar (Gurmukhi vocabulary anchor for initial_prompt) ---
MOOL_MANTAR = "ੴ ਸਤਿ ਨਾਮੁ ਕਰਤਾ ਪੁਰਖੁ ਨਿਰਭਉ ਨਿਰਵੈਰੁ ਅਕਾਲ ਮੂਰਤਿ ਅਜੂਨੀ ਸੈਭੰ ਗੁਰ ਪ੍ਰਸਾਦਿ"

# --- Dataset ---
DATASET_NAME = "surindersinghssj/gurbani-asr"
# Optional auxiliary kirtan-aligned dataset mixed into train stream.
AUX_TRAIN_DATASET_NAME = "surindersinghssj/gurbani-asr-whisper-aligned"
# Fraction of batches sampled from AUX_TRAIN_DATASET_NAME when enabled.
AUX_TRAIN_PROBABILITY = 0.0
TRAIN_SPLIT = "train"
VAL_SPLIT = "validation"   # Proper split created by scripts/create_val_split.py
TEXT_COLUMN = "transcription"  # Column name for transcription text in Gurbani ASR dataset
VAL_SIZE = 300             # Number of validation examples to materialize eagerly
SHUFFLE_BUFFER = 500       # Buffer size for streaming shuffle

# --- Weights & Biases ---
# Auto-enabled when WANDB_API_KEY is set in environment.
# On RunPod: export WANDB_API_KEY="your-key" in ~/.bashrc
WANDB_PROJECT = "surt-pilot"
WANDB_ENTITY = "sabysurinder-surinder"

print(f"[config] GPU: {GPU_NAME}")
print(f"[config] Batch: {BATCH_SIZE} x Accum: {GRAD_ACCUM} = Effective: {EFFECTIVE_BATCH}")
print(f"[config] Max steps: {MAX_STEPS}, LR: encoder={ENCODER_LR}, decoder={DECODER_LR}")
print(f"[config] Train repo: {TRAINING_HUB_REPO}")
print(f"[config] Final repo: {HF_MODEL_REPO}")
