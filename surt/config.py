import torch

# --- GPU detection ---
GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

# Target effective batch size = 32
# Adjust per_device_train_batch_size and gradient_accumulation_steps
# to maintain this while fitting in VRAM
if "A100" in GPU_NAME:
    BATCH_SIZE = 16
    GRAD_ACCUM = 2
elif "A40" in GPU_NAME:
    BATCH_SIZE = 16
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
HF_MODEL_REPO = "surindersingh/surt-small-v1"

# --- Training constants ---
EVAL_STEPS = 300
SAVE_STEPS = 300
SAVE_TOTAL_LIMIT = 3
WARMUP_STEPS = 400
GENERATION_MAX_LENGTH = 448  # Gurmukhi tokenizer expansion: 3-5x longer than English

# --- Mool Mantar (Gurmukhi vocabulary anchor for initial_prompt) ---
MOOL_MANTAR = "ੴ ਸਤਿ ਨਾਮੁ ਕਰਤਾ ਪੁਰਖੁ ਨਿਰਭਉ ਨਿਰਵੈਰੁ ਅਕਾਲ ਮੂਰਤਿ ਅਜੂਨੀ ਸੈਭੰ ਗੁਰ ਪ੍ਰਸਾਦਿ"

print(f"[config] GPU: {GPU_NAME}")
print(f"[config] Batch: {BATCH_SIZE} x Accum: {GRAD_ACCUM} = Effective: {EFFECTIVE_BATCH}")
