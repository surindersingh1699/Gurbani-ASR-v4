#!/bin/bash
# Auto-restart training wrapper with multi-GPU (torchrun) support.
# Automatically restarts training on crash (spot preemption, OOM, etc.)
# Usage: bash scripts/run_training.sh [extra args for surt.train]
# Example: bash scripts/run_training.sh --max-steps 3000

set -euo pipefail

export PATH=/opt/conda/bin:$PATH
export HF_HOME=/workspace/.cache/huggingface

# Load tokens from environment or .bashrc
if [ -z "${HF_TOKEN:-}" ]; then
    export HF_TOKEN=$(grep HF_TOKEN /root/.bashrc 2>/dev/null | tail -1 | cut -d= -f2 | tr -d '"' | tr -d "'" || echo "")
fi
if [ -z "${WANDB_API_KEY:-}" ]; then
    export WANDB_API_KEY=$(grep WANDB_API_KEY /root/.bashrc 2>/dev/null | tail -1 | cut -d= -f2 | tr -d '"' | tr -d "'" || echo "")
fi
if [ -z "${SURT_BASE_MODEL:-}" ]; then
    export SURT_BASE_MODEL=$(grep SURT_BASE_MODEL /root/.bashrc 2>/dev/null | tail -1 | cut -d= -f2 | tr -d '"' | tr -d "'" || echo "")
fi

cd /workspace/Gurbani-ASR-v4
export PYTHONPATH=/workspace/Gurbani-ASR-v4:${PYTHONPATH:-}

# Pre-cache training data using all CPU cores before CUDA is initialised.
# On cache hit this completes in seconds; on first run ~20 min with 24 cores.
echo "[launcher] Pre-caching training data (fork-safe, no CUDA)..."
python scripts/prepare_data.py || { echo "[launcher] Data prep failed — aborting"; exit 1; }

# Detect GPUs
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
echo "[launcher] Detected $NUM_GPUS GPU(s)"
echo "[launcher] Extra args: $*"

MAX_RESTARTS=10
RESTART_COUNT=0
RESTART_DELAY=30  # seconds between restarts
PREV_EXIT_CODE=""
CONSECUTIVE_SAME=0

while [ $RESTART_COUNT -lt $MAX_RESTARTS ]; do
    echo ""
    echo "========================================================"
    echo "[launcher] Starting training (attempt $((RESTART_COUNT + 1))/$MAX_RESTARTS)"
    echo "[launcher] $(date)"
    echo "========================================================"

    EXIT_CODE=0
    if [ "$NUM_GPUS" -gt 1 ]; then
        OMP_NUM_THREADS=1 torchrun \
            --nproc_per_node="$NUM_GPUS" \
            --master_port=29500 \
            surt/train.py --skip-preflight "$@" || EXIT_CODE=$?
    else
        python -m surt.train --skip-preflight "$@" || EXIT_CODE=$?
    fi

    if [ $EXIT_CODE -eq 0 ]; then
        echo "[launcher] Training completed successfully!"
        exit 0
    fi

    RESTART_COUNT=$((RESTART_COUNT + 1))

    # Detect deterministic failures: if same exit code 3 times in a row,
    # likely a corrupted checkpoint or code bug — retrying won't help.
    if [ "$EXIT_CODE" = "$PREV_EXIT_CODE" ]; then
        CONSECUTIVE_SAME=$((CONSECUTIVE_SAME + 1))
    else
        CONSECUTIVE_SAME=1
    fi
    PREV_EXIT_CODE=$EXIT_CODE

    if [ $CONSECUTIVE_SAME -ge 3 ]; then
        echo ""
        echo "[launcher] ERROR: Same exit code ($EXIT_CODE) 3 times in a row."
        echo "[launcher] Likely a deterministic error (corrupted checkpoint, code bug)."
        echo "[launcher] Aborting to avoid burning GPU time."
        exit 1
    fi

    echo ""
    echo "[launcher] Training exited with code $EXIT_CODE"
    echo "[launcher] Restarting in ${RESTART_DELAY}s... ($RESTART_COUNT/$MAX_RESTARTS)"
    sleep $RESTART_DELAY
done

echo "[launcher] Max restarts ($MAX_RESTARTS) reached. Giving up."
exit 1
