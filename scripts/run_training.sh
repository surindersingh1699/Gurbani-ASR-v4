#!/bin/bash
# Auto-restart training wrapper with multi-GPU (torchrun) support.
# Automatically restarts training on crash (spot preemption, OOM, etc.)
# Usage: bash scripts/run_training.sh [extra args for surt.train]
# Example: bash scripts/run_training.sh --max-steps 3000

set -euo pipefail

export PATH=/opt/conda/bin:$PATH

# Load tokens from environment or .bashrc
if [ -z "${HF_TOKEN:-}" ]; then
    export HF_TOKEN=$(grep HF_TOKEN /root/.bashrc 2>/dev/null | tail -1 | cut -d= -f2 | tr -d '"' | tr -d "'" || echo "")
fi
if [ -z "${WANDB_API_KEY:-}" ]; then
    export WANDB_API_KEY=$(grep WANDB_API_KEY /root/.bashrc 2>/dev/null | tail -1 | cut -d= -f2 | tr -d '"' | tr -d "'" || echo "")
fi

cd /workspace/Gurbani-ASR-v4
export PYTHONPATH=/workspace/Gurbani-ASR-v4:${PYTHONPATH:-}

# Detect GPUs
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
echo "[launcher] Detected $NUM_GPUS GPU(s)"
echo "[launcher] Extra args: $*"

MAX_RESTARTS=10
RESTART_COUNT=0
RESTART_DELAY=30  # seconds between restarts

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
    echo ""
    echo "[launcher] Training exited with code $EXIT_CODE"
    echo "[launcher] Restarting in ${RESTART_DELAY}s... ($RESTART_COUNT/$MAX_RESTARTS)"
    sleep $RESTART_DELAY
done

echo "[launcher] Max restarts ($MAX_RESTARTS) reached. Giving up."
exit 1
