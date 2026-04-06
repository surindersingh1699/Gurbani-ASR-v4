#!/bin/bash
# Setup and run 1000-step pilot on RunPod
# Usage: bash scripts/setup_and_run_pilot.sh
set -e

echo "=== Step 0: Environment ==="
export PATH="/opt/conda/bin:$PATH"
source /root/.bashrc 2>/dev/null || true
python --version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo ""
echo "=== Step 1: Pull latest code ==="
cd /workspace/Gurbani-ASR-v4 || { echo "Repo not found at /workspace/Gurbani-ASR-v4"; exit 1; }
git pull origin codex/complete-phase-4-work

echo ""
echo "=== Step 2: Install/update dependencies ==="
pip install -q --upgrade datasets huggingface_hub jiwer audiomentations wandb

echo ""
echo "=== Step 3: Create validation split on HuggingFace ==="
python scripts/create_val_split.py

echo ""
echo "=== Step 4: Run 1000-step pilot ==="
echo "Starting pilot with W&B logging, eval every 100 steps..."
python -m surt.train --mode pilot --skip-preflight

echo ""
echo "=== Done ==="
