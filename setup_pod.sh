#!/bin/bash
# setup_pod.sh -- Run once after SSH into a fresh RunPod pod
set -euo pipefail

# Ensure conda bin is on PATH (RunPod uses /opt/conda)
export PATH="/opt/conda/bin:$PATH"

echo "=== Installing system dependencies ==="
apt-get update && apt-get install -y tmux libsndfile1

echo "=== Installing Python dependencies ==="
pip install -r requirements.txt

echo "=== Verifying Python imports ==="
python -c "import transformers, datasets, audiomentations, jiwer, huggingface_hub; print('All imports OK')"

echo "=== Verifying HuggingFace auth ==="
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN environment variable is not set"
    echo "Set it with: export HF_TOKEN=hf_your_token_here"
    exit 1
fi
python -c "from huggingface_hub import HfApi; user = HfApi().whoami(); print(f'Logged in as: {user[\"name\"]}')"

echo "=== Verifying GPU ==="
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')"

echo "=== Creating workspace directories ==="
mkdir -p /workspace/surt/checkpoints /workspace/surt/logs

echo "=== Verifying config module ==="
python -c "from surt.config import GPU_NAME, BATCH_SIZE, GRAD_ACCUM, EFFECTIVE_BATCH; print(f'Config OK: {GPU_NAME} -> batch={BATCH_SIZE} accum={GRAD_ACCUM} eff={EFFECTIVE_BATCH}')"

echo "=== Starting tmux session ==="
tmux new-session -d -s surt || echo "tmux session 'surt' already exists"
echo "tmux session 'surt' created. Attach with: tmux attach -t surt"

echo "=== Setup complete ==="
echo "Next: tmux attach -t surt"
