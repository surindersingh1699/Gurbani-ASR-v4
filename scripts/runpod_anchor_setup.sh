#!/bin/bash
# Setup + launch on a fresh RunPod A40 pod for the first-letter anchor CTC run.
# Designed to be idempotent — re-running picks up where it left off.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/surindersingh1699/Gurbani-ASR-v4.git}"
BRANCH="${BRANCH:-main}"
WORK_DIR="${WORK_DIR:-/workspace}"
REPO_DIR="${REPO_DIR:-${WORK_DIR}/Gurbani-ASR-v4}"
HF_CACHE="${HF_HOME:-${WORK_DIR}/.cache/huggingface}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_CACHE}/datasets}"

export HF_HOME="${HF_CACHE}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE}"
export PATH="/opt/conda/bin:${PATH}"

mkdir -p "${HF_CACHE}" "${HF_DATASETS_CACHE}" "${WORK_DIR}/data/manifests" \
         "${WORK_DIR}/data/audio_anchor" "${WORK_DIR}/checkpoints"

echo "=== System info ==="
nvidia-smi || true
python -V

if [ ! -d "${REPO_DIR}/.git" ]; then
    echo "=== Cloning repo ==="
    git clone --branch "${BRANCH}" "${REPO_URL}" "${REPO_DIR}"
else
    echo "=== Updating repo ==="
    git -C "${REPO_DIR}" fetch origin
    git -C "${REPO_DIR}" checkout "${BRANCH}"
    git -C "${REPO_DIR}" pull --ff-only origin "${BRANCH}"
fi

cd "${REPO_DIR}"

echo "=== Installing system deps ==="
apt-get update -y && apt-get install -y --no-install-recommends \
    tmux libsndfile1 ffmpeg sox build-essential git-lfs

echo "=== Installing Python deps ==="
# NeMo wants nemo_toolkit[asr]. Pin datasets<4 (project memory).
pip install --upgrade pip
pip install \
    'nemo_toolkit[asr]==1.23.0' \
    'datasets<4' \
    'soundfile' \
    'huggingface_hub>=0.24' \
    'wandb' \
    'jiwer' \
    'pytorch-lightning==2.0.7'

echo "=== Verifying installs ==="
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import nemo.collections.asr as a; print('nemo OK')"
python -c "from huggingface_hub import HfApi; print('hf user:', HfApi().whoami()['name'])"

echo "=== Mapper round-trip ==="
python scripts/sttm_first_letter_map.py --limit 5000

echo "=== Building manifests (sehajpath only) ==="
if [ ! -s "${WORK_DIR}/data/manifests/anchor_first_letter_train.jsonl" ]; then
    python scripts/build_first_letter_anchor_manifests.py \
        --data-root "${WORK_DIR}/data" \
        --audio-root "${WORK_DIR}/data/audio_anchor"
else
    echo "manifests already built, skipping"
fi

wc -l "${WORK_DIR}/data/manifests"/anchor_first_letter_*.jsonl

# Push the manifests + vocab snapshot to HF dataset repo
python - <<'PYEOF'
import os, sys
from pathlib import Path
from huggingface_hub import HfApi
work = Path("/workspace/data/manifests")
repo = "surindersinghssj/gurbani-anchor-first-letter-v1"
api = HfApi()
api.create_repo(repo, repo_type="dataset", private=False, exist_ok=True)
for fp in sorted(work.glob("anchor_first_letter_*")):
    api.upload_file(path_or_fileobj=str(fp), path_in_repo=f"manifests/{fp.name}",
                    repo_id=repo, repo_type="dataset",
                    commit_message=f"upload {fp.name}")
    print("pushed", fp.name)
PYEOF

echo "=== Smoke training (100 steps) ==="
python scripts/train_first_letter_anchor.py \
    --config training/conformer_ctc_medium_first_letter.yaml \
    --smoke 2>&1 | tee "${WORK_DIR}/checkpoints/smoke.log"

echo "=== Smoke complete. Launching full run in tmux ==="
tmux new-session -d -s anchor || tmux kill-session -t anchor && tmux new-session -d -s anchor
tmux send-keys -t anchor "cd ${REPO_DIR} && export HF_HOME=${HF_CACHE} && export HF_DATASETS_CACHE=${HF_DATASETS_CACHE} && \
python scripts/train_first_letter_anchor.py \
    --config training/conformer_ctc_medium_first_letter.yaml \
    --hf-model-repo surindersinghssj/surt-anchor-ctc-first-letter-v1 \
    --hf-push-every 2000 2>&1 | tee ${WORK_DIR}/checkpoints/full.log" C-m

echo "=== Done. tmux attach -t anchor to view ==="
