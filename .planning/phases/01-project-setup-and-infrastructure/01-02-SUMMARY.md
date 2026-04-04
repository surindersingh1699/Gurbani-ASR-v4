# Plan 01-02: RunPod Validation — Summary

**Status:** Complete
**Completed:** 2026-04-04
**Environment:** NVIDIA A40, 47.7 GB VRAM, RunPod spot GPU

## Validation Results

| Criterion | Command | Result |
|-----------|---------|--------|
| 1. Python imports | `python -c "import transformers, datasets, audiomentations, jiwer, huggingface_hub; print('ok')"` | All imports OK |
| 2. HuggingFace auth | `HfApi().whoami()` | Logged in as: surindersinghssj |
| 3. tmux persistence | `tmux ls` after SSH reconnect | surt: 1 windows (persists) |
| 4. GPU config detection | `from surt.config import ...` | NVIDIA A40 -> batch=32 accum=1 eff=32 |

## Environment Details

- **GPU:** NVIDIA A40 (47.7 GB VRAM)
- **Template:** PyTorch 2.7.1 + CUDA 12.6 (conda-based)
- **Python:** 3.11 (via /opt/conda/bin)
- **SSH:** Direct TCP on port 22058

## Issues Found and Fixed During Validation

1. **`pip` not on PATH:** RunPod uses conda at `/opt/conda/bin/`. Fixed by adding `export PATH="/opt/conda/bin:$PATH"` to setup_pod.sh.
2. **`total_mem` API change:** PyTorch renamed to `total_memory`. Fixed in setup_pod.sh.
3. **Batch size optimization:** A40 has 46GB VRAM — bumped from batch=16/accum=2 to batch=32/accum=1 for full GPU utilization (same effective batch of 32, faster training).

## Requirements Validated

- **INFRA-01:** Pipeline scripts run via SSH on RunPod spot GPU (A40)
- **INFRA-02:** tmux session 'surt' persists across SSH disconnect
- **INFRA-03:** HF_TOKEN loaded from environment variable, authenticated as surindersinghssj
- **INFRA-04:** All dependencies installed via single `pip install -r requirements.txt`
