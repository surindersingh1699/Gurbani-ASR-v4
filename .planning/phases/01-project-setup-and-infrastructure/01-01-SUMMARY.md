# Plan 01-01: Project Scaffolding — Summary

**Status:** Complete
**Completed:** 2026-04-04
**Commit:** d9f0291

## What Was Built

4 files creating the foundation for the Surt training pipeline:

| File | Purpose |
|------|---------|
| `surt/__init__.py` | Python package marker |
| `surt/config.py` | GPU auto-detection, batch sizing, training constants, paths |
| `requirements.txt` | 7 Python dependencies with version pins (no torch) |
| `setup_pod.sh` | One-shot idempotent RunPod pod provisioning script |

## Key Design Decisions

- **GPU detection via `torch.cuda.get_device_name(0)`** with if/elif chain mapping 8 GPU types to batch_size/grad_accum targeting effective batch of 32
- **Conservative fallback** (batch=4, accum=8) for unknown GPUs
- **No torch in requirements.txt** — provided by RunPod template, pip installing it breaks CUDA
- **setup_pod.sh is idempotent** — safe to re-run (tmux uses `|| echo "already exists"`)
- **HF_TOKEN validated early** in setup script with clear error message if not set

## Verification Results

| Check | Result |
|-------|--------|
| config.py syntax (ast.parse) | Pass |
| surt/__init__.py exists | Pass |
| setup_pod.sh shell syntax (bash -n) | Pass |
| setup_pod.sh is executable | Pass |
| requirements.txt has 7 deps | Pass |
| GPU detection present | Pass |
| MOOL_MANTAR present | Pass |
| tmux in setup script | Pass |
| HF_TOKEN in setup script | Pass |

## Requirements Addressed

- **INFRA-01**: Pipeline runs as Python scripts; setup_pod.sh provisions RunPod spot GPU
- **INFRA-02**: setup_pod.sh installs tmux and creates persistent session
- **INFRA-03**: setup_pod.sh validates HF_TOKEN env var; config.py never hardcodes tokens
- **INFRA-04**: requirements.txt enables single `pip install -r requirements.txt` command

## What's Next

Plan 01-02: Human checkpoint to validate on a live RunPod pod (all 4 success criteria must pass on actual hardware).
