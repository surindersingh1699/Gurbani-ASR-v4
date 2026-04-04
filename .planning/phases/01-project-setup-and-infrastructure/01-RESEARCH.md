# Phase 1: Project Setup and Infrastructure - Research

**Researched:** 2026-04-04
**Domain:** RunPod spot GPU environment, Python ML dependency management, GPU auto-detection
**Confidence:** HIGH

## Summary

Phase 1 establishes the RunPod spot GPU development environment for the Surt training pipeline. The work is straightforward infrastructure: provision a GPU pod, install Python dependencies, configure HuggingFace authentication, set up tmux for session persistence, and build a config module that auto-detects the GPU and sets batch size / gradient accumulation accordingly.

RunPod provides pre-built PyTorch templates with CUDA, SSH, and JupyterLab already configured. The recommended template is "PyTorch 2.7.1 + CUDA 12.6" which comes with PyTorch, Transformers, Datasets, and Accelerate pre-installed. The remaining dependencies (audiomentations, jiwer, huggingface_hub) install via a single `pip install` command. tmux must be installed separately via `apt-get` as it is not included in RunPod templates. The HuggingFace Hub library natively supports the `HF_TOKEN` environment variable with automatic authentication -- no explicit `login()` call is needed if `HF_TOKEN` is set.

The key design decision is the GPU auto-detection config module. The pattern is simple: read `torch.cuda.get_device_name(0)`, match against known GPU names, and set `batch_size` and `gradient_accumulation_steps` to maintain a target effective batch size of 32. This module will be used by all subsequent phases.

**Primary recommendation:** Use RunPod's "PyTorch 2.7.1 + CUDA 12.6" template as the base, install remaining deps via a single pip command, and build a minimal `config.py` that maps GPU name to batch/accumulation settings.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| INFRA-01 | Entire pipeline runs as Python scripts via SSH on RunPod spot GPU (< $0.30/hr) | RTX 3090 ($0.22/hr on-demand, spot cheaper), RTX A5000 ($0.16/hr) both qualify. RunPod SSH on port 22 is standard. PyTorch template provides base environment. |
| INFRA-02 | Training runs inside tmux for session persistence across SSH disconnects | tmux must be installed via `apt-get install -y tmux`. Sessions survive SSH disconnects but NOT pod restarts. RunPod docs confirm this pattern. |
| INFRA-03 | HuggingFace token loaded from environment variable (never hardcoded) | `HF_TOKEN` env var is the standard pattern. huggingface_hub auto-detects it with priority over cached tokens. `HfApi().whoami()` works automatically when `HF_TOKEN` is set. |
| INFRA-04 | All dependencies installable via single `pip install` command | Single command: `pip install transformers datasets audiomentations jiwer huggingface_hub accelerate evaluate`. PyTorch/CUDA come from the RunPod template. |
</phase_requirements>

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.7.1+ | Deep learning framework | Pre-installed in RunPod template, CUDA 12.6 support |
| transformers | >=4.46 (v5 API) | Whisper model, Seq2SeqTrainer | HuggingFace standard; `eval_strategy` replaces deprecated `evaluation_strategy` in v4.46+ |
| datasets | >=2.16 | Streaming data loading from HuggingFace Hub | Required for `load_dataset(..., streaming=True)` |
| audiomentations | >=0.42 | Waveform augmentation (Gaussian noise, room sim, time stretch, pitch shift) | Standard audio augmentation; lighter than torchaudio transforms |
| jiwer | >=4.0 | WER (Word Error Rate) computation | Standard ASR evaluation metric, uses RapidFuzz C++ backend |
| huggingface_hub | >=0.20 | Authentication, model push, API access | `HF_TOKEN` env var auto-detection, `HfApi().whoami()` |
| accelerate | >=0.26 | Training utilities, device placement | Required by Seq2SeqTrainer for gradient accumulation, bf16 |
| evaluate | >=0.4 | Metric loading framework | Used for `evaluate.load("wer")` pattern in compute_metrics |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| tmux | system package | Session persistence | Always -- every training run must be inside tmux |
| pyroomacoustics | (audiomentations dep) | Room impulse response simulation | Pulled in by audiomentations `RoomSimulator` |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| audiomentations | torchaudio transforms | torchaudio is already installed but audiomentations has better augmentation primitives (RoomSimulator, etc.) and requirements specify it |
| jiwer | evaluate + "wer" metric | evaluate wraps jiwer internally; direct jiwer is simpler for custom compute_metrics |
| RunPod | Google Colab Pro | Original training plan used Colab; shifted to RunPod for SSH automation and session reliability |

**Installation:**
```bash
# On RunPod pod (PyTorch 2.7.1 + CUDA 12.6 template -- PyTorch, transformers, datasets, accelerate already present)
pip install audiomentations jiwer huggingface_hub evaluate

# OR full standalone install (if template has older versions):
pip install transformers datasets audiomentations jiwer huggingface_hub accelerate evaluate
```

**System dependencies (run before pip):**
```bash
apt-get update && apt-get install -y tmux libsndfile1
```

## Architecture Patterns

### Recommended Project Structure

```
surt/
├── config.py              # GPU detection, batch size, paths, constants
├── train.py               # Main training script (Phase 3)
├── data.py                # Data pipeline: streaming, augmentation, preprocessing (Phase 2)
├── model.py               # Model init: Whisper config, generation config (Phase 2)
├── callbacks.py           # PushBestToHub callback (Phase 3)
├── smoke_test.py          # Pre-flight validation (Phase 4)
├── requirements.txt       # All pip dependencies
└── setup_pod.sh           # One-shot pod setup script (tmux, apt deps, pip deps, HF auth check)
```

### Pattern 1: GPU Auto-Detection Config Module

**What:** A `config.py` that reads the GPU name at import time and exposes batch/accumulation settings as module-level constants.
**When to use:** Every script imports `config` to get environment-appropriate settings.
**Example:**
```python
# config.py
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
    # Conservative fallback
    BATCH_SIZE = 4
    GRAD_ACCUM = 8

EFFECTIVE_BATCH = BATCH_SIZE * GRAD_ACCUM

# --- Paths ---
OUTPUT_DIR = "/workspace/surt/checkpoints"
LOG_DIR = "/workspace/surt/logs"

# --- Model ---
BASE_MODEL = "openai/whisper-small"
HF_MODEL_REPO = "surindersingh/surt-small-v1"  # placeholder -- user sets this

# --- Training constants ---
EVAL_STEPS = 300
SAVE_STEPS = 300
SAVE_TOTAL_LIMIT = 3
WARMUP_STEPS = 400
GENERATION_MAX_LENGTH = 448

# --- Mool Mantar (Gurmukhi vocabulary anchor) ---
MOOL_MANTAR = "ੴ ਸਤਿ ਨਾਮੁ ਕਰਤਾ ਪੁਰਖੁ ਨਿਰਭਉ ਨਿਰਵੈਰੁ ਅਕਾਲ ਮੂਰਤਿ ਅਜੂਨੀ ਸੈਭੰ ਗੁਰ ਪ੍ਰਸਾਦਿ"

print(f"[config] GPU: {GPU_NAME}")
print(f"[config] Batch: {BATCH_SIZE} x Accum: {GRAD_ACCUM} = Effective: {EFFECTIVE_BATCH}")
```

### Pattern 2: One-Shot Pod Setup Script

**What:** A bash script that installs all dependencies and validates the environment in one step.
**When to use:** Every time a new RunPod pod is provisioned (spot instances are ephemeral).
**Example:**
```bash
#!/bin/bash
# setup_pod.sh -- Run once after SSH into a fresh RunPod pod
set -euo pipefail

echo "=== Installing system dependencies ==="
apt-get update && apt-get install -y tmux libsndfile1

echo "=== Installing Python dependencies ==="
pip install transformers datasets audiomentations jiwer huggingface_hub accelerate evaluate

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

echo "=== Starting tmux session ==="
tmux new-session -d -s surt
echo "tmux session 'surt' created. Attach with: tmux attach -t surt"

echo "=== Setup complete ==="
```

### Pattern 3: HuggingFace Token from Environment Variable

**What:** The `HF_TOKEN` environment variable is automatically detected by huggingface_hub.
**When to use:** Always. Never hardcode tokens.
**Example:**
```python
import os
from huggingface_hub import HfApi

# HF_TOKEN is automatically used by huggingface_hub when set as env var
# No explicit login() call needed
api = HfApi()
user_info = api.whoami()
print(f"Authenticated as: {user_info['name']}")

# Verify token is from env, not cached
assert os.environ.get("HF_TOKEN"), "HF_TOKEN must be set as environment variable"
```

### Anti-Patterns to Avoid

- **Hardcoding HF token:** Never put tokens in source code, config files, or git. Always use `HF_TOKEN` env var.
- **Running training without tmux:** A single SSH disconnect kills the process. Always `tmux new -s surt` first.
- **Installing PyTorch via pip on RunPod:** The template already has PyTorch+CUDA compiled for the GPU. Re-installing via pip can break CUDA compatibility.
- **Using `num_train_epochs` with streaming:** Streaming datasets have no length; use `max_steps` instead (this is a Phase 3 concern but config.py should anticipate it).
- **Using `evaluation_strategy` instead of `eval_strategy`:** Deprecated in transformers v4.46, removed in v5. Use `eval_strategy`.
- **Using `fp16` on Ampere GPUs:** Use `bf16=True` instead for better numerical stability (TRAIN-03 requirement).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| WER computation | Custom edit-distance WER | `jiwer.wer()` | Handles edge cases (empty strings, normalization), uses C++ backend |
| Audio augmentation | Custom numpy noise/reverb | `audiomentations.Compose` | Validated transforms, correct sample rate handling, probability-based |
| Token authentication | Custom token file management | `HF_TOKEN` env var + huggingface_hub | Auto-detected, priority over cached tokens, standard pattern |
| Model checkpointing to Hub | Custom upload logic | `HfApi().upload_folder()` | Handles large files, resumable uploads, commit messages |
| GPU memory management | Custom batch size tuning | config.py lookup table | Deterministic, no OOM surprises, easy to update |

**Key insight:** This phase is pure infrastructure. Every component has a standard solution. The only custom code needed is `config.py` (GPU-to-batch-size mapping) and `setup_pod.sh` (one-shot provisioning).

## Common Pitfalls

### Pitfall 1: RunPod Spot Instance Preemption Loses tmux Sessions
**What goes wrong:** Spot instances can be reclaimed at any time. tmux sessions survive SSH disconnects but NOT pod restarts/preemptions.
**Why it happens:** Spot pricing is cheaper because RunPod can reclaim capacity.
**How to avoid:** (1) Use Network Volumes for persistent storage at `/workspace`. (2) Checkpoint to HuggingFace Hub frequently (every 300 steps). (3) The `setup_pod.sh` script should be idempotent -- re-runnable on a fresh pod. (4) Store `HF_TOKEN` in RunPod pod environment variables, not just shell export.
**Warning signs:** Pod suddenly terminates; reconnect shows fresh environment.

### Pitfall 2: PyTorch/CUDA Version Mismatch After pip install
**What goes wrong:** Running `pip install torch` on a RunPod pod replaces the pre-compiled PyTorch+CUDA with a CPU-only or wrong-CUDA version.
**Why it happens:** pip defaults to the latest PyTorch wheel which may not match the pod's CUDA version.
**How to avoid:** Never `pip install torch` on RunPod. The template provides the correct version. Only install additional packages (transformers, audiomentations, etc.).
**Warning signs:** `torch.cuda.is_available()` returns False after pip install.

### Pitfall 3: HF_TOKEN Not Set or Using Wrong Token
**What goes wrong:** `HfApi().whoami()` fails or returns wrong user; model push fails with 401.
**Why it happens:** Token not exported, or stale cached token takes priority.
**How to avoid:** (1) Set `HF_TOKEN` in RunPod pod environment variables (persists across SSH sessions). (2) Verify with `python -c "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])"` before any training run. (3) The `HF_TOKEN` env var has priority over cached tokens per huggingface_hub docs.
**Warning signs:** Push to Hub fails; whoami returns unexpected user.

### Pitfall 4: Missing libsndfile for audiomentations
**What goes wrong:** `import audiomentations` fails or certain transforms (RoomSimulator) fail at runtime.
**Why it happens:** RunPod templates don't include `libsndfile1` system package.
**How to avoid:** Run `apt-get install -y libsndfile1` before pip install. Include in `setup_pod.sh`.
**Warning signs:** ImportError or runtime error mentioning libsndfile/soundfile.

### Pitfall 5: tmux Not Installed
**What goes wrong:** `tmux: command not found` after SSH.
**Why it happens:** RunPod templates don't include tmux by default.
**How to avoid:** Install via `apt-get install -y tmux` in setup script.
**Warning signs:** Command not found error.

### Pitfall 6: Config Module Doesn't Handle Unknown GPUs
**What goes wrong:** Script crashes if GPU name doesn't match any known pattern.
**Why it happens:** RunPod GPU availability varies; may get an unexpected GPU type.
**How to avoid:** Always include a conservative fallback (batch=4, accum=8) in the else branch. Log a warning but don't crash.
**Warning signs:** KeyError or missing config value; OOM errors from too-large batch size.

## Code Examples

Verified patterns from official sources:

### HuggingFace Token Verification
```python
# Source: huggingface_hub official docs
import os
from huggingface_hub import HfApi

# HF_TOKEN env var is auto-detected by huggingface_hub
# Priority: env var > cached token on disk
assert os.environ.get("HF_TOKEN"), "HF_TOKEN environment variable must be set"

api = HfApi()
user = api.whoami()
print(f"Authenticated as: {user['name']}")
```

### GPU Detection with torch.cuda
```python
# Source: PyTorch docs (torch.cuda.get_device_name)
import torch

assert torch.cuda.is_available(), "No GPU detected -- RunPod pod must have a GPU"
gpu_name = torch.cuda.get_device_name(0)
vram_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
print(f"GPU: {gpu_name}, VRAM: {vram_gb:.1f} GB")
```

### tmux Session Management
```bash
# Source: RunPod documentation (https://docs.runpod.io/tips-and-tricks/tmux)
# Install tmux (not pre-installed on RunPod)
apt-get update && apt-get install -y tmux

# Create named session
tmux new-session -d -s surt

# Attach to session
tmux attach -t surt

# Detach from session (inside tmux)
# Press: Ctrl+B, then D

# List sessions (verify persistence after SSH reconnect)
tmux ls

# Kill session when done
tmux kill-session -t surt
```

### Full Dependency Install (Single Command)
```bash
# Source: transformers, audiomentations, jiwer official docs
# System deps first
apt-get update && apt-get install -y tmux libsndfile1

# Python deps (PyTorch already from template)
pip install transformers datasets audiomentations jiwer huggingface_hub accelerate evaluate
```

### Import Verification (Success Criterion 1)
```python
# Matches roadmap success criterion exactly
import transformers, datasets, audiomentations, jiwer, huggingface_hub
print("ok")
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `evaluation_strategy` in TrainingArguments | `eval_strategy` | transformers v4.46 (deprecated), removed in v5 | Must use `eval_strategy` |
| `fp16=True` for mixed precision | `bf16=True` on Ampere+ GPUs | PyTorch 1.10+ / Ampere GPUs | Better numerical stability, no loss scaling needed |
| Google Colab Pro for training | RunPod spot GPU via SSH | Project decision (INFRA-01) | SSH automation, session reliability, cost control |
| Manual HF token in scripts | `HF_TOKEN` env var auto-detection | huggingface_hub >=0.19 | Cleaner, more secure, priority-based |
| `pip install audiomentations` only | `apt-get install libsndfile1` + pip install | Always needed on server envs | RoomSimulator and audio I/O require system lib |

**Deprecated/outdated:**
- `evaluation_strategy`: Renamed to `eval_strategy` in transformers >=4.46. Training plan code uses the old name -- must update.
- `generation_max_length=225`: Training plan used 225. Requirements specify 448 for Gurmukhi tokenizer expansion. Config must use 448.
- `fp16=True`: Training plan used fp16. Requirements specify bf16 for Ampere GPUs (A40, RTX 3090, RTX 4090, L4 are all Ampere or newer).

## Open Questions

1. **Exact RunPod GPU to target for spot pricing**
   - What we know: RTX 3090 ($0.22/hr on-demand), RTX A5000 ($0.16/hr on-demand), L4 ($0.44/hr). Spot pricing is 50-70% cheaper but fluctuates.
   - What's unclear: Exact spot price at time of provisioning; GPU availability varies by region and demand.
   - Recommendation: Target RTX 3090 or RTX A5000 as primary (both well under $0.30/hr even on-demand). Config module handles whatever GPU is assigned.

2. **Network Volume for checkpoint persistence**
   - What we know: RunPod Network Volumes persist across pod restarts. Spot preemption loses all local data.
   - What's unclear: Whether to use Network Volume (adds cost) or rely solely on HuggingFace Hub push for checkpoint safety.
   - Recommendation: For Phase 1 setup, prepare for both. Config should point to `/workspace/` which is the Network Volume mount point. Hub push is the ultimate safety net.

3. **audiomentations RoomSimulator dependency on pyroomacoustics**
   - What we know: RoomSimulator uses room impulse response simulation. Docs reference pyroomacoustics for ray tracing.
   - What's unclear: Whether pyroomacoustics is auto-installed as a dependency or needs explicit install.
   - Recommendation: Test `from audiomentations import RoomSimulator` after pip install. If it fails, add `pyroomacoustics` to the install command. This is LOW confidence -- needs validation during setup.

## Sources

### Primary (HIGH confidence)
- Context7 `/huggingface/huggingface_hub` - HF_TOKEN env var authentication, `HfApi().whoami()`, login patterns
- Context7 `/huggingface/transformers` - Seq2SeqTrainer setup, `eval_strategy` parameter, Whisper fine-tuning
- Context7 `/iver56/audiomentations` - Installation, Compose pattern, augmentation transforms
- [RunPod tmux documentation](https://docs.runpod.io/tips-and-tricks/tmux) - tmux setup, persistence behavior, best practices
- [PyTorch torch.cuda.get_device_name docs](https://docs.pytorch.org/docs/stable/generated/torch.cuda.get_device_name.html) - GPU name detection API

### Secondary (MEDIUM confidence)
- [RunPod GPU pricing page](https://www.runpod.io/gpu-pricing) - A40 $0.35/hr, RTX 3090 $0.22/hr, RTX A5000 $0.16/hr on-demand
- [RunPod templates guide](https://dev.to/vishva_ram/the-complete-guide-to-runpod-templates-cuda-pytorch-environments-for-every-ai-project-4i94) - PyTorch 2.7.1 + CUDA 12.6 template, pre-installed packages, SSH config
- [jiwer on PyPI](https://pypi.org/project/jiwer/) - Version 4.0, RapidFuzz C++ backend, WER/CER metrics
- [transformers eval_strategy migration](https://discuss.huggingface.co/t/solved-difference-between-eval-strategy-and-evaluation-strategy/96657) - Deprecated in v4.46, removed in v5
- [audiomentations RoomSimulator docs](https://iver56.github.io/audiomentations/waveform_transforms/room_simulator/) - Parameters, room simulation behavior

### Tertiary (LOW confidence)
- audiomentations pyroomacoustics dependency for RoomSimulator -- docs reference it but unclear if auto-installed. Needs validation.
- RunPod spot pricing -- on-demand prices confirmed but spot prices fluctuate; 50-70% discount is approximate.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All libraries verified via Context7 and official docs; versions confirmed
- Architecture: HIGH - Project structure and patterns follow standard ML training pipeline conventions
- Pitfalls: HIGH - tmux persistence limits, PyTorch/CUDA mismatch, libsndfile all confirmed by official RunPod/library docs
- GPU pricing: MEDIUM - On-demand prices confirmed; spot prices are approximate

**Research date:** 2026-04-04
**Valid until:** 2026-05-04 (stable infrastructure; library versions unlikely to change significantly)
