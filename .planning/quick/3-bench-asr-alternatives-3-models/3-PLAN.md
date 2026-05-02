# Quick Task 3: Benchmark 3 ASR Alternatives vs faster-whisper-small-turbo

**Status:** in_progress
**Created:** 2026-05-01
**Branch:** `codex/complete-phase-4-work`

## Why

Before committing weeks of GPU time fine-tuning a new ASR architecture on the v3
500h kirtan corpus, we need a fast empirical answer to:

> "Of the 2026-class fast/streaming CPU ASR models, which one(s) deserve our
> fine-tune budget — measured by speed AND closeness to Punjabi out-of-box?"

The current production stack (faster-whisper-small-turbo) is the baseline. We
do NOT expect any of the 3 candidates to beat it on out-of-box accuracy
(except IndicConformer-pa, which already has Punjabi). This is a **decision-
support** bench, not a "pick a winner" bench.

## Models under test

| # | Model | Role | Punjabi out-of-box? |
|---|-------|------|---------------------|
| 0 | `faster-whisper-small-turbo` (Surt v2 fine-tune) | **Baseline** | Yes (fine-tuned) |
| 1 | `ai4bharat/indicconformer_stt_pa_hybrid_ctc_rnnt_large` | **Accuracy floor #1** | ✅ Native Gurmukhi (designed for Punjabi) |
| 2 | `facebook/mms-1b-all` (added: user request) | **Accuracy floor #2 / w2v2-CTC speed reference** | ✅ Native Gurmukhi via `pan` adapter |
| 3 | `Qwen/Qwen3-ASR-0.6B` | **Best transfer candidate** | ❌ Hindi only (Devanagari) — needs translit for fair WER |
| 4 | `nvidia/parakeet-tdt-0.6b-v3` | **Speed ceiling** | ❌ 25 EU langs only — expect ~100% WER, measure speed only |

## Eval datasets

Both are HF-hosted, canonical-corrected (post caption-chunks alignment fix):

- `surindersinghssj/gurbani-sehajpath-yt-captions-eval-canonical` — 444 clips
- `surindersinghssj/gurbani-kirtan-yt-captions-eval-canonical` — 573 clips

Reference column: `final_text` (canonical-corrected Gurmukhi). Audio is the
`audio` column. Texts are normalized via `surt.data.normalize_gurbani_text`
(strips ॥ markers and verse numbers) before WER/CER.

## Metrics per (model, dataset)

| Metric | Notes |
|--------|-------|
| WER %  | After normalization. Cross-script (Devanagari) translit'd to Gurmukhi for Qwen3. |
| CER %  | Same normalization. |
| RTF (CPU, 4-thread) | Wall-clock transcribe / audio duration. Lower is better. |
| RTF (CPU, 1-thread) | For "single-stream live" worst case. |
| Peak RAM (MB) | RSS via psutil. |
| First-decode latency | Time to first decoded text on a 1s warm clip — proxy for live feel. |

## Script

`scripts/bench_asr_alternatives.py`

- Lazy-imports each backend's deps so a missing dep skips that backend, not the bench.
- Reuses `surt.data.normalize_gurbani_text` (the canonical normalizer used in training).
- Default device: `--device cpu` (CPU is the whole point — `cuda` is opt-in for sanity check).
- Default subset: `--max-samples 100` per dataset (full eval = ~1000 inferences × 4 models which is hours on CPU; 100 is enough to rank).
- Outputs: `bench_results/<timestamp>/results.csv` + `summary.md`.

## RunPod execution recipe

We already have the canonical eval datasets cached on RunPod NFS. **Do not re-download.**

### Decision: baseline = production config

The faster-whisper baseline uses **`surindersinghssj/surt-small-v3`** (matches
`apps/transcribe/backend.py:27`), beam=5 (per user direction; production runs
beam=1 for live latency), VAD **off** (eval clips are pre-trimmed
caption-aligned speech, and VAD is faster-whisper-only — keeping it on would
give the baseline a unfair speed advantage no other backend has).

### A — RunPod (recommended for full 5-backend bench)

```bash
# One-time deps install (heavy: NeMo + transformers main):
pip install -U "transformers>=4.45" jiwer librosa soundfile psutil
pip install faster-whisper                   # baseline
pip install nemo_toolkit[asr]==2.0.0         # IndicConformer + Parakeet
pip install qwen_asr                         # Qwen3-ASR
# (MMS uses transformers — already installed)

# Critical env (per project memory):
export HF_HOME=/workspace/.cache/huggingface
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
export SURT_DOWNLOAD_WORKERS=8

cd /workspace/Gurbani-ASR-v4
git checkout codex/complete-phase-4-work
git pull
python scripts/bench_asr_alternatives.py \
    --device cpu \
    --threads 4 \
    --beam 5 \
    --max-samples 100
```

### B — Mac M5 Pro (subset only — NeMo doesn't run on macOS)

NeMo (IndicConformer, Parakeet) is broken on macOS; skip those backends and
use MPS for the transformers ones:

```bash
brew install ffmpeg     # for librosa / soundfile decode
python3.11 -m venv .venv && source .venv/bin/activate    # NOT 3.14
pip install -U "transformers>=4.45" jiwer librosa soundfile psutil \
    faster-whisper qwen_asr datasets

python scripts/bench_asr_alternatives.py \
    --device mps \
    --threads 8 \
    --beam 5 \
    --only baseline,mms,qwen3 \
    --max-samples 100
```

Notes for M5 Pro speed:
- `--device mps` routes MMS + Qwen3 to the M5 Pro GPU. faster-whisper baseline auto-falls-back to CPU+int8 (CT2 has no MPS support; int8 on Apple Silicon is already its fastest path).
- `--threads 8` — M5 Pro has 8+ performance cores; matching gives the best CPU saturation for CT2 + the env vars OMP/MKL pick up.
- For absolute max baseline speed on M5 Pro, also set `SURT_CT2_DIR=~/models/surt-small-v3-int8` if you've already converted the model locally (avoids re-converting from HF on each run).

### Subset run when a backend fails to install

```bash
python scripts/bench_asr_alternatives.py --only baseline,mms,qwen3 --device cpu --beam 5
```

## CPU optimization knobs (already wired in the script)

The script sets these BEFORE any heavy imports so PyTorch/MKL/OpenMP pick them
up at thread-pool init:

| Var | Value |
|---|---|
| `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS` | `--threads` (default 4) |
| `OMP_WAIT_POLICY` | `PASSIVE` (idle threads sleep instead of busy-wait) |
| `KMP_BLOCKTIME` | `0` (Intel MKL) |
| `torch.set_num_threads` / `set_num_interop_threads` | matches `--threads` |
| `torch.set_grad_enabled(False)` | skip grad bookkeeping |
| `torch.backends.quantized.engine` | `qnnpack` (ARM macOS) / `fbgemm` (x86) |
| faster-whisper `compute_type` | `int8` on CPU, `float16` on GPU |
| MMS `torch_dtype` | `float32` (faster than bf16/fp16 on CPU without AVX512-BF16) |

For more speed later, look at IPEX (Intel) or ONNX export — out of scope for v1 of this bench.

## Why I cannot run this from the local agent session

This Mac doesn't have the required environment:
- Python 3.14 (NeMo only supports 3.10/3.11)
- PEP 668 externally-managed pip → can't install deps
- macOS Darwin → NeMo native ops are notoriously broken
- ~10 GB of model weights + ~1 GB of dataset cache to download
- ~5 backends × ~1000 clips × CPU = many hours of wall-clock

You need to run the command above on the RunPod where:
- HF dataset cache for the canonical evals is already on NFS (per your reminder)
- NeMo / qwen_asr deps are installable cleanly on Linux + Python 3.10
- Wall-clock for `--max-samples 100` is reasonable (~30-60 min on CPU; ~5 min on GPU)

When you paste back the contents of `bench_results/<timestamp>/summary.md`, I'll
interpret which model deserves fine-tune budget per the decision rule below.

## Acceptance

- Script runs end-to-end on at least 1 backend without unhandled exceptions.
- `summary.md` contains the 4×2 results table.
- We can answer: "should I fine-tune Qwen3-ASR on v3 data, or stick with Whisper-small architecture?" — based on (a) IndicConformer's out-of-box WER as the realistic Punjabi-native floor, (b) Qwen3 / Parakeet speed numbers vs faster-whisper baseline.

## Out of scope (explicit non-goals)

- Fine-tuning any of the 3 candidates. That's the *next* decision, gated on this bench.
- Beating Surt-v2 baseline accuracy. We expect Surt-v2 to win out-of-box on both evals — it was fine-tuned on related data.
- Streaming infrastructure changes. Speed numbers here just inform whether streaming is even worth migrating to a new model family.

## Decisions to make AFTER this bench

1. If `indicconformer` (or `mms`) WER < 70% on kirtan eval out-of-box → fine-tune it on v3 (cheap win, no tokenizer surgery, native Gurmukhi).
2. If `qwen3` RTF on CPU < 0.5× baseline AND Hindi WER on Punjabi < 90% → strong fine-tune candidate; budget 1-2 days GPU.
3. If `parakeet` RTF on CPU < 0.2× baseline → worth the tokenizer-extension work for Gurmukhi later.
4. If none of the above → stay on Whisper-small architecture, focus effort on data scale (the actual v3 plan).
