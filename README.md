# Gurbani ASR v4 (Surt)

Automatic Speech Recognition for Gurbani — sehaj path and kirtan.

## Current Direction — Surt v3

**See [`PLAN.md`](PLAN.md) for the active plan.** Any agent or contributor starting work on this repo should read `PLAN.md` first.

**In one line:** build a 500-hour kirtan training dataset from YouTube (Ragi + AKJ + SGPC), label it with Gemini 2.5 Flash Lite via the pre-split batched pipeline in `scripts/kirtan_bulk_transcribe.py`, and retrain Surt.

## What You Should NOT Assume

Agents reading this codebase have historically latched onto older plans. Explicit disclaimers:

- The v1 plan ([`surt_training_plan.md`](surt_training_plan.md)) is **historical**. It only covered sehaj path.
- The v2 plan ([`surt_v2_training_plan.md`](surt_v2_training_plan.md)) is **historical**. Its HF kirtan dataset (~28h) is frozen — we are not adding to it or training against it any more.
- The approach of pulling kirtan from existing HF datasets is **superseded**. v3 builds its own dataset from YouTube end-to-end.
- Forced-alignment, OCR-from-subtitles, and Whisper-Large-transcription approaches are **abandoned** (see [memory/kirtan_data_approaches.md](../.claude/projects/-Users-surindersingh-Developer-Gurbani-ASR-v4/memory/kirtan_data_approaches.md)).

## Shipped Artifacts (reference only)

| Name | What it is |
|---|---|
| `surindersinghssj/surt-small-v1` | v1 model, sehaj path only, WER≈24 on sehaj / hallucinates on kirtan |
| `surindersinghssj/surt-small-v2` | v2 model, trained on sehaj + 28h kirtan v2, kirtan WER≈55% |
| `surindersinghssj/gurbani-sehajpath` | Sehaj path dataset, 63.1k samples, ~66h — **still used in v3 as regularizer** |
| `surindersinghssj/gurbani-kirtan-dataset-v2` | v2 kirtan dataset, ~28h — frozen, not extended |

## v3 Dataset (target, in progress)

- `surindersinghssj/gurbani-kirtan-v3-500h` (not yet pushed)
- Tagged by `kirtan_type`: `ragi | akj | sgpc`
- Labeled by Gemini 2.5 Flash Lite, 20s fixed clips

## Key Paths

- Plan: [`PLAN.md`](PLAN.md)
- Transcription pipeline: [`scripts/kirtan_bulk_transcribe.py`](scripts/kirtan_bulk_transcribe.py)
- Training code: [`surt/`](surt/)
- Eval: [`scripts/eval_kirtan.py`](scripts/eval_kirtan.py)

## Training env vars

| Var | Default | What it does |
| --- | --- | --- |
| `HF_HOME` | `/workspace/.cache/huggingface` | HF cache root — must be on the big volume, not the root overlay. |
| `SURT_MAP_WORKERS` | `os.cpu_count()` | CPU workers for `.map()` feature extraction in [`scripts/prepare_data.py`](scripts/prepare_data.py). |
| `SURT_DOWNLOAD_WORKERS` | `8` | **Parallel parquet-shard downloads** from HF Hub (`num_proc` on non-streaming `load_dataset`). HF caps single-stream at ~8 MB/s — without this the 18-shard sehaj dataset takes ~16 min; with 8 it takes ~2 min. Only affects `streaming=False` loads. See [`surt/data.py::_load_dataset_with_retry`](surt/data.py). |
| `SURT_VAL_SIZE` | `200` | Number of examples in the materialized val split. |
| `SURT_NO_CUDA_INIT` | unset | Set by [`scripts/prepare_data.py`](scripts/prepare_data.py) so fork-based multiprocessing is safe. Do not set in training. |
