# Project Research Summary

**Project:** Gurbani ASR v4 (Surt — Gurmukhi Speech Recognition)
**Domain:** Low-resource Indic ASR — Whisper Small fine-tuning on Gurmukhi audio
**Researched:** 2026-04-04
**Confidence:** HIGH

## Executive Summary

This project is a production-quality Automatic Speech Recognition system for Gurbani (Sikh sacred scripture) using Gurmukhi script. The core challenge is a severe low-resource problem: Punjabi constituted only ~0.06% of Whisper's pretraining data, meaning the model has almost no prior exposure to Gurmukhi phonetics or script. Expert practice for this domain calls for full parameter fine-tuning (not LoRA/adapters), discriminative learning rates that update the decoder faster than the encoder, waveform-level audio augmentation to bridge the gap between clean sehaj path recordings and noisy kirtan audio, and aggressive checkpoint safety mechanisms given the ephemeral nature of RunPod GPU sessions. The entire pipeline is built around HuggingFace Transformers v5 tooling (`Seq2SeqTrainer`, streaming datasets, `HfApi` callbacks) running on a single A40 GPU via SSH.

The recommended approach is a modular Python training pipeline (7 files: config, data, collator, model, callbacks, metrics, train) that streams the 100h sehaj path FLAC dataset directly from HuggingFace Hub with no disk footprint, applies audiomentations augmentation inline, and pushes best checkpoints to HuggingFace Hub after every evaluation cycle. This design is explicitly optimized for unattended training under Claude Code SSH management, where session interruptions are expected and checkpoint recovery is a first-class concern. The pipeline is designed as Phase 1 of a 5-phase roadmap: baseline fine-tune → forced alignment of 700h kirtan → curriculum training on 800h → model distillation → production inference system.

The primary risks are all concentrated in Phase 1 data preprocessing and training configuration: sampling rate mismatches that silently corrupt the entire training run, wrong generation config tokens that teach the model the wrong language, streaming dataset incompatibility with epoch-based training, and checkpoint resume that restores model weights but not optimizer state (causing learning rate spikes mid-training). All 8 critical pitfalls identified are Phase 1 concerns with HIGH recovery costs (some requiring full retraining from scratch). The mitigation strategy is a mandatory pre-flight checklist and a 10-step smoke test before committing to the full 12-18 hour training run.

## Key Findings

### Recommended Stack

The stack is fully resolved at specific version numbers. The core is PyTorch 2.11.0 (CUDA 12.8) + transformers 5.5.0 + datasets 4.8.4 + accelerate 1.13.0, running on RunPod's `runpod/pytorch:2.11.0-py3.11-cuda12.8.1-devel-ubuntu22.04` template. The A40 GPU is Ampere architecture, so `bf16=True` is strongly preferred over `fp16=True` — bf16 has larger dynamic range, no loss scaling needed, and is slightly faster on Ampere silicon. Audio augmentation uses audiomentations 0.43.1 (AddGaussianNoise, RoomSimulator, TimeStretch, PitchShift — all confirmed in current API). Experiment tracking uses trackio 0.20.2, HuggingFace's new local-first tracker that replaces wandb as the default in transformers v5. Full version details and install ordering in `STACK.md`.

One critical transformers v5 API change from the existing `surt_training_plan.md` (which was written against v4.x): `evaluation_strategy` is renamed to `eval_strategy` in v5. This will cause a silent failure or error if not updated.

**Core technologies:**
- **PyTorch 2.11.0 + CUDA 12.8**: Deep learning framework — only framework Whisper training supports; A40 Ampere fully supported
- **transformers 5.5.0**: Whisper model + Seq2SeqTrainer — canonical HuggingFace library for Whisper fine-tuning
- **datasets 4.8.4**: Streaming audio from HuggingFace Hub — avoids 9GB FLAC download to ephemeral RunPod disk
- **accelerate 1.13.0**: Mixed precision + device management — required by transformers.Trainer
- **audiomentations 0.43.1**: Waveform augmentation — bridges clean sehaj path to noisy kirtan target domain
- **huggingface_hub 1.9.0**: Checkpoint push to Hub — critical safety net for ephemeral RunPod sessions
- **jiwer 4.0.0**: WER computation — direct use preferred over `evaluate` library overhead
- **soundfile 0.13.1**: Audio I/O backend — must be explicitly installed on RunPod, not always in base images

**What NOT to use:** LoRA/PEFT (insufficient for 0.06% language coverage), DeepSpeed/FSDP (single-GPU, unnecessary), openai/whisper package (not fine-tunable), faster-whisper (inference-only), librosa (heavy dep, datasets handles resampling natively), wandb (requires account/network; trackio is local-first and the new default).

### Expected Features

The feature research distinguishes sharply between table stakes (training crashes or loses money without these) and differentiators (Gurmukhi-specific quality advantages). All 12 must-have features are P1 and must be in the first training run.

**Must have (table stakes):**
- HuggingFace dataset streaming with `cast_column("audio", Audio(sampling_rate=16000))` — avoids 9GB disk usage, handles resampling
- `Seq2SeqTrainer` with `predict_with_generate=True` — required for autoregressive generation during eval
- WER compute_metrics via jiwer — the only meaningful ASR metric
- Gradient checkpointing + bf16 — fits training in A40 VRAM at reasonable batch sizes
- Checkpoint save every 300 steps + HuggingFace Hub push via custom callback — zero lost work guarantee on ephemeral RunPod
- Training resume from checkpoint — handles RunPod interruptions cleanly
- `language="pa"`, `task="transcribe"` on both processor AND `model.generation_config` — prevents English hallucinations
- Label masking with -100 for pad tokens — prevents model from learning trivial padding prediction
- Warmup steps (400) — Whisper fine-tuning diverges without warmup
- GPU auto-detection for batch size — handles different RunPod GPU allocations

**Should have (Gurmukhi-specific differentiators):**
- Discriminative learning rates (encoder 5e-5, decoder 1e-4) — encoder retains acoustic knowledge; decoder learns Gurmukhi nearly from scratch; critical for low-resource language quality
- Audio augmentation (noise, reverb, time stretch, pitch shift) — bridges clean studio sehaj path to real-world noisy kirtan target domain
- Mool Mantar as `initial_prompt` during eval/inference — free accuracy gain, anchors decoder to Gurmukhi vocabulary
- Full fine-tuning (not LoRA) — 0.06% Punjabi pretraining requires deep weight adaptation that low-rank updates cannot achieve
- Fixed 300-example validation subset (not streamed) — reproducible WER tracking
- Gurmukhi-specific text normalization for WER — handles diacritics and Unicode variants

**Defer (v2+):**
- Curriculum data loading (gold before silver) — Phase 3 specific
- Confidence-based data filtering — Phase 2/3 for auto-aligned kirtan
- Distillation trainer — Phase 4 specific
- ONNX export + INT8 quantization — Phase 5 specific
- Multi-GPU / FSDP — Phase 3 only if 800h dataset requires it (unlikely on A40)
- Formal evaluation harness with automated exit criteria — Phase 3 go/no-go decisions

### Architecture Approach

The architecture is a 7-module Python pipeline with clean separation of concerns. `config.py` is the single source of truth for all hyperparameters, paths, and GPU-adaptive batch sizing. `data.py` owns all HuggingFace Datasets and audiomentations logic. `collator.py` handles variable-length label padding with -100 masking. `model.py` loads the model, configures generation settings, and constructs the discriminative LR optimizer (these three concerns are intentionally co-located because they are tightly coupled). `callbacks.py` implements the HuggingFace Hub push safety mechanism. `metrics.py` is a standalone WER computation function. `train.py` is a thin orchestration entrypoint. The build order is strict: config → data + model + metrics → collator + callbacks → train. Each layer must be verified before moving to the next.

The key architectural decision is using `streaming=True` for the training dataset (IterableDataset, no disk footprint) while loading a fixed 300-example validation set eagerly into memory for reproducible WER tracking. This asymmetry is intentional and correct. The streaming mode requires `max_steps` instead of `num_train_epochs` and `dataloader_num_workers=0`.

**Major components:**
1. **Config Module** — Central hyperparameters, GPU detection, HF token loading; single source of truth, read-only at runtime
2. **Data Pipeline** — Streaming audio from HF Hub, 16kHz resampling, log-Mel feature extraction, Gurmukhi tokenization, waveform augmentation (train only)
3. **Training Loop (Seq2SeqTrainer)** — Orchestrates forward/backward, gradient accumulation, bf16, gradient checkpointing, eval, checkpointing
4. **Checkpoint Safety Callback** — Fires on every eval, pushes best checkpoint folder to HuggingFace Hub via `HfApi.upload_folder`; at most 300 steps of work lost on session death
5. **Resume Logic** — Checks local checkpoints first, then HuggingFace Hub; handles full resume including optimizer state from local checkpoint
6. **WER Evaluation** — Fixed validation subset, `predict_with_generate`, jiwer WER computation with special token stripping

### Critical Pitfalls

All 8 critical pitfalls are Phase 1 concerns. The 5 highest-priority based on recovery cost:

1. **Sampling rate mismatch (recovery: full retrain)** — `cast_column("audio", Audio(sampling_rate=16000))` must be called immediately after dataset load; add `assert audio["sampling_rate"] == 16000` assertion in preprocessing
2. **Wrong generation config language/task (recovery: full retrain)** — Set `model.generation_config.language = "pa"`, `task = "transcribe"`, `forced_decoder_ids = None`; verify by generating one sample before training starts
3. **Checkpoint resume without optimizer state (recovery: MEDIUM, steps wasted)** — Custom AdamW optimizer must NOT be re-created before `trainer.train(resume_from_checkpoint=...)`; let Trainer load optimizer from checkpoint; push full checkpoint directory (including `optimizer.pt`) not just model weights
4. **Streaming dataset with `num_train_epochs` (recovery: LOW, fix config and restart)** — Use `max_steps` instead; set `dataloader_num_workers=0`; for 100h at ~30s avg (~12,000 examples), 3 epochs at effective batch 32 = ~1,125 steps
5. **Whisper hallucinations on silence/non-speech (recovery: MEDIUM)** — Raise `no_speech_threshold` to 0.7-0.8; post-filter outputs shorter than 5 Gurmukhi characters or containing Latin script; VAD pre-filter is a Phase 5 concern but generation config adjustments are Phase 1

Additional Phase 1 checklist items: label padding not masked with -100 (causes pad-token learning), double BOS token in label sequences (causes constant WER offset), label truncation from Gurmukhi tokenizer starvation (Whisper BPE tokenizer learned very few Gurmukhi subword units, set `generation_max_length=448`).

## Implications for Roadmap

The project training plan already defines 5 phases at a high level. Research confirms and refines that structure. The roadmap should map directly onto these phases, with Phase 1 receiving the most detailed task breakdown given the density of critical pitfalls.

### Phase 1: Environment + Training Pipeline Foundation
**Rationale:** All 8 critical pitfalls are Phase 1 data and config concerns. The model is only as good as the data pipeline and training configuration. A single mistake here requires a full retrain ($18-27 compute cost, 12-18 hours). This phase must be verified with a 10-step smoke test before committing to a real training run.
**Delivers:** Verified data pipeline, correct model initialization, checkpoint safety mechanism, and a full training run producing `surt_small_v1` on HuggingFace Hub with WER < 15% on sehaj path
**Addresses:** All 12 table-stakes features + discriminative LR + augmentation + Mool Mantar prompt
**Avoids:** Sampling rate mismatch, wrong generation tokens, streaming/epoch incompatibility, pad-token masking errors, BOS doubling, tokenizer truncation, checkpoint resume corruption
**Stack:** Full stack — transformers 5.5.0, datasets 4.8.4, audiomentations 0.43.1, jiwer 4.0.0, huggingface_hub 1.9.0
**Build order:** config.py → data.py → model.py → collator.py + metrics.py → callbacks.py → train.py (10-step test) → full run

### Phase 2: Forced Alignment Pipeline (700h Kirtan)
**Rationale:** `surt_small_v1` must exist and meet exit criteria (WER < 15% sehaj path, WER < 35% kirtan, no English hallucinations) before forced alignment is viable. A poor base model produces unusable alignment confidence scores.
**Delivers:** ~700h of weakly-labeled kirtan audio with alignment confidence scores; ready for curriculum training input
**Uses:** `surt_small_v1` as alignment model; WhisperX or ctc-forced-aligner for forced alignment
**Implements:** Confidence-based data filtering pipeline; kirtan-specific audio chunking
**Note:** This phase has the least well-documented patterns for Gurmukhi-specific alignment. Needs `/gsd:research-phase` during planning.

### Phase 3: Curriculum Training on 800h
**Rationale:** Gold sehaj path data (100h) must be presented before silver kirtan data (700h) to prevent the model from learning noise patterns before Gurmukhi fundamentals.
**Delivers:** `surt_small_v2` trained on full 800h corpus with curriculum ordering
**Implements:** `concatenate_datasets` with gold-first ordering; confidence-based filtering for silver data; enhanced checkpoint resume with optimizer state to HuggingFace Hub (Phase 3 runs are 36-42h, spanning multiple RunPod sessions)
**Avoids:** Curriculum loading from wrong checkpoint (must start from `surt_small_v1`, not `openai/whisper-small`)
**Note:** Multi-session training introduces the optimizer-state resume problem at scale. Enhanced checkpoint logic needed.

### Phase 4: Model Distillation
**Rationale:** `surt_small_v2` (244M params) is too large for real-time inference. Distillation produces a smaller model with competitive WER.
**Delivers:** Distilled student model (~80M params) with WER within 5% of teacher
**Implements:** Custom distillation trainer (teacher-student cross-entropy + intermediate layer projection); projection layer must map 768→384 dimensions exactly
**Avoids:** Distillation projection dimension mismatch (crashes on first batch if wrong); distilling from wrong teacher checkpoint
**Note:** Distillation trainer patterns are well-documented via Distil-Whisper codebase. May not need deep research phase.

### Phase 5: Production Inference System
**Rationale:** Distilled model ready; focus shifts from training to inference optimization and FAISS retrieval integration.
**Delivers:** Real-time Gurbani transcription + shabad identification pipeline
**Implements:** VAD pre-filter (silero-vad or webrtcvad) to prevent hallucinations on silence; ONNX export + INT8 quantization (faster-whisper); FAISS retrieval with hallucination detection (Latin script filter, short-output filter)
**Avoids:** Hallucinations on silence/tabla/harmonium segments reaching the retrieval pipeline

### Phase Ordering Rationale

- Phase 1 must come first because all downstream phases depend on a working fine-tuned model; the critical pitfalls here have the highest recovery cost
- Phase 2 before Phase 3 because forced alignment requires a quality base model and produces the silver data that Phase 3 trains on
- Phase 3 before Phase 4 because distillation requires a high-quality teacher model trained on the full corpus
- Phase 5 last because it depends on having a compact distilled model suitable for real-time inference
- The build order within Phase 1 (config → data → model → collator → callbacks → train) is dictated by strict dependency ordering discovered in architecture research

### Research Flags

Phases likely needing `/gsd:research-phase` during planning:
- **Phase 2 (Forced Alignment):** Gurmukhi-specific forced alignment tooling (WhisperX vs ctc-forced-aligner), confidence thresholding for low-resource language, kirtan audio chunking for variable-length recordings with music segments — sparse documentation for this specific domain
- **Phase 5 (Inference Pipeline):** FAISS retrieval index construction from Gurbani corpus, VAD tuning for kirtan audio (music vs speech discrimination), real-time latency requirements and chunking strategy

Phases with standard patterns (skip research-phase or minimal research):
- **Phase 1 (Training Pipeline):** All patterns are fully documented via HuggingFace official blog and transformers/datasets docs. Stack is verified at specific version numbers. Architecture is a standard fine-tuning pipeline. Research is complete.
- **Phase 4 (Distillation):** Distil-Whisper codebase provides a well-documented reference implementation. The projection dimension and training loop patterns are standard. Minor research into dimension specifics may be needed.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All versions verified via PyPI JSON API (2026-04-04). Core libraries verified via Context7 documentation. transformers v5 API changes from v4 explicitly enumerated. No significant uncertainty. |
| Features | HIGH | Features verified against HuggingFace transformers v4.56.2 + audiomentations docs via Context7. MVP definition is clear and unambiguous. Anti-features are well-reasoned with explicit trade-offs documented. |
| Architecture | HIGH | Module structure and data flow verified against HuggingFace Datasets streaming docs, transformers Trainer API, and official Whisper fine-tuning blog. Build order derived from actual dependency analysis. |
| Pitfalls | MEDIUM-HIGH | Phase 1 pitfalls are HIGH confidence (verified via HuggingFace official blog and Context7 docs). RunPod session behavior is MEDIUM confidence (training data + community patterns, not verified against current RunPod docs). Pitfalls for Phases 2-5 are MEDIUM confidence (domain-specific, limited documentation for Gurmukhi alignment). |

**Overall confidence:** HIGH

### Gaps to Address

- **Exact dataset size for `max_steps` calculation:** The training plan assumes ~36,000 examples for 100h at 10s average, but PITFALLS.md calculates ~12,000 examples at 30s average. These differ by 3x and produce very different `max_steps` values (3,375 vs 1,125 for effective batch 32). The actual average clip duration must be checked by inspecting the dataset before setting `max_steps`.
- **Kirtan test set for Phase 1 eval:** Research flags that evaluating only on sehaj path is a critical anti-pattern (model may overfit to clean audio), but Phase 1 doesn't yet have a labeled kirtan test set. Even 10-20 manually labeled kirtan clips would close this gap. This must be addressed during Phase 1 planning.
- **Optimizer state persistence to HuggingFace Hub:** The current PushBestToHub callback pattern only pushes model weights. For Phase 3 multi-session training (36-42h), full optimizer state must be persisted. The architecture for this is described in PITFALLS.md but not implemented in the Phase 1 callback design. This gap becomes critical in Phase 3 planning.
- **Gurmukhi text normalization implementation:** FEATURES.md flags this as a P2 feature that corrects misleading WER metrics (diacritics, Unicode variants). No normalization code exists yet. This should be added before Phase 1 exit criteria are evaluated formally.
- **trackio verification in transformers v5.5.0:** trackio 0.20.2 is flagged as MEDIUM confidence (relatively new library). If `report_to="trackio"` fails on RunPod, fallback is `report_to="none"` with console logging. Low risk, easy mitigation.

## Sources

### Primary (HIGH confidence)
- Context7 `/llmstxt/huggingface_co_transformers_v5_2_0_llms_txt` — Whisper model, Seq2SeqTrainer, Seq2SeqTrainingArguments, eval_strategy API, bf16 training, v5 breaking changes
- Context7 `/llmstxt/huggingface_co_datasets_main_en_llms_txt` — Audio feature, streaming mode, cast_column, IterableDataset patterns, dataloader_num_workers=0 requirement
- Context7 `/iver56/audiomentations` — AddGaussianNoise, RoomSimulator, TimeStretch, PitchShift API
- Context7 `/huggingface/huggingface_hub` — upload_folder, push_to_hub patterns, checkpoint management
- PyPI JSON API (2026-04-04) — Version numbers for all packages: transformers 5.5.0, datasets 4.8.4, accelerate 1.13.0, audiomentations 0.43.1, huggingface_hub 1.9.0, jiwer 4.0.0, soundfile 0.13.1, safetensors 0.7.0, torch 2.11.0, trackio 0.20.2
- HuggingFace Blog "Fine-Tune Whisper" (https://huggingface.co/blog/fine-tune-whisper) — Data collator pattern, BOS token deduplication, language token configuration, sampling rate handling

### Secondary (MEDIUM confidence)
- OpenAI Whisper GitHub Discussions — Hallucination behavior on low-resource languages, no_speech_threshold guidance, community fine-tuning experiences
- AI4Bharat IndicWhisper — Indic language ASR patterns, multi-language handling
- Distil-Whisper paper and codebase — Distillation training patterns for Phase 4 reference
- RunPod session behavior patterns — Ephemeral storage, SSH stability, tmux requirements (community patterns, not official docs)

### Tertiary (LOW confidence)
- trackio library (PyPI + transformers v5 references) — Listed as new default in transformers v5, but relatively young library; fallback to `report_to="none"` if issues arise

---
*Research completed: 2026-04-04*
*Ready for roadmap: yes*
