# Roadmap: Surt Phase 1 -- Fine-tune Whisper Small on Gurbani

## Overview

This roadmap delivers `surt_small_v1` -- a Gurmukhi-specialized ASR model fine-tuned from OpenAI's Whisper Small on 100h of sehaj path audio. The work progresses from RunPod environment setup, through data pipeline and model configuration (the highest-risk area where all critical pitfalls concentrate), into training loop wiring with checkpoint safety for spot instance resilience, and culminates in a validated full training run that pushes the final model to HuggingFace Hub. Every phase builds on the previous one and can be independently verified before proceeding.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Project Setup and Infrastructure** - RunPod environment, project scaffolding, dependencies, and config module
- [x] **Phase 2: Data Pipeline and Model Initialization** - Streaming data with augmentation, model with correct Gurmukhi generation config
- [ ] **Phase 3: Training Loop and Checkpoint Safety** - Seq2SeqTrainer wiring, discriminative LR, Hub push callback, resume logic
- [ ] **Phase 4: Smoke Test and Full Training Run** - Pre-flight validation, 10-step smoke test, full run producing surt_small_v1

## Phase Details

### Phase 1: Project Setup and Infrastructure
**Goal**: A RunPod spot GPU environment is ready to develop and run the training pipeline, with all dependencies installed and project structure in place
**Depends on**: Nothing (first phase)
**Requirements**: INFRA-01, INFRA-02, INFRA-03, INFRA-04
**Success Criteria** (what must be TRUE):
  1. Running `python -c "import transformers, datasets, audiomentations, jiwer, huggingface_hub; print('ok')"` succeeds on RunPod
  2. HuggingFace token is loaded from environment variable and `huggingface_hub.HfApi().whoami()` returns the correct user
  3. A tmux session persists across SSH disconnect and reconnect
  4. config.py correctly detects the GPU model and sets batch size and gradient accumulation accordingly
**Plans**: 2 plans

Plans:
- [x] 01-01-PLAN.md — Create surt/ package with config.py, requirements.txt, and setup_pod.sh
- [x] 01-02-PLAN.md — Validate infrastructure on live RunPod pod (checkpoint)

### Phase 2: Data Pipeline and Model Initialization
**Goal**: The streaming data pipeline produces correctly preprocessed Gurmukhi training examples, and the model is initialized with the right language, task, and generation settings to prevent catastrophic misconfiguration
**Depends on**: Phase 1
**Requirements**: DATA-01, DATA-02, DATA-03, DATA-04, DATA-05, MODEL-01, MODEL-02, MODEL-03, MODEL-04, MODEL-05
**Success Criteria** (what must be TRUE):
  1. Training dataset streams from HuggingFace Hub without downloading to disk, and each example has audio resampled to 16kHz with correct log-Mel features
  2. Augmentation (noise, reverb, time stretch, pitch shift) is applied to training examples but not to validation examples
  3. A fixed 300-example validation subset is loaded into memory and produces identical examples across runs
  4. Labels are tokenized as Gurmukhi with pad tokens masked to -100 and no double BOS tokens
  5. `model.generate()` on a single audio sample produces Gurmukhi text (not English) with generation_max_length=448
**Plans**: 2 plans

Plans:
- [ ] 02-01-PLAN.md — Model and processor initialization with Gurmukhi language/task config
- [ ] 02-02-PLAN.md — Streaming data pipeline with augmentation, validation subset, and data collator

### Phase 3: Training Loop and Checkpoint Safety
**Goal**: The Seq2SeqTrainer is fully wired with discriminative learning rates, gradient checkpointing, bf16 precision, WER evaluation, and a checkpoint safety system that preserves progress to HuggingFace Hub every 300 steps
**Depends on**: Phase 2
**Requirements**: TRAIN-01, TRAIN-02, TRAIN-03, TRAIN-04, TRAIN-05, TRAIN-06, TRAIN-07, TRAIN-08, CKPT-01, CKPT-02, CKPT-03
**Success Criteria** (what must be TRUE):
  1. AdamW optimizer has three parameter groups with correct learning rates (encoder 5e-5, decoder 1e-4, proj_out 1e-4)
  2. Training uses max_steps (not num_train_epochs) with dataloader_num_workers=0 for streaming compatibility
  3. After an eval step, the best checkpoint folder appears on HuggingFace Hub with a commit message containing the step number and WER
  4. Simulating a restart and calling `trainer.train(resume_from_checkpoint=...)` resumes from the last local checkpoint without errors
  5. WER is computed via jiwer at each eval step (every 300 steps) using predict_with_generate
**Plans**: TBD

Plans:
- [ ] 03-01: TBD
- [ ] 03-02: TBD

### Phase 4: Smoke Test and Full Training Run
**Goal**: The pipeline is validated with pre-flight checks and a 10-step smoke test, then committed to a full training run that produces surt_small_v1 on HuggingFace Hub
**Depends on**: Phase 3
**Requirements**: TEST-01, TEST-02, TEST-03, CKPT-04
**Success Criteria** (what must be TRUE):
  1. Pre-flight generates one sample and confirms Gurmukhi output (no English hallucination)
  2. Pre-flight batch check confirms 16kHz sample rate, -100 padding on pad tokens, and single BOS token in labels
  3. A 10-step mini training run shows decreasing loss and correct LR schedule (warmup increasing from 0)
  4. The full training run completes and the final model is accessible at the surt_small_v1 HuggingFace repo
**Plans**: TBD

Plans:
- [ ] 04-01: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Project Setup and Infrastructure | 2/2 | Complete | 2026-04-04 |
| 2. Data Pipeline and Model Initialization | 0/2 | Not started | - |
| 3. Training Loop and Checkpoint Safety | 0/2 | Not started | - |
| 4. Smoke Test and Full Training Run | 0/1 | Not started | - |
