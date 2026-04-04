# Requirements: Surt Phase 1 -- Fine-tune Whisper Small

**Defined:** 2026-04-04
**Core Value:** Accurately transcribe Gurbani audio in Gurmukhi script -- the foundation for all downstream phases

## v1 Requirements

Requirements for Phase 1 training pipeline. Each maps to roadmap phases.

### Data Pipeline

- [x] **DATA-01**: Training dataset streams from HuggingFace Hub with no local disk copy required
- [x] **DATA-02**: Audio is resampled to 16kHz via `cast_column("audio", Audio(sampling_rate=16000))`
- [x] **DATA-03**: Waveform augmentation applied to training data only (Gaussian noise p=0.4, room simulation p=0.3, time stretch p=0.2, pitch shift p=0.2)
- [x] **DATA-04**: Fixed 300-example validation subset loaded eagerly into memory for reproducible WER tracking
- [x] **DATA-05**: Labels tokenized as Gurmukhi with pad tokens masked to -100

### Model Configuration

- [x] **MODEL-01**: Base model is `openai/whisper-small` with full fine-tuning (encoder + decoder, not LoRA)
- [x] **MODEL-02**: Language set to `punjabi` and task set to `transcribe` on both processor AND `model.generation_config` (Whisper tokenizer stores the full language name internally, not the ISO code `pa`)
- [x] **MODEL-03**: `forced_decoder_ids` set to `None` to prevent forced token generation
- [x] **MODEL-04**: Mool Mantar used as `initial_prompt` during generation/eval for Gurmukhi vocabulary anchoring
- [x] **MODEL-05**: `generation_max_length` set to 448 to handle Gurmukhi tokenizer expansion (3-5x longer sequences than English)

### Training

- [x] **TRAIN-01**: Discriminative learning rates -- encoder at 5e-5, decoder at 1e-4, proj_out at 1e-4
- [x] **TRAIN-02**: Gradient checkpointing enabled to fit in VRAM with reasonable batch sizes
- [x] **TRAIN-03**: bf16 mixed precision (preferred over fp16 on Ampere GPUs for better numerical stability)
- [x] **TRAIN-04**: 400 warmup steps to prevent early divergence
- [x] **TRAIN-05**: Training uses `max_steps` (not `num_train_epochs`) since streaming datasets have no length
- [x] **TRAIN-06**: `Seq2SeqTrainer` with `predict_with_generate=True` for autoregressive eval
- [x] **TRAIN-07**: WER computed via jiwer at each eval step (every 300 steps)
- [x] **TRAIN-08**: GPU auto-detection adjusts batch size and gradient accumulation automatically

### Checkpoint Safety

- [x] **CKPT-01**: Checkpoints saved locally every 300 steps with `save_total_limit=3`
- [x] **CKPT-02**: Best checkpoint pushed to HuggingFace Hub after every evaluation via custom callback
- [x] **CKPT-03**: Training can resume from last checkpoint if spot instance is preempted
- [ ] **CKPT-04**: Final trained model pushed to HuggingFace Hub as `surt_small_v1`

### Infrastructure

- [ ] **INFRA-01**: Entire pipeline runs as Python scripts via SSH on RunPod spot GPU (< $0.30/hr)
- [ ] **INFRA-02**: Training runs inside tmux for session persistence across SSH disconnects
- [ ] **INFRA-03**: HuggingFace token loaded from environment variable (never hardcoded)
- [ ] **INFRA-04**: All dependencies installable via single `pip install` command

### Smoke Test

- [ ] **TEST-01**: Pre-flight validation generates one sample before training starts -- verifies Gurmukhi output (not English)
- [ ] **TEST-02**: Pre-flight checks one batch for correct sample rate (16kHz), -100 padding, single BOS token
- [ ] **TEST-03**: 10-step mini training run verifies loss decreases and LR schedule is correct before committing to full run

## v2 Requirements

Deferred to future phases. Tracked but not in current roadmap.

### Evaluation

- **EVAL-01**: Gurmukhi text normalization for accurate WER (handle diacritics, Unicode variants)
- **EVAL-02**: Automated exit criteria checking (WER < 15% sehaj path, WER < 35% kirtan, retrieval hit@3 > 75%)
- **EVAL-03**: CER (Character Error Rate) alongside WER for Gurmukhi-specific accuracy measurement

### Enhanced Training

- **ETRAIN-01**: Learning rate scheduler (cosine/linear decay) after warmup
- **ETRAIN-02**: TensorBoard logging for loss curve visualization
- **ETRAIN-03**: Optimizer state persistence to HuggingFace Hub for multi-session resume

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| LoRA / QLoRA / Adapter fine-tuning | Whisper has 0.06% Punjabi exposure; low-rank updates cannot achieve the encoder adaptation needed |
| Multi-GPU / DeepSpeed / FSDP | Single GPU with gradient accumulation is sufficient for 100h dataset |
| Weights & Biases / MLflow | Single training run, no hyperparameter sweeps -- console logging is sufficient |
| SpecAugment / spectrogram augmentation | Fragile with Whisper's feature extractor; waveform augmentation via audiomentations is better studied |
| Custom tokenizer / extended vocabulary | Whisper's tokenizer already covers Gurmukhi Unicode; extending it destabilizes pretrained decoder weights |
| Data quality filtering pipeline | Sehaj path is human-labeled gold data; quality filtering is a Phase 2/3 concern for silver data |
| Colab notebook delivery | Spot RunPod GPU is the target platform; no notebook-only patterns |
| Phases 2-5 (alignment, curriculum, distillation, quantization) | Future milestones |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 | Phase 2 | Complete |
| DATA-02 | Phase 2 | Complete |
| DATA-03 | Phase 2 | Complete |
| DATA-04 | Phase 2 | Complete |
| DATA-05 | Phase 2 | Complete |
| MODEL-01 | Phase 2 | Complete |
| MODEL-02 | Phase 2 | Complete |
| MODEL-03 | Phase 2 | Complete |
| MODEL-04 | Phase 2 | Complete |
| MODEL-05 | Phase 2 | Complete |
| TRAIN-01 | Phase 3 | Complete |
| TRAIN-02 | Phase 3 | Complete |
| TRAIN-03 | Phase 3 | Complete |
| TRAIN-04 | Phase 3 | Complete |
| TRAIN-05 | Phase 3 | Complete |
| TRAIN-06 | Phase 3 | Complete |
| TRAIN-07 | Phase 3 | Complete |
| TRAIN-08 | Phase 3 | Complete |
| CKPT-01 | Phase 3 | Complete |
| CKPT-02 | Phase 3 | Complete |
| CKPT-03 | Phase 3 | Complete |
| CKPT-04 | Phase 4 | Pending |
| INFRA-01 | Phase 1 | Pending |
| INFRA-02 | Phase 1 | Pending |
| INFRA-03 | Phase 1 | Pending |
| INFRA-04 | Phase 1 | Pending |
| TEST-01 | Phase 4 | Pending |
| TEST-02 | Phase 4 | Pending |
| TEST-03 | Phase 4 | Pending |

**Coverage:**
- v1 requirements: 29 total
- Mapped to phases: 29
- Unmapped: 0

---
*Requirements defined: 2026-04-04*
*Last updated: 2026-04-04 after roadmap creation*
