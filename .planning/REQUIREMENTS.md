# Requirements: Surt Phase 1 — Fine-tune Whisper Small

**Defined:** 2026-04-04
**Core Value:** Accurately transcribe Gurbani audio in Gurmukhi script — the foundation for all downstream phases

## v1 Requirements

Requirements for Phase 1 training pipeline. Each maps to roadmap phases.

### Data Pipeline

- [ ] **DATA-01**: Training dataset streams from HuggingFace Hub with no local disk copy required
- [ ] **DATA-02**: Audio is resampled to 16kHz via `cast_column("audio", Audio(sampling_rate=16000))`
- [ ] **DATA-03**: Waveform augmentation applied to training data only (Gaussian noise p=0.4, room simulation p=0.3, time stretch p=0.2, pitch shift p=0.2)
- [ ] **DATA-04**: Fixed 300-example validation subset loaded eagerly into memory for reproducible WER tracking
- [ ] **DATA-05**: Labels tokenized as Gurmukhi with pad tokens masked to -100

### Model Configuration

- [ ] **MODEL-01**: Base model is `openai/whisper-small` with full fine-tuning (encoder + decoder, not LoRA)
- [ ] **MODEL-02**: Language set to `pa` and task set to `transcribe` on both processor AND `model.generation_config`
- [ ] **MODEL-03**: `forced_decoder_ids` set to `None` to prevent forced token generation
- [ ] **MODEL-04**: Mool Mantar used as `initial_prompt` during generation/eval for Gurmukhi vocabulary anchoring
- [ ] **MODEL-05**: `generation_max_length` set to 448 to handle Gurmukhi tokenizer expansion (3-5x longer sequences than English)

### Training

- [ ] **TRAIN-01**: Discriminative learning rates — encoder at 5e-5, decoder at 1e-4, proj_out at 1e-4
- [ ] **TRAIN-02**: Gradient checkpointing enabled to fit in VRAM with reasonable batch sizes
- [ ] **TRAIN-03**: bf16 mixed precision (preferred over fp16 on Ampere GPUs for better numerical stability)
- [ ] **TRAIN-04**: 400 warmup steps to prevent early divergence
- [ ] **TRAIN-05**: Training uses `max_steps` (not `num_train_epochs`) since streaming datasets have no length
- [ ] **TRAIN-06**: `Seq2SeqTrainer` with `predict_with_generate=True` for autoregressive eval
- [ ] **TRAIN-07**: WER computed via jiwer at each eval step (every 300 steps)
- [ ] **TRAIN-08**: GPU auto-detection adjusts batch size and gradient accumulation automatically

### Checkpoint Safety

- [ ] **CKPT-01**: Checkpoints saved locally every 300 steps with `save_total_limit=3`
- [ ] **CKPT-02**: Best checkpoint pushed to HuggingFace Hub after every evaluation via custom callback
- [ ] **CKPT-03**: Training can resume from last checkpoint if spot instance is preempted
- [ ] **CKPT-04**: Final trained model pushed to HuggingFace Hub as `surt_small_v1`

### Infrastructure

- [ ] **INFRA-01**: Entire pipeline runs as Python scripts via SSH on RunPod spot GPU (< $0.30/hr)
- [ ] **INFRA-02**: Training runs inside tmux for session persistence across SSH disconnects
- [ ] **INFRA-03**: HuggingFace token loaded from environment variable (never hardcoded)
- [ ] **INFRA-04**: All dependencies installable via single `pip install` command

### Smoke Test

- [ ] **TEST-01**: Pre-flight validation generates one sample before training starts — verifies Gurmukhi output (not English)
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
| Weights & Biases / MLflow | Single training run, no hyperparameter sweeps — console logging is sufficient |
| SpecAugment / spectrogram augmentation | Fragile with Whisper's feature extractor; waveform augmentation via audiomentations is better studied |
| Custom tokenizer / extended vocabulary | Whisper's tokenizer already covers Gurmukhi Unicode; extending it destabilizes pretrained decoder weights |
| Data quality filtering pipeline | Sehaj path is human-labeled gold data; quality filtering is a Phase 2/3 concern for silver data |
| Colab notebook delivery | Spot RunPod GPU is the target platform; no notebook-only patterns |
| Phases 2-5 (alignment, curriculum, distillation, quantization) | Future milestones |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 | TBD | Pending |
| DATA-02 | TBD | Pending |
| DATA-03 | TBD | Pending |
| DATA-04 | TBD | Pending |
| DATA-05 | TBD | Pending |
| MODEL-01 | TBD | Pending |
| MODEL-02 | TBD | Pending |
| MODEL-03 | TBD | Pending |
| MODEL-04 | TBD | Pending |
| MODEL-05 | TBD | Pending |
| TRAIN-01 | TBD | Pending |
| TRAIN-02 | TBD | Pending |
| TRAIN-03 | TBD | Pending |
| TRAIN-04 | TBD | Pending |
| TRAIN-05 | TBD | Pending |
| TRAIN-06 | TBD | Pending |
| TRAIN-07 | TBD | Pending |
| TRAIN-08 | TBD | Pending |
| CKPT-01 | TBD | Pending |
| CKPT-02 | TBD | Pending |
| CKPT-03 | TBD | Pending |
| CKPT-04 | TBD | Pending |
| INFRA-01 | TBD | Pending |
| INFRA-02 | TBD | Pending |
| INFRA-03 | TBD | Pending |
| INFRA-04 | TBD | Pending |
| TEST-01 | TBD | Pending |
| TEST-02 | TBD | Pending |
| TEST-03 | TBD | Pending |

**Coverage:**
- v1 requirements: 29 total
- Mapped to phases: 0
- Unmapped: 29

---
*Requirements defined: 2026-04-04*
*Last updated: 2026-04-04 after initial definition*
