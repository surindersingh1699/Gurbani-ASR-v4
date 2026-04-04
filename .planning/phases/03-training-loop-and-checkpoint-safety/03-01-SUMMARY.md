---
phase: 03-training-loop-and-checkpoint-safety
plan: 01
subsystem: training
tags: [whisper, seq2seq-trainer, discriminative-lr, wer, jiwer, adamw, cosine-scheduler]

# Dependency graph
requires:
  - phase: 02-data-pipeline-and-model-initialization
    provides: "load_model_and_processor(), get_train_dataset(), get_val_dataset(), DataCollatorSpeechSeq2SeqWithPadding, config constants"
provides:
  - "SurtTrainer with discriminative LR (encoder=5e-5, decoder=1e-4, proj_out=1e-4)"
  - "make_compute_metrics factory for WER via jiwer"
  - "build_training_args with all Seq2SeqTrainingArguments"
  - "Training constants: MAX_STEPS, LEARNING_RATE, ENCODER_LR, DECODER_LR, WEIGHT_DECAY, TRAINING_HUB_REPO"
affects: [03-02, phase-04]

# Tech tracking
tech-stack:
  added: [jiwer]
  patterns: [discriminative-lr-via-trainer-subclass, id-param-deduplication-for-tied-weights]

key-files:
  created: [surt/train.py]
  modified: [surt/config.py]

key-decisions:
  - "proj_out checked before decoder in parameter name matching to avoid misclassification"
  - "id(param) deduplication handles tied proj_out.weight / embed_tokens.weight"
  - "jiwer direct (not evaluate.load) for WER computation simplicity"

patterns-established:
  - "Trainer subclass pattern: override create_optimizer() for custom parameter groups"
  - "Factory function pattern: make_compute_metrics(processor) returns closure with injected processor"
  - "Builder function pattern: build_training_args() centralizes all training configuration"

requirements-completed: [TRAIN-01, TRAIN-02, TRAIN-03, TRAIN-04, TRAIN-05, TRAIN-06, TRAIN-07, TRAIN-08, CKPT-01]

# Metrics
duration: 3min
completed: 2026-04-04
---

# Phase 3 Plan 1: Training Loop Core Summary

**SurtTrainer with three-group discriminative LR (encoder 5e-5, decoder 1e-4), WER compute_metrics via jiwer, and full Seq2SeqTrainingArguments configuration**

## Performance

- **Duration:** 3 min
- **Started:** 2026-04-04T15:03:23Z
- **Completed:** 2026-04-04T15:06:25Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Training constants added to config.py: MAX_STEPS=5000, discriminative LRs, TRAINING_HUB_REPO
- SurtTrainer subclass with create_optimizer override implementing three AdamW parameter groups with id(param) deduplication for tied weights
- make_compute_metrics factory producing WER computation via jiwer with -100 padding replacement
- build_training_args returning fully configured Seq2SeqTrainingArguments (bf16, gradient checkpointing, cosine scheduler, max_steps=5000, eval/save every 300 steps)

## Task Commits

Each task was committed atomically:

1. **Task 1: Add training constants to config.py** - `3420b05` (feat)
2. **Task 2: Create surt/train.py with SurtTrainer, compute_metrics, and training arguments** - `ed9b099` (feat)

## Files Created/Modified
- `surt/config.py` - Added MAX_STEPS, LEARNING_RATE, ENCODER_LR, DECODER_LR, WEIGHT_DECAY, TRAINING_HUB_REPO constants
- `surt/train.py` - New file (183 lines): SurtTrainer class, make_compute_metrics factory, build_training_args builder

## Decisions Made
- proj_out parameter name checked before decoder in the matching order, since proj_out does not contain "decoder" in its name and should not fall through to the decoder group
- id(param) set deduplication used to handle the tied weight between proj_out.weight and model.decoder.embed_tokens.weight, preventing double gradient updates
- jiwer used directly (not evaluate.load("wer")) for simplicity and to avoid the evaluate library dependency
- All transformers v5 API conventions used: eval_strategy (not evaluation_strategy), processing_class (not tokenizer)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Installed missing jiwer dependency**
- **Found during:** Task 2 (train.py verification)
- **Issue:** jiwer package not installed locally, causing ImportError during verification
- **Fix:** Ran `pip3 install jiwer` (jiwer 4.0.0 installed)
- **Files modified:** None (local pip install, not committed)
- **Verification:** Import succeeds, all assertions pass
- **Committed in:** Not applicable (runtime dependency, not source code)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** jiwer is a required runtime dependency already specified in research. Local install needed for verification only. No scope creep.

## Issues Encountered
None beyond the jiwer install.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- SurtTrainer, make_compute_metrics, and build_training_args are ready for Plan 02
- Plan 02 will add HubPushCallback, auto-resume via get_last_checkpoint, and the main() entry point
- All imports from surt.config and surt.train verified working

## Self-Check: PASSED

- [x] surt/config.py exists with new constants
- [x] surt/train.py exists (183 lines)
- [x] 03-01-SUMMARY.md exists
- [x] Commit 3420b05 found (Task 1)
- [x] Commit ed9b099 found (Task 2)

---
*Phase: 03-training-loop-and-checkpoint-safety*
*Completed: 2026-04-04*
