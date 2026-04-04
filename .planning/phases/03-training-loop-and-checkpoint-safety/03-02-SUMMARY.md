---
phase: 03-training-loop-and-checkpoint-safety
plan: 02
subsystem: training
tags: [huggingface-hub, checkpoint, callback, auto-resume, whisper]

# Dependency graph
requires:
  - phase: 03-training-loop-and-checkpoint-safety plan 01
    provides: SurtTrainer, make_compute_metrics, build_training_args
  - phase: 02-data-pipeline-and-model-initialization
    provides: DataCollatorSpeechSeq2SeqWithPadding, get_train_dataset, get_val_dataset, load_model_and_processor
provides:
  - HubPushCallback with WER-gated + periodic Hub push and best_wer.json persistence
  - main() entry point with auto-resume from latest checkpoint
  - Complete runnable training pipeline via python -m surt.train
affects: [04-smoke-test-and-full-run]

# Tech tracking
tech-stack:
  added: [huggingface_hub.HfApi, transformers.TrainerCallback, transformers.trainer_utils.get_last_checkpoint]
  patterns: [deferred-imports-in-main, best-wer-json-persistence, wer-gated-hub-push]

key-files:
  created: []
  modified: [surt/train.py]

key-decisions:
  - "Deferred surt.data and surt.model imports into main() to avoid import-time audiomentations dependency -- keeps module importable for testing"
  - "Synchronous Hub push (no run_as_future) -- eval happens every 300 steps so blocking time is acceptable for v1"
  - "processing_class=processor.feature_extractor used instead of deprecated tokenizer= parameter (transformers v5)"

patterns-established:
  - "Deferred heavy imports: surt.data/surt.model imported inside main() not at module level"
  - "best_wer.json persistence: JSON file in OUTPUT_DIR tracks best WER across resume cycles"

requirements-completed: [CKPT-02, CKPT-03]

# Metrics
duration: 3min
completed: 2026-04-04
---

# Phase 3 Plan 02: Hub Push Callback and Auto-Resume Summary

**HubPushCallback with WER-gated + periodic safety push to Hub, best_wer.json persistence, and main() entry point with auto-resume from latest checkpoint**

## Performance

- **Duration:** 3 min
- **Started:** 2026-04-04T15:12:36Z
- **Completed:** 2026-04-04T15:15:41Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments
- HubPushCallback pushes to TRAINING_HUB_REPO on WER improvement ("best") and every 3rd eval ("periodic")
- best_wer.json persists best WER to disk so metric survives across spot instance resume cycles
- main() auto-detects checkpoints in OUTPUT_DIR via get_last_checkpoint and resumes without flags
- Complete training pipeline runnable via `python -m surt.train` or `python surt/train.py`

## Task Commits

Each task was committed atomically:

1. **Task 1: Add HubPushCallback with best_wer.json persistence** - `1503e16` (feat)
2. **Task 2: Add main() entry point with auto-resume and trainer wiring** - `5b8436f` (feat)

## Files Created/Modified
- `surt/train.py` - Added HubPushCallback class (WER-gated + periodic Hub push, best_wer.json persistence), main() entry point with auto-resume, and __main__ block

## Decisions Made
- Deferred surt.data and surt.model imports into main() to avoid import-time dependency on audiomentations (not installed locally, only on RunPod). This keeps `import surt.train` lightweight for testing.
- Used `processing_class=processor.feature_extractor` instead of deprecated `tokenizer=` parameter for transformers v5 compatibility.
- Synchronous Hub push (no `run_as_future=True`) -- eval happens every 300 steps so blocking time is acceptable for v1 simplicity.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Deferred surt.data/surt.model imports to main()**
- **Found during:** Task 1 (HubPushCallback implementation)
- **Issue:** Module-level `from surt.data import ...` caused ImportError due to audiomentations not being installed locally. This blocked import verification.
- **Fix:** Moved surt.data and surt.model imports inside main() with a comment explaining the deferral. The classes in train.py (HubPushCallback, SurtTrainer, etc.) don't need these at import time.
- **Files modified:** surt/train.py
- **Verification:** `python3 -c "from surt.train import HubPushCallback, SurtTrainer, main"` succeeds
- **Committed in:** 1503e16 (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Import deferral is a standard Python pattern for training scripts. No scope creep -- the deferred imports are functionally identical at runtime.

## Issues Encountered
None beyond the import deferral documented above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 3 is now complete: all training loop components (SurtTrainer, discriminative LR, compute_metrics, training args, HubPushCallback, auto-resume, main entry point) are implemented in surt/train.py
- Phase 4 (Smoke Test and Full Run) can proceed -- the training pipeline is fully wired and runnable via `python -m surt.train`
- RunPod pod needs to be started and SSH connection established before Phase 4 execution

## Self-Check: PASSED

- [x] surt/train.py exists
- [x] 03-02-SUMMARY.md exists
- [x] Commit 1503e16 found in git log
- [x] Commit 5b8436f found in git log

---
*Phase: 03-training-loop-and-checkpoint-safety*
*Completed: 2026-04-04*
