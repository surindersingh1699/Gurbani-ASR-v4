---
phase: 02-data-pipeline-and-model-initialization
plan: 02
subsystem: data
tags: [whisper, streaming, augmentation, audiomentations, huggingface, datasets, collator]

# Dependency graph
requires:
  - phase: 02-01
    provides: "WhisperProcessor and model via load_model_and_processor() for feature extraction and tokenization"
provides:
  - "Streaming training dataset with waveform augmentation (get_train_dataset)"
  - "Fixed-size validation dataset materialized in memory (get_val_dataset)"
  - "Data collator with -100 masking and double-BOS guard (DataCollatorSpeechSeq2SeqWithPadding)"
  - "HuggingFace rate limit retry with exponential backoff"
affects: [03-training-loop-and-checkpoints]

# Tech tracking
tech-stack:
  added: [audiomentations, datasets-streaming]
  patterns: [dependency-injection-processor, waveform-augmentation-before-mel, retry-with-backoff]

key-files:
  created: [surt/data.py]
  modified: [surt/config.py]

key-decisions:
  - "HF rate limit retry: 5 retries with exponential backoff (2s initial, 2x factor) on load_dataset"
  - "Augmentation on raw waveform before feature extraction (not on log-Mel spectrograms)"
  - "Processor passed as parameter (dependency injection) rather than imported at module level"
  - "Validation taken from train split (dataset has no separate val split)"
  - "Runtime verification deferred: RunPod SSH unavailable, AST verification passed"

patterns-established:
  - "Dependency injection: processor passed to all public functions, not imported at module level"
  - "Config as single source of truth: data.py imports constants from config.py, never redefines them"
  - "Retry pattern: _load_dataset_with_retry wraps load_dataset with exponential backoff for HF Hub"

requirements-completed: [DATA-01, DATA-02, DATA-03, DATA-04, DATA-05]

# Metrics
duration: 3min
completed: 2026-04-04
---

# Phase 2 Plan 02: Data Pipeline Summary

**Streaming Gurbani data pipeline with audiomentations waveform augmentation, -100 pad masking collator, and HF Hub retry backoff**

## Performance

- **Duration:** 3 min
- **Started:** 2026-04-04T13:23:10Z
- **Completed:** 2026-04-04T13:26:04Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Streaming training dataset from HuggingFace Hub with 16kHz resampling and waveform augmentation (noise p=0.4, reverb p=0.3, stretch p=0.2, pitch p=0.2)
- Fixed 300-example validation dataset materialized eagerly with no augmentation
- Data collator with pad token masking to -100 and double-BOS token stripping
- HuggingFace rate limit retry logic with exponential backoff (5 retries, 2s initial, 2x factor)
- Dataset constants centralized in surt/config.py as single source of truth

## Task Commits

Each task was committed atomically:

1. **Task 1: Create surt/data.py with streaming pipeline, augmentation, and data collator** - `9f6c5e3` (feat)
2. **Task 2: Add dataset configuration to surt/config.py and verify data pipeline** - `63ef162` (feat)

## Files Created/Modified
- `surt/data.py` - Streaming data pipeline with get_train_dataset, get_val_dataset, DataCollatorSpeechSeq2SeqWithPadding, and HF retry logic
- `surt/config.py` - Added DATASET_NAME, TRAIN_SPLIT, VAL_SPLIT, TEXT_COLUMN, VAL_SIZE, SHUFFLE_BUFFER constants

## Decisions Made
- **HF rate limit retry:** Added `_load_dataset_with_retry` with 5 retries and exponential backoff (2s initial, 2x factor) per user request. Retries on 429, 502, 503, connection, and timeout errors.
- **Dependency injection pattern:** Processor is passed as parameter to all public functions rather than imported at module level. This keeps surt/data.py decoupled from surt/model.py and makes testing easier.
- **Runtime verification deferred:** RunPod SSH authentication failed (pod likely restarted with new port). AST verification passed all structural checks. Runtime verification against actual dataset will be done when pod is available.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Added HF Hub retry with exponential backoff**
- **Found during:** Task 1 (data pipeline creation)
- **Issue:** User explicitly requested retry logic for HuggingFace rate limits on streaming dataset loading
- **Fix:** Created `_load_dataset_with_retry()` wrapper with 5 retries, 2s initial backoff, 2x exponential factor. Retries on 429, 502, 503, connection, and timeout errors.
- **Files modified:** surt/data.py
- **Verification:** AST check confirms function exists and backoff logic present
- **Committed in:** 9f6c5e3 (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 missing critical -- user-requested feature)
**Impact on plan:** Retry logic adds robustness for production HF Hub streaming. No scope creep.

## Issues Encountered
- RunPod SSH unavailable (authentication denied on port 22006). Pod likely restarted and SSH port changed. Runtime verification against actual Gurbani dataset deferred. All AST/structural verification passed.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- surt/data.py ready for Phase 3 training loop integration
- get_train_dataset() and get_val_dataset() accept processor from load_model_and_processor()
- DataCollatorSpeechSeq2SeqWithPadding ready to pass to Seq2SeqTrainer
- Runtime verification on RunPod recommended before starting Phase 3

## Self-Check: PASSED

- FOUND: surt/data.py
- FOUND: surt/config.py
- FOUND: 02-02-SUMMARY.md
- FOUND: commit 9f6c5e3 (Task 1)
- FOUND: commit 63ef162 (Task 2)

---
*Phase: 02-data-pipeline-and-model-initialization*
*Completed: 2026-04-04*
