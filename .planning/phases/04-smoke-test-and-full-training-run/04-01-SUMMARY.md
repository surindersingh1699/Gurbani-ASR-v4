---
phase: 04-smoke-test-and-full-training-run
plan: 01
subsystem: training
tags: [smoke-test, phase4, orchestration]
requires:
  - phase: 03-training-loop-and-checkpoint-safety
    provides: trainer wiring, callback, auto-resume baseline
provides:
  - pre-flight checks implementation
  - smoke run implementation
  - full run + final push hook implementation
affects:
  - .planning/ROADMAP.md
  - .planning/STATE.md
  - surt/train.py
  - surt/smoke_test.py
completed: 2026-04-04
---

# Phase 4 Plan 01 Summary

## Outcome

Phase 4 implementation is now present in code. Runtime validation on RunPod is still required to mark the milestone fully complete.

## What Was Implemented

- Added `surt/smoke_test.py` with:
  - `run_generation_preflight()` for TEST-01
  - `run_batch_preflight()` for TEST-02
  - `run_preflight_checks()` orchestrator
- Extended `surt/train.py` with:
  - `run_training_job()` shared training runner
  - `validate_smoke_training()` for TEST-03
  - `push_final_model_to_hub()` for CKPT-04
  - CLI modes: `full`, `smoke`, `phase4`
- Updated planning docs to reflect Phase 4 decisions in cross-phase state.

## Key Decisions (Cross-Phase)

1. Keep heavy imports deferred in `train.py` runtime functions to preserve the lightweight import behavior established in Phase 3.
2. Isolate pre-flight checks in a dedicated module so TEST-01/02 can be reused independently of full training.
3. Use a separate smoke output directory to avoid contaminating long-run checkpoint state.
4. Disable checkpoint Hub callback during smoke runs; smoke artifacts should not pollute the training checkpoint repo.
5. Push final artifacts explicitly to `HF_MODEL_REPO` at end of full run, keeping training and final repos separated as decided in prior phases.

## Verification Performed

- `python3 -m py_compile surt/train.py surt/smoke_test.py` (pass)

## Remaining Runtime Steps

1. Run `python -m surt.train --mode phase4` on RunPod.
2. Confirm TEST-01/02 logs pass in the session output.
3. Confirm TEST-03 logs show loss improvement and LR warmup increase.
4. Confirm final model is visible at `surindersingh/surt-small-v1`.
