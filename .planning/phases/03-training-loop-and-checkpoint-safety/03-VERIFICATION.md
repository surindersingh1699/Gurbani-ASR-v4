---
phase: 03-training-loop-and-checkpoint-safety
verified: 2026-04-04T16:00:00Z
status: passed
score: 9/9 must-haves verified
re_verification: false
---

# Phase 3: Training Loop and Checkpoint Safety Verification Report

**Phase Goal:** Implement the training loop (SurtTrainer with discriminative LRs), WER evaluation, Hub push callback, and auto-resume entry point.
**Verified:** 2026-04-04
**Status:** PASSED
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| #  | Truth | Status | Evidence |
|----|-------|--------|----------|
| 1  | SurtTrainer creates three AdamW parameter groups (encoder 5e-5, decoder 1e-4, proj_out 1e-4) with no duplicate or missing parameters | VERIFIED | `create_optimizer()` in train.py lines 144-192; `seen_ids = set()` deduplication confirmed; encoder/proj_out/decoder groups built in correct priority order |
| 2  | Seq2SeqTrainingArguments uses max_steps=5000, bf16=True, gradient_checkpointing with use_reentrant=False, cosine LR scheduler, eval every 300 steps | VERIFIED | `build_training_args()` verified by live assertion: `max_steps=5000`, `bf16=True`, `gradient_checkpointing_kwargs={'use_reentrant': False}`, `scheduler=SchedulerType.COSINE`, `eval_steps=300` |
| 3  | WER is computed via jiwer by decoding predictions and labels, replacing -100 with pad_token_id before decoding | VERIFIED | `make_compute_metrics()` lines 195-228; `-100` replacement confirmed; `jiwer.wer(label_str, pred_str)` × 100 returns `{"wer": wer}` |
| 4  | Training uses dataloader_num_workers=0 for streaming IterableDataset safety | VERIFIED | `build_training_args()` line 267: `dataloader_num_workers=0`; confirmed by live assertion |
| 5  | Checkpoints are saved every 300 steps with save_total_limit=3 | VERIFIED | `save_steps=SAVE_STEPS (300)`, `save_total_limit=SAVE_TOTAL_LIMIT (3)`; confirmed by live assertion |
| 6  | After an eval step, if WER improved or it is a periodic safety push, the model+processor+generation_config are uploaded to the training Hub repo | VERIFIED | `HubPushCallback.on_evaluate()` lines 108-129; WER-gated push on `is_best` + periodic push every 3rd eval via `is_periodic`; `upload_folder(repo_id=self.hub_repo, ...)` confirmed |
| 7  | best_wer.json is written to OUTPUT_DIR on every WER improvement and loaded on callback init, so best WER survives across resume cycles | VERIFIED | `_save_best_wer()` writes `{"best_wer": wer, "step": step}` to `self.output_dir / "best_wer.json"`; `_load_best_wer()` called in `__init__`; key link pattern `best_wer\.json` confirmed present |
| 8  | Running `python -m surt.train` when OUTPUT_DIR contains checkpoints automatically resumes from the latest checkpoint without any flag | VERIFIED | `main()` line 315: `last_ckpt = get_last_checkpoint(OUTPUT_DIR)`; line 364: `trainer.train(resume_from_checkpoint=last_ckpt)` |
| 9  | Running `python -m surt.train` when OUTPUT_DIR is empty starts fresh training | VERIFIED | `if last_ckpt:` branch falls through to `print("[train] Starting fresh training run")`; `last_ckpt=None` passed to `trainer.train()`, which starts fresh |

**Score:** 9/9 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `surt/config.py` | MAX_STEPS, LR constants, TRAINING_HUB_REPO | VERIFIED | `MAX_STEPS=5000`, `ENCODER_LR=5e-5`, `DECODER_LR=1e-4`, `LEARNING_RATE=1e-4`, `WEIGHT_DECAY=0.01`, `TRAINING_HUB_REPO='surindersinghssj/surt-small-v1-training'` — all confirmed by live import |
| `surt/train.py` | SurtTrainer, compute_metrics, training_args configuration | VERIFIED | 369 lines (well above 80-line minimum); `SurtTrainer`, `make_compute_metrics`, `build_training_args`, `HubPushCallback`, `main()` all present and importable |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `surt/train.py` | `surt/config.py` | `from surt.config import ... MAX_STEPS` | VERIFIED | Pattern `from surt\.config import.*MAX_STEPS` matches lines 24-40 (multi-constant import block) |
| `surt/train.py` | `surt/model.py` | `from surt.model import load_model_and_processor` | VERIFIED | Pattern matches line 302 (deferred inside `main()`) |
| `surt/train.py` | `surt/data.py` | `from surt.data import` | VERIFIED | Pattern matches lines 297-301 (deferred inside `main()`) |
| `HubPushCallback.on_evaluate` | `huggingface_hub.HfApi.upload_folder` | WER-gated push logic | VERIFIED | `upload_folder.*repo_id` pattern confirmed; `self.api.upload_folder(repo_id=self.hub_repo, ...)` at line 101-105 |
| `main()` | `transformers.trainer_utils.get_last_checkpoint` | auto-resume detection | VERIFIED | `get_last_checkpoint` imported at line 22 and called at line 315 |
| `HubPushCallback.__init__` | `best_wer.json` | persistent WER tracking file | VERIFIED | `best_wer\.json` pattern confirmed; `self.best_wer_path = self.output_dir / "best_wer.json"` at line 75 |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| TRAIN-01 | 03-01 | Discriminative LRs: encoder 5e-5, decoder 1e-4, proj_out 1e-4 | SATISFIED | `SurtTrainer.create_optimizer()` builds three groups with those exact LR values |
| TRAIN-02 | 03-01 | Gradient checkpointing enabled | SATISFIED | `gradient_checkpointing=True`, `gradient_checkpointing_kwargs={"use_reentrant": False}` |
| TRAIN-03 | 03-01 | bf16 mixed precision | SATISFIED | `bf16=True` in `build_training_args()` |
| TRAIN-04 | 03-01 | 400 warmup steps | SATISFIED | `warmup_steps=WARMUP_STEPS` where `WARMUP_STEPS=400` in config.py |
| TRAIN-05 | 03-01 | max_steps (not num_train_epochs) for streaming | SATISFIED | `max_steps=MAX_STEPS=5000`; `num_train_epochs` is not set |
| TRAIN-06 | 03-01 | Seq2SeqTrainer with predict_with_generate=True | SATISFIED | `SurtTrainer` extends `Seq2SeqTrainer`; `predict_with_generate=True` confirmed |
| TRAIN-07 | 03-01 | WER via jiwer at each eval step (every 300 steps) | SATISFIED | `make_compute_metrics()` uses `jiwer.wer`; eval every 300 steps |
| TRAIN-08 | 03-01 | GPU auto-detection adjusts batch size and gradient accumulation | SATISFIED | `config.py` lines 9-36: GPU name detection for A100/A40/4090/3090/A5000/L4/T4/V100 with fallback |
| CKPT-01 | 03-01 | Checkpoints saved locally every 300 steps, save_total_limit=3 | SATISFIED | `save_steps=300`, `save_total_limit=3` |
| CKPT-02 | 03-02 | Best checkpoint pushed to Hub after every evaluation via callback | SATISFIED | `HubPushCallback.on_evaluate()` handles WER-gated + periodic push; wired into trainer via `callbacks=[hub_callback]` |
| CKPT-03 | 03-02 | Training can resume from last checkpoint if preempted | SATISFIED | `get_last_checkpoint(OUTPUT_DIR)` + `trainer.train(resume_from_checkpoint=last_ckpt)` |

**All 11 requirements from Phase 3 plans: SATISFIED**

No orphaned requirements. REQUIREMENTS.md traceability table maps TRAIN-01 through TRAIN-08 and CKPT-01 through CKPT-03 to Phase 3, all marked Complete. Phase 3 plans claim exactly those 11 IDs — no gaps or extras.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `surt/train.py` | 238 | `evaluation_strategy` appears in docstring | Info | Docstring note documenting the deprecated name for context; NOT in live code. No impact. |

No blockers. No warnings. One informational note only.

---

### Human Verification Required

#### 1. Hub push actually reaches HuggingFace Hub

**Test:** On RunPod with HF_TOKEN set, run a short training run and verify a commit appears in `surindersinghssj/surt-small-v1-training` on the Hub after the first eval at step 300.
**Expected:** Repository shows a new commit with message matching `step 300 | WER XX.XX | best` or `| periodic`.
**Why human:** Cannot verify network upload or HF token authentication without the live RunPod environment.

#### 2. Discriminative LR groups cover all model parameters

**Test:** On RunPod, run `python -c "from surt.train import SurtTrainer; ..."` against the actual loaded Whisper Small model and log the three group sizes. Verify total equals `model.num_parameters()` minus frozen parameters.
**Expected:** All trainable parameters are assigned to exactly one of the three groups; no parameter is missing or duplicated.
**Why human:** The deduplication logic is correct in source, but the actual parameter count from a loaded Whisper Small model needs live verification on GPU.

#### 3. Auto-resume restores training state correctly

**Test:** Run `python -m surt.train` to step 300, interrupt, then re-run. Verify training resumes at step 300 (not step 0) with optimizer state restored.
**Expected:** Second run log shows `[train] Resuming from checkpoint: .../checkpoint-300` and training continues from step 300.
**Why human:** Requires an actual interrupted run on RunPod to exercise the resume path.

---

### Gaps Summary

No gaps. All 9 observable truths pass. All 11 requirement IDs are satisfied. All 6 key links are wired. All 2 artifacts are substantive and connected. No deprecated APIs in live code. No stubs, placeholders, or empty implementations found.

The one note is that `surt.data` and `surt.model` imports are deferred inside `main()` rather than at module level — this is a documented, intentional decision (audiomentations not installed locally) that does not affect runtime correctness on RunPod.

---

_Verified: 2026-04-04_
_Verifier: Claude (gsd-verifier)_
