# Surt v3 Training Plan — 8900-step `phase4` run (smoke + full)

> **Supersedes:** `surt_training_plan.md` (v1, sehaj-only), `surt_v2_training_plan.md` (v2, 28h kirtan).
> **Context:** v1/v2 sehaj WER numbers turned out to be inflated by a data leak in the prior sehaj eval set. Starting fresh is the only way to get trustable metrics.

---

## TL;DR

1. **Code + config are already wired for v3** — Phase 0.1/0.2 are applied (see below). The agent only needs to run Phase 0.3 pre-flight, set the pod env (0.4), then launch the single `phase4` run (Phase 1), then post-training validation (Phase 2).
2. Launch: `python -m surt.train --mode phase4`. Chains preflight + 10-step smoke + 8900-step full + final push. No separate pilot — smoke covers load/config bugs and early-stop (patience 3 on kirtan WER+CER) guards against plateau.
3. Final pushed to `surindersinghssj/surt-small-v3`. Best-WER checkpoint is also on `surindersinghssj/surt-small-v3-training` throughout the run.

No A/B, no curriculum experiment, no warm-start from v1 or v2, no separate pilot — single clean mixed-training run from `openai/whisper-small` on the new v3 data via `phase4`. **This is the only approach to execute** — do not revisit v1/v2 warm-start, do not propose curriculum training, do not run a standalone pilot.

---

## Starting state (verify before launch)

### Training data

- **450h kirtan (canonical):** `surindersinghssj/gurbani-kirtan-yt-captions-300h-canonical` — text column `final_text`. Repo name says 300h, may be ~450h after AKJ additive push — confirm `train.num_rows` before launch.
- **~160h sehaj (new, canonical):** `surindersinghssj/gurbani-sehajpath-yt-captions-canonical` — primary sehaj stream (`DATASET_NAME`), text column `final_text`.
- **~66h sehaj (old, studio-recorded):** `surindersinghssj/gurbani-sehajpath` — non-canonical; uses `gurmukhi_text` column (harmonized to `final_text` on load via `EXTRA_SEHAJ_TEXT_COLUMN = "gurmukhi_text"`). Concatenated with primary sehaj before aux mixing. Total sehaj after concat ≈ 220–230h.

The three-dataset setup is wired in [surt/data.py](surt/data.py)'s `get_train_dataset` — primary + extra sehaj concat happens first (renaming the extra sehaj text column to `final_text`), then canonical kirtan is mixed in as aux at `AUX_TRAIN_PROBABILITY=0.64`. No separate combined HF dataset is built. The four canonical datasets (primary sehaj, aux kirtan, both evals) use `final_text` natively; only the non-canonical old sehaj uses `gurmukhi_text` and is renamed on load.

### Held-out eval (1h each)

- Kirtan eval: `surindersinghssj/gurbani-kirtan-yt-captions-eval`
- Sehaj eval: `surindersinghssj/gurbani-sehajpath-yt-captions-eval`

### Base model

`openai/whisper-small` (fresh). **Do not warm-start from v1 or v2** — v1 sehaj WER was inflated by leak, so its "prior knowledge" is untrustable; v2 has baked-in kirtan hallucination priors from the 28h noisy dataset.

### Target run layout

- Primary train stream = sehaj (smaller, ~45k clips at 20s)
- Aux train stream = kirtan (larger, ~80k clips at 20s)
- `AUX_TRAIN_PROBABILITY = 0.64` (matches the natural 450:250 kirtan:sehaj hour ratio)
- Eval: dual — sehaj eval set + kirtan eval set, fired separately after each `eval_steps` interval

---

## Phase 0 — Config and code state

### 0.1 `surt/config.py` — DONE (committed)

The following values are already set in [surt/config.py](surt/config.py). Do not re-edit; just sanity-check before launching:

```python
BASE_MODEL = "openai/whisper-small"
HF_MODEL_REPO = "surindersinghssj/surt-small-v3"
TRAINING_HUB_REPO = "surindersinghssj/surt-small-v3-training"

LEARNING_RATE = 3e-5
ENCODER_LR = 5e-5
DECODER_LR = 3e-5
WARMUP_STEPS = 900

DATASET_NAME = "surindersinghssj/gurbani-sehajpath-yt-captions"
AUX_TRAIN_DATASET_NAME = "surindersinghssj/gurbani-kirtan-yt-captions-300h"
AUX_TRAIN_PROBABILITY = 0.64

SEHAJ_EVAL_DATASET_NAME = "surindersinghssj/gurbani-sehajpath-yt-captions-eval"
KIRTAN_EVAL_DATASET_NAME = "surindersinghssj/gurbani-kirtan-yt-captions-eval"

EARLY_STOP_PATIENCE = 3
EARLY_STOP_METRIC = "kirtan"
```

Also already set: `MAX_STEPS=8906` (190k × 3 / 64), `EVAL_STEPS=SAVE_STEPS=500`, `VAL_SIZE=400`, `SHUFFLE_BUFFER=8000`, `GENERATION_MAX_LENGTH=448`.

### 0.2 `surt/train.py` — DONE (committed)

- `SEHAJ_EVAL_DATASET_NAME` and `KIRTAN_EVAL_DATASET_NAME` are imported from config.
- `run_training_job` reads eval from the dedicated eval repos (not from train repos' validation splits). Train data is never used for eval.
- `PlateauEarlyStopCallback` is wired in, watching `eval_kirtan_{wer,cer}` with patience 3.
- `HubPushCallback` now tracks **best WER AND best CER** for **both** sehaj and kirtan splits (4 independent bests). Pushes fire whenever any of the 4 metrics improves, plus a periodic safety push every 3 evals. State persists to `best_metrics.json` (migrates from the legacy single-WER `best_wer.json` on first resume).
- Final push commit message reads `surt_small_v3 final (step N)`.

No action needed in this subsection — the agent should verify by grep only:

```bash
grep -n "SEHAJ_EVAL_DATASET_NAME\|KIRTAN_EVAL_DATASET_NAME\|PlateauEarlyStopCallback" surt/train.py
```

### 0.3 Pre-flight — verify dataset shapes

Before anything else, run this on the pod:

```bash
python - <<'EOF'
from datasets import load_dataset
for name, split in [
    ("surindersinghssj/gurbani-sehajpath-yt-captions-canonical", "train"),       # primary sehaj (canonical)
    ("surindersinghssj/gurbani-sehajpath", "train"),                             # extra sehaj (old, non-canonical)
    ("surindersinghssj/gurbani-kirtan-yt-captions-300h-canonical", "train"),     # aux kirtan (canonical)
    ("surindersinghssj/gurbani-sehajpath-yt-captions-eval-canonical", "validation"),
    ("surindersinghssj/gurbani-kirtan-eval-pure-canonical", "validation"),
]:
    try:
        ds = load_dataset(name, split=split, streaming=True)
        ex = next(iter(ds))
        print(name, split, "→ cols:", sorted(ex.keys()))
    except Exception as e:
        print(f"FAIL {name} {split}: {e!r}")
EOF
```

**Must confirm:**

- All 5 datasets load cleanly
- All canonical datasets (primary sehaj, aux kirtan, both evals) expose `final_text` — matches `TEXT_COLUMN = "final_text"`
- Extra sehaj (`gurbani-sehajpath`, non-canonical) exposes `gurmukhi_text` — matches `EXTRA_SEHAJ_TEXT_COLUMN = "gurmukhi_text"`
- All have an `audio` column
- Eval splits are named `validation`

If any column name differs from the defaults above, fix it in `surt/config.py` (`TEXT_COLUMN` or `EXTRA_SEHAJ_TEXT_COLUMN`) before launching. The aux kirtan mapping is no longer hardcoded in `surt/data.py` — it reads directly from `TEXT_COLUMN`, so both primary sehaj and aux kirtan MUST share the same column name.

All 5 canonical / non-canonical repos are ready as of 2026-04-20. Pre-flight should pass on all rows. If any one fails, fix before launching Phase 1.

### 0.4 Pod environment

```bash
export HF_HOME=/workspace/.cache/huggingface
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
export WANDB_API_KEY=...           # optional — enables wandb logging
export SURT_DOWNLOAD_WORKERS=8     # parallel parquet shard download
```

---

## Phase 1 — Training launch (`phase4` mode)

The standalone 500-step pilot is intentionally skipped. `--mode phase4` chains preflight + 10-step smoke test + full 8900-step run + final push, which covers the critical load/config/tokenizer sanity checks in <1 minute without needing a separate pilot phase. Early-stop (patience 3 on kirtan WER+CER) guards against plateau; `HubPushCallback` uploads the best-WER model regardless of the final step.

### 1.1 Command

```bash
python -m surt.train --mode phase4
```

That's it. No flags required — `surt/config.py` holds all schedule / LR / dataset / early-stop settings.

Stages the command runs, in order:

1. **Preflight checks** (`run_preflight_checks`) — TEST-01/TEST-02: model loads, tokenizer returns Gurmukhi tokens, dataset loads one example
2. **10-step smoke run** — trains 10 steps on streaming data, verifies loss decreases and LR warmup ticks up (`validate_smoke_training`)
3. **Full 8900-step training** — eval every 500 steps, save every 500 steps, hub push WER-gated + every 3rd eval
4. **Final push** — uploads the final model + processor to `surindersinghssj/surt-small-v3` on exit (early-stopped or completed)

Resume is automatic — if the pod restarts mid-run, re-invoke the same command. `resume_from_last_checkpoint=True` picks up the latest checkpoint from `/workspace/surt/checkpoints/`.

### 1.2 Expected behavior

- **Duration:** ~6–8h on A40, ~4–5h on A100
- **Evals:** every 500 steps → ~17 evals over the full run
- **Hub pushes:** WER-gated best push (sehaj) + periodic safety push every 3rd eval. Best-WER checkpoint is on the Hub *regardless of how training ends* — crash, early-stop, or completion
- **Early stop:** watches `eval_kirtan_wer` + `eval_kirtan_cer`, patience=3. If neither kirtan metric improves for 3 consecutive evals (1500 steps), training halts and final push still fires
- **Final push target:** `surindersinghssj/surt-small-v3`

### 1.3 Smoke-stage pass criteria (after step 10)

If the smoke-stage `validate_smoke_training` assertion fires, the full run is aborted automatically. Investigate before re-running:

| Symptom | Likely cause | Fix |
|---|---|---|
| Loss didn't decrease over 10 steps | Column / tokenizer / audio bug | Re-run Phase 0.3 column check; verify TEXT_COLUMN values |
| LR didn't ramp up | Scheduler misconfigured | Check `WARMUP_STEPS` and `lr_scheduler_type="cosine"` in `build_training_args` |
| Crash on `load_model_and_processor` | Wrong `BASE_MODEL` string / no HF auth | Confirm `BASE_MODEL = "openai/whisper-small"` and `huggingface-cli login` |
| Crash during dataset cast | Column-name mismatch in primary / extra sehaj / kirtan | Adjust `TEXT_COLUMN`, `EXTRA_SEHAJ_TEXT_COLUMN`, or the kirtan `gurmukhi_text` mapping |

### 1.4 Full-run monitoring — first 500 steps

Same early-warning signals as a pilot, just embedded in the full run. Watch wandb (or stdout) during the first eval (step 500):

| # | Check | Threshold | Action if fail |
|---|---|---|---|
| 1 | Loss trending down from step 50 → 500 | loss@500 < 0.7 × loss@50 | Kill run, inspect data pipeline |
| 2 | Kirtan WER @ step 500 | < 75% | Column-mapping bug — stop and fix |
| 3 | Sehaj WER @ step 500 | < 60% | Tokenizer / audio bug — stop and fix |
| 4 | No NaN / inf | — | Drop `DECODER_LR` to 2e-5 and restart |
| 5 | No OOM | — | `export SURT_BATCH_SIZE=32` and restart |
| 6 | Hub push succeeded at step 500 | — | `surt-small-v3-training` commit exists |

Kill the run manually (`Ctrl-C`) if any of the above fails — don't wait for early-stop, it triggers on plateau not broken-pipeline.

### 1.5 Full-run monitoring — ongoing

- [ ] Loss trends down through ~step 5000, may plateau after
- [ ] Both WER and CER (kirtan + sehaj) improving or within noise vs prior eval
- [ ] `[train] Hub push (best|periodic)` log lines on every eval with best-so-far
- [ ] GPU utilization >80% in `nvidia-smi` (low = dataloader starvation; bump `SURT_DL_WORKERS`)
- [ ] `[early-stop] STOP:` only fires on genuine plateau, not before step ~3000

### 1.6 Expected WER trajectory (sanity benchmark)

If the run is healthy, eval metrics should roughly track:

| Step | Kirtan WER | Sehaj WER | Note |
|---|---|---|---|
| 500 | 60–70% | 40–50% | First eval — full-run pilot gate |
| 2000 | 40–50% | 25–30% | Kirtan should be beating v2's 55% floor |
| 5000 | 30–35% | 15–20% | Clean gains from 700h scale |
| 8900 | 25–30% | 10–15% | Target final WER |

If at any milestone kirtan WER is >1.5× the band above, **kill the run** — don't let it burn to completion on a broken pipeline.

### 1.7 If early-stop fires

This is the designed-for outcome if training plateaus. When `[early-stop] STOP:` logs:

1. Training stops, final checkpoint is saved
2. Final push to `surt-small-v3` fires (contains the last-step model)
3. **Best-WER checkpoint is already on Hub** via `HubPushCallback` periodic pushes — the main repo just happens to hold the last-step model, not the best
4. If the best and final differ meaningfully, manually re-push the best checkpoint to `surt-small-v3` from `surt-small-v3-training` (check `best_wer.json` in the output dir for which step was best)

---

## Phase 2 — Post-training validation

### 2.1 Final held-out WER

After final push, run a clean eval of `surt-small-v3` against both held-out eval repos. Use a size well above the 400-clip training-time eval (e.g., full 1h eval set, not the 400-row sample):

```bash
python scripts/evaluate_model.py \
    --model surindersinghssj/surt-small-v3 \
    --eval-sehaj surindersinghssj/gurbani-sehajpath-yt-captions-eval \
    --eval-kirtan surindersinghssj/gurbani-kirtan-yt-captions-eval \
    --out reports/v3_final_eval.md
```

(If `scripts/evaluate_model.py` doesn't exist, write a minimal one using the same WER/CER path as `make_compute_metrics`.)

### 2.2 Report

Write final numbers, eval set sizes, and commit/SHA of the model to `reports/v3_final_eval.md`. Include:

- Sehaj WER / CER (full 1h)
- Kirtan WER / CER (full 1h)
- Training wall-clock
- Final step count (may be <8900 if early-stop fired)
- Sample transcriptions (best + worst 5 per split)

### 2.3 Release gate

Only flip `surt-small-v3` to a production tag if:

- Kirtan WER < 35% on the full 1h held-out eval
- Sehaj WER < 20% on the full 1h held-out eval
- No regression vs v1 on sehaj (accounting for the v1 leak — use the same clean held-out eval)

If these gates fail, keep the model in `surt-small-v3-training` for analysis, do not promote.

---

## Rollback / escalation

If anything in Phase 1 or Phase 2 fails unexpectedly:

- **Don't delete training checkpoints** — they're the only forensic record
- **Don't force-push** to any Hub repo
- Stash the run directory: `mv /workspace/surt/checkpoints /workspace/surt/checkpoints.failed-$(date +%s)`
- Log the failure mode + last passing step in `reports/v3_failures.md` before retrying

---

## What this plan intentionally does NOT include

The following were evaluated and rejected. Do not revisit in the current run:

- Curriculum training (sehaj-then-kirtan sequential) — rejected: risks catastrophic forgetting; mixed is the ASR fine-tuning default
- LoRA / PEFT — full fine-tune continues to be the approach
- Warm-start from v1 or v2 — rejected: v1 sehaj WER was inflated by leak; v2 has kirtan hallucination priors
- A/B pilot experiments — rejected: 500 steps is too short to distinguish strategies, mixed is the safe default
- Standalone 500-step pilot — rejected: `phase4`'s 10-step smoke stage + early-stop + best-WER hub push cover the failure modes a pilot would catch, without the extra compute

## Cross-references

- Pipeline code: `surt/config.py`, `surt/data.py`, `surt/train.py`, `surt/model.py`
- Early-stop callback: `surt/train.py::PlateauEarlyStopCallback`
- Hub push callback: `surt/train.py::HubPushCallback`
- Label-length filter: `surt/data.py::_make_label_fits_filter`
- Dataset allowlist for kirtan source material: `kirtan.txt`
- v3 dataset build plan (upstream of this training plan): `PLAN.md`
