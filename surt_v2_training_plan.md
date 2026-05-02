# Surt v2 Training Plan — Kirtan-Adapted Model

> ## ⚠️ HISTORICAL — DO NOT FOLLOW
>
> This is the **v2** training plan. It used the HF-hosted kirtan dataset (~28h, 5.8k samples) assembled from mixed OCR + auto-caption + Whisper sources. Kirtan WER plateaued around 55–62%, which is why we pivoted.
>
> **Current plan:** [`PLAN.md`](PLAN.md) — v3 builds a **500-hour** kirtan dataset from YouTube and labels it with **Gemini 2.5 Flash Lite**. This supersedes the v2 approach entirely (no more HF kirtan v2 dataset, no more streaming interleave experiments).
>
> Keep this file for historical context only. Do not use its numbers, configs, or phase definitions when planning new work.

## Goal

Fine-tune `surt-small-v1` (trained on sehaj path) into a kirtan-capable model using **sehaj path + kirtan** interleaved training. Primary use case: kirtan / gurbani audio search.

## What We Know (from v1 experiments)

| Experiment | Data | Steps | Best Step | WER (eval) |
|---|---|---|---|---|
| v1 sehaj path training | 90h sehaj path | 5000 | 3400 | 24 (sehaj path) |
| v1 evaluated on kirtan | — | — | — | 104 (kirtan) — hallucinating |
| Quick kirtan fine-tune | 1-2h kirtan (291 samples) | 500 | ~350 | 34 (kirtan) |

**Key insight:** 1-2h of kirtan dropped WER from 104 to 34. The decoder (Gurbani vocabulary) transferred perfectly. The encoder just needs acoustic adaptation for singing/harmonium/multi-speaker.

---

## Verified Dataset Info

| Key | Value |
|---|---|
| **Sehaj path HF path** | `surindersinghssj/gurbani-asr` |
| Sehaj path text column | `transcription` |
| Sehaj path train samples | 63,100 |
| Sehaj path val samples | 300 (already exists) |
| **Kirtan HF path** | `surindersinghssj/gurbani-kirtan-dataset-v2` |
| Kirtan text column | **`gurmukhi_text`** (NOT `transcription` — needs mapping) |
| Kirtan total segments | 14,527 |
| Kirtan ≤30s segments | 6,544 (~28.5h) |
| Kirtan >30s segments | 7,983 → dropped |
| Kirtan train samples | 5,818 (≤30s, after val/test carve-out) |
| Kirtan val samples | 363 (≤30s, split by video_id) |
| Kirtan test samples | 363 (≤30s, split by video_id) |
| **Combined train samples** | ~68,918 |
| GPU type | `[FILL]` |
| GPU VRAM | `[FILL]` |

### Critical: Text Column Mismatch

The pipeline hardcodes `TEXT_COLUMN = "transcription"` in `data.py`. The kirtan dataset uses `gurmukhi_text`. Options:
1. **Rename on HF** — rename `gurmukhi_text` → `transcription` on the HF dataset (preferred, no code change)
2. **Add column mapping** — map the column in `data.py` when loading the aux dataset

The kirtan dataset also has `gurmukhi_ocr` (raw OCR from slides) vs `gurmukhi_text` (STTM-corrected canonical text). Use `gurmukhi_text` — it's the clean ground truth.

### Critical: Strip `॥` (Double Danda) From Transcriptions

Both datasets contain `॥` (Gurmukhi double danda) and verse markers like `॥੧॥` in the text. These are **visual/structural markers, never spoken aloud**. The model wastes capacity predicting them and every misplaced `॥` inflates WER/CER.

**Action:** Add a text normalization function in `data.py` applied before tokenization:

```python
import re

def normalize_gurbani_text(text: str) -> str:
    """Strip non-spoken structural markers from Gurbani text."""
    # Remove verse numbers: ॥੧॥ ॥੨॥ ॥੧੨॥ etc.
    text = re.sub(r'॥[੦-੯]+॥', '', text)
    # Remove standalone double danda ॥
    text = re.sub(r'॥', '', text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text
```

Apply in both `prepare_train` and `prepare_val` before `processor.tokenizer(text)`. Also apply in `eval_kirtan.py` to both predictions and references for consistent metric computation.

**Note:** v1 WER (24) was measured WITH `॥` in the text. v2 WER will be measured WITHOUT. Numbers are not directly comparable — v2 metrics will be stricter (fewer free-match characters).

### Interleave Probability — Recalculated

With actual sizes (63k sehaj vs ~5.8k kirtan), proportional sampling = `5.8k / 69k ≈ 0.08`. Setting `AUX_TRAIN_PROBABILITY = 0.35` means kirtan is **~4x oversampled** — aggressive but justified since kirtan is the target domain with only 5.8k unique samples. Streaming mode ensures fresh augmentation each pass, reducing overfitting risk from repeated samples. No need to reduce sehaj path — it serves as a regularizer preventing catastrophic forgetting.

---

## Phase 0: Dataset Preparation

1. **Filter kirtan by duration** — keep only segments ≤30s (6,544 samples, ~28.5h). Drop all segments >30s.
2. **Resplit kirtan ≤30s pool** — 5,818 train / 363 val / 363 test. Split by `video_id` to prevent leakage. Push updated dataset to HF.
3. **Kirtan text column** — rename `gurmukhi_text` → `transcription` on HF (or add mapping in code)
4. **Add `normalize_gurbani_text()`** to `data.py` — strip `॥` and verse markers before tokenization
5. **Combine validation sets** for training eval:
   - Sehaj path validation (existing 300 samples) — regression guard
   - Kirtan validation (363 samples) — target metric

---

## Phase 1: Code Changes

### 1a. Config Updates (`surt/config.py`)

```python
# --- Model ---
BASE_MODEL = "openai/whisper-small"                                # fallback
HF_MODEL_REPO = "surindersinghssj/surt-small-v2"                  # final model
TRAINING_HUB_REPO = "surindersinghssj/surt-small-v2-training"

# --- Datasets ---
DATASET_NAME = "surindersinghssj/gurbani-asr"                         # sehaj path (63k samples)
AUX_TRAIN_DATASET_NAME = "surindersinghssj/gurbani-kirtan-dataset-v2"  # kirtan (≤30s only, ~5.8k samples, ~28.5h)
AUX_TRAIN_PROBABILITY = 0.35  # ~4x oversample kirtan (proportional = 0.08)

# --- Learning rates ---
# v2 strategy: encoder LR > decoder LR (encoder adapts acoustics, decoder preserves vocab)
# Encoder is LOWER than v1 (4e-5 vs 5e-5) but 4x HIGHER than decoder.
ENCODER_LR = 4e-5
DECODER_LR = 1e-5
LEARNING_RATE = 1e-5       # base LR (matches decoder)

# --- Training schedule ---
# 5000 steps * 64 effective batch / 68,918 combined samples ≈ 4.6 epochs
MAX_STEPS = 5000
WARMUP_STEPS = 150         # ~3% — model already warm from v1

# --- W&B ---
WANDB_PROJECT = "surt"     # same project, new run name
```

### 1b. Model Loading (`surt/model.py`)

Point to v1 checkpoint at runtime via existing env var override:

```bash
export SURT_BASE_MODEL="surindersinghssj/surt-small-v1-training"
```

No code changes needed — env var override exists in `model.py:37`.

### 1c. Data Pipeline (`surt/data.py`)

Interleave infrastructure is already built. Changes needed:
1. **Add `normalize_gurbani_text()`** — strip `॥`, `॥੧॥`, and collapse whitespace. Call it in `prepare_train()`, `prepare_train_batch()`, and `prepare_val()` before `processor.tokenizer(text)`.
2. **Text column** — kirtan dataset must use `transcription` by the time training starts (rename on HF or add mapping).
3. **Augmentation** — keep as-is for v2. Tune later if needed.

Also update `scripts/eval_kirtan.py` to apply the same normalization to both references and predictions for consistent metrics.

### 1d. Validation (`surt/train.py`)

Combine both val sets into one for training eval. Use `scripts/eval_kirtan.py` for per-domain WER/CER breakdown at notable checkpoints.

---

## Phase 2: Hyperparameters

| Parameter | v1 | v2 | Rationale |
|---|---|---|---|
| Base model | `whisper-small` | `surt-v1-training` | Continue from v1 |
| Encoder LR | 5e-5 | **4e-5** | Lower than v1 but 4x decoder — acoustic adaptation |
| Decoder LR | 1e-4 | **1e-5** | 10x lower than v1 — Gurbani vocab is golden |
| Warmup | ~416 (8%) | **150 (3%)** | Warm start |
| Max steps | 5000 | **5000** | ~4.6 epochs over 69k combined samples |
| Eval/Save | 200 | **200** | Same cadence (~2% overhead) |
| Batch / accum | 32 / 2 | **32 / 2** | Effective 64; adjust for GPU VRAM |
| Scheduler | Cosine | **Cosine** | — |
| Weight decay | 0.01 | **0.01** | — |
| bf16 | Yes | **Yes** | Requires Ampere+ GPU |
| Streaming | Yes | **Yes** | Fresh augmentation each pass |
| Interleave | 0.0 | **0.35** | ~4x kirtan oversample |

### Data Mix Per Step

With `AUX_TRAIN_PROBABILITY = 0.35`:
- ~65% batches from sehaj path (63,100 samples)
- ~35% batches from kirtan (5,818 samples)
- Epochs: `5000 * 64 / 68,918 ≈ 4.6 passes`
- Kirtan effective passes: `4.6 * (0.35 / 0.08) ≈ 20x` (~4x oversampled, fresh augmentation each pass)

---

## Phase 3: Training Runs

### Step 1: Smoke Test (10 steps)

```bash
python -m surt.train --mode smoke --smoke-steps 10
```

Verify: loss decreases, LR warmup works, both datasets load, interleave ratio looks right.

### Step 2: Pilot Run (500 steps)

```bash
python -m surt.train --mode pilot --max-steps 500 --eval-steps 100
```

At step 500, run `eval_kirtan.py` for per-domain WER/CER. Decision points:
- Kirtan WER not improving by step 200 → bump encoder LR to 5e-5
- Sehaj path WER rising significantly → reduce `AUX_TRAIN_PROBABILITY` to 0.20

### Step 3: Full Training (5000 steps)

```bash
python -m surt.train --mode full --preset full
```

### Step 4: Monitor on W&B

- **eval_wer** and **eval_cer** trending down
- **train_loss** smooth, no spikes
- **learning_rate** cosine curve

---

## Phase 4: Evaluation

```bash
python scripts/eval_kirtan.py \
  --model surindersinghssj/surt-small-v2-training \
  --baseline \
  --max-samples 363
```

### Results Table

| Model | Sehaj WER | Sehaj CER | Kirtan WER | Kirtan CER | Notes |
|---|---|---|---|---|---|
| Whisper Small (baseline) | ~85 | — | ~120 | — | Measured |
| surt-small-v1 | 24 | — | 104 | — | Measured |
| surt-small-v1-kirtan (1-2h) | ? | — | 34 | — | Measured |
| **surt-small-v2 (target)** | **< 28** | **TBD** | **< 25** | **TBD** | **Goal** |

### Success Criteria

- Kirtan WER **< 25** (major improvement over 34)
- Sehaj path WER **< 28** (< 4 points regression from 24)
- No hallucination on kirtan (WER < 100 is the floor)
- Track CER alongside WER — CER is more meaningful for Gurmukhi (agglutinative script, word boundaries are ambiguous)

---

## Phase 5: Push Final Model

1. Identify best checkpoint from W&B (lowest combined eval WER)
2. Run full eval on both domains at that checkpoint
3. Push to `surindersinghssj/surt-small-v2` with model card including:
   - Training data composition
   - Per-domain WER/CER results
   - Base model lineage (whisper-small → v1 → v2)
4. Tag the training repo checkpoint

---

## Phase 6: SGPC Live Kirtan Dataset Cleanup

**Dataset:** `surindersinghssj/sgpc-amritsar-kirtan-live` — 1,747 segments, 2.1h, from 1 SGPC Golden Temple live stream
**Status:** NOT USABLE AS-IS — needs full rebuild before training

### Known Issues

| Issue | Severity | Detail |
|-------|----------|--------|
| **Raw YouTube auto-captions** | Critical | Text is NOT human-verified. Wrong characters (`ਚੁਣ`→`ਝੁਣ`), merged words (`ਮੈਰੋਵੰਦੀ`→`ਮੈ ਰੋਵੰਦੀ`), garbled alaap, wrong words entirely (`ਦਾਦੀ`→`ਆਢੀ`) |
| **Systematic word shift** | Critical | Caption timing is offset ~1 word: first word of each segment belongs to previous pangati, last word is missing (leaked to next segment) |
| **Inconsistent spellings** | High | Same line transcribed differently on repetition (`ਛੁਟੀ` vs `ਸੁਟੀ`) |
| **Segments > 30s exist** | High | Duration filter broken — closing "ਖਾਲਸਾ ਵਾਹਿਗੁਰੂ..." segment is 212s. Whisper truncates at 30s so text label would be wrong |
| **Only 1 source video** | Medium | No speaker/raagi diversity — model will overfit to one jatha |
| **Gaps between segments** | Medium | Segments are NOT always contiguous — 2-5s gaps where YouTube couldn't detect speech (instrumental, pauses) |
| **Garbled alaap/intro** | Medium | Opening sections before shabad starts produce nonsense captions |
| **No val/test split** | Low | All 1,747 in train split |
| **Text column name** | Low | Uses `gurmukhi_text` not `transcription` — needs mapping |

### Fix Plan: Forced Alignment Rebuild

Rather than patching individual issues, rebuild the dataset from scratch using forced alignment.

**Prerequisites:**
- Original full video audio (already have via yt-dlp)
- STTM database (`database.sqlite`) for canonical SGGS text
- Trained Surt model (v1 or v2) for forced alignment
- `whisper-timestamped` library

**Steps:**

1. **Identify shabads in the video (manual, ~5 min)**
   - Listen to first 30s of each section or check less-garbled captions
   - Record shabad IDs from STTM database
   - Known shabads so far: ਭੈਣੇ ਸਾਵਣੁ ਆਇਆ (Suhi M1), ਮਾਧਵੇ ਤੁਮ ਨ ਤੋਰਹੁ

2. **Pull canonical pangati sequences from STTM**
   - Query `database.sqlite` for each shabad's ordered pangatis
   - Strip vishraams and dandas via `normalize_gurbani_text()`

3. **Forced alignment with Surt model**
   - Load full video audio (16kHz mono)
   - Run `whisper-timestamped` with canonical shabad text as `initial_prompt`
   - Get real word-level timestamps from audio analysis (not interpolated from VTT)

4. **Segment at pangati boundaries**
   - Group timestamped words by canonical pangati
   - Add 0.3-0.5s padding at each boundary to capture complete words
   - Keep repeated pangatis as separate segments (good for training variety)

5. **Quality filter**
   - Drop segments < 1.5s or > 30s
   - Drop segments with alignment confidence < 0.6
   - Drop garbled alaap sections (no SGGS match)

6. **Build clean dataset**
   - Audio: re-extracted from full video at corrected boundaries with padding
   - Text: canonical SGGS pangatis (not auto-captions)
   - Split: carve out val/test by shabad_id

7. **Scale to more videos**
   - SGPC channel has hundreds of live streams
   - Once pipeline works for 1 video, batch-process 10-20 videos
   - Target: 10-20h of clean, diverse kirtan data

### Expected Outcome

| Before | After |
|--------|-------|
| 1,747 noisy segments | ~1,200-1,400 clean segments (alaap/instrumental dropped) |
| YouTube auto-caption text | Canonical SGGS pangatis |
| Word-shifted audio boundaries | Proper pangati-aligned boundaries with padding |
| 1 raagi jatha | 1 raagi (expandable to 10-20 with more videos) |

---

## Risk Mitigation

| Risk | Signal | Mitigation |
|---|---|---|
| Encoder doesn't adapt | Kirtan WER plateaus after 500 steps | Bump encoder LR to 5e-5 or 8e-5 |
| Sehaj path regression | Sehaj WER > 28 at any eval | Reduce `AUX_TRAIN_PROBABILITY` to 0.20 |
| Kirtan overfitting | Kirtan train loss drops but val WER stalls | Reduce `AUX_TRAIN_PROBABILITY` to 0.22 (proportional) |
| Loss spikes early | train_loss jumps in first 200 steps | Reduce encoder LR to 3e-5, increase warmup to 300 |
| Text column mismatch | Dataset load error or garbled text | Rename `gurmukhi_text` → `transcription` on HF |
| RunPod preemption | Pod killed mid-training | Hub callback every 600 steps; resume from checkpoint |
| OOM | CUDA OOM error | Reduce batch to 16, increase accum to 4 (same effective batch) |
