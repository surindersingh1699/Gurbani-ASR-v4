# Surt v3 Training — Design Spec

**Date:** 2026-04-18
**Scope:** Retrain Surt on ~580h (250h sehaj + 330h kirtan) from base `whisper-small`, with 2-stage curriculum, canonical-text snapping, strict eval discipline, SGPC noise handling, and multi-GPU speed stack.
**Supersedes:** single-stage mixed training approach discussed earlier in brainstorm.

---

## Goal

Produce `surt-small-v3` with:
- Kirtan WER dropped from v2's ~55% into the 15–25% range.
- Sehaj WER held near v1's ~24% (no regression).
- Per-stratum WER reported on a frozen, human-verified eval set (no leakage).
- Reproducible training run in ≤ 5h wall-clock on 1× H100.

---

## 1. Curriculum — 2-Stage, Not Single-Stage Mixed

### Rationale

- Gemini kirtan labels are ~57% exact-match to canonical SGGS — meaningful noise.
- If training mixes from step 0, the model sees noisy Gurmukhi tokens before it has a Gurmukhi LM prior to push back.
- Sehaj-first warm-up builds a clean language prior that acts as a soft regularizer against Gemini label noise in Stage B.

### Shape

| Stage | Data | Init | Purpose | Epochs |
|---|---|---|---|---|
| **A. Sehaj warm-up** | 250h sehaj only | base `whisper-small` | Gurmukhi tokenizer routing, clean language prior | 1.5–2 |
| **B. Kirtan + sehaj mix** | 330h kirtan + sehaj at 30% sampling prob | Stage A best checkpoint | Acoustic domain adaptation with sehaj replay to prevent forgetting | 3 |

**Stage A stopping criterion:** not sehaj WER floor. Stop at ~WER 30 or 1.5 epochs, whichever first. It's a bootstrap, not a v1 reproduction.

**Stage B sampling:** `aux_probability=0.3` for sehaj replay. Primary training dataset is `gurbani-kirtan-v3-500h`; aux is `gurbani-sehajpath`.

---

## 2. Mool Mantar — Eval/Inference Only

- **Training:** NO `prompt_ids`. Whisper trains teacher-forced; a training-time prompt biases the model to a specific decode pattern.
- **Eval/inference:** YES, set `model.generation_config.prompt_ids = get_mool_mantar_prompt_ids(processor)` before `trainer.evaluate()` and in demo scripts.
- **Wiring change:** `surt/train.py` in `run_training_job()`, right after `load_model_and_processor()` — one line.

The helper `surt.model.get_mool_mantar_prompt_ids` already exists and is correct.

---

## 3. Canonical Text Snapping (STTM Integration)

### Purpose

Recover clean ground-truth labels for the ~57% of clips that ARE canonical SGGS tuks. Effectively free label denoising.

### Design

New column in `data/transcripts_final/<track_id>.jsonl`:

```json
{
  "clip_id": "sikhnet-11220-clip_00042",
  "audio_path": "data/clips/sikhnet-11220/clip_00042.wav",
  "gemini_text": "ਜਨਮ ਮਰਣ ਦੁਖ ਤੇ ਮਨ ਜਾਗੁ",
  "canonical_text": "ਜਨਮ ਮਰਨ ਦੁਖਹੁ ਮਨੁ ਜਾਗੈ",
  "canonical_source": "sggs",
  "canonical_ang": 624,
  "canonical_confidence": 0.91,
  "label_source": "canonical"
}
```

**Label selection rule:** `label_source = "canonical"` when `canonical_confidence >= 0.85`, else `"gemini"`. Training reads from whichever is authoritative for that clip.

### Index scope

- SGGS (primary)
- Dasam Granth banis commonly sung in kirtan
- Bhai Gurdas Vaaran
- Amrit Keertan
- Nitnem banis (Japji, Jaap, Tvai Prasad Svaiye, Chaupai, Anand, Rehras, Kirtan Sohila)

Total ~50k canonical rows. Use multilingual-e5-large or LaBSE for embedding.

### Skip-list (do NOT snap)

- Simran phrases (`ਵਾਹਿਗੁਰੂ ਵਾਹਿਗੁਰੂ`, `ਸਤਿਨਾਮ ਵਾਹਿਗੁਰੂ`) — repetition, not tuk
- Contemporary translations
- Sangat-participation mantras
- Any clip where Gemini output duplicates one word ≥ 10 times (simran pattern)

### Pipeline placement

New step **7.5** between normalize (7) and final filter (8).

### Time budget

| Step | Time |
|---|---|
| Build canonical embedding index (one-time) | ~30min on H100 |
| Embed + top-1 search over 240k clips | ~1.5h on H100 |
| Write canonical_text column | ~5min |
| Human spot-check ~200 matches | ~1h |
| **Total one-shot** | **~3h wall-clock** |

### Deliverable

`scripts/v3_snap_canonical.py` — idempotent, checkpoint-safe, reruns skip already-matched clips.

---

## 4. SGPC Noise Handling — Segment, Don't Filter

### Principle

SGPC PA-mic'd live darbar audio IS the real-world acoustic we want the model to handle. Reverb, crowd, PA compression = KEEP. The noise problem is non-kirtan content (ardas, hukamnama, announcements) mixed into the same recording.

### Two separate fixes

| Problem | Fix | Pipeline step |
|---|---|---|
| Tracks contain ardas/hukamnama/announcements | Silero VAD + music-vs-speech classifier → keep only sung segments | New step **3.5** |
| Clip is mostly silence / PA glitch | SNR floor (5dB), silence-ratio (<50%) | Step 7 filter |
| Reverb, crowd on actual kirtan | **KEEP** — domain distribution | — |

### New step 3.5 (between download and clip-split)

```
3.5. Segment out non-kirtan  →  data/segments/<track_id>.jsonl
     - Silero VAD (speech regions)
     - YAMNet or panns_inference for music probability per second
     - Keep [segment_start, segment_end] where music_prob > 0.7
     - Drop speech-only regions (= ardas / hukamnama / announcements)
```

Clip-splitting (step 4) reads segment ranges and cuts 20s clips only within kept segments.

### Scope

Apply to SGPC only initially. Ragi/AKJ tracks are clean enough that segmentation is unnecessary — add later if eval shows otherwise.

### Time budget

~2h wall-clock for 580h of audio on a single CPU pod (embarrassingly parallel; scales to ~15min on 16-core machine).

### Deliverable

`scripts/v3_segment_non_kirtan.py` — takes `data/raw_audio/<stratum>/*.wav`, writes `data/segments/<track_id>.jsonl`.

---

## 5. Data Scope — Hold at ~580h, Expand Strategically

### Current targets (no change)

| Stratum | Current | v3 target | Source |
|---|---|---|---|
| Sehaj | 250h | 250h | existing `gurbani-sehajpath` dataset |
| Ragi kirtan | ~60h ceiling | 60h | SikhNet 12 playlists (capped) |
| AKJ | ~70h | **350h** (raised from 170h) | akj.org archive (full scrape) |
| SGPC | ~200h | **500h** (raised from 160h) | `sgpc.net/recorded-kirtan-ragi-wise/`, 2h/ragi cap |

**Total kirtan target: ~910h** (up from 500h). Diversity first — the 2h/ragi cap means more voices, not more hours per voice.

### Why raise targets

Data is free for the user now. The previous "stop at 800h" guidance was cost-benefit; zero marginal cost flips the calculation. Diversity returns don't saturate at 800h for this model.

### New stratum: Nitnem (v2 candidate)

Not in v3 scope — queued in `kirtan.txt` v2 section. Add once authoritative sources are identified.

### Hard rule preserved

Only URLs above the `=== v2 ===` marker in `kirtan.txt` are in scope. Enumeration scripts must stop reading at the marker.

---

## 6. Eval Discipline — Frozen, Verified, Leakage-Checked

### Problems with current setup

- Val split is 300 random clips from training datasets → no speaker/ragi hold-out → WER measures memorization.
- `scripts/kirtan_eval_manifest.yaml` has 3 YouTube URLs (contradicts v3 allowlist) and unverified labels.
- No train/eval leakage check runs before training.

### New system: `gurbani-eval-v1`

**Frozen, versioned, human-verified eval dataset. 4 strata, ~2.5h total.**

#### a) Eval manifest

```yaml
# eval/manifest.yaml
version: v1
frozen_at: 2026-04-18
strata:
  sehaj:          { target_hours: 0.5, held_out_readers: [...] }
  kirtan_ragi:    { target_hours: 0.75, held_out_ragis: [...] }  # 3 of 12 SikhNet ragis
  kirtan_akj:     { target_hours: 0.5,  held_out_samagams: [...] }
  kirtan_sgpc:    { target_hours: 0.75, held_out_ragis: [...] }  # ~20 of 201 SGPC ragis

sources:
  - stratum: kirtan_ragi
    source_url: https://play.sikhnet.com/track/...
    track_id: sikhnet-XXXXX
    duration_sec: 637
    transcript_source: human_verified
    transcript_path: eval/verified/kirtan_ragi/sikhnet-XXXXX.txt
```

#### b) Held-out rules (written, not implied)

- **Kirtan ragi:** 3 of 12 SikhNet ragis are eval-only. Never in training.
- **Kirtan SGPC:** ~20 of 201 ragis are eval-only. Date-disjoint from training if same ragi appears.
- **Kirtan AKJ:** whole samagam recordings held out, not random chunks.
- **Sehaj:** whole reader sessions held out.

#### c) Leakage check — MUST pass before training starts

`scripts/check_eval_leakage.py`:
1. Load eval manifest — extract `track_id`, `source_url`, SHA1 of audio bytes, SHA1 of normalized transcript.
2. Load training manifest (`data/transcripts_final/*.jsonl`).
3. Assert intersection on each key is empty.
4. Exit nonzero if any overlap.

Wire into `surt.smoke_test.run_preflight_checks()` — training cannot start with leakage.

#### d) Per-stratum reporting

Current: `eval_sehaj_path_wer`, `eval_kirtan_wer`.
New: `eval_sehaj_wer`, `eval_kirtan_ragi_wer`, `eval_kirtan_akj_wer`, `eval_kirtan_sgpc_wer`.

Best-checkpoint gate: weighted composite `0.3·sehaj + 0.3·ragi + 0.2·akj + 0.2·sgpc`.

#### e) Transcript verification workflow

For each eval clip:
1. Gemini produces a draft transcript.
2. Human reviewer corrects against SGGS canonical where applicable.
3. Diff from Gemini draft to human-verified stored in git so verification effort is auditable.

Verification effort: ~2.5h of audio × 2.5× realtime ≈ 6–8h of human review total.

### Deliverables

- `eval/manifest.yaml`
- `eval/verified/<stratum>/<clip_id>.{wav,txt}`
- `scripts/build_eval_set.py` — reads manifest, produces HF dataset `surindersinghssj/gurbani-eval-v1`
- `scripts/check_eval_leakage.py` — pre-train leakage assertion

---

## 7. Training Speed Stack

### Actual hardware: 1× A40 on-demand (chosen 2026-04-18)

User rented 1× A40 on-demand at ~$0.44/hr. This is the canonical hardware for v3.

| Config | EB | Wall-clock (A+B) | Cost (on-demand) |
|---|---|---|---|
| **1× A40 48GB on-demand (ACTUAL)** | **64→128 via grad_accum** | **~10–13h** | **$4.40–5.72** |
| 1× A40 spot (if available) | same | same | $2.00–2.60 |
| 2× A40 DDP | 128 native | ~5–7h | $4.40–6.16 |
| 1× H100 spot | 128 native | ~4–5h | $7.00–8.75 |

**On-demand vs spot trade:** pay ~$2.50 extra per run for deterministic wall-clock and no preemption. Worth it for Stage B production run; overkill for ablations (re-run those on spot if supply is healthy).

### Config for 1× A40

`surt/config.py` already has the correct A40 branch:
```python
elif "A40" in GPU_NAME:
    BATCH_SIZE = 64  # 48GB VRAM, fits without grad_checkpointing on Whisper-small
# GRAD_ACCUM = 64 // (64*1) = 1
# EFFECTIVE_BATCH = 64*1*1 = 64
```

To hit EB=128 (the sweet spot for Whisper-small), bump `GRAD_ACCUM` to 2:
```python
# Manual override for 1× A40 v3 training:
GRAD_ACCUM = 2  # gives EB=128 at batch 64 × 1 GPU
```

VRAM budget at batch 64 without grad_checkpointing:
- Activations: ~14GB
- Params + optimizer + gradients: ~8GB
- Total: ~22GB (26GB headroom on 48GB)

Safe to disable `gradient_checkpointing` per the speed stack.

### Speed optimizations (stack them)

| # | Change | Expected speedup | Where |
|---|---|---|---|
| 1 | Disable `gradient_checkpointing` | ~35–40% | `surt/train.py:build_training_args` — `gradient_checkpointing=False` on H100/A100 |
| 2 | Flash Attention 2 (3 on H100) | ~25–30% | `surt/model.py:load_model_and_processor` — pass `attn_implementation="flash_attention_2"` |
| 3 | Re-enable cuDNN | ~5–10% if works | `surt/train.py:31` — test first, keep disabled if CUDNN_STATUS_NOT_INITIALIZED returns |
| 4 | `generation_max_length=256` for eval | ~15–25% of total | `surt/config.py` — 20s Gurmukhi clips ≈ 60–90 tokens, 256 is plenty |
| 5 | `eval_steps=400` (up from 200) | ~10% of total | halve eval overhead |
| 6 | Dataloader: `persistent_workers=True`, `prefetch_factor=4`, `pin_memory=True` | ~3–5% | `surt/train.py:build_training_args` |
| 7 | `torch.compile(model)` | ~15–25% | attempt last, breaks generate path sometimes |

**Stacked realistic gain on 1× H100:** (1)+(2)+(4)+(5)+(6) ≈ **~2.2× over unoptimized**.

---

## 8. Deliverables (code + data artifacts)

### Code

| File | Action |
|---|---|
| `surt/config.py` | Add H100 branch, `STAGE_A_MAX_STEPS` / `STAGE_B_MAX_STEPS`, `AUX_TRAIN_PROBABILITY=0.3` for Stage B, `generation_max_length=256`, canonical-snap confidence threshold |
| `surt/model.py` | Add `attn_implementation="flash_attention_2"` flag |
| `surt/train.py` | Wire Mool Mantar `prompt_ids` into `model.generation_config` for eval, add `--stage {a,b}` CLI, disable grad_ckpt on H100/A100, re-cadence eval (400 steps) |
| `surt/smoke_test.py` | Add leakage check to preflight |
| `scripts/v3_snap_canonical.py` | NEW — STTM canonical snapping |
| `scripts/v3_segment_non_kirtan.py` | NEW — SGPC pre-segmentation |
| `scripts/build_eval_set.py` | NEW — frozen eval dataset builder |
| `scripts/check_eval_leakage.py` | NEW — train/eval leakage assertion |
| `scripts/v3_enumerate_*.py` | UPDATE — stop reading `kirtan.txt` at `=== v2 ===` marker |

### Data

| Artifact | Source |
|---|---|
| `eval/manifest.yaml` | NEW |
| `eval/verified/**` | NEW — ~2.5h of human-verified transcripts |
| `surindersinghssj/gurbani-eval-v1` | NEW HF dataset |
| `surindersinghssj/gurbani-kirtan-v3-500h` | Scoped up to ~910h kirtan with canonical_text column |
| `surindersinghssj/surt-small-v3-training` | NEW training checkpoints repo |
| `surindersinghssj/surt-small-v3` | NEW final model repo |

### Updates

| File | Change |
|---|---|
| `kirtan.txt` | Already updated: `=== v2 ===` marker + YouTube v2 sources (pending commit) |
| `PLAN.md` | Raise AKJ target 170h → 350h, SGPC 160h → 500h; add step 3.5 (segmentation) + step 7.5 (canonical snap) to pipeline |
| `memory/project_v3_kirtan_500h.md` | Rename project to `project_v3_kirtan_900h.md` or similar; update targets |

---

## 9. Execution Order & Wall-Clock Budget

Critical path, 1× H100:

| Step | Wall-clock | Depends on |
|---|---|---|
| 1. Commit `kirtan.txt` v2-marker change | 2 min | — |
| 2. Update enumeration scripts to respect v2 marker | 30 min | (1) |
| 3. Enumerate AKJ + SGPC to new targets (350h + 500h) | 1h | (2) |
| 4. Download + segment SGPC | ~6h (can parallelize) | (3) |
| 5. Split into 20s clips | 1h | (4) |
| 6. Gemini batch transcription (~910h audio @ 17.6× realtime) | ~1h | (5) |
| 7. Normalize + STTM canonical snap | ~3h | (6) |
| 8. Build `gurbani-eval-v1` + human verification | 6–8h (human) | — (parallelizable) |
| 9. Leakage check + preflight | 10 min | (7), (8) |
| 10. Stage A training | ~2–3h | (9) |
| 11. Stage B training | ~4–5h | (10) |
| 12. Final WER eval + model card | 30 min | (11) |

**Parallelizable:** (8) can run while (4)–(7) run. (3)–(7) can pipeline across strata.

**Critical path total: ~20h wall-clock end-to-end** including data prep. Training alone ~7–8h on 1× H100.

---

## 10. Success Criteria

v3 is shippable when:

1. `gurbani-eval-v1` composite WER < v2 model's measured composite WER by at least 20 percentage points.
2. Per-stratum: kirtan_ragi WER < 25%, kirtan_akj WER < 35%, kirtan_sgpc WER < 30%, sehaj WER ≤ v1 (no regression).
3. Leakage check passes (no train/eval overlap).
4. Model card published to `surindersinghssj/surt-small-v3` with training + eval details.
5. Memory file `MEMORY.md` updated with v3 results and next-iteration notes.

---

## 11. Out of Scope for v3

- Real-world noisy fine-tune (Stage D — YouTube phones, Instagram reels). Queued for post-v3 once baseline is measured.
- Whisper-medium migration. Queued if v3 shows model-capacity bottleneck (kirtan WER plateaus > 25% despite more data).
- LoRA / PEFT. Full fine-tune continues.
- Audiomentations-based Stage C noise augmentation. Evaluate if v3 production deployment shows robustness gap.

---

## 12. Open Questions for User Review

Before moving to writing-plans:

1. **Held-out ragis** — which 3 of 12 SikhNet ragis are eval-only? Suggest: the 3 least-represented in Stage B training (fewest tracks in playlist).
2. **Stage A length** — fixed at 1.5–2 epochs, or early-stop on validation WER plateau?
3. **Canonical confidence threshold** — 0.85 default. Tighten to 0.90 (fewer snaps, higher precision) or loosen to 0.80 (more snaps, more risk of wrong-tuk)?
4. **Human verification budget** — 6–8h for eval set. Comfortable with that time cost, or start with smaller 1h eval and expand later?
5. **Enumeration script v2-marker handling** — error on finding v2-section URLs, or silently skip? Recommend silent-skip with log warning.
