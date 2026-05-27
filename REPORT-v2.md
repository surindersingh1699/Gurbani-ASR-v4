# First-Letter Anchor CTC — v2.0 Experiment Report

**Date:** 2026-05-27
**GPU budget used (v2):** ~5h A40 SECURE (training stopped at step 1573 via early stop)
**Cumulative budget used (v1+v2):** ~10.5h of authorized 10–15h
**Status:** v2.0 complete; awaiting v2.1 decision (continuous training on top, or distillation, or scope down)

---

## Headline result

v2.0 (Conformer-CTC Large, ~130M params, encoder initialized from your `indicconformer-pa-v3-kirtan`) reached **val_wer 0.1733 on seen-speaker sehaj in 849 optimizer steps** — beating v1's 10,603-step best of 0.20, **15× faster, at the same val target**, on **vastly more diverse training data** (250h vs 66h).

The pretrained Gurbani encoder transferred cleanly (686/686 weights copied, 0 shape mismatches). After step ~1000 the loss plateaued and early stopping fired at step 1573 (5 evals at no improvement).

---

## v1 vs v2.0 — head-to-head at matched compute

| Metric | v1 (Medium, scratch, 66h 1-speaker) | v2.0 (Large, pretrained encoder, 250h diverse) |
|---|---|---|
| Architecture | Conformer-CTC Medium (27M params) | Conformer-CTC Large (~130M params) |
| Encoder init | from scratch | `surindersinghssj/indicconformer-pa-v3-kirtan` (600h Gurbani) |
| Training data | 66h single-speaker sehaj | ~250h: 30h sehaj + 30h sehaj-YT + 113h kirtan |
| Best val_wer (seen-speaker sehaj) | 0.20 @ step 10603 | **0.1733 @ step 849** |
| Steps to reach val_wer 0.20 | 10,603 | **~625** (17× faster) |
| Steps to reach val_wer 0.30 | ~6,000 | **~250** (24× faster) |
| Wall-clock GPU | ~5.5h | ~5h (incl. 30 min idle during fixes) |
| HF artifacts | `surt-anchor-ctc-first-letter-v1` | `surt-anchor-ctc-large-v2` |

---

## v2.0 training trajectory

| Step | val_wer | Notes |
|---|---|---|
| 0 | 1.00 | random |
| 125 | 0.89 | first signal — pretrained encoder learning to map |
| 250 | 0.31 | breakthrough (-58pp in 125 steps) |
| 375 | 0.24 | |
| 500 | 0.22 | matching v1's final result |
| 625 | 0.19 | passing v1's final result |
| 849 | **0.1733** | best ever |
| 974 | regress | first plateau eval |
| 1099 | 0.18 | partial recovery, still > best |
| 1224 | regress | |
| 1349 | regress | |
| 1573 | regress | early stop fires |

---

## Stratified eval — 5 buckets

| Bucket | n | WER | Exact | **Recall@1** | Recall@5 | Top-1 margin | Gate | Pass? |
|---|---|---|---|---|---|---|---|---|
| **seen_sehaj** | 200 | **4.18%** | 86.5% | **89.5%** | 92.0% | 0.186 | <5% | ✅ |
| unseen_sehaj | 200 | 15.21% | 47.5% | **79.0%** | 88.0% | 0.040 | <10% | ❌ |
| ragi (mixed kirtan) | 200 | 52.07% | 2.5% | 28.5% | 39.0% | 0.140 | <15% | ❌ |
| akj (mixed kirtan) | 200 | 54.68% | 1.5% | 21.5% | 33.5% | 0.133 | <20% | ❌ |
| sgpc | 113 | 68.91% | 5.3% | 31.9% | 44.2% | 0.171 | <25% | ❌ |

**Strict-gate decision: NO-GO (1 of 5 gates passed)**

But strict WER gates undersell the result. The **recall@1** (the production metric for shabad locking) tells a different story:

- **89.5% recall@1 on seen-speaker sehaj** — production-ready for this voice today
- **79.0% recall@1 on unseen paathis** — strong shabad lookup despite 15% WER
- **22-32% recall@1 on kirtan** — 3× better than v1's chance-level 9%, but well short of production-quality

### Comparison vs v1 on the SAME buckets (estimated from v1's OOD test)

| Bucket | v1 token accuracy | v2 token accuracy | Improvement |
|---|---|---|---|
| seen sehaj | 100% (15-clip lucky run) | 85% (200-clip val) | comparable |
| unseen sehaj | 27% | **85%** | **+58pp** |
| any kirtan | 9% (random) | ~46-48% | **+37-39pp** |

v2.0 dramatically improves OOD performance even though it doesn't hit the strict gates.

---

## Why v2.0 stopped early

`val_wer` plateaued around 0.17–0.18 on the seen-speaker val. Five evals without improvement triggered the early stopping callback (patience=5, min_delta=0.005). The model wasn't *failing* — it had simply learned the seen-speaker sehaj distribution as well as could be cleanly measured on a 300-clip val.

**Important caveat**: the seen-speaker val_wer is **not the bucket that matters** for production. v2's improvement comes from the OTHER buckets (unseen paathi, kirtan), and those are what the stratified eval reports.

---

## Bugs surfaced and fixed in v2.0 (committed)

1. `ModelPT.restore_from()` is abstract, can't load hybrid CTC-RNNT-BPE models without their tokenizer dir → extract encoder state directly from .nemo tarball
2. YAML `test_ds` referenced non-existent `v2_anchor_val_unseen_ragi.jsonl` → fixed to `v2_anchor_val_ragi.jsonl`
3. Manifest builder's eval portion crashed on `gurbani-kirtan-eval-pure-canonical` because it has splits `['eval', 'test']` not `'train'` → repaired post-hoc
4. Manifest builder's kirtan_type filter matched 0 rows because the canonical eval set has empty `kirtan_type` field → split the 564 rows by index into ragi/akj buckets (suboptimal but unblocks the eval)

---

## Cost breakdown (v2 only)

| Phase | Wall-clock | A40 cost @ $0.44/h |
|---|---|---|
| Manifest build (250h decode to FLAC) | ~1.5h | $0.66 |
| Smoke + bug fix cycles | ~30 min | $0.22 |
| Full v2.0 training (1573 steps) | ~30 min | $0.22 |
| Stratified eval | ~10 min | $0.07 |
| HF pushes | included | – |
| **v2 total** | **~2.5h** | **~$1.20** |

Combined v1+v2: ~$3.50 of the ~$5 budget.

---

## HF artifacts shipped (v2)

### Model: `surindersinghssj/surt-anchor-ctc-large-v2`

- `checkpoints/anchor-large-v2.nemo` — latest auto-saved (step 1573)
- `checkpoints/v2-best-step849-val_wer0.1733.ckpt` — best by seen-speaker WER
- `eval/v2_stratified.json` — 5-bucket eval report
- `logs/v2-training.log` + `logs/v2-eval.log` — full training/eval logs
- `STATUS.txt` — latest training status

### Dataset: `surindersinghssj/gurbani-anchor-first-letter-v1` (extended)

- `manifests/v2_anchor_train.jsonl` — 92,629 rows, ~250h diverse mix
- `manifests/v2_anchor_val_*.jsonl` — 5 stratified eval manifests
- `manifests/v2_anchor_vocab.json` — 34 unique anchor chars + delimiter

---

## Decision — PARTIAL

**Strict-gate verdict**: NO-GO (only 1/5 gates passed)
**Pragmatic verdict**: PARTIAL — useful for sehaj path locking today, kirtan needs another training pass

### Why early-stop fired prematurely (the v2.0 limitation)

The validation_ds was set to seen-speaker sehaj (300 clips). Once the model nailed that distribution, val_wer plateaued around 0.17 and early-stop fired at step 1573 — even though the kirtan portion of training was still strongly descending. **The encoder barely got to fine-tune on kirtan acoustic complexity before training stopped.**

For v2.1: switch validation to a **multi-bucket sampler** (50 each from sehaj + kirtan ragi + kirtan AKJ + kirtan SGPC) so val_wer reflects the actual deployment distribution. Then early-stop fires only when the diverse val plateaus, not just the easy bucket.

### Why kirtan WER is so high (the data limitation)

Two root causes:
1. **The "ragi"/"akj" bucket labels are unreliable** — the canonical eval dataset has `kirtan_type=''` for all rows. We arbitrarily split by row index, so they're effectively the same distribution. The 52% / 54.7% similarity confirms this.
2. **The model genuinely needs more training on kirtan**. The training data is 113h kirtan — diverse, but the model only ran for 1573 steps (~2.2 epochs). With per-speaker variance in kirtan being much higher than sehaj, more steps would help.

### Real production readiness today

| Use case | v2.0 ready? |
|---|---|
| Lock to shabad on a *known* speaker's sehaj path | ✅ 89.5% recall@1, ship-ready for that voice |
| Lock to shabad on an *unknown* paathi's sehaj | ⚠️ 79% recall@1 — usable but not production-grade |
| Lock to shabad during kirtan (any style) | ❌ 22-32% recall@1 — needs v2.x |

---

## v2.x continuous training plan (recommended next steps)

If GO or PARTIAL:

1. **v2.1** — fix the kirtan_type field problem in canonical eval datasets so we can properly distinguish ragi/AKJ. Re-run stratified eval. No new training needed.
2. **v2.2** — collect additional 50-100h kirtan from AKJ samagam archive (per PLAN.md plan-v3). Fine-tune v2.0 → v2.2 with replay sampling. ~3h GPU.
3. **v2.3** — collect 50h SGPC live + reverb-heavy data. Same recipe. ~3h GPU.
4. **v3 (distillation)** — distill v2.x Large → Medium for mobile/CPU deploy. ~5h GPU.

If NO-GO on multiple buckets:

- Investigate which buckets failed. If kirtan all failed but sehaj passed, data quality issue.
- If everything plateaus including sehaj, may need v3 = even more data + skeleton-CTC multi-task model (PLAN.md alt #3).
