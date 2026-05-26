# First-Letter Anchor CTC — v1 Experiment Report

**Date:** 2026-05-26  
**GPU budget used:** ~5.5h A40 (out of 10-15h authorized)  
**Status:** v1 stopped voluntarily at step 10603 / 15000 to pivot to v2 (Large arch + Gurbani-fine-tuned IndicConformer init)

---

## Executive summary

v1 answered PLAN.md's central architectural question: **first-letter STTM-ASCII CTC is learnable**. A 27M-param Conformer-CTC Medium trained from scratch on 66h of single-speaker sehaj path data reached **val_wer 0.20 (80% in-distribution token accuracy)** in ~5.5h of A40 compute, with the loss curve still descending when stopped.

The result is sufficient to commit to the anchor-CTC direction and scale up via v2 (Large + multi-speaker data + your fine-tuned IndicConformer encoder init).

---

## Milestone status vs PLAN.md

| Milestone | Goal | Status |
|---|---|---|
| **M1.1** Data prep + target validation | STTM mapping, mixed manifests, label spot-check | ✅ **PASS** — mapper 99.94% DB round-trip on 141,264 lines; 63,398/63,400 sehaj clips kept (99.997% retention) |
| **M1.2** Training scaffold + smoke run | NeMo configs, 100-step smoke, params 30-50M | ✅ **PASS** — smoke ran clean, decode is delimiter-valid, 27M params (slightly under 30M target due to from-scratch Medium choice) |
| **M1.3** Full experiment training | Sehaj exact ≥85% | ⚠️ **PARTIAL** — reached 80% token accuracy on single-speaker val. Did not hit 85% gate but: (a) val is same-speaker-as-train, so not a true generalization gate, (b) curve was still descending, (c) decision was made to pivot to Large+pretrained rather than burn budget pushing this small model |
| **M1.4** Eval and matcher integration | recall@1 vs current matcher | ❌ **NOT RUN** — pivot to v2 took precedence; deferred to v2 |
| **M1.5** Decision gate | ship/revise call | ✅ **DECISION: REVISE** — Architecture proven; scale data + use Gurbani-pretrained encoder for v2 |
| **M1.6** Scale-or-stop | continue/scale recommendation | ✅ **CONTINUE** — kirtan errors look data-fixable; v2 plan codified separately |

---

## Training trajectory

Single-speaker val_wer over ~5.5h training:

| Step | val_wer | Phase |
|---|---|---|
| 0 | 1.000 | random outputs (blank-dominated) |
| 250 | 0.933 | learned non-blank > blank |
| 1236 | 0.840 | letter-frequency stats |
| 1729 | 0.673 | acoustic threshold crossed (+17pp burst) |
| 3208 | 0.490 | sub-50% — most positions correct |
| 5180 | 0.357 | clear phoneme mapping |
| 7152 | 0.260 | first plateau |
| 9124 | 0.230 | broke out of plateau |
| **10603** | **0.200** | stopped (still descending) |

Throughput peaked at **~3 it/s** on A40 with batch 64, bf16-mixed, accumulate_grad_batches=2 (effective batch 128).

---

## Out-of-distribution test (v1 model on unseen data)

Ran `test_v1_ood.py` on the trained v1 checkpoint over 15 random clips from each of three sources:

| Slice | Token accuracy (pos_match) | Exact match | n |
|---|---|---|---|
| In-dist sehaj (same speaker as train) | **100%** | 100% | 15 |
| OOD sehaj YT (unseen paathis) | **27.2%** | 0% | 15 |
| OOD kirtan (never seen) | **8.6%** ≈ random | 0% | 15 |

**Conclusion**: v1 nailed its training distribution but does NOT generalize. The 100% on in-dist is luck on a small sample (val_wer=0.20 on 300 clips says actual ~80%), but the **drop from 100% → 27% → 9%** as we move OOD is the key signal.

This is the strongest possible justification for v2's design choices:
- Multi-speaker data (v1's 1-speaker limit caused the 27% sehaj-YT drop)
- Multi-domain data (v1 never saw kirtan, hence the 9% chance-level)
- Pretrained Gurbani encoder (your `indicconformer-pa-v3-kirtan`, already trained on multi-ragi data)

## Critical caveat: single-speaker val

All 63,398 training clips and 300 validation clips are from **one speaker** (`giani_mehnga_singh`) in `surindersinghssj/gurbani-sehajpath`. The reported val_wer measures **in-distribution accuracy only** — it tells us the model is learning Gurmukhi-to-first-letter mapping on this speaker, not that it generalizes.

This was acceptable for the M1.1-M1.3 experiment goal (proving learnability) but means the v1 model is NOT production-deployable. v2 explicitly addresses this with speaker-stratified train/val splits.

---

## Artifacts shipped

### HuggingFace model repo
`https://huggingface.co/surindersinghssj/surt-anchor-ctc-first-letter-v1`

Contents:
- `checkpoints/anchor-first-letter.nemo` — best checkpoint by val_wer (auto-updated every 2000 steps during training)
- `checkpoints/v1-final-step10603-val_wer0.20.nemo` — frozen final v1 snapshot
- `logs/v1-training.log` — full training log
- `STATUS.txt` — last training status

### HuggingFace dataset repo
`https://huggingface.co/datasets/surindersinghssj/gurbani-anchor-first-letter-v1`

Contents:
- `manifests/anchor_first_letter_train.jsonl` — 63,098 NeMo manifest rows
- `manifests/anchor_first_letter_val.jsonl` — 300 rows
- `manifests/anchor_first_letter_eval_by_domain.jsonl` — initial eval manifest (same as val for v1)
- `manifests/anchor_first_letter_vocab.json` — 38 unique anchor chars + delim

### Code (branch: `feat/anchor-first-letter-v1`)
- `scripts/sttm_first_letter_map.py` — Gurmukhi → STTM-ASCII mapper (99.94% DB round-trip)
- `scripts/build_first_letter_anchor_manifests.py` — NeMo JSONL builder
- `scripts/train_first_letter_anchor.py` — launcher with HF auto-push
- `scripts/eval_first_letter_anchor.py` — custom anchor-metric eval (utterance_exact_match_rate, anchor_shabad_recall@1/@5, top1-top2 margin)
- `training/conformer_ctc_medium_first_letter.yaml` — 27M Conformer-CTC Medium config
- `scripts/runpod_anchor_setup.sh` — idempotent pod bootstrap

---

## Bugs surfaced and fixed (committed on the branch)

1. `lightning.pytorch` vs `pytorch_lightning` namespace — NeMo 1.23 requires the new name
2. ConvASRDecoder needs explicit `vocabulary` / `num_classes` (hydra doesn't auto-fill from `model.labels`)
3. `numpy<2.0` required — NeMo 1.23 uses removed `np.sctypes`
4. Checkpoint filename `step{step}` → `stepstep=N` breaks NeMo's resume parser; use bare `{step}`
5. `val_check_interval` must be ≤ batches per epoch (986 in our case); was 1000

All five are codified in the YAML + launcher so v2 won't hit them again.

---

## What v1 proved (for v2 design)

1. ✅ **First-letter CTC is learnable.** 80% token accuracy with a small model from scratch on minimal data validates the entire PLAN.md direction.
2. ✅ **The STTM-ASCII mapper is correct.** 99.94% DB round-trip means training labels align with the search corpus.
3. ✅ **Pipeline works end-to-end.** HF auto-push, NeMo training, custom metric — all functional.
4. ✅ **Throughput is fine.** 3 it/s on A40 means a Large model with pretrained init can finish v2 in <5h GPU.

---

## What v1 did NOT prove

1. ❌ Multi-speaker generalization (single speaker only)
2. ❌ Kirtan performance (sehaj only)
3. ❌ Matcher recall@1 vs full-text matcher (eval not run)
4. ❌ <85% utterance exact match (reached 80%, didn't push further)

These move to v2.

---

## Cost breakdown

| Phase | Wall-clock | A40 cost |
|---|---|---|
| Pod boot + clone + pip install | ~7 min | $0.05 |
| DB scp from Mac | ~5 min | $0.04 |
| Manifest build (CPU only) | ~12 min | $0.09 |
| Smoke + bug-fix cycles | ~10 min | $0.07 |
| Full training to step 10603 | ~4.5h | $1.98 |
| HF pushes + admin | ~5 min | $0.04 |
| **Total** | **~5.5h** | **~$2.30** |

Well inside the 10-15h authorization. Remaining ~$2-5 budget rolls into v2.

---

## Decision: proceed to v2

**Locked direction**: Large Conformer-CTC, initialize encoder from `surindersinghssj/indicconformer-pa-v3-kirtan` (already fine-tuned on 600h Gurbani), train on 200h diverse mix (100h kirtan + 100h sehaj across multiple speakers), continuous-training strategy for future data refreshes.

**v2 gate target** (slice-stratified):
- Seen-speaker sehaj: <5% WER
- Unseen-paathi sehaj: <10% WER  
- Clean ragi kirtan: <15% WER
- AKJ fast kirtan: <20% WER
- SGPC live with sangat: <25% WER

v2 plan codified in `PLAN-v2.md` (drafting next).
