# PLAN-v3 — Kirtan-only push toward 80-90% recall@1

> Locked 2026-05-28 after the v2 matcher benchmark showed window voting hurts on short kirtan clips. The model — not the matcher — is the bottleneck. Path to 80-90% requires multiple training waves.

## Stack of waves (each builds on the previous)

### v3.0 — kirtan-only fine-tune from v2.0 encoder (IN FLIGHT)
- 200h kirtan-only from `gurbani-kirtan-yt-captions-300h-canonical`
- Init encoder from `surt-anchor-ctc-large-v2`, fresh CTC head
- 10k steps, batch 16 × accum 8, val on v3_kirtan_eval (677 clips)
- Expected: **recall@1 50-65%** (vs v2's 22-32%)
- Cost: ~5h L4 GPU

### v3.1 — Skeleton-CTC multi-task head (NEXT)
Add a second CTC head to the same encoder. Two outputs per word:
- Head A (existing): first letter `s|n|k|p`
- Head B (new):     consonant skeleton `st|nm|krt|prk`

At search time:
1. Head A output → 4-gram overlap on `first_letters` → candidate pool of top-50 shabads
2. Head B output → edit-distance rerank against pre-computed skeletons of those 50
3. Top-1 = highest rerank score with margin > threshold

Why: first letter alone loses information ("s" matches dozens of shabads). Skeleton adds enough identity to disambiguate collisions.

**Code lift** (estimated ~400 lines):
- `training/conformer_ctc_large_kirtan_skeleton_v3_1.yaml` — two ConvASRDecoder heads, joint CTC loss (sum of both heads' losses)
- `scripts/build_skeleton_anchor_targets.py` — extend manifests with `skeleton_text` column derived from `scripts/canonical/gurmukhi_skeleton.py`
- `scripts/anchor_matcher_v3.py` — two-head decode + rerank matcher
- Init from v3.0 encoder (load encoder only, both heads fresh)

Expected: **recall@1 65-75%**. Wave needed for 80%+.

### v3.2 — Rolling-window matcher (for STREAMING use only)
Once a clip has > 15 anchor chars (longer than a single shabad line), slide a 15-20 char window with stride 5, vote across windows with continuity bias. NOT useful for the current 677-clip offline eval (clips are too short), but critical for live transcribe-app integration.

### v3.3 — Fresh diverse data
- 50h AKJ samagam from akj.org/keertan.php (per `kirtan.txt` allowlist)
- 50h SGPC live (per `scripts/v3_enumerate_sgpc.py` with ragi-balanced sampling)
- Continuous training v3.1 → v3.3 with 20% replay sample
- Expected: **recall@1 75-85%**

### v3.4 — n-gram LM rescoring (if v3.3 hasn't hit gate)
Build a 5-gram LM from `database.sqlite.lines.first_letters` (no external corpus). Rescore CTC beam outputs against LM. Should add ~5pp.

### v3.5 — Audio augmentation push
Reverb sim + multi-speaker mixup + pitch/tempo. Conformer-CTC is robust to this in NeMo. Final ~3-5pp.

## Realistic ceiling

| Wave | Cumulative recall@1 (kirtan) | Cumulative GPU |
|---|---|---|
| v2.0 (already shipped) | 22-32% | 5h (done) |
| v3.0 (in flight) | 50-65% | +5h |
| v3.1 (skeleton head) | 65-75% | +5h |
| v3.3 (more data) | 75-85% | +8h |
| v3.4 (LM rescore) | 80-88% | +1h |
| v3.5 (augmentation) | 82-90% | +5h |

Total path to 82-90%: ~25h GPU = ~$10. Realistic, not magic.

## Hard truths

- Single-pass solutions for 80-90% kirtan recall@1 don't exist on 200h with current architecture
- Skeleton-CTC multi-task is the **single biggest lever** after v3.0 — do it first
- Data scaling has diminishing returns past 300h without architectural changes
- Distillation to Medium for mobile is a v4 problem, not blocking accuracy
