# PLAN-v2 — First-Letter Anchor CTC v2 (Large, continuous training)

> **Status:** Locked direction as of 2026-05-26. Supersedes PLAN.md's M1.3 architectural choice (Medium / from-scratch). v1 results documented in `REPORT-v1.md`.

---

## What v1 told us (and why v2 changes course)

v1 took a 27M-param Conformer-CTC Medium **from scratch** on **66h of single-speaker sehaj** and reached:
- ✅ In-distribution (same speaker): 100% on 15-clip OOD test, ~80% on 300-clip val
- ⚠️ Unseen-paathi sehaj: 27% token accuracy
- ❌ Unseen kirtan: 9% (random chance)

The architecture works. The data doesn't generalize. v2 fixes both.

---

## v2 locked decisions

| Decision | v1 (superseded) | v2 (locked) |
|---|---|---|
| Architecture | Conformer-CTC Medium ~32M | **Conformer-CTC Large ~120M** (matches IndicConformer-pa-large shape) |
| Encoder init | from scratch | **Init from `surindersinghssj/indicconformer-pa-v3-kirtan`** (already trained on 600h Gurbani) |
| Decoder | new CTC head on STTM-ASCII vocab | same — `model.change_vocabulary(["A","B",...,"|"], decoder_type="ctc")` |
| Data scope | 66h single-speaker sehaj | **All public canonical Gurbani datasets, diversity-filtered (~250–300h)** |
| Training mode | one-shot to convergence | **Continuous: load latest .nemo → add new data → fine-tune → push → repeat** |
| Eval | one in-dist val_wer | **Slice-stratified eval** (5 buckets, each with its own gate) |
| Forgetting protection | n/a (first run) | **Replay sampling** (20% of previous mix in every refresh) + decaying encoder LR |

---

## Data plan — all public, filtered, diversity-balanced

### Sources (public, cc-by-4.0, surindersinghssj)

| Dataset | Raw rows | Raw hours | Use |
|---|---|---|---|
| `gurbani-sehajpath` | 63.1K | ~66h | sehaj baseline (1 speaker) |
| `gurbani-sehajpath-yt-captions-canonical` | 63K | ~30–40h post-filter | multi-paathi sehaj |
| `gurbani-kirtan-yt-captions-300h-canonical` | 207.5K | ~200–280h post-filter | multi-ragi kirtan |
| **Total raw** | **~333K rows** | **~300–400h** | — |

### Filter chain (per-row)

```python
keep_row = (
    # canonical match quality (where available)
    row.get("canonical_match_score", 1.0) >= 0.70
    and row.get("training_usable", True) is True

    # text quality
    and passes_simran_filter(row.text)
    and not row.get("is_simran", False)
    and len(row.text.split()) >= 2

    # audio quality
    and 0.5 <= row.duration <= 18.0

    # anchor sanity (first-letter task specific)
    and 0.15 <= row.duration / len(search_anchor) <= 1.5
    and len(search_anchor) >= 3
)
```

Expected drop rate after filtering: ~10–20% per source.

### Diversity caps (post-filter)

Stop any one source/speaker/ragi from dominating the gradient:

```python
# Per-source caps to land near 250-300h with good balance
caps = {
    "gurbani-sehajpath":                          {"max_hours": 30, "by": "video_id"},   # cap single-speaker dominance
    "gurbani-sehajpath-yt-captions-canonical":   {"max_hours_per_video": 1.5},            # 30-40h spread across ~30 paathis
    "gurbani-kirtan-yt-captions-300h-canonical": {"max_hours_per_video": 0.5,             # ~200h spread across ~400 ragis
                                                  "max_hours_per_kirtan_type": 80,        # ragi/akj/sgpc balance
                                                  "min_kirtan_type_coverage": ["ragi","akj","sgpc"]},
}
```

Round-robin sample across (video_id, kirtan_type) buckets with seed=42 for reproducibility.

### Target v2 mix

| Slice | Hours | % of train |
|---|---|---|
| Sehaj original (single speaker) | 30h | 12% |
| Sehaj YT (unseen paathis) | 30h | 12% |
| Kirtan ragi (multi-voice studio) | 80h | 32% |
| Kirtan AKJ (fast + sangat) | 60h | 24% |
| Kirtan SGPC (live + reverb) | 50h | 20% |
| **v2 total** | **~250h** | **100%** |

### Replay sampling for v2.x refreshes

When new data is added in v2.1, v2.2, etc., always include **20% of v2's manifest sample** (frozen, seed=42) so the encoder doesn't forget what it learned. Across refreshes the encoder LR halves; the decoder LR stays constant.

---

## Architecture

Match IndicConformer-pa-large's encoder exactly so `change_vocabulary` works without surgery.

```yaml
# training/conformer_ctc_large_first_letter.yaml
encoder:
  _target_: nemo.collections.asr.modules.ConformerEncoder
  feat_in: 80
  n_layers: 18
  d_model: 512
  n_heads: 8
  ff_expansion_factor: 4         # d_ff = 2048
  self_attention_model: "rel_pos"
  conv_kernel_size: 31
  subsampling: "striding"
  subsampling_factor: 4

decoder:
  _target_: nemo.collections.asr.modules.ConvASRDecoder
  feat_in: ${model.encoder.d_model}
  vocabulary: ${model.labels}    # 38 STTM-ASCII chars + delimiter
  num_classes: -1
```

Total: ~120M params. Larger than v1 by ~4× but matches the pretrained shape.

---

## Training recipe

### v2.0 initial run

```bash
python scripts/train_anchor_continuous.py \
  --config training/conformer_ctc_large_first_letter.yaml \
  --init-from-hf surindersinghssj/indicconformer-pa-v3-kirtan \
  --hf-model-repo surindersinghssj/surt-anchor-ctc-large-v2 \
  --hf-push-every 1000 \
  --max-steps 8000        # half of v1's cap — pretrained init means fewer steps needed
```

Expected wall-clock on A40: **~3–5h**.

### v2.x refresh template

```bash
python scripts/train_anchor_continuous.py \
  --init-from-hf surindersinghssj/surt-anchor-ctc-large-v2 \
  --manifest path/to/v2.1_refresh.jsonl \
  --replay-sample-frac 0.20 \
  --encoder-lr 2.5e-5 \    # half of v2.0's 5e-5
  --decoder-lr 3e-4 \
  --max-steps 4000
```

---

## Eval — slice-stratified gates

Five evaluation buckets, each with its own quality bar. v2 is GO if all gates pass.

| Bucket | Eval source | v2 gate (WER) | Why this matters |
|---|---|---|---|
| 1. Seen-speaker sehaj | `gurbani-sehajpath` validation | **< 5%** | Sanity — should be trivial with pretrained init |
| 2. Unseen-paathi sehaj | `gurbani-sehajpath-yt-captions-eval-canonical` | **< 10%** | Speaker generalization on the easy domain |
| 3. Clean ragi kirtan | `gurbani-kirtan-yt-captions-eval-canonical` filtered to `kirtan_type=ragi` | **< 15%** | The realistic production target |
| 4. AKJ kirtan | same eval, filtered to `kirtan_type=akj` | **< 20%** | Fast-tempo + sangat overlap |
| 5. SGPC live | `gurbani-kirtan-eval-pure-canonical` SGPC-tagged | **< 25%** | Hardest case; reverb + PA + sangat |

Also report:
- `anchor_shabad_recall@1` and `@5` against `database.sqlite.lines.first_letters` per bucket — this is the real production metric
- `top1 - top2` overlap margin (locking confidence)
- per-position token-match curve

---

## Continuous training plan beyond v2.0

| Version | What's added | New data scope | Trigger |
|---|---|---|---|
| **v2.0** | Initial run | ~250h public | After PLAN-v2 lands |
| **v2.1** | Additive HF push fix (project memory item) | same data | Bug fix |
| **v2.2** | Add 100h newly-collected AKJ | +100h | When `@akjdotorg` scrape is filtered/canonical |
| **v2.3** | Add 50h SGPC live with ragi diversity | +50h | When SGPC v3 enumerate completes |
| **v2.4** | Add YT-caption diverse pool (more paathis) | +X h | Ongoing collection |
| **v3** | Distill v2.x → Medium for mobile | n/a | When v2.x quality is locked |

Each refresh: load latest `.nemo` → manifest with new + 20% replay → train ~1–2h → push to HF with version tag.

---

## Files to create for v2

1. **`scripts/build_v2_anchor_manifests.py`** — pulls all three public canonical sources, applies filter chain + diversity caps, writes 5 stratified eval manifests + 1 train manifest
2. **`scripts/train_anchor_continuous.py`** — supports `--init-from-hf <repo>` (uses `change_vocabulary` for first time, `restore_from` for refreshes), `--replay-sample-frac`, decaying encoder LR, HF auto-push
3. **`training/conformer_ctc_large_first_letter.yaml`** — Conformer-Large arch matching IndicConformer-pa
4. **`scripts/eval_anchor_stratified.py`** — runs the 5-bucket eval with anchor_shabad_recall@1/@5

---

## Risk register

| Risk | Mitigation |
|---|---|
| `change_vocabulary` on hybrid CTC-RNNT model breaks — needs special handling | Test on smoke first; fall back to extracting encoder weights and building a fresh EncDecCTCModel if needed |
| Kirtan labels still noisy after filtering | Match-score floor 0.70 + alignment-ratio sanity check; track decision flag distribution per-bucket in the manifest |
| 250h is too much for one A40 run | Caps at 8000 steps via `max_steps`; can early-stop and resume |
| Hyperparameters wrong for fine-tune vs from-scratch | Default to encoder_lr 5e-5 (10× lower than v1's), decoder_lr 3e-4 (similar to v1) — these are NeMo IndicConformer fine-tune defaults |
| HF push race conditions during continuous training | Per-version repo tags (`surt-anchor-ctc-large-v2.0`, `v2.1`, ...) — push to a fresh tag, not overwrite |
| Decode is fine but matcher recall@1 lags | `eval_anchor_stratified.py` reports both; if recall@1 lags but val_wer is good, the matcher needs tuning, not the model |

---

## Budget

| Phase | GPU hours | Cost (A40 spot) |
|---|---|---|
| v2.0 initial training | ~4h | $1.80 |
| Eval + reports | ~30 min | $0.20 |
| Headroom for retries | ~3h | $1.30 |
| **v2.0 total** | **~7.5h** | **~$3.30** |

Well under the 10–15h GPU authorization. ~$5 already used in v1.

---

## Decision gate (after v2.0)

- **GO**: all 5 bucket gates met → ship as anchor matcher backbone, build `apps/transcribe/anchor_searcher.py` (PLAN.md M1.4), schedule v2.2 with new AKJ data
- **PARTIAL**: 3+ bucket gates met → ship for production sehaj path, defer kirtan-locking; investigate which buckets failed and target data collection accordingly
- **NO-GO**: ≤2 bucket gates met → investigate. If sehaj buckets pass and kirtan all fail, data quality issue. If everything plateaus, may need v3 = even more data + distillation; reconsider the anchor approach in favor of the skeleton-CTC multi-task model (PLAN.md alt #3)

---

## Out of scope for v2

- Skeleton-CTC multi-task head — only if v2 plateaus on collisions
- Mobile distillation — defer to v3 once quality is locked
- Live wiring into `apps/transcribe/` — defer to M1.4 once GO is locked
- Adding non-public datasets — explicit scope decision: only `surindersinghssj/*` cc-by-4.0 data
