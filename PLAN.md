# CURRENT PLAN - Surt Anchor CTC: first-letter shabad locking

> **READ THIS FIRST.** This file is the single source of truth for the current direction.
> Any agent picking up this repo should align with the first-letter anchor CTC plan below.
>
> **Superseded plans (do NOT execute as the active path):**
> - [`surt_training_plan.md`](surt_training_plan.md) - v1 (sehaj path only)
> - [`surt_v2_training_plan.md`](surt_v2_training_plan.md) - v2 (HF-hosted kirtan dataset ~28h, 5k samples)
> - [`surt_v3_training_plan.md`](surt_v3_training_plan.md) - Whisper-small full-transcript retrain
> - The old 500h kirtan dataset plan preserved lower in this file is now reference material only.

## Active Goal

Build a small custom CTC model, roughly **30-50M parameters**, whose job is not
full transcription. Its job is to emit a **delimited STTM-ASCII first-letter
sequence**: one acoustically grounded first-letter token per spoken word, with an
explicit word-boundary delimiter for CTC training.

Target behavior:

```text
Audio:            ਸਤਿ ਨਾਮੁ ਕਰਤਾ ਪੁਰਖੁ ...
Training target:  s|n|k|p...
Search target:    snkp...
Search:           compact STTM-ASCII anchor -> database.sqlite.lines.first_letters
App:              lock shabad, track current pangti, optionally push to STTM
```

This is intentionally a different task from full ASR:

- Full ASR target: `audio -> exact Gurmukhi words`
- Anchor CTC target: `audio -> STTM-ASCII first-letter sequence -> canonical search`

The intended outcome is a **~32M-param NeMo Conformer-CTC Medium** model, chosen
after a scratch-vs-pretrained acoustic encoder ablation, wired into the matcher
as an alternative or pre-filter lookup path. This removes the need for the
600 MB MuRIL+FAISS stack on the anchor path and optimizes directly for shabad
locking.

## Context

Current Gurbani ASR (`surt-small-v3`) produces full Gurmukhi transcripts and
feeds them into the matcher. It is useful, but not anchor-quality: clean sehaj is
around 5% CER and kirtan around 28% CER. The full-text path also carries the
MuRIL+FAISS inference dependency.

The anchor CTC model narrows the task aggressively:

- Output vocab collapses from full Gurmukhi text to roughly 30-50 STTM-ASCII
  symbols plus one delimiter symbol.
- One output token represents one source word's first letter; the delimiter makes
  word positions explicit for CTC and evaluation.
- `database.sqlite.lines.first_letters` already contains the lookup target for
  all **141,264** SGGS lines.
- `apps/transcribe/retriever.py` literal helpers (`_char_4grams`, `_overlap`)
  are string-agnostic and can score first-letter strings too.

Two anchor forms are required:

```text
training_anchor: s|n|k|p
search_anchor:   snkp
```

The model predicts the delimited `training_anchor`. Search strips delimiters and
any training-only boundary markers before querying `lines.first_letters`.

## Locked Decisions

- **Target vocab:** delimited STTM-ASCII first letters for training, convertible
  losslessly to compact `database.sqlite.lines.first_letters` anchors for search.
- **Architecture:** NeMo Conformer-CTC Medium, target ~32M params.
- **Training ablation:** run from-scratch CTC and pretrained-acoustic-encoder
  CTC-head variants before committing to a final backbone.
- **HF reuse:** existing Hugging Face datasets/models owned by this project may
  be reused for manifests, pretrained encoder initialization, baselines, and
  comparison runs.
- **HF persistence:** intermediate datasets, manifests, checkpoints, and eval
  reports should be pushed incrementally to Hugging Face so progress can be
  inspected without waiting for the full experiment to finish.
- **Data v1:** small, deliberate experiment mix:
  clean sehaj path plus **30-40h of diverse, high-confidence kirtan**.
- **Dataset philosophy:** rough hours are secondary. Correct labels, diversity,
  and failure-mode coverage matter more than scale.
- **Primary metric:** custom anchor token-match, collision-aware shabad recall,
  and product lock metrics, not WER/CER.
- **Linear:** milestone is `First-letter anchor CTC v1` under the existing Gurbani
  ASR Linear project once approved.

## Feasibility

Expected strengths:

- CTC is a natural fit for monotonic audio-to-label alignment.
- A delimited first-letter target is more CTC-friendly than compact anchors
  because repeated adjacent initials remain distinct (`s|s` cannot collapse into
  one `s` the way compact `ss` can under CTC decoding).
- Sehaj data is studio-quality, well-aligned, and already available.
- Existing NeMo CTC infra in `training/indicconformer_pa_v3_kirtan.yaml` and
  manifest patterns in `scripts/build_indicconformer_manifests.py` cover much of
  the scaffolding.
- First-letter matching tolerates partial anchors better than full transcript WER
  suggests. A 10-letter anchor can survive 1-2 wrong tokens and still retrieve
  well via 4-gram overlap/window voting.

Main risks:

- Per-utterance exact-anchor-match of 100% is not realistic. A reasonable sehaj
  target is 85-95% utterance-exact-match.
- First-letter sequences collide across common Gurbani phrases, especially short
  windows and short lines.
- Kirtan repeats, simran, sangat vocals, and stretched vowels can cause insertions/deletions.
- Too much easy sehaj could hide kirtan failure modes; v1 must include a small
  but diverse kirtan slice from the start.
- Non-spoken STTM/database markers (`<`, `>`, `]`, digits, verse notation) can
  poison acoustic supervision if trained as ordinary spoken tokens.

Realistic experiment target:

- `utterance_exact_match_rate >= 85%` on canonical sehaj eval.
- `anchor_shabad_recall@1 >= current matcher top-1 recall` on the same eval clips,
  with recall@5, margin, wrong-lock rate, and time-to-lock reported.
- Kirtan pilot metrics do not need to beat final production baselines yet, but
  they must show the approach is learnable and worth scaling.

## Experiment Dataset Mix

This is not an end-product dataset. It is a compact proof dataset designed to
answer one question: **is first-letter CTC worth pursuing for Gurbani locking?**

Use a deliberately selected mix:

- **Sehaj clean core:** enough clean sehaj path to teach stable Gurbani word-order
  and first-letter emission.
- **Extra sehaj, intelligently sampled:** add more sehaj only where it increases
  speaker, recording, paath speed, or pronunciation diversity. Do not add easy
  duplicates just to increase hours.
- **Kirtan pilot:** add **30-40h** of high-confidence kirtan, balanced across:
  - studio ragi kirtan: clean melody/harmonium/tabla
  - AKJ/fast kirtan: speed, repetition, sangat participation
  - SGPC/live kirtan: reverb, PA, background noise, live ambience

Kirtan selection rules:

1. Prefer clips where the text can be confidently mapped to canonical Gurbani.
2. Avoid long katha, announcements, tabla-only intros, ardaas, and non-Gurbani speech.
3. Cap per-ragi/per-source contribution so one voice or recording style cannot dominate.
4. Keep difficult but valid examples: stretched vowels, repeated pangtis, sangat overlap.
5. Preserve source tags so eval can report sehaj vs ragi vs AKJ vs SGPC separately.
6. Include kirtan in training only when transcript-to-canonical matching is
   high-confidence. If canonical alignment is uncertain, keep the clip for a
   diagnostic eval bucket or manual review, not supervised CTC training.

Suggested first experiment size:

```text
Sehaj:  40-80h, sampled for cleanliness and variety
Kirtan: 30-40h, sampled for diversity and confidence
Total:  70-120h
```

If the model cannot learn useful anchors on this mix, a larger dataset is unlikely
to fix the core approach. If it works, then scale the same curation policy.

## Alternative Target Ablations

The active path remains delimited first-letter CTC. Run only small ablations here
until the first-letter baseline is measured.

1. **Delimited first-letter CTC (primary):**
   - Target: one first-letter token per spoken word.
   - Strength: shortest CTC sequence, smallest vocab, fastest search.
   - Risk: collisions on short/common phrases.

2. **Gurbani skeleton CTC (best fallback):**
   - Target: a compact consonant/akhar skeleton per word, with word delimiters.
   - Example shape: first consonant plus selected consonant skeleton, not full
     orthographic Gurmukhi.
   - Strength: still easier than full transcription, but carries more identity
     than first letters and should reduce collisions.
   - Risk: harder labels and more spelling/normalization edge cases.

3. **Multi-task anchor model:**
   - Shared encoder with two CTC heads: first-letter anchor head plus skeleton
     head.
   - Search uses first-letter head for fast candidate generation and skeleton
     head for rerank/margin.
   - Strength: preserves the simple anchor path while adding disambiguation only
     when needed.
   - Risk: more training and eval complexity.

4. **Shazam-style audio fingerprinting (diagnostic, not primary):**
   - Useful for identifying the exact same recording or near-duplicate audio.
   - Weak for Gurbani search across different ragis, keys, tempo, sangat vocals,
     reverb, and live paath because the same shabad can sound very different.
   - Keep as a dedup/provenance tool, not the main shabad-locking model.

If first-letter CTC learns but collides too often, try the multi-task
first-letter+skeleton model before returning to full transcription.

## Files To Create

### `scripts/sttm_first_letter_map.py`

Self-contained STTM-ASCII mapping:

```python
gurmukhi_text_to_training_anchor(text: str) -> str
training_anchor_to_search_anchor(anchor: str) -> str
gurmukhi_text_to_search_anchor(text: str) -> str
```

Build it by dumping `(gurmukhi, first_letters)` pairs from
`database.sqlite.lines`, inferring the per-consonant mapping, then freezing the
mapping table. It should handle vowel-initial words and denukta normalization
informed by `scripts/canonical/gurmukhi_skeleton.py`.

Important normalization rule:

- The **training anchor** should contain only acoustically meaningful first-letter
  tokens plus the delimiter.
- Non-spoken database/STTM markers such as sentence markers, verse endings,
  digits, brackets, and punctuation should be stripped or mapped in a documented
  way before training.
- The **search anchor** should be compact and DB-compatible after delimiter and
  training-only marker removal.

Required tests:

- round-trip at least **99%** of the 141,264 DB rows after documented DB
  normalization
- verify repeated initials survive training-target conversion, e.g. compact `ss`
  becomes `s|s` and not `s`
- verify `training_anchor_to_search_anchor()` emits a string comparable to
  `database.sqlite.lines.first_letters`

### `scripts/build_first_letter_anchor_manifests.py`

Build NeMo JSONL manifests where `text` is the delimited STTM-ASCII first-letter
training sequence.

Pipeline:

1. Load selected sehaj and kirtan HF/local datasets using the retry/download
   pattern from `surt/data.py`.
2. Normalize source text using `surt.data.normalize_gurbani_text`.
3. Apply `passes_simran_filter` from `scripts/build_indicconformer_manifests.py`.
4. Convert normalized Gurmukhi to STTM-ASCII anchor text.
5. Preserve source metadata (`source_type`, `kirtan_type`, `speaker/ragi` when
   available) for stratified eval.
6. Preserve label confidence (`canonical_match_score`, `label_confidence`,
   `alignment_source` when available). Low-confidence kirtan labels are excluded
   from train and retained only for diagnostics/manual review.
7. Emit NeMo-style rows where `text` is the delimited training anchor and
   `search_anchor` is optional metadata for eval/debug:

```text
{"audio_filepath": "...", "duration": 7.34, "text": "s|n|k|p", "search_anchor": "snkp"}
```

Primary outputs:

- `data/manifests/anchor_first_letter_train.jsonl`
- `data/manifests/anchor_first_letter_val.jsonl`
- `data/manifests/anchor_first_letter_eval_by_domain.jsonl`

### `training/conformer_ctc_medium_first_letter.yaml`

NeMo Conformer-CTC Medium config, with a required scratch-vs-pretrained ablation.

Key choices:

- Scratch run: `init_from_nemo_model: null`
- Pretrained-acoustic run: initialize the encoder from a compatible Punjabi/Indic
  or multilingual CTC/ASR checkpoint when available, including existing project
  models on Hugging Face, replace the decoder with the first-letter CTC head, and
  compare early learning curves plus anchor metrics.
- `EncDecCTCModel` or `EncDecCTCModelBPE` with a char tokenizer over STTM symbols.
- Encoder target: 16 layers, `d_model: 256`, `n_heads: 4`, `d_ff: 1024`.
- Decoder: linear CTC head over the anchor vocab.
- Trainer: 80-120 epochs on sehaj, bf16 where available.

### Hugging Face Artifacts

Push incrementally rather than waiting for the final model:

- anchor manifest snapshots after each data-curation pass
- train/val/eval split summaries with source/domain counts
- mapping and normalization audit reports
- 100-step smoke checkpoints and decoded samples
- full-training checkpoints at useful intervals
- eval reports by domain and collision bucket
- final model card documenting target format, delimiter, DB normalization, and
  intended search pipeline

Suggested repo names:

- dataset: `surindersinghssj/gurbani-anchor-first-letter-v1`
- model: `surindersinghssj/surt-anchor-ctc-first-letter-v1`
- eval/artifacts, if separated: `surindersinghssj/surt-anchor-ctc-reports-v1`

Each push should include enough metadata to reproduce the result: git commit,
manifest hash, source dataset names/revisions, target normalization version, and
training config.

### `scripts/eval_first_letter_anchor.py`

Evaluate against canonical sehaj eval plus the curated kirtan pilot eval split.

Reports:

- `utterance_exact_match_rate`
- `mean_pos_match_rate`
- `anchor_shabad_recall@1`
- `anchor_shabad_recall@5`
- top-1 vs top-2 margin
- false/wrong-shabad lock rate
- time-to-lock on rolling windows
- current-pangti accuracy after shabad lock
- unlock recovery time after wrong/noisy windows
- per-position token-match rate
- collision bucket breakdown by anchor length and duplicate-anchor count
- side-by-side recall comparison against the current full-text matcher
- breakdown by `source_type` / `kirtan_type`

CER/WER may be included only as debug aids, never as primary shipping metrics.

### `apps/transcribe/anchor_searcher.py`

Feature-flagged wrapper:

```python
search_by_anchor(anchor: str, n: int = 5) -> list[dict]
```

It should accept delimited or compact anchors, normalize to the compact
DB-compatible search form, and reuse `_char_4grams` and `_overlap` logic against
precomputed 4-gram sets from `database.sqlite.lines.first_letters`. Keep the
existing `ShabadRetriever.search_topn` path untouched until v1 ships.

Search should be window-aware:

- line-level search for simple eval/debug
- rolling multi-line/shabad-window search for live locking
- continuity bias once a shabad is locked, so short ambiguous anchors do not
  bounce the pointer across unrelated shabads

## Reuse, No Changes Initially

- `surt/data.py` - `normalize_gurbani_text`, HF loading patterns.
- `scripts/build_indicconformer_manifests.py` - simran filter and manifest write pattern.
- `scripts/canonical/gurmukhi_skeleton.py` - denukta/skeleton edge-case guidance.
- `apps/transcribe/retriever.py` - `_char_4grams` and `_overlap`.
- `database.sqlite` - `lines.first_letters` as vocab source and lookup target.

## Custom Metric

Training still uses CTC loss for gradients, but validation, checkpoint selection,
early stopping, and the GO/NO-GO gate use the anchor metric:

```python
def anchor_token_match(pred: str, ref: str) -> dict:
    """
    Score predicted anchor vs ground-truth anchor at the word-position level.
    Each delimiter-separated token is one source word's first letter.

    Returns:
      exact_match: bool
      per_position: list[bool]
      pos_match_rate: float
      shabad_hit: bool
    """
```

Aggregate metrics:

- `utterance_exact_match_rate`: fraction of clips where `pred == ref`.
- `mean_pos_match_rate`: average position-match rate using reference length as denominator.
- `anchor_shabad_recall@1`: predicted compact search anchor's top DB lookup lands
  on the same shabad as the reference compact search anchor's top DB lookup.
- `anchor_shabad_recall@5`: correct/reference shabad appears in top 5.
- `top1_margin`: top-1 score minus top-2 score; low margin means do not lock.
- `false_lock_rate`: live-window locks to the wrong shabad.
- `time_to_lock`: seconds of audio needed before a stable correct lock.
- `current_pangti_accuracy`: once locked, whether the pointer is on the correct
  line/pangti.

Why not CER/WER:

- CER over full Gurmukhi penalizes characters the anchor model never attempts to output.
- WER on one-letter-per-word output is degenerate.
- The product question is: did the anchor land on the right shabad?

Implementation details:

- Tokenization for the metric: split by the configured delimiter, after stripping
  CTC blanks and whitespace.
- Order matters; compare by position, not bag-of-tokens.
- `shabad_hit` uses `_overlap()` + `_char_4grams()` against
  `database.sqlite.lines.first_letters` after converting predictions to compact
  search anchors.
- Wire the metric into the NeMo validation callback so checkpoint selection uses
  anchor recall, false-lock rate, and `utterance_exact_match_rate`, not `val_loss`
  alone.

## Linear Milestones

| Milestone | Goal | Output | Gate |
|---|---|---|---|
| M1.1 Data prep and target validation | Derive STTM mapping, build mixed sehaj+kirtan first-letter manifests, spot-check labels | `sttm_first_letter_map.py`, train/val JSONL | DB round-trip >=99%, repeated initials preserved, 100 random clips clean across domains |
| M1.2 Training scaffold and smoke run | Author NeMo configs and run 100-step smoke tests on tiny subset | Scratch YAML + pretrained-ablation YAML + smoke checkpoints | Loss decreases, decode is delimiter-valid and anchor-like, params 30-50M |
| M1.3 Full experiment training | Train on curated 70-120h sehaj+kirtan mix | Best anchor CTC checkpoint | Sehaj exact >=85%; kirtan shows learnable anchor signal |
| M1.4 Eval and matcher integration | Run custom eval and build feature-flagged anchor searcher | Eval report + `anchor_searcher.py` | recall@1 >= current matcher, recall@5/margin/false-lock acceptable |
| M1.5 Decision gate | Decide ship / revise using custom and product metrics only | GO/NO-GO note | GO if exact >=85%, recall >= current matcher, and wrong-lock/time-to-lock are production-safe |
| M1.6 Scale-or-stop decision | Decide whether to scale data/model after pilot | Recommendation | Continue only if kirtan errors look data-fixable |

## Verification Plan

- Mapping unit test: DB round-trip should pass on at least 99% of 141,264 lines
  after documented DB normalization.
- Delimiter unit test: repeated adjacent initials remain distinct in the training
  target and collapse only when deliberately converted to search form.
- Manifest sanity: `wc -l data/manifests/anchor_first_letter_train.jsonl` matches
  expected row count after filtering.
- Label-confidence sanity: all train rows meet the kirtan canonical-confidence
  floor; uncertain rows appear only in diagnostic/manual-review outputs.
- HF artifact sanity: each major stage has a pushed dataset/checkpoint/eval
  snapshot with commit, config, manifest hash, and source revisions recorded.
- Smoke training: 100 steps, loss drops substantially, sample decode shows ASCII
  delimiter-valid anchor sequence.
- Eval: `scripts/eval_first_letter_anchor.py --model <checkpoint>` reports
  `utterance_exact_match_rate`, `mean_pos_match_rate`, and
  `anchor_shabad_recall@1/@5`, margin, wrong-lock rate, time-to-lock, and
  current-pangti accuracy against current matcher.
- Matcher integration: query a known sehaj utterance anchor and confirm top-1
  returns the correct `shabad_id`.
- Live dry run: feature-flag anchor path in `apps/transcribe/app.py` and confirm
  sehaj lock fires faster or more confidently than full-text-only path.

## Risks And Mitigations

| Risk | Mitigation |
|---|---|
| STTM-ASCII mapping misses edge characters | M1.1 DB round-trip test catches gaps before training |
| 32M params too small | M1.5 can scale to ~50M before declaring the approach dead |
| Kirtan diversity is too small to generalize | Include 30-40h kirtan in v1, balanced across ragi/AKJ/SGPC |
| Kirtan labels are noisy | Train only on high-confidence canonical matches; keep uncertain clips for diagnostics/manual review |
| Easy sehaj dominates the loss | Domain tags and sampling weights keep kirtan visible during training/eval |
| Compact anchors lose repeated initials under CTC | Train with delimiter-separated word initials and strip delimiters only for search |
| Non-spoken DB markers become fake acoustic labels | Strip/map visual markers before training; document conversion to DB search form |
| Per-letter errors compound on long utterances | 4-gram overlap and window voting tolerate partial anchors; M1.4 verifies this |
| Short anchors collide across shabads | Use recall@5, margin thresholds, rolling windows, and continuity bias before locking |
| Anchor path conflicts with current matcher | Keep `anchor_searcher.py` feature-flagged and leave `ShabadRetriever.search_topn` unchanged |

## Archived Reference - old 500h full-transcript plan

The content below is preserved as reference for source allowlists, chunking lessons, and
data-cleaning gotchas. It is **not** the active execution plan.

---

## Goal

Build a **500-hour kirtan training dataset** from the **allowlisted sources in [`kirtan.txt`](kirtan.txt)** (SikhNet ragi playlists, AKJ archive, SGPC recorded archive), labeled end-to-end with **Gemini 2.5 Flash Lite**, and retrain Surt on it.

> **Source allowlist is hard.** Only the URLs in [`kirtan.txt`](kirtan.txt) are in scope for v3. Do NOT pull from any YouTube channel, SoundCloud account, or other site — even if previous notes or earlier drafts of this plan mentioned them. If a source is not in `kirtan.txt`, it is out of scope.

## Why We Pivoted

- v2 kirtan dataset (~28h, from mixed OCR + auto-caption sources) produced best kirtan WER ≈ 55% — too noisy and too small.
- Whisper-based transcription hallucinates on sung Gurmukhi.
- Gemini 2.5 Flash Lite + pre-split audio batching (proven in `scripts/kirtan_bulk_transcribe.py`) gives clean Gurmukhi transcripts at **$0 on free tier / ~$0.035/hr paid**, 40–100× cheaper than Google Cloud STT.
- Data scale is the bottleneck, not model architecture. Going from 28h → 500h of clean labels is the highest-leverage move.
- YouTube channels are excluded in v3 because they introduced copyright, geo-block, and dedup headaches. The SikhNet / AKJ / SGPC official archives in `kirtan.txt` give clean provenance and stable URLs.

## Top-Level Architecture

```
Audio taxonomy:
  source_type: paath | kirtan

  paath_type:   sehaj | akhand | nitnem          (paath only)
  kirtan_type:  ragi  | akj    | sgpc            (kirtan only)
```

Kirtan split (target 500h total):

- **Ragi kirtan** — studio/clean, harmonium+tabla, trained vocalists (target ~170h, capped by the 12 SikhNet playlists in `kirtan.txt`)
- **AKJ** — Akhand Kirtani Jatha, fast tempo, sangat participation, crowd vocals (target ~170h, capped by what the `akj.org/keertan.php` archive contains)
- **SGPC kirtan** — live PA from Darbar Sahib and SGPC gurdwaras, reverb + ambient (floor ~160h, **absorbs any shortfall** from ragi + AKJ up to 500h total; see "Scale note + overflow policy" below)

## Hard Rules — Transcription Pipeline

These rules are non-negotiable. They come from burned experience — every deviation in the past introduced alignment bugs or cost overruns.

1. **Audio is always cut to fixed 20-second clips before it touches Gemini.**
   - Last clip of a video may be short; pad with silence or drop if < 5 s.
   - Clips are named `clip_{i:05d}.wav` where `i * 20` is the start second of that clip in the source audio. This naming IS the timestamp — no metadata file needed to recover it.

2. **Never ask Gemini for timestamps.** Gemini is bad at timestamping long audio — every approach that did this produced misaligned text (see `memory/kirtan_data_approaches.md`).
   - We own the timestamps locally because we cut the clips locally.
   - The API contract is: send N clips → receive N text strings in the same order. Nothing else.

3. **Batch 5–10 minutes of audio per API call — not more, not less.**
   - At 20 s per clip, that is **15–30 clips per call**.
   - Default: **15 clips = 5 min audio per call** (proven in `scripts/kirtan_bulk_transcribe.py`, 17.6× realtime).
   - Going higher than 30 clips risks truncated responses, malformed JSON, and higher per-call retry cost if anything fails.
   - Going lower than 15 wastes API-call overhead and inflates cost.
   - **Run many 5-min batches in parallel** to saturate throughput. Gemini 2.5 Flash Lite paid-tier RPM is **1,000–2,000 req/min**, so ~50–100 concurrent workers (each sending one 15-clip batch at a time) is well within limits. Free tier is ~15 RPM — cap workers at 4–8 there. Target: transcribe 500 h of audio in under an hour of wall-clock time on paid tier.

4. **Persist Gemini output to disk immediately — before any post-processing.**
   - Write raw responses to `data/gemini_raw/<video_id>/batch_{k:04d}.json` as each batch returns.
   - Write per-clip transcripts to `data/transcripts/<video_id>.jsonl` (append mode, one line per clip).
   - This is checkpointing: if the run crashes or we want to change normalization rules, we do NOT re-call the API. Gemini calls are the expensive, irreversible step.
   - A re-run must be idempotent: if `data/transcripts/<video_id>.jsonl` has N lines and video has M clips, only process clips N..M.

## Pipeline

```text
0. Read allowlist from kirtan.txt         →  data/manifests/sources.csv  (url, kirtan_type)
1. Enumerate tracks per allowlisted URL   →  data/manifests/<kirtan_type>_manifest.csv  (id, title, duration)
     - SikhNet playlist URL      → per-track MP3 URLs via play.sikhnet.com API
     - SikhNet single track URL  → one-row manifest
     - akj.org/keertan.php       → scrape samagam MP3 links
     - sgpc.net/recorded-kirtan* → scrape per-ragi archive MP3 links
2. Filter + dedup                         →  data/manifests/<kirtan_type>_filtered.csv
3. Audio-only download (direct MP3)       →  data/raw_audio/<kirtan_type>/<track_id>.wav  (16kHz mono)
4. Split into FIXED 20s clips             →  data/clips/<track_id>/clip_{i:05d}.wav
5. Batch 15 clips (5 min) per Gemini call →  data/gemini_raw/<track_id>/batch_{k:04d}.json   (persist raw)
6. Parse batch → per-clip JSONL           →  data/transcripts/<track_id>.jsonl              (persist parsed)
7. Normalize (strip ॥, ॥੧॥, whitespace)   →  data/transcripts_clean/<track_id>.jsonl
8. Quality filter + dedup                 →  data/transcripts_final/<track_id>.jsonl
9. Build HF dataset                       →  surindersinghssj/gurbani-kirtan-v3-500h (kirtan_type tag)
10. Retrain Surt on sehaj + v3 kirtan
```

> `track_id` replaces the old `video_id` since sources are no longer YouTube. Use the SikhNet track numeric id (e.g. `sikhnet-11220`), the AKJ samagam filename slug, or the SGPC archive filename slug. The `i * 20 sec` filename → timestamp invariant is unchanged.

Each step reads from the previous step's output directory and writes to its own — so any stage can be re-run independently without re-downloading or re-calling Gemini.

## Data Sources (ALLOWLIST — single source of truth is [`kirtan.txt`](kirtan.txt))

All kirtan audio for v3 comes from the URLs in `kirtan.txt` — nothing else. If you add a new source, edit `kirtan.txt` first and land that as its own commit so intent is explicit.

### Ragi kirtan — SikhNet playlists (play.sikhnet.com)

- Bhai Harjinder Singh (Srinagar) — `/playlist/bhai-harjinder-singh-srinagar-gurbani-jukebox`
- Bhai Ravinder Singh (Hazuri Ragi) — `/playlist/bhai-ravinder-singh-hazuri-ragi-gurbani-jukebox`
- Bhai Dalbir Singh (Hazuri Ragi) — `/playlist/bhai-dalbir-singh-hazuri-ragi-gurbani-jukebox`
- Bhai Kamaljeet Singh (Hazuri Raagi) — `/playlist/bhai-kamaljeet-singh-hazuri-raagi-gurbani-jukebox`
- Bhai Davinder Singh Ji Nirman (Amritsar) — `/playlist/bhai-davinder-singh-ji-nirman-amritsar-gurbani-jukebox`
- Bhai Randhir Singh (Patiala) — `/playlist/bhai-randhir-singh-patiala-gurbani-jukebox`
- Bhai Amarjit Singh (Patiala) — `/playlist/bhai-amarjit-singh-patiala-gurbani-jukebox`
- Bhai Jaskaran Singh (Patiala) — `/playlist/bhai-jaskaran-singh-patiala-gurbani-jukebox`
- Bhai Surinder Singh (Jodhpuri) — `/playlist/bhai-surinder-singh-jodhpuri-gurbani-jukebox`
- Bhai Anantvir Singh — `/playlist/bhai-anantvir-singh-gurbani-jukebox`
- Bibi Jaskiran Kaur — `/playlist/bibi-jaskiran-kaur-gurbani-jukebox`
- Bhai Lakhwinder Singh (Hazuri Ragi) — `/playlist/bhai-lakhwinder-singh-hazuri-ragi-gurbani-jukebox`

### AKJ

- [`play.sikhnet.com/track/jaag-ray-mann-jaganhare`](https://play.sikhnet.com/track/jaag-ray-mann-jaganhare) — single seed track
- [`akj.org/keertan.php`](https://akj.org/keertan.php) — samagam archive (Rainsbai ~8–10h, Kirtan Darbar ~6h)

### SGPC

- [`sgpc.net/recorded-kirtan-ragi-wise/`](https://sgpc.net/recorded-kirtan-ragi-wise/) — official recorded archive, grouped by ragi

### Excluded (previously considered, NOT in scope for v3)

YouTube (`@sikhnet`, `@amrittsaagar`, `@GurbaniMediaCentre`, `@NirbaanKeertan`, `@SGPCSriAmritsar`), SoundCloud (`akjdotorg`), live streams (`sgpclive.com`), third-party sites (`hukamnamasahib.com`, `gurmatsagar.com`). Do not add scripts or manifest rows for these.

### Scale note + overflow policy

The allowlist is narrower than the 500h target. Back-of-envelope: ~12 SikhNet ragi playlists × ~50 tracks × ~6 min avg ≈ ~60h ragi; AKJ samagam archive is a few hundred hours worth of long recordings; SGPC archive is by far the largest single source (the `recorded-kirtan-ragi-wise/` index aggregates decades of Harmandir Sahib recordings grouped by ragi).

**Overflow policy — SGPC is the fill source.** After enumeration (P1), if ragi + AKJ fall short of 500h, pull the remainder from `sgpc.net/recorded-kirtan-ragi-wise/`. Do NOT rebalance by adding non-allowlisted sources. Concretely:

1. Enumerate ragi (all 12 SikhNet playlists) → record actual hours.
2. Enumerate AKJ (`akj.org/keertan.php` + the one SikhNet seed track) → record actual hours.
3. Compute `sgpc_target_hours = max(160, 500 - ragi_hours - akj_hours)`. The 160h floor keeps the SGPC split meaningful even if AKJ over-delivers; the `500 - ragi - akj` formula lets SGPC absorb any shortfall.
4. Pull SGPC tracks until `sgpc_target_hours` is hit **using the diversity policy below** (not "grab the biggest ragi first").
5. Record the final split (ragi / akj / sgpc hours) in the dataset card so downstream training can weight / stratify.

If the allowlist still can't reach 500h even after maxing SGPC, surface this explicitly — either lower the target or extend `kirtan.txt` with a new explicit commit. Do not silently add hidden sources in scripts.

### SGPC ragi diversity policy (required for scraping)

The SGPC archive has **201 ragi directories** and is heavily skewed — a handful of prolific ragis (e.g. Bhai Amandeep Singh, Bhai Gurdev Singh K) each have 1,000+ MP3s, while most ragis have under 100. If we pulled SGPC top-down we would get ~150h from 3 voices, which is a training disaster for a per-speaker-rare model. Variety in timbre, tempo, vocal style, and reverb conditions matters more than raw hours.

**Rule — any SGPC download selection MUST be ragi-balanced:**

1. **Per-ragi cap first.** Pick a target of `max_hours_per_ragi` (default starts at **2 h/ragi**; raise only if the 160h floor can't otherwise be met). This caps how much one voice can dominate.
2. **Then round-robin across ragis** until the SGPC hour target is reached. Do not sort ragis by size — iterate them in a shuffled order and take one track from each before looping.
3. **Random pick within each ragi, seeded.** Within a ragi, sample tracks with a fixed RNG seed (default `seed=42`) so the selection is reproducible but not biased toward the archive's storage order (which roughly correlates with upload time and thus recording era).
4. **If a ragi has fewer tracks than its cap, take all of them** and redistribute the shortfall across the remaining ragis (the round-robin handles this automatically by just skipping exhausted ragis).
5. **Prefer more ragis over more hours per ragi.** If forced to choose between "160h from 80 ragis × 2h" vs "160h from 40 ragis × 4h", pick the 80-ragi split.

Rule of thumb for planning: at 2 h/ragi across 201 ragis the ceiling is ~400h of balanced SGPC material — comfortably above the 160h floor. Use `max_hours_per_ragi` to tune, not "go wide and hope".

**Implementation** — `scripts/v3_enumerate_sgpc.py` supports this directly:

```bash
# balanced 160h SGPC manifest, 2h/ragi cap, seeded, ~80 ragis contributing
python scripts/v3_enumerate_sgpc.py \
  --out /root/v3_data/manifests/sgpc.csv \
  --max-hours-per-ragi 2 \
  --target-total-hours 160 \
  --seed 42
```

When the downloader consumes the manifest it just downloads every row — the diversity has already been baked into row selection. Do NOT re-sort or re-filter the manifest downstream in a way that re-introduces ragi skew (e.g. don't sort by `size_bytes desc` to "prioritize long tracks").

## Why the Hard Rules Matter (~10× savings)

The four hard rules above together produce roughly a **10× reduction in cost and wall-clock time** vs. the naive approach we tried earlier:

| Lever | Naive approach | v3 approach | Effect |
|---|---|---|---|
| Clip length | Variable, Gemini-estimated | **Fixed 20 s, owned locally** | No alignment bugs, no re-transcription of misaligned segments |
| Batch size | 1 clip per call | **15 clips (5 min) per call** | ~15× fewer API calls → ~15× less per-call overhead cost |
| Timestamps | Returned by Gemini (unreliable) | **Derived from filename `clip_{i}.wav` → `i*20` sec** | Zero re-runs due to bad timestamps |
| Output persistence | Re-run regenerates from scratch | **Raw responses + parsed JSONL both persisted** | Re-runs skip expensive Gemini calls, only re-do free local steps |

End result: proven 17.6× realtime throughput, ~$0.035/hr paid (or $0 free tier) — down from the $0.30–$3.84/hr range of prior approaches.

## Build Order

| Phase | Output | Status |
|---|---|---|
| P1. Enumerate | `data/manifests/<kirtan_type>_manifest.csv` with real durations | TODO |
| P2. Filter | Dedup (chromaprint), drop <3 min / >2 hr, dedup by title+duration | TODO |
| P3. Download | `data/raw_audio/<kirtan_type>/<video_id>.wav` (16 kHz mono) | TODO |
| P4. Split | `data/clips/<video_id>/clip_{i:05d}.wav` (fixed 20 s) | TODO |
| P5a. Gemini raw | `data/gemini_raw/<video_id>/batch_{k:04d}.json` (persist first) | TODO |
| P5b. Parse | `data/transcripts/<video_id>.jsonl` (clip_id → text, start_sec = i*20) | TODO |
| P6. Normalize | `data/transcripts_clean/<video_id>.jsonl` (strip `॥`/`॥੧॥`) | TODO |
| P7. Filter + dedup | `data/transcripts_final/<video_id>.jsonl` | TODO |
| P8. Dataset build | HF dataset `surindersinghssj/gurbani-kirtan-v3-500h` | TODO |
| P9. Retrain | Surt v3 on sehaj + v3 kirtan | TODO |

## Known Gotchas

1. **Dedup by audio fingerprint**, not title — same kirtan gets re-uploaded 10+ times.
2. **Katha contamination** — many "kirtan" videos have long spoken intros. Need VAD + music/speech split, or the model learns katha cadence.
3. **SGPC 24/7 stream bias** — raw stream contains ardaas, hukamnama, announcements. Need segmentation.
4. **Copyright/channel suspensions** — stick to official channels; SGPC has already been suspended once.
5. **Gemini spelling drift** — transcripts are ~57% exact-match to canonical SGGS. Post-process against `tuks.json` / STTM where possible.
6. **Text normalization** — always strip `॥`, `॥੧॥` verse markers before tokenization (they are visual, never spoken).

## What is NOT in scope for v3

- LoRA / PEFT — full fine-tune continues to be the approach.
- Changing the Whisper-Small backbone — scale data, not model.
- Rebuilding the sehaj path dataset — reuse `surindersinghssj/gurbani-sehajpath` as-is for regularization.
- Forced alignment / OCR pipelines — Gemini replaces both.

## Key Commands (reference)

```bash
# Enumerate a channel (no download)
yt-dlp --flat-playlist --print "%(id)s,%(title)s,%(duration)s" "URL" > manifest.csv

# Audio-only download, 16kHz mono
yt-dlp -x --audio-format wav --postprocessor-args "-ar 16000 -ac 1" \
       --download-archive done.txt -a urls.txt

# Transcription (existing script, proven at scale)
python scripts/kirtan_bulk_transcribe.py <video_id>
```
