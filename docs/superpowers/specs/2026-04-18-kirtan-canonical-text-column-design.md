# Kirtan Canonical Text Column — Design

**Date:** 2026-04-18
**Status:** Spec (awaiting user review)
**Target dataset:** `surindersinghssj/gurbani-kirtan-yt-captions-300h` (HF)
**Source of truth:** local STTM `database.sqlite` (ShabadOS, 141,264 lines, SGGS + Dasam + Bhai Gurdas + Nand Lal etc.)
**Depends on (blockers, must ship first):**
- AKJ YouTube ingest (`@akjdotorg`, ~100–150h) merged into the same HF repo.
- Additive HF push (new parquet shards, no rewrite) implemented — tracked in `memory/project_yt_pipeline_todos.md`. If additive push is not yet available when this ships, this step will force-rewrite the dataset (acceptable fallback, but flag in PR).

## 1. Goal

Add three new columns (`sggs_line`, `final_text`, `decision`) to the kirtan HF dataset by matching each YT caption clip to SGGS/STTM via **consonant-skeleton phrase alignment**. The `final_text` column corrects spellings and matras in place while preserving the caption's structure (word order, repetitions, newlines). `>>` segment markers are stripped from `final_text` only. The existing `text`, `raw_text`, and every other current column stay untouched.

**Primary motivation:** the current YT caption text is ~95% correct. We want to test whether STTM-grounded correction gets us to ~100% — specifically fixing matras and 1-consonant-off spellings where we can do so verbatim from scripture, while falling back to the caption when we cannot.

**Non-goal:** this is not a full rewrite of the caption into scripture form. Structural divergences (ragi repetitions, partial lines, cross-shabad sung-order) are preserved; only in-place token substitutions (match / fix / merge / split) are applied.

## 1.5 Pre-cleaning (runs BEFORE the STTM + Gemini pipeline)

Before any canonical-text work starts, run a cleaning pass that **drops rows** the pipeline cannot meaningfully handle. These are structural/quality filters on the existing HF dataset — they operate on row metadata, not on the `text` column content (which stays untouched).

Filters (row is dropped if ANY is true):

1. **Duration < 1.0 second** — too short for ASR supervision; usually segmentation artifacts. Check `duration_s` column. A 0.5s clip is rarely a complete sung tukk.
2. **Caption resolves to ≤ 1 Gurmukhi token after stripping `>>` and non-Gurmukhi characters.** Single-letter orphans like `ਜ`, `ਸ` (caused by upstream panel-line splitting) carry no meaningful signal. Check via tokenization.
3. **Caption is only `>>` markers / punctuation / ASCII** — dead row, no Gurmukhi content. Drop.
4. **(Optional, tuneable)** Caption duration-to-token ratio outliers: if `duration_s / n_gurmukhi_tokens > 4` (clip is very slow) or `< 0.1` (clip has far too many tokens for its length), flag for manual review. Default: flag only, don't drop.

Normalization (applied to all surviving rows, does not drop):

5. **Strip ASR `<unk>` artifacts** (`STRIP_UNK_ARTIFACTS=true`, default). Pattern: `<unk?>?|<un|<k>?|unk>|un>|<k` replaced with single space, then re-tokenize. Catches sehaj-path ASR artifacts like `ਜਾਲunk>`, `ਕਢਾਇ<un`, `ਸਾਲਾਹਿਹਿ<k` that corrupt STTM alignment. Should run BEFORE tokenization.

Implementation: add as Phase −1 in `scripts/add_canonical_column.py`, before any STTM/LLM work. Emit a summary to stderr: `dropped_short=N, dropped_single_token=N, dropped_no_gurmukhi=N, unk_artifacts_stripped=N`.

**Expected drop rate** (from inspecting the sample data): ~5–10% of rows — worth removing before the expensive STTM + Gemini passes, and avoids polluting the training set with meaningless clips.

## 1.6 Simran quota (runs AFTER §1.5, BEFORE the canonical pipeline)

AKJ kirtan videos include long ਵਾਹਿਗੁਰੂ simran sessions. Simran is a real sung pattern, not a transcription error — so it shouldn't be "fixed" or flagged — but at raw ingest volumes it will dominate the dataset and bias the ASR decoder toward always outputting ਵਾਹਿਗੁਰੂ when uncertain. Left unchecked, AKJ simran could easily be 17–25% of the final dataset.

### Detection

A row is marked `is_simran = true` if:
- It contains ≥ `SIMRAN_DETECT_MIN_REPS` consecutive `ਵਾਹਿਗੁਰੂ` tokens, AND
- Those tokens are > `SIMRAN_DETECT_RATIO` of the clip's Gurmukhi tokens.

Defaults: `SIMRAN_DETECT_MIN_REPS = 5`, `SIMRAN_DETECT_RATIO = 0.70`.

### Downsample strategy (absolute target, adaptive per-video cap)

Simran teaches ONE thing — the ਵਾਹਿਗੁਰੂ repetition pattern. The teaching signal is fixed regardless of dataset size, so simran count is an **absolute target**, not a ratio of dataset size.

**Target**: `SIMRAN_TARGET_COUNT = 750` total simran rows kept (midpoint of empirical 500–1000 range for adequate voice/tempo diversity).

**Step 1 — compute adaptive per-video cap:**

```python
n_simran_videos = count distinct video_ids among is_simran=true rows
per_video_cap = clamp(ceil(SIMRAN_TARGET_COUNT / n_simran_videos),
                      min=SIMRAN_PER_VIDEO_MIN,
                      max=SIMRAN_PER_VIDEO_MAX)
```

Defaults: `SIMRAN_PER_VIDEO_MIN = 1`, `SIMRAN_PER_VIDEO_MAX = 10`.

Expected behavior across AKJ ingest sizes:

| Videos with simran | per_video_cap | Total kept |
|---|---|---|
| ~300 (dense) | 3 | ~900 |
| ~150 | 5 | ~750 |
| ~100 | 8 | ~800 |
| ~50 (sparse) | 10 (capped) | ~500 |
| ~1 (pathological) | 10 (capped) | ~10 |

The cap prevents a single voice from dominating the simran set.

**Step 2 — per-video cap + round-robin stratified sampling:**

- Group simran rows by `video_id`.
- Within each video, random sample up to `per_video_cap` rows (not the first N — those cluster at session start).
- If total surviving rows > `SIMRAN_TARGET_COUNT`, select rows in **round-robin order** across videos: take video₁[0], video₂[0], …, videoₙ[0], then video₁[1], video₂[1], …, until target met. This ensures each video contributes proportionally before any video contributes multiply.

Dropped simran rows are **removed from the dataset**, not flagged. The `is_simran = true` column stays on the surviving rows for training-side samplers.

**Why 750, not 5% of dataset**:
- Simran is a single repeated word — ASR models learn repetitive patterns fast. ~500-1000 rows across diverse voices is enough.
- Absolute count decouples simran from dataset growth: if the dataset doubles, you don't accidentally balloon simran to 1,200 rows (wastefully over-teaching).
- If empirical simran WER turns out high after training, bump `SIMRAN_TARGET_COUNT` up; easier to add data than un-bias a trained model.

### Short-circuit (applies to surviving simran rows in the canonical pipeline)

Rows with `is_simran = true` that survive the quota skip both stages of the canonical pipeline:

- Skip STTM retrieval — ਵਾਹਿਗੁਰੂ is not in SGGS (it lives in Bhai Gurdas Vaaran), so retrieval always fails anyway.
- Skip LLM pass — the text is already correct; sending it to Gemini wastes API calls.
- Apply waheguru normalization (see below) to produce `final_text`.
- Emit: `decision = "simran"`, `sggs_line = null`, `final_text = normalized caption`, `final_text_llm = null`, `llm_verified = null`.

New decision value: `simran` — joins `matched / replaced / review / unchanged`.

### Waheguru normalization (hardcoded, always-correct treatment)

ਵਾਹਿਗੁਰੂ is definitionally the correct spelling in Gurmat context. YT captions commonly have matra variants — `ਵਾਹੇਗੁਰੂ`, `ਵਾਹਿਗੁਰ`, `ਵਾਹਿਗੁਰੁ`, `ਵਹਿਗੁਰੂ` — which all collapse to the same consonant skeleton `ਵਹਗਰ`. Any token matching that skeleton is rewritten to the canonical `ਵਾਹਿਗੁਰੂ` with zero LLM involvement.

**Where it applies:**
1. **Primary target**: simran-decision rows. Every ਵਾਹਿਗੁਰੂ-ish token in the caption becomes the canonical form in `final_text`.
2. **Secondary pass**: applied to ALL rows' captions after the main pipeline, regardless of decision. Even in the middle of a non-simran clip, a stray `ਵਾਹੇਗੁਰੂ` becomes `ਵਾਹਿਗੁਰੂ`. Cheap safety net.

**Implementation** (tight, deterministic):

```python
CANONICAL_WAHEGURU = "ਵਾਹਿਗੁਰੂ"
WAHEGURU_SKEL = skel(CANONICAL_WAHEGURU)  # = "ਵਹਗਰ"

def normalize_waheguru(tokens: list[str]) -> list[str]:
    return [CANONICAL_WAHEGURU if skel(t) == WAHEGURU_SKEL else t for t in tokens]
```

Applied unconditionally, runs in microseconds. Zero API cost. No risk of corruption since it's skeleton-exact-match (4 consonants, unique enough — no other common Gurbani word collapses to `ਵਹਗਰ`).

### Tuneable constants

| Constant | Default | Purpose |
|---|---|---|
| `SIMRAN_DETECT_MIN_REPS` | 5 | Min consecutive ਵਾਹਿਗੁਰੂ tokens to trigger detection |
| `SIMRAN_DETECT_RATIO` | 0.70 | Min fraction of clip tokens that must be ਵਾਹਿਗੁਰੂ |
| `SIMRAN_TARGET_COUNT` | 750 | Absolute target count of simran rows in final dataset |
| `SIMRAN_PER_VIDEO_MIN` | 1 | Lower clamp on adaptive per-video cap |
| `SIMRAN_PER_VIDEO_MAX` | 10 | Upper clamp on adaptive per-video cap |

### Expected outcome

- Pre-downsample simran count: ~10–15K rows (at 17–25% of 60K) from AKJ alone.
- Post-downsample: ~500–900 rows depending on AKJ video count; guaranteed diversity via adaptive per-video cap clamped to [1, 10].
- Waheguru normalization: ~5–10% of non-simran rows will have ≥1 matra-variant `ਵਾਹੇਗੁਰੂ`-style token that gets canonicalized in `final_text`.
- Logs: `simran_detected=N, n_simran_videos=N, per_video_cap=N, after_per_video_cap=N, after_round_robin=N, waheguru_normalized=N` printed to stderr.

## 2. Inputs & outputs

### Input: existing HF dataset schema (current)
`audio, text, raw_text, clip_id, start_s, end_s, duration_s, n_cues, clip_mode, caption_offset_s, video_id, caption_lang`

### Hard invariant
**No existing column is modified.** `text`, `raw_text`, and every other current column are passed through byte-identical. This pipeline is strictly additive.

### Output: three new columns added
| Column | Type | Description |
|---|---|---|
| `sggs_line` | str \| null | SGGS canonical reference text (verbatim from STTM DB, Unicode, vishraams stripped): the matched shabad line(s) that cover the caption span, joined with single spaces, in traversal order. Null when shabad retrieval fails (decision = `unchanged`). |
| `final_text` | str | Structure-preserved, spelling/matra-corrected text. Same whitespace/newlines/word-order/repetitions as caption. `>>` markers stripped. On `decision == "unchanged"`, `final_text` equals the caption's `text` with `>>` stripped (so the column is always populated and usable directly). |
| `decision` | str | One of: `matched` (caption already correct; no token replacements), `replaced` (at least one token replaced via fix/merge/split), `review` (borderline — human should verify), `unchanged` (shabad retrieval failed or score too low; `sggs_line` is null and `final_text` is caption-with-`>>`-stripped). |

### Audit sidecar (not pushed to HF)
Diagnostic info for debugging + threshold tuning is written to a **local parquet sidecar** (`canonical_audit.parquet`), not to the HF dataset, keyed on `clip_id`: `shabad_id`, `line_ids` (list), `match_score` (float), `op_counts` (dict), `top1_top2_margin` (float). Stays local; regenerated on each rerun.

## 3. Algorithm

Five phases per HF row. All phases reuse the caption stored in `text` (which is already lightly cleaned). `raw_text` is not used.

### Phase 0 — Caption preprocessing (for matching only; source `text` column is never modified)

1. Strip `>>` markers (record positions to drop from `final_text` too).
2. Normalize whitespace (collapse multiple spaces, keep newlines as soft-break markers).
3. Tokenize on whitespace. Each token tagged as `gurmukhi` or `other` (punctuation, digits, ASCII).

### Phase 1 — Shabad retrieval (coarse, with sliding-window context)

Pre-built once per run:
- Pull all content lines (`type_id IN (1,3,4)` — pankti/rahao/manglacharan) from STTM source `sggs` (source_id=1) via the existing `scripts/v3_build_canonical.py` SQL pattern. Apply `gurmukhi_converter.strip_vishraams` + `ascii_to_unicode` + per-token vishraam/digit scrubbing.
- For each SGGS shabad, compute the multiset of **consonant 4-grams** over the shabad's concatenated skeleton (matras, spaces, digits, vishraams all stripped).
- Index: `shabad_id → {4gram → count}` + inverse index `4gram → [shabad_ids]` for fast lookup.

**Per row — sliding-window retrieval:**
- Concatenate the consonant skeletons of **rows `[i-1, i, i+1]`** (within the same `video_id`, ordered by `start_s`). `>>`-only rows are excluded from the window. This is the retrieval query — it uses neighbor rows **only** to identify which shabad the current row belongs to.
- Tokenize this combined skeleton into 4-grams.
- Score each candidate shabad by **TF-IDF-weighted 4-gram overlap**, IDF computed over shabad corpus. Pick top-1.
- Record top-1 and top-2 scores; the **margin** top1 − top2 becomes a confidence signal.
- **Critical invariant**: the alignment in Phase 3 uses **only row `i`'s tokens**. Neighbors inform shabad selection, nothing else. No token leakage.
- Window size `W=1` (i.e., `±1` row) by default, tuneable via `RETRIEVAL_WINDOW`. Validated on the 67-row kirtan sample in §4 — bumped per-row shabad accuracy from ~50% (per-clip retrieval) to ~95% (±1 window).

### Phase 2 — Shabad-scoped vocab + optional global fallback

**Tier 1 (primary):** build a shabad-scoped vocab from top-1:
- `shabad_vocab: {canonical_token_skel → (canonical_token_unicode, line_id)}` for every unique token in the retrieved shabad.

**Tier 2 (fallback, conditional):** if top-1 margin is low (margin < `M_MIN`, default 0.05) OR if Phase 3 leaves a contiguous run of `≥ R_MIN` (default 3) Gurmukhi tokens unaligned, consult a **global SGGS inverted index**: `{canonical_token_skel → [(unicode_token, line_id, shabad_id)]}`. Used only for orphan spans, with a secondary_shabad_id tagged per span in `canonical_line_ids` metadata.

### Phase 2.5 — Sirlekh (shabad-header) indexing and normalization

(Enabled when `INCLUDE_SIRLEKH=true`, default `true` for both kirtan and sehaj.)

Extend the STTM SQL query from `type_id IN (1, 3, 4)` to `type_id IN (1, 3, 4, 5)` so that shabad headers like `ਸ੍ਰੀਰਾਗੁ ਮਹਲਾ ੫ ॥` (`type_id=5`, Sirlekh) are part of the consonant-skeleton index. Without this, any row containing header text falls through to `unchanged`.

Additionally, build a small canonical-header normalization table for the most common variants that ASR produces (observed on sehaj path):

```
"ਸੀ ਰਾਗ ਮਹਲਾ ਪੰ"       → "ਸ੍ਰੀਰਾਗੁ ਮਹਲਾ ੫"
"ਸੀਰਾਗ ਮਹਲਾ"           → "ਸ੍ਰੀਰਾਗੁ ਮਹਲਾ"
"ਸ੍ਰੀਰਾਗੁ ਮਹਲਾ>"       → "ਸ੍ਰੀਰਾਗੁ ਮਹਲਾ"
"ਸਿ੍ਰ ਰਾਗੁ ਮਹਲਾ"      → "ਸ੍ਰੀਰਾਗੁ ਮਹਲਾ"
"ਸਿ੍ਰੀਰਾਗੁ ਮਹਲਾ"       → "ਸ੍ਰੀਰਾਗੁ ਮਹਲਾ"
```

Applied as a regex sweep before tokenization, only when flag is true.

### Phase 2.6 — Multi-shabad row splitting

(Enabled when `SPLIT_MULTI_SHABAD_ROWS=true`, default `true` for both datasets.)

Sehaj path (and occasionally kirtan) has rows that straddle a shabad boundary:

```
Row:  ਕਢਾਇ ਸੀਰਾਗ ਮਹਲਾ ਘੜੀ ਮੁਹਤ ਕਾ ਪਾਹੁਣਾ
         └ shabad A ┘ └ header ┘ └─ shabad B ──┘
```

Detect mid-row Sirlekh patterns via:

```python
SIRLEKH_RE = re.compile(
    r'(ਸ੍ਰੀ\s*ਰਾਗੁ?|ਸੀ\s*ਰਾਗ|ਸਿ੍ਰ\s*ਰਾਗੁ?|ਸੀਰਾਗ|ਸਿਰੀੀ?\s*ਰਾਗੁ?)\s*ਮਹਲਾ'
)
```

If a match is found at position `p > 3` (i.e., not at row start), split the row into sub-parts at the Sirlekh boundary and run alignment on each sub-part independently. Merge the results with the shabad header between them in the final `final_text`. Emit row-level `split_at=[p1, p2, ...]` in the audit sidecar.

### Phase 2.7 — Sequential shabad retrieval (sehaj-specific)

(Enabled when `SEQUENTIAL_SHABAD_RETRIEVAL=true`. Default `true` for sehaj, `false` for kirtan.)

In sehaj path, reading progresses linearly through SGGS. If row `i-1` matched shabad X, row `i` is almost certainly in shabad X or the immediately-following shabad (X+1 in `order_id` sequence). Apply a strong prior:

```python
# Bias retrieval scores by recency of match in this video
for candidate_sid, score in shabad_scores.items():
    if candidate_sid == prev_shabad_id:
        score *= (1 + SEQUENTIAL_CURRENT_BOOST)   # default 0.8
    elif candidate_sid == next_shabad_in_sequence(prev_shabad_id):
        score *= (1 + SEQUENTIAL_NEXT_BOOST)      # default 0.5
```

`next_shabad_in_sequence(sid)` looks up the shabad with the smallest `order_id` greater than `sid`'s last line's `order_id`. Pre-computed once.

This is **NOT** enabled for kirtan because ragis often medley across shabads — a sequential prior would bias alignment toward the wrong shabad in that case.

### Phase 3 — Phrase-level alignment (fine, monotonic)

**Full Needleman–Wunsch DP** over `(caption_tokens × retrieved_shabad_token_stream)` with **monotonic advance on the SGGS side** (required to fix the prototype's greedy bug where `ਰੋਤੀ` aligned to an earlier `ਰੁਖੀ` instead of the correct forward `ਰੋਟੀ`). The DP table is O(m·n) with m = caption tokens (typically < 20) and n = shabad tokens (typically < 200), so ~10ms per row on CPU.

Edit operations (scoring via skeleton-Levenshtein):

| Op | Caption span | SGGS span | Gate | Emit |
|---|---|---|---|---|
| MATCH | 1 | 1 | skeleton equal | keep caption token (already correct) |
| FIX | 1 | 1 | skeleton Levenshtein ≤ 1 | replace with SGGS unicode token |
| MERGE | N (≤ 3) | 1 | joined-caption-skeleton Levenshtein vs SGGS-skeleton ≤ 1 | replace the N-span with the one SGGS token |
| SPLIT | 1 | N (≤ 3) | caption-skeleton Levenshtein vs joined-SGGS-skeleton ≤ 1 | replace the 1 token with the N SGGS tokens |
| INSERT | 0 | 1 | SGGS token with no caption counterpart | **do not inject** — respect audio-faithful structure |
| DELETE | 1 | 0 | caption token with no SGGS counterpart | keep caption token, mark `unmatched` |

Gap penalty (unaligned runs in either sequence) is negative but finite, so the aligner prefers short gaps over spurious far-apart matches. Scoring constants are listed in §7 and live at the top of the code as tunable constants.

**Repetition handling:** repetitions are handled naturally by the aligner — when the ragi sings a tukk twice, the aligner matches each caption repetition independently against the same SGGS lines (the aligner is allowed to revisit SGGS positions, not consume-once). Concretely: run one alignment pass per contiguous repeat group detected by a simple n-gram duplicate detector over caption tokens; concatenate results.

### Phase 4 — Output assembly

Walk caption tokens in original order:
- Emit canonical token per the aligner's decision.
- Drop `>>` markers from `canonical_text` stream.
- Preserve whitespace and newlines from the caption.
- Accumulate `op_counts` and collect `line_id`s encountered into `canonical_line_ids` (deduplicated, in traversal order).

Compute `match_score = (match_count + fix_count + merge_count + split_count) / (match_count + fix_count + merge_count + split_count + delete_count)`. `match` rewards already-correct tokens in the denominator-numerator; pure-caption tokens (delete) penalize.

### Phase 5 — Flag thresholds (initial; tuneable after dry-run)

- `match_score ≥ 0.92` AND shabad retrieval succeeded → `auto_accept`
- `0.75 ≤ match_score < 0.92` OR shabad retrieval margin < `M_MIN` → `manual_review`
- `match_score < 0.75` OR no shabad locked → `reject_use_caption` (`canonical_text = ""`, `canonical_shabad_id = null`)

## 4. Preprocessing & normalization rules

### Consonant skeleton
Strip from input text, keeping only consonant letters:
- Matras (dependent vowel signs): U+0A3E–U+0A4D (ਾਿੀੁੂੇੈੋੌ੍ etc.)
- Bindi / tippi / addak / visarga: ਂ ਃ ੱ ੰ
- Nukta-modified consonants: normalize to base form (ਸ਼→ਸ, ਖ਼→ਖ, ਗ਼→ਗ, ਜ਼→ਜ, ਫ਼→ਫ, ਲ਼→ਲ)
- Vishraams: ॥ ॰
- Digits: ੦–੯ (and ASCII 0–9)
- ASCII punctuation, spaces, zero-width joiners (ZWJ U+200D, ZWNJ U+200C)
- `>>` markers

Keep: bare Gurmukhi consonants (ੳ–ਹ, range U+0A05–U+0A3B consonants subset — exact whitelist enumerated in the module).

### Worked example (your kirtan clip, Ang 105 shabad EZ7)

Verified end-to-end against STTM DB. Output after this pipeline (structure preserved, `>>` stripped in canonical):

```
Caption token run                         →  canonical_text                              [ops]
─────────────────────────────────────────────────────────────────────────────────────────────
ਸਿਆਸਤ ਸੋਇਨ ਚਵਾਰੇ                           →  ਸੇ ਅਸਥਲ ਸੋਇਨ ਚਉਬਾਰੇ                           [split, match, fix]
ਜਿਥੈ ਨਾਮੁ ਨ ਜਪੀਐ ਮੇਰੇ ਗੋਇੰਦਾ                →  ਜਿਥੈ ਨਾਮੁ ਨ ਜਪੀਐ ਮੇਰੇ ਗੋਇਦਾ                   [match*6, fix]
>> >> ਜਿਥੈ ਨਾਮ                             →  ਜਿਥੈ ਨਾਮੁ                                   [match, fix; >> dropped]
ਨ ਜਪੀਐ ਮੇਰੇ ਗੋਇੰਦਾ ਸੇਈ ਨਦਰ                  →  ਨ ਜਪੀਐ ਮੇਰੇ ਗੋਇਦਾ ਸੇਈ ਨਗਰ                  [match*5, fix; ਨਦਰ→ਨਗਰ]
ਉਜਾੜੀ ਜੀਉ                                  →  ਉਜਾੜੀ ਜੀਉ                                  [match*2]
ਸੇ ਨਗਰ >>                                  →  ਸੇਈ ਨਗਰ                                    [fix, match; >> dropped]
>> ਉਦਾੜੀ ਦੀਆ ਸੇਈ ਨਗਰ                        →  ਉਜਾੜੀ ਜੀਉ ਸੇਈ ਨਗਰ                          [fix, fix, match*2]
ਉਦਾੜੀ ਦੀਆ ਸੇਈ ਨਗਰ                           →  ਉਜਾੜੀ ਜੀਉ ਸੇਈ ਨਗਰ                          [fix, fix, match*2]
ਉਦਾੜੀ ਦੇਆ ਤੇਰੀ ਸਰਣ ਮੇਰੇ                    →  ਉਜਾੜੀ ਜੀਉ ਤੇਰੀ ਸਰਣਿ ਮੇਰੇ                   [fix, fix, match, fix, match]
ਦੀਨ ਦਇਆਲਾ ਸੁਖ ਸਾਗਰ                         →  ਦੀਨ ਦਇਆਲਾ ਸੁਖ ਸਾਗਰ                         [match*4]
ਮੇਰੇ ਗੁਰ ਗੋਪਾਲਾ ਸੁਖ ਸਾਗਰ                   →  ਮੇਰੇ ਗੁਰ ਗੋਪਾਲਾ ਸੁਖ ਸਾਗਰ                  [match*5]
ਮੇਰੇ ਗੁਣ ਗੋਪਾਲਾ ਤੇਰੀ ਸਰਨ ਮੇਰੇ              →  ਮੇਰੇ ਗੁਰ ਗੋਪਾਲਾ ਤੇਰੀ ਸਰਣਿ ਮੇਰੇ             [match, fix, match*2, fix, match]
ਰੁਖੀ ਰੋਤੀ ਖਾਇ ਸਮਾਲੇ                        →  ਰੁਖੀ ਰੋਟੀ ਖਾਇ ਸਮਾਲੇ                        [match, fix, match*2]
ਹਰਿ ਅੰਤਰ ਬਾਹਰ ਨਦਰਿ ਨਿਹਾਲੇ                 →  ਹਰਿ ਅੰਤਰਿ ਬਾਹਰਿ ਨਦਰਿ ਨਿਹਾਲੇ               [match, fix*2, match*2]
ਖਾਇ ਖਾਇ ਕਰੇ ਮਦ ਫੈਲੀ                        →  ਖਾਇ ਖਾਇ ਕਰੇ ਬਦਫੈਲੀ                         [match*3, merge]
ਜਾਣ ਵਿਸੋ ਕੀ ਵਾੜੀ ਜੀਆ                       →  ਜਾਣੁ ਵਿਸੂ ਕੀ ਵਾੜੀ ਜੀਉ                      [fix*2, match*2, fix]
ਦੁਲਭਦੇ ਹੋਈ ਅਗਿਆਨੀ                          →  ਦੁਲਭ ਦੇਹ ਖੋਈ ਅਗਿਆਨੀ                        [split (with fix), match]  or UNMATCHED if per-op edit-dist > 1
...
```

The row with ਦੁਲਭਦੇ ਹੋਈ is a borderline case: split+fix combined. Default config keeps it `unmatched` (per-op edit-dist cap = 1); bumping the cap catches it but increases false-positive risk. Tune via dry-run.

Note: SGGS actually has `ਗੋਇਦਾ` (no tippi on ਗੋ) on this Ang — not `ਗੋਇੰਦਾ` as the caption writes. The DB is authoritative, so the canonical follows DB. If stakeholders disagree (some ShabadOS revisions normalize differently), we add a post-normalization rule.

## 5. Implementation

### New files

- `scripts/gurmukhi_skeleton.py` — pure function `to_consonant_skeleton(text: str) -> str` + `skeleton_distance(a, b) -> int`. Unit-tested with pytest (test cases for matras, nukta, vishraams, digits, ZWJ, `>>` edge cases).
- `scripts/canonical_align.py` — core alignment library:
  - `load_sggs_index(db_path) -> (shabad_idx, global_token_idx)`: one-shot DB load, reuses `v3_build_canonical` logic.
  - `retrieve_shabad(caption_tokens, shabad_idx) -> (shabad_id, margin)`: 4-gram TF-IDF.
  - `align_phrase(caption_tokens, shabad_tokens, ...) -> list[AlignOp]`: phrase-level NW.
  - `render(caption_tokens, ops, gloablidx, fallback_cfg) -> CanonicalResult`: builds `canonical_text`, `line_ids`, `op_counts`, `match_score`, `match_flag`.
- `scripts/add_canonical_column.py` — driver:
  - Load HF dataset (`load_dataset("surindersinghssj/gurbani-kirtan-yt-captions-300h", split="train", streaming=False)`).
  - For each row, run Phases 0–5, emit a sidecar parquet keyed on `clip_id`.
  - Merge sidecar into dataset via `dataset.map(...)` lookup-on-`clip_id`.
  - `--dry-run N` flag: run on first N rows only, print `match_flag` histogram and sample triples.
  - `--push` flag: push to HF (otherwise writes local parquet only).
- `scripts/review_canonical_flags.py` — lightweight triage CLI:
  - Read `manual_review` rows.
  - For each: play audio clip (via `ffplay`), show `text` vs `canonical_text` side by side, capture user decision (`accept` / `reject` / `edit`) → append to `review_decisions.csv`.
  - A follow-up pass reads the CSV and applies user decisions to produce the final column values.

### Reused / extended modules
- `scripts/gurmukhi_converter.py` — `ascii_to_unicode`, `strip_vishraams` (unchanged).
- `scripts/v3_build_canonical.py` — SQL pattern reused via import, not re-implemented.

### New tests
`tests/test_gurmukhi_skeleton.py` and `tests/test_canonical_align.py`:
- Skeleton round-trips for SGGS sample lines.
- Alignment ops exhaustive tests: match / fix / merge / split / insert / delete, each against hand-crafted pairs.
- End-to-end test on the Ang-105 EZ7 sample in §4 (assert expected `op_counts` and `canonical_text`).

## 6. Output & push strategy

- Write sidecar parquet `canonical_sidecar.parquet` keyed on `clip_id`, columns: `clip_id, canonical_text, canonical_shabad_id, canonical_line_ids, canonical_source, match_score, match_flag, op_counts`.
- Join: `dataset.map(lookup_by_clip_id, batched=True, num_proc=8)`.
- Push: new revision of `surindersinghssj/gurbani-kirtan-yt-captions-300h`.
- **Additive push gate:** if `memory/project_yt_pipeline_todos.md` item (1) — additive HF push — is done, use it. Otherwise, this step rewrites the dataset (non-ideal but acceptable; flag in PR description).

## 7. Tuneable constants (exposed as CLI flags on `add_canonical_column.py`)

### Dataset-specific defaults

Same pipeline code for both datasets; configuration flags differ:

| Flag | Kirtan default | Sehaj-path default | Purpose |
|---|---|---|---|
| `STRIP_UNK_ARTIFACTS` | `true` | `true` | Pre-clean `<unk>`/`<un`/`<k>` ASR noise |
| `INCLUDE_SIRLEKH` | `true` | `true` | Index shabad headers (`type_id=5`) |
| `SPLIT_MULTI_SHABAD_ROWS` | `true` | `true` | Split rows with mid-row Sirlekh |
| `SEQUENTIAL_SHABAD_RETRIEVAL` | `false` | `true` | Bias retrieval toward previous row's shabad (sehaj is linear; kirtan medleys) |
| `USE_LLM_FALLBACK` | `true` | `true` | Gemini on `unchanged`/`review` (catches the ~5% tail for sehaj, ~23% tail for kirtan) |
| `SIMRAN_DETECTION` | `true` | `false` | AKJ-specific; sehaj doesn't have simran blocks |

Expected confident-correction rates with defaults applied:

| Dataset | STTM alone | + sehaj-specific additions | + LLM fallback |
|---|---:|---:|---:|
| Kirtan | 61–76% | 65–80% | **~92%** |
| Sehaj path | 74% | **~95%** | **~99%** |

### Core alignment tuneables (both datasets)


| Constant | Default | Purpose |
|---|---|---|
| `NGRAM_N` | 4 | Shabad-retrieval n-gram size |
| `M_MIN` (retrieval margin) | 0.05 | Triggers Tier-2 global fallback if top1−top2 < M_MIN |
| `R_MIN` (orphan run length) | 3 | Triggers Tier-2 for contiguous unaligned runs |
| `MAX_MERGE_SPLIT` | 3 | Max tokens on either side of a merge/split op |
| `MAX_OP_EDIT_DIST` | 1 | Max skeleton Levenshtein per alignment op |
| `ACCEPT_THRESHOLD` | 0.92 | `auto_accept` score cutoff |
| `REVIEW_THRESHOLD` | 0.75 | `manual_review` lower bound |

Dry-run flow: run on 1000 rows, inspect `match_flag` histogram and spot-check 20 `auto_accept` + 20 `manual_review`, then lock constants before full run.

## 8. Validation plan

**Prototype results (67-row sample covering Ang 105 EZ7 + TD3 shabads):**

- Per-row shabad accuracy: ~95% (with ±1 window retrieval) vs ~50% with per-clip retrieval.
- Correct `final_text`: ~58/67 rows (~87%). Remaining failures are:
  - Rows with 3+ consonant errors per token (alignment drift).
  - Very-short/fragment clips (single token).
  - Clips spanning shabad transitions with the current row near the boundary.
- Zero silent corruption of already-correct captions (the primary risk with per-clip retrieval — fixed by window retrieval + monotonic NW).
- Prototype: `scripts/proto_canonical.py` (not a shipping artifact; kept for regression).

**Production validation checklist:**

1. **Unit tests** pass (`tests/test_gurmukhi_skeleton.py`, `tests/test_canonical_align.py`).
2. **Dry-run on 1000 rows** from the live HF dataset:
   - `decision` histogram printed.
   - Spot-check 20 `matched` + 20 `replaced` rows → confirm canonical matras/spellings are correct.
   - Spot-check 20 `review` rows → confirm they're genuinely ambiguous.
3. **Sanity counter**: how many rows were already full-Unicode-identical to the retrieved SGGS line (baseline — "how much does this help at all").
4. **Regression probe**: take 20 rows known to be non-SGGS (e.g., intro announcements, sangat's spoken words) and confirm they route to `unchanged`, not `replaced`.
5. **Cost & runtime**: expect ~10 min on a single CPU for full 300h (~60K rows), ~200MB audit sidecar.

## 9. Scope & ordering

- **Source scope first pass:** SGGS only (source_id=1). Add Dasam (2) + Bhai Gurdas Vaaran (3) + Kabit (4) only if unmatched-rate on full run > 10%.
- **Gating order**, strict:
  1. AKJ ingest (`@akjdotorg`) merged into kirtan HF repo.
  2. Additive HF push landed (per memory TODO).
  3. This canonical-column work.
- Rerunning this script is idempotent and cheap; tuning constants via repeat dry-runs is the expected workflow.

## 10. Out of scope (for the DB-grounded pipeline)

- Fine-tuning a neural corrector (ByT5/IndicBART). Valid follow-up project using pairs produced by this pipeline, tracked separately.
- Cross-shabad medley handling beyond the Tier-2 global fallback. Ragis who sing shlok → shabad → shlok transitions may split across clips in ways that need per-video segmentation; that's a separate problem.
- Editing or regenerating `text` / `raw_text`. Those columns stay exactly as shipped.

## 11. LLM fallback pass (ADDITIVE, runs AFTER §1.5 cleaning + DB pipeline)

**Both datasets use LLM fallback** — but sehaj path needs it for a much smaller tail after sehaj-specific STTM additions bring the confident-correction rate to ~95%.

Empirical validation:

- **Kirtan** (`gurbani-kirtan-yt-captions-300h`): STTM alone hits 61–76% confident corrections (varies by shabad). LLM on `unchanged+review` (~23% of rows) lifts this to ~90–95%.
- **Sehaj path** (`gurbani-sehajpath`): STTM alone hits 74% (validated on 100-row sample). Adding sehaj-specific pre-cleaning (§1.5 `STRIP_UNK_ARTIFACTS`) + Sirlekh indexing (§3.5) + multi-shabad row splitting (§3.6) + sequential retrieval (§3.7) projects ~95%. LLM picks up the remaining ~5% tail.

Same code path for both datasets; the LLM stage decides whether to run per-row based on `decision`. The only difference is how many rows qualify.

For rows the DB pipeline marks `decision IN (unchanged, review)` (when enabled), run a second-stage LLM pass that adds three MORE columns. The main `final_text` / `sggs_line` / `decision` columns stay untouched. The LLM pass never modifies DB-grounded outputs — it only adds its own opinion as separate columns that downstream can choose to use or ignore.

### Additional columns (additive, never overwriting DB pipeline)

| Column | Type | Description |
|---|---|---|
| `final_text_llm` | str \| null | LLM-corrected text (SGGS-grounded). Null for rows where `decision IN (matched, replaced)`. |
| `llm_model` | str \| null | Model ID that produced `final_text_llm` (e.g. `gemini-3.1-pro-preview`). |
| `llm_verified` | bool \| null | True iff every token in `final_text_llm` is verbatim in SGGS OR ≤ 1-edit skeleton from some SGGS token, AND token count is within ±2 of caption. |

### Model choice (locked after empirical validation, 2026-04-18)

**`gemini-3.1-pro-preview`**, batch size **30**, `temperature=0.0`, `response_mime_type="application/json"`.

Validated on the "ਅਬ ਮੋਹਿ ਜਾਨੀ ਰੇ" and "ਤੇਰੀ ਸਰਣਿ" sample runs:

- `gemini-3.1-pro-preview` — 100% clean JSON, structure-preserving, handles SPLIT/MERGE cases the DB pipeline cannot. **Chosen.**
- `gemini-3-flash-preview` — quality is equivalent to Pro on rows where it succeeds, but **truncates JSON unpredictably on Gurmukhi batches ≥ 10**. Saves ~$1 on the whole 300h dataset in exchange for retry complexity. Not worth it at this scale.
- `gemini-2.5-flash-lite` — 3.3× cheaper than Pro but drops/expands token counts regularly. Fails the structure-preservation invariant.

### Prompt rules (critical — empirically validated)

The prompt MUST include these exact rules (the first one is non-obvious):

1. **"You MUST return EXACTLY ONE corrections entry per input clip_id. No omissions. The output's corrections array MUST have the same length as the input captions array."** Without this, Gemini silently drops rows it decides don't need correction — observed in practice.
2. "If the caption is already correct or cannot be confidently corrected, still include it in the output — copy the caption verbatim."
3. "Preserve word order, repetitions, and token count (± 1 max)."
4. "You may ONLY use words that appear in the SGGS shabad provided below."
5. "Do NOT add lines, repetitions, or words not present in the caption. Do NOT drop trailing tokens."

**Post-parse assertion**: `len(corrections) == len(batch)`. If not, log the missing `clip_id`s and retry at half the batch size.

### Verifier (hallucination guard)

For each LLM output, verify:

1. **Token count drift**: `abs(len(llm_tokens) - len(caption_tokens)) ≤ 2`. Otherwise `llm_verified=false, reason=len_drift`.
2. **Token inventory**: every Gurmukhi token in the LLM output must be either verbatim in SGGS (anywhere, not just the retrieved shabad), OR within skeleton-Levenshtein ≤ 1 of some SGGS token. Otherwise `llm_verified=false, reason=invented(<tok>)`. **Critical**: check against the full-SGGS token inventory, not just the retrieved shabad — valid tokens like `ਸਬਦੁ` appear across many shabads and a shabad-scoped check produces false negatives.

### Batching + cost

- Group rows by `retrieved_shabad_id` (from DB-pipeline audit sidecar) so each batch shares a shabad context in the prompt.
- Within each shabad group, chunk at **batch_size=30**.
- **Dedup identical captions** before sending (many caption values repeat verbatim in kirtan).
- Prompt caching is implicit on Gemini — the repeated shabad prefix within a video → 25% input cost.

**Measured cost** (confirmed on 100-row sample, batch=30):
- 31 unique LLM candidates, 3 batches, $0.0185 total.
- Extrapolated to full 300h (~60K rows, ~23% candidate rate → ~14K LLM-touched rows): **~$2.30–3.00 total**.

### Implementation (new script)

`scripts/llm_canonical_pass.py`:
1. Read the audit sidecar produced by `add_canonical_column.py`.
2. Filter rows with `decision IN (unchanged, review)`.
3. Group by `retrieved_shabad_id`, dedup captions, chunk at 30.
4. For each batch: build prompt, call Gemini, parse JSON, run verifier.
5. Write `llm_sidecar.parquet` keyed on `clip_id`: `final_text_llm`, `llm_model`, `llm_verified`.
6. A final merge step joins both sidecars into the HF dataset as new columns.

Rerunning is cheap (~$3) and idempotent; can iterate on prompts/models without touching the DB-grounded columns.

### Downstream training use

For ASR training labels, pick per row:

- `decision IN (matched, replaced)` → use `final_text` (DB-grounded, highest confidence)
- `decision IN (unchanged, review)` AND `llm_verified` → use `final_text_llm`
- `decision IN (unchanged, review)` AND NOT `llm_verified` → use `text` (original caption) OR drop the row

This gives 3 confidence tiers, explicit per row, with traceability back to the source of each correction.

## 12. Out of scope (long-term follow-ups)

- Fine-tuning a small seq2seq corrector (ByT5-small) on the `(caption, final_text)` pairs this pipeline produces. Valuable follow-up project; not required for the canonical column.
- Video-level shabad segmentation (detecting when a ragi transitions from shabad A to shabad B mid-video). Currently handled implicitly via the ±1 retrieval window; a dedicated segmentation pass would improve edge cases but is a bigger project.
- An interactive review UI for the `review`-decision rows. Deferred until after initial dataset ships and review-row volume is empirically known.
