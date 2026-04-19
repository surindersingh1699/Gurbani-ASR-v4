# STTM + LLM Pipeline — Learnings & Current State

**Last updated:** 2026-04-19
**Branch:** `codex/complete-phase-4-work`
**Scope:** Gurmukhi caption correction for the YT-captions kirtan + sehaj-path datasets using STTM (local alignment) + LLM fallback (Gemini / GPT-5.4-nano).

---

## TL;DR

- Built a 9-module canonical-text pipeline (`scripts/canonical/`) + 3 CLIs. 87 tests green.
- **STTM Stage 1 alone produces ~4-10% bad rewrites** on the kirtan sample due to loose fix-eligibility + caption-reorder pathology.
- **LLM Stage 2 (Gemini 2.5 Flash Lite) added ~5 new hallucinations** because the verifier didn't check caption preservation or length growth.
- **GPT-5.4-nano at $10 / Gemini 3 Flash Preview at $62 for 230k rows.** Gemini 3 is 6× more expensive and 7× slower but had zero true bad rewrites on the 46-row test. GPT had 2 real hallucinations.
- **Ship the additive cleanup first** (`scripts/clean_dataset.py`) — no STTM, no LLM, zero risk. Adds `text_cleaned` + `is_simran` + `drop_candidate` columns to the 4 HF datasets. Currently **mid-push** to `*-clean` repos.
- **Fix STTM algorithm issues before running STTM+LLM at scale.** Four specific fixes identified below.

---

## Current state (2026-04-19)

### Shipped & tested
- `scripts/canonical/` — 9 modules (gurmukhi_skeleton, sttm_index, preprocess, waheguru, simran, sirlekh, retrieval, align, decision, config, pipeline, llm_pass, merge_hf).
- `scripts/add_canonical_column.py` — Stage 1 CLI.
- `scripts/merge_canonical_into_hf.py` — Stage 2 + merge CLI.
- `scripts/canonical_dry_run.py` — 100-row sample validator.
- `scripts/clean_dataset.py` — **additive** cleanup (no STTM, no LLM; safe).
- Runbook at `docs/superpowers/runbooks/canonical-text-pipeline.md`.
- Full spec at `docs/superpowers/specs/2026-04-18-kirtan-canonical-text-column-design.md`.

### In flight
- **4 HF pushes running on Hetzner server** (as of 2026-04-19):
  - `gurbani-kirtan-yt-captions-eval-clean`
  - `gurbani-sehajpath-yt-captions-eval-clean`
  - `gurbani-sehajpath-yt-captions-clean`
  - `gurbani-kirtan-yt-captions-300h-clean`
  - These are **clean-only** (no STTM, no LLM). Each is a new repo; originals untouched.

### Blocked / pending
- Full STTM+LLM pass on the 4 datasets — blocked on 4 algorithm fixes below.
- Consensus-vote pipeline (STTM + GPT + Gemini 2-of-3) — not built yet.

---

## Dataset dry-run stats (all 4 datasets)

| Dataset | Rows | `>>` markers | `<unk>` artifacts | Waheguru norm | Simran flag | Drop candidates |
|---|---|---|---|---|---|---|
| kirtan-yt-captions-eval | 573 | 0 | 7 (1.2%) | 0 | 0 | 0 |
| sehajpath-yt-captions-eval | 444 | 0 | 38 (8.6%) | 0 | 0 | 0 |
| sehajpath-yt-captions (full) | 63,085 | 94 | 14,737 (23%) | 0 | 0 | 95 |
| **kirtan-yt-captions-300h** | **228,921** | **37,198 (16%)** | **1,620** | **47** | **13,295 (5.8%)** | **8,901 (3.9%)** |

**Key insight:** kirtan-300h has 16% of rows with `>>` split markers — these are the primary source of both STTM mis-alignments and LLM hallucinations. Stripping `>>` at clean stage is a big quality win regardless of downstream corrections.

---

## STTM Stage 1 learnings

### What works well
1. **Matra fixes** land cleanly. Examples from the 88-row test:
   - `ਸਭ ਜਗਤ ਵਣਜਾਰਾ ਰਾਮ ਰਾਜ` → `ਸਭੁ ਜਗਤੁ ਵਣਜਾਰਾ ਰਾਮ ਰਾਜੇ`
   - `ਹਮ ਬਾਰਕ ਤੂੰ ਗੁਰ ਪਿਤਾ ਹੈ` → `ਹਮ ਬਾਰਿਕ ਤੂੰ ਗੁਰੁ ਪਿਤਾ ਹੈ`
2. **Sirlekh normalization** fires correctly on sehaj:
   - `ਸੀ ਰਾਗ ਮਹਲਾ ਪੰ` → `ਸਿਰੀਰਾਗੁ ਮਹਲਾ ੫` (matched shabad GFF)
3. **Simran short-circuit** caught all-waheguru rows correctly.
4. **`<unk>` artifact stripping** — zero leaks into `final_text`.
5. **4-gram TF-IDF retrieval** found the correct shabad in 46/46 test cases; scores ranged 13–28, well above the 2.0 threshold.

### Bad-rewrite patterns (6/46 = ~13% on test, mostly fixable)

| Pattern | Example | Root cause | Fix |
|---|---|---|---|
| **Word substitution with different-length skeleton** | `ਹਰਿ` → `ਕਰੇ`, `ਤੂੰ` → `ਤਿਨੑੀ` | `_fix_eligible` accepts lev≤1 across different lengths | Require `len(cap_skel) == len(sgs_skel)` for 1-consonant swaps |
| **Reordered-caption pathology** | `ਮੇਰੇ ਗੁਰ ਗੋਪਾਲਾ ਸੁਖ ਸਾਗਰ` → scripture word at wrong position | Semi-global NW picks SOMETHING when caption reorders vs scripture | Detect reorder (>2 delete ops in run) → fall back to `unchanged` |
| **Over-confident `replaced` on 4/5 match rate** | accept_threshold=0.92 lets 1-bad-fix through | 4 matches + 1 bad fix = 100% if 0 deletes | Raise `accept_threshold` → 0.95, lower `review_threshold` → 0.60 |
| **Split-caption fragments** | `ਤੂੰ ਤੂ ਧਨੀ >> ਸਚ ਸਾਹੁ ਹਮਾਰਾ` → `ਤਿਨੑੀ ਧੁਰਿ ਸਚੁ ਸਾਹੁ ਹਮਾਰਾ` | `>>` caused retrieved shabad's first words to "replace" the repeat | Strip `>>` before retrieval (already done in cleanup) + tighter fix-eligibility |

### Retrieval is effectively perfect
- In every test case the best-scoring shabad was correct (EZ7, 79E, 6NM, 3SX, X5G, GFF, HYC, AMT, X5N).
- 4-gram skeleton TF-IDF + video-memory prior works.
- Don't change retrieval; focus fixes on alignment.

---

## LLM Stage 2 learnings

### Verifier was the weakest link
Original `llm_pass.py::verify_llm_output` only checked:
1. Token count within ±2 of caption
2. Each output token's skeleton appears in the shabad (lev ≤ 1)

**Missed**:
- **Hallucination** — `ਸਾਜਿਆ ਹਾ` → `ਸਭ ਭਾਂਡੇ ਤੁਧੈ ਸਾਜਿਆ` (extra word within ±2 drift, all tokens in shabad)
- **Reordering** — `ਤੂ ਧਨੀ ਸਚ ਸਾਹੁ ਹਮਾਰਾ` → `ਸਚੁ ਸਾਹੁ ਹਮਾਰਾ ਤੂੰ ਧਣੀ`
- **Length growth** — LLM completing partial fragments into full scripture lines

### Fixed verifier (v2)
Added two checks:
- **Monotonic caption preservation**: ≥60% of caption tokens must appear in LLM output in same relative order (skeleton-lev ≤ 1)
- **Length growth cap**: output length ≤ caption length + 1 (was ±2)

### Provider comparison (46-row kirtan sample)

| Model | Bad rate (verifier) | True bad rate (manual) | Sample cost | **230k rows** | Avg latency | JSON reliability |
|---|---|---|---|---|---|---|
| **GPT-5.4-nano** | 15% (7 flagged) | **4%** (2 real — r009 + r037 hallucinations) | $0.002 | **$9.77** | 0.9s | Perfect (native JSON schema) |
| **Gemini 3 Flash Preview** | 9% (4 flagged) | **0%** (all false positives) | $0.012 | **$61.52** | 6.5s | Poor — truncation needs regex extraction |
| Gemini 2.5 Flash Lite | ~30% | ~10-15% | — | ~$5 | ~1-2s | OK when `response_mime_type` set |

**Observations**:
- On identical inputs, GPT-5.4-nano and Gemini 3 Flash Preview produced **near-identical corrections for 42 of 46 rows**.
- GPT-5.4-nano's 2 real hallucinations both occurred on `>>`-split fragments — the model tried to "complete" the fragment from shabad context.
- Gemini 3 Flash Preview was bulletproof on this sample but 6× more expensive and 7× slower.
- The large "bad rate" gap (15% vs 9%) was almost entirely **false positives** from my strict verifier (e.g. flagging `ਭਗਤਿ` as "invented" when it's valid scripture just not in that specific shabad).

### Prompt design — what worked
The tightened system prompt (v2) emphasised:
1. **Preserve word order + count** (no ±1 drift allowed)
2. **Spell-corrector only, not translator** (stop completing fragments)
3. **Only scripture words with skeleton-lev ≤ 1 as replacements** (no different-length swaps)

With v2 prompt, bad rewrites on 46-row sample:
- GPT-5.4-nano: 6 → 4 → **2 true bad** (-70%)
- Gemini 3 Flash Preview: all true bad eliminated (truncation false-positives remained)

### Prompt design — what didn't
- **Batching 15 rows per shabad** (in original `llm_pass.py`) — amortises shabad context across calls, ~10× cost reduction. But: single-row prompts were more consistent in output format in this test. Need to re-test batching with v2 prompt.
- **Chaining S1 → S2** (feed STTM output as LLM input) — was the original design. Worse than feeding raw caption to LLM independently, because S2 now sees a potentially-corrupted input.

### Fix: pass RAW caption to LLM, not S1 output
Originally `llm_pass.py` received `r["final_text"]` (the STTM-corrected text). Should receive `r["text"]` (the raw YT caption). Ensures Stage 2 is an independent correction path, not a patch-over.

---

## Cost modelling

### Gemini 2.5 Flash Lite (original plan)
- Input: $0.10/M, Output: $0.40/M
- Per-row tokens (batched 15 per call, shabad amortized): ~100 in / ~40 out
- **230k rows: ~$6**

### GPT-5.4-nano (current cheapest viable)
- Input: $0.05/M, Output: $0.40/M (realtime)
- With OpenAI Batch API: $0.025/$0.20 (50% off, 24h SLA)
- Per-row tokens (per-call, not batched): ~550 in / ~35 out
- **230k rows realtime: $9.77**
- **230k rows batch: ~$5**

### Gemini 3 Flash Preview
- Input: $0.30/M, Output: $2.50/M
- Per-row tokens: ~580 in / ~35 out
- **230k rows: $61.52**

### Consensus strategy (recommended for quality)
- Run GPT-5.4-nano on everything (~$10 for 230k)
- For rows where GPT disagrees with STTM (~20-30% ≈ 60k rows), also run Gemini 3 Flash Preview (~$16)
- Use consensus: if 2 of 3 agree, accept. If all disagree, keep `text_cleaned` as-is.
- **Total: ~$26 for near-zero bad rewrites** across 230k rows

### Hard-line minimum (pure STTM, no LLM)
- $0. Uses Gemini / GPT only if user wants a second opinion. But leaves ~10% bad rewrites in dataset.

---

## Additive cleanup script (what's currently running)

`scripts/clean_dataset.py` — safe, no STTM, no LLM. Operations:
1. Strip `>>`, `<unk>`, `<un`, `<k`, `unk>` artifacts from `text`
2. Remove non-Gurmukhi characters (keep only `U+0A00–U+0A7F` + whitespace)
3. Waheguru skeleton-normalization (ਵਾਹੇਗੁਰੂ / ਵਾਹਿਗੁਰ / etc. → ਵਾਹਿਗੁਰੂ)
4. Flag (don't drop) simran-heavy rows
5. Flag (don't drop) rows too short (<1s) or too few tokens (<2 Gurmukhi)

**Does NOT:**
- Modify `text` column (original preserved verbatim)
- Drop any rows
- Touch audio or other columns
- Run retrieval or alignment

**Adds 4 new columns:** `text_cleaned`, `is_simran`, `drop_candidate`, `n_waheguru_normalized`.

### Streaming mode
`--streaming` option reads rows without downloading audio shards. Lets you get cleanup stats on a 450h dataset in ~5 min without the ~65 GB audio download. But can't `--push` in streaming mode (need full data to re-pack parquets).

---

## Known bugs / open issues

### Must fix before STTM+LLM scale run
1. **`align.py::_fix_eligible`** — require equal-length skeletons for 1-consonant swaps.
2. **`decision.py::DecisionConfig`** — tighten thresholds (accept 0.92→0.95, review 0.75→0.60).
3. **`llm_pass.py::verify_llm_output`** — add caption-preservation + length-growth checks.
4. **`llm_pass.py::run_llm_pass`** — pass `row["text"]` (raw) not `row["final_text"]` (S1 output) to the LLM prompt.

### Non-blocking, nice-to-have
5. **LLM provider abstraction** — swap Gemini ↔ OpenAI without rewriting pipeline glue.
6. **Exact-match short-circuit** — if `text_cleaned` exact-matches some SGGS line, skip both STTM and LLM, save ~10-20% of work.
7. **Batch API integration** — OpenAI's 50% discount + 24h SLA is ideal for one-shot dataset cleanup.
8. **Dataset column filter** — load only `text` + `duration_s` columns, skip audio shards. Would speed up dry-runs 10-100×.

### Bugs in messy-commit `9abab1e`
- One Task 16 agent over-staged 44 files in a single commit ("feat(canonical): dry-run validation script"). Content is correct but commit history is ugly. Low priority to split.

---

## Operational references

### Hetzner server
- `ssh root@138.199.174.101` — keys already authorized
- Repo: `/root/Gurbani_ASR_v4`, venv: `/root/venv/bin/python`
- Env: `set -a && source /root/.env && set +a` loads HF_TOKEN, GEMINI_API_KEY, OPEN_API_KEY
- Logs: `/root/clean_logs/<repo>-push.log` (current 4 pushes)
- Free disk: 187GB of 301GB

### Local Mac
- Cache-heavy: `~/.cache/huggingface` can balloon past 50 GB on dataset downloads; clear periodically
- `/private/tmp/claude-501/` accumulates task transcripts; safe to trim

### HF dataset patterns
- Original: `surindersinghssj/gurbani-kirtan-yt-captions-300h`
- Cleaned: `<original>-clean` (convention)
- STTM-corrected: TBD (likely `<original>-canonical` once algo fixes land)

---

## Next steps (in order)

1. ☐ Wait for 4 additive-cleanup pushes to finish
2. ☐ Sanity-check one row from each `*-clean` repo: original `text` byte-matches source, `text_cleaned` is correctly stripped
3. ☐ Ship the 4 STTM algorithm fixes (bullets 1-4 above)
4. ☐ Re-test STTM+LLM on 46-row kirtan sample, target true-bad-rate ≤ 2%
5. ☐ Add provider abstraction + wire up GPT-5.4-nano batch API
6. ☐ Add exact-match short-circuit
7. ☐ Run STTM+LLM on `*-clean` repos, add `text_sttm` + `text_llm` columns as a third additive layer
8. ☐ Optional: retrain Surt on `text_cleaned` first to measure baseline lift; then on `text_llm` to measure STTM+LLM lift
