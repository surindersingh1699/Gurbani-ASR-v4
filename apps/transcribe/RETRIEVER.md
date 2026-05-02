# Transcribe App — Retriever Notes

**Scope:** `apps/transcribe/retriever.py` — the 2-layer MuRIL + FAISS retriever
powering the live transcribe app's shabad search. Separate from
`apps/live_lab/tracker.py` (EMA tracker) and `scripts/canonical/retrieval.py`
(offline STTM pipeline).

**Last updated:** 2026-04-21

---

## Search modes

`search_topn(text, n, mode=...)` dispatches on `mode`:

| Mode | What it does |
|---|---|
| `banidb` | Remote BaniDB API (currently returns empty — fallback path) |
| `literal` | Pure char-4gram overlap over the 56k-tuk index |
| `semantic` (default) | MuRIL → FAISS tuk search + shabad-level confirm layer |
| `semantic+literal` | Semantic cascade, then blend a literal overlap score into the top-K rerank |

Literal index (`_tuk_4grams`) is lazily built once per session via
`_ensure_literal_index()` — a dict of `{row_idx: set[4gram]}` over all 56,152
tuks. Rough cost: ~2s build, ~50 ms to scan all tuks per query.

---

## Known failure (2026-04-21): default + semantic+literal miss verbatim tuks

### Symptom
On ang 694 (observed in live-mic session), the ragi sings a tuk verbatim but
neither `semantic` nor `semantic+literal` surfaces it. Top-20 FAISS hits all
tie at cosine ≈ 0.897 on unrelated tuks.

### Root cause
MuRIL's embedding for repeated chant-style kirtan text drifts into a dense
"chant cluster" where many unrelated tuks become near-equidistant from the
query. Because `semantic+literal` only reranks the MuRIL top-K, if the
correct tuk isn't in that pool, the literal boost has nothing correct to
rerank. Improving the rerank doesn't help — the candidate set is the
bottleneck.

The `literal` mode alone *does* find the correct tuk (4-gram overlap is high
for verbatim matches), which confirms the tuk is discoverable — just not via
MuRIL top-K on this class of query.

---

## Proposed fix: candidate fusion

Pull top-K from **both** MuRIL and the literal 4-gram index, union the row
sets, then rerank the union by the blended score. That way:

- MuRIL's paraphrase tolerance stays intact (its top-K is still in the pool).
- Any verbatim-match tuk is guaranteed to be in the pool via the literal
  index (which already exists from `literal` mode — no new index build).
- The existing blend formula (`best_cosine + LITERAL_WEIGHT * overlap`) does
  the final ranking unchanged.

### Shape (~15 lines inside `_search_semantic`)

```python
D, I = self._tuk_idx.search(q, k)                       # semantic top-K
self._ensure_literal_index()                            # lazy, once/session
lit_scored = [(r, _overlap(q_grams, g))
              for r, g in self._tuk_4grams.items()]
lit_scored.sort(key=lambda t: t[1], reverse=True)
lit_rows = [r for r, s in lit_scored[:k] if s > 0]

candidates = set(I[0].tolist()) | set(lit_rows)         # fusion
# … compute semantic cosine for lit-only rows, blend, sort, return top-n
```

### Trade-offs
- First `semantic+literal` call pays the one-time ~2s literal-index build
  (already the case for `literal` mode; amortised across the session).
- Every query scans all 56k tuks for overlap (~50 ms, fine for live-mic).
- Blending two score distributions can surface literal false positives on
  short queries with coincidental n-gram overlap. **Mitigation:** keep a
  semantic-cosine floor (`TUK_SCORE_FLOOR`, currently 0.50) on lit-only
  candidates so a random high-overlap tuk can't beat a real paraphrase
  match.

### Suggested rollout
1. Ship behind a flag (e.g. a 4th mode `semantic+literal+fusion` or an env
   var `SURT_FUSION=1`) so we can A/B it against current `semantic+literal`
   on the known-broken queries (ang 694 + any others collected).
2. Once validated on a handful of failure cases, promote fusion to the
   default behaviour of `semantic+literal` (and optionally `semantic`).
3. Leave `literal` and `banidb` modes untouched.

---

## Why the retriever is in `apps/transcribe/` not `apps/live_lab/`

- `apps/transcribe/retriever.py` serves the **transcribe** app's shabad
  search (search-topn + lock-mode pointer via `score_within_shabad`).
- `apps/live_lab/tracker.py` is a different consumer of the same FAISS
  indices for the EMA-tracker live-mic demo — it does its own
  MuRIL+FAISS+EMA smoothing and does not call this retriever.
- Both share the on-disk artifacts under `index/` (built by
  `scripts/01_build_tuk_index.py` + `scripts/02_build_shabad_index.py`).

If the fusion fix proves out, consider whether `tracker.py` has the same
candidate-pool blind spot — it uses the same MuRIL embedding space so the
chant-cluster pathology is likely the same.

---

## Lock / pointer model (2026-04-21)

Two distinct score floors govern the locked state. Keeping them decoupled is
the whole reason the pointer can follow the ragi on noisy ASR.

| Constant | Default | Env var | Role |
| --- | --- | --- | --- |
| `LOCK_SCORE_FLOOR` | 0.80 | `SURT_LOCK_SCORE` | Literal overlap a top hit must hit (UNLOCKED) before it becomes lock-eligible. |
| `UNLOCK_SCORE_FLOOR` | 0.50 | `SURT_UNLOCK_FLOOR` | Within-shabad overlap below which a window is a "miss". N misses in a row → unlock. |
| `POINTER_MOVE_FLOOR` | 0.15 | `SURT_POINTER_MOVE` | Within-shabad overlap required to *move the pointer* to the best-matching pangti. Weaker than UNLOCK_SCORE_FLOOR on purpose. |

Why three floors: UNLOCKED needs a strong literal signal to trust a shabad
pick out of ~15 k. LOCKED needs only a weak signal to move the pointer —
we already committed to the shabad, so within it, even partial/garbled ASR
is enough to identify the current pangti.

### Per-path behavior (all three share the same `_refresh_matches` / `_handle_locked`)

| Input path | Query source when locked | Update cadence | Pointer-advance benefit |
| --- | --- | --- | --- |
| **Live mic** (`on_stream`) | `st.tentative or st.committed` | Every throttle tick (~1.2 s) | Biggest — pointer advances between commits, not only every ~10 s. |
| **Play from URL / loaded file** (`_on_stream_url`) | `st.committed` (no tentative in this path) | Every playhead-sized window (~10 s) | Medium — windows with weak/noisy ASR now advance instead of being misses. |
| **Upload file** (`on_upload`) | One-shot; whole audio transcribed once | Single call, no pointer to advance over time | None. If you want per-line advancement on a file, load it in the **Play from file / URL** tab instead. |

### Query source differs by state

- **LOCKED:** prefer `st.tentative` (freshest ASR of the current pangti) over
  `st.committed`. Fall back to committed only when tentative is empty (e.g.
  URL/playback path, which never sets tentative).
- **UNLOCKED:** prefer `st.committed` (longer, cleaner tail) — picking the
  right shabad from 15 k needs more context than identifying a pangti inside
  one locked shabad.

### Relationship to the UI

- `_handle_locked` sets `st.locked_tuk_row` and rewrites `st.matches` to a
  single locked hit with `highlight_idx` pointing at the current pangti.
- The stage hero renders the full shabad with the highlighted line.
- Unlock decision is counted **independently** of pointer advance — a
  window can move the pointer (overlap ≥ 0.15) and still count as a miss
  toward unlock (overlap < 0.50). That's intentional: follow the ragi on
  partial signal, but still release the lock if signal stays weak for
  `unlock_misses_target` windows (default 3).

### Tuning knobs

- Pointer chatters on noise → raise `SURT_POINTER_MOVE` (e.g. 0.25).
- Pointer lags on clean audio → raise `SURT_POINTER_MOVE` is the wrong fix;
  usually means hero isn't rendering. Check that `st.locked_shabad_id` is
  set and that `hero_mode` short-circuits on lock (see below).
- Unlocks too eagerly → raise `SURT_UNLOCK_FLOOR` or `unlock_misses_target`.

### Gotcha: hero card must unconditionally render while locked

Pointer-advance writes `ui_score = 0.5 + 0.5 * overlap`. On weak signal
(overlap 0.15–0.40) that lands below the default `hero_threshold = 0.70`,
so the stage falls back to "transcript-only" — user sees committed text
but no highlighted shabad, and reports "pointer isn't moving" even though
`st.locked_tuk_row` is updating correctly. Fix: gate `hero_mode` on
`st.locked_shabad_id is not None` in addition to the score threshold.
