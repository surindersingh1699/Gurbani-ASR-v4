# Locked-pangti sync — design

**Date:** 2026-04-22
**Scope:** `apps/transcribe/` — four targeted fixes so the locked-shabad
experience is fast, correct, and STTM stays in sync with the pangti the
ragi is actually singing.

## Problems

1. **STTM stuck on first pangti.** Once the app locks on a shabad, STTM
   opens it but the projector highlight stays on the first pushed
   `verseId` even when our local pointer (`st.locked_tuk_row`) has clearly
   advanced. Confirmed in conversation.
2. **Pointer advance is too slow.** When the ragi starts a new pangti
   (or loops back to Rahao), the in-app highlight takes ~3-5s to follow.
   Target: ≤2s.
3. **Raw transcript doubles at the commit seam.** `committed` is built by
   naive append of every Whisper commit-window output. With 2s carry-over
   between adjacent 10s windows, the same audio is transcribed twice and
   the duplicated words show up glued together in the visible strip.
4. **Default retrieval mode is `literal`.** Char-4gram-only is fast but
   brittle on noisy ASR — paraphrase-tolerant matching with MuRIL +
   literal rerank is a strict upgrade for kirtan use.

## Non-goals

- UI layout changes — confirmed in conversation that the current hero +
  transcript-strip layout is fine.
- Lock state machine changes — auto-lock and unlock-floor logic stays as
  designed.
- Backfilling the dedup retroactively over already-committed text. Only
  forward seams between new commit windows are merged.
- Whisper decoder changes — explicitly ruled out by
  `feedback_whisper_gurbani_decoder.md` (temperature fallback +
  `no_repeat_ngram_size` wreck legitimate Gurbani repetition).

## Design

### 1. STTM follows the locked pointer

**Problem source.** Auto-push dedup is keyed on `shabadId` only:

```python
# apps/transcribe/app.py:1713
sid = top.get("shabadId")
if not sid or sid == st.last_pushed_sid:
    return
```

So after the first push of a shabad, no further `/api/bani-control`
calls are made — even though `_handle_locked` keeps updating
`st.locked_tuk_row` and the rendered match dict carries a fresh
`verseId`.

**Fix.** Dedup on `(shabadId, verseId)`.

- Replace `StreamState.last_pushed_sid: int | None` with
  `StreamState.last_pushed_key: tuple[int, int] | None`.
- In `_try_auto_push`: build `key = (int(top["shabadId"]),
  int(top["verseId"]))`; push when `key != st.last_pushed_key`.
- Apply the same change in the two ad-hoc auto-push call sites inside
  `_on_stream_url` and `on_play_sync` (they each maintain a local
  `last_pushed_sid` variable today).
- `_unlock` clears `last_pushed_key`; `_lock` clears it too so a fresh
  lock-on-same-shabad re-pushes with the right verseId.
- `on_clear` clears it.

**Outcome.** Every time `_handle_locked` advances the pointer, the next
auto-push tick sees a new `verseId`, sends a fresh payload, and STTM's
highlight follows the ragi.

### 2. Faster pointer advance (target ≤2s)

**Problem source.** Three live-mic constants conspire to slow detection:

| Constant | Value | Effect |
| --- | --- | --- |
| `min_transcribe_s` | 3.0 | Whisper won't run on <3s of buffered audio |
| `throttle_s` | 1.2 | Whisper runs at most once per 1.2s |
| `carry_over_s` | 2.0 | First 2s of every new buffer is leftover from the prior pangti, so the prefix matcher anchors on stale text |

Worst-case current detection ≈ ~3-5s after a pangti transition.

**Fix.** Add a dedicated fast-pointer path that runs **only when
LOCKED**. The full commit pipeline is unchanged for unlocked mode.

- New `StreamState` fields:
  - `fast_pointer_min_s: float = 1.2` — minimum buffered audio for the
    fast path
  - `fast_pointer_throttle_s: float = 0.6` — minimum gap between fast
    calls
  - `last_fast_pointer_t: float = 0.0` — separate throttle clock from
    the regular `last_call_t`
- New `_refresh_pointer(st, tail_text)` helper:
  - Bypasses `_retrieval_query` (no 140-char tail trim — the tail audio
    IS the query).
  - Calls `score_within_shabad_prefix` and `score_within_shabad`
    directly via the same pointer/unlock decision logic in
    `_handle_locked` (factor the pointer/unlock body into a private
    helper that takes a query string, so both refresh paths reuse it).
- In `on_stream` (live mic loop): when `st.locked_shabad_id` is set,
  use a parallel branch:
  - Gate by `fast_pointer_min_s` and `fast_pointer_throttle_s`.
  - Transcribe a **fresh tail slice**:
    `tail = st.buffer[-int(1.5 * TARGET_SR):]`. No carry contamination.
  - Pass the resulting text to `_refresh_pointer`.
  - Still triggers `_try_auto_push` afterwards.
- The slower commit-and-tentative path keeps running in parallel —
  it's still responsible for `committed` text, unlock decisions on long
  silences, and the unlocked-mode retrieval.
- **Scope:** live mic only. The `_on_stream_url` and `on_play_sync`
  paths are intentionally not touched — both are gated by the
  browser/file playhead in 10s windows, so a faster pointer would have
  no real-time benefit and would race ahead of audio playback.

**Why hallucinations are tolerable on short buffers here.** Matches are
constrained to the ~30 tuks of the locked shabad. A noise-driven
"ਵਾਹਿਗੁਰੂ ਵਾਹਿਗੁਰੂ" loop that doesn't match any tuk simply produces no
pointer advance — and sustained noise still trips the existing 3-miss
unlock floor in `_handle_locked`. The `_suppress_repeat_hallucination`
gate also still fires.

**Outcome.** New-pangti detection ≈ 1.2s of audio + ~0.3s inference +
≤0.6s throttle ≈ **1.5–2.1s** worst-case from the moment the ragi
starts the next line.

### 3. Raw-transcript seam dedup

**Problem source.** Commit-and-carry transcribes the carry-over twice.
Whisper is context-sensitive, so the two transcriptions differ slightly
and `committed` ends up with near-duplicates glued together.

**Fix.** Word-level longest-suffix/prefix merge.

- New helper in `apps/transcribe/app.py`:

```python
def _merge_committed(prev: str, new: str, *, max_overlap_words: int = 12) -> str:
    """Append `new` to `prev`, dropping the longest token-level overlap
    between prev's suffix and new's prefix.

    Word-level (not char-level) so we never split a Gurmukhi grapheme
    cluster mid-word. Capped at `max_overlap_words` to keep the cost
    constant on long buffers.
    """
    prev = (prev or "").strip()
    new = (new or "").strip()
    if not prev:
        return new
    if not new:
        return prev
    p = prev.split()
    n = new.split()
    cap = min(len(p), len(n), max_overlap_words)
    k = 0
    for i in range(cap, 0, -1):
        if p[-i:] == n[:i]:
            k = i
            break
    return (prev + " " + " ".join(n[k:])).strip() if k < len(n) else prev
```

- Replace every `st.committed = (st.committed + " " + txt).strip()` with
  `st.committed = _merge_committed(st.committed, txt)`. Affected sites
  (verified by grep):
  - Live mic commit: `app.py:1576`
  - URL stream commit (live + drain + final tail): `app.py:1910`,
    `app.py:1958`, `app.py:1966`
  - File play-sync commit: `app.py:2017`
- Tentative is unaffected — it's overwritten each call, never appended.

**Tests.** New `tests/transcribe/test_merge_committed.py`:

- empty prev / empty new
- no overlap → straight join with single space
- 2-word overlap → drops exactly those 2 tokens
- new is fully contained in prev's suffix → returns prev unchanged
- overlap longer than `max_overlap_words` → only `max_overlap_words`
  trimmed (preserves rest)
- whitespace normalization on input

**Outcome.** Raw transcript strip stops doubling at every commit
boundary; each pangti appears once.

### 4. Default retrieval = `semantic+literal`

**Problem source.** `DEFAULT_MODE = MODE_LITERAL` in
`apps/transcribe/retriever.py:74` — char-4gram only. Brittle on noisy
ASR. The semantic+literal mode (MuRIL embeddings + 2-layer FAISS +
4-gram rerank) is strictly better for kirtan retrieval at ~50ms/call,
which is well under the 1.2s throttle.

**Fix.**

- `apps/transcribe/retriever.py:74` — `DEFAULT_MODE = MODE_SEMANTIC_LITERAL`.
- `apps/transcribe/app.py:1032` — dropdown initial `value=DEFAULT_MODE`
  (already references the constant; will pick up the new value
  automatically). Verify no hardcoded "literal" string elsewhere in the
  UI defaults.
- `StreamState.retrieval_mode: str = DEFAULT_MODE` — already references
  the constant.

**Outcome.** First-launch users get the better retriever without
touching settings. Existing users who have explicitly switched modes are
unaffected (mode is per-session, not persisted).

## Files touched

- `apps/transcribe/app.py` — STTM dedup key, fast-pointer path,
  `_merge_committed`, replace four `committed` append sites.
- `apps/transcribe/retriever.py` — `DEFAULT_MODE` constant.
- `tests/transcribe/test_merge_committed.py` (new) — seam-dedup unit tests.

## Verification

Manual UAT (live mic against a known shabad):

1. Start the app, mic on, sing a known shabad.
2. After auto-lock, advance pangti. Confirm:
   - In-app highlight follows within ~2s.
   - STTM projector highlight follows the same line (verify by checking
     STTM's UI — should change line, not just the shabad).
3. Loop back to Rahao. Same expectation.
4. Inspect raw transcript strip — no obvious doubling at 10s
   boundaries.
5. Reload the app fresh — settings dropdown shows
   "Semantic + literal …" as the selected mode.

Automated:

- `pytest tests/transcribe/test_merge_committed.py` — all cases pass.
- Existing `apps/transcribe/` tests still pass.
