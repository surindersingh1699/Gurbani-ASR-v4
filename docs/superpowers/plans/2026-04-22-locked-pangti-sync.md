# Locked-pangti sync — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the locked-shabad experience fast and STTM-synchronised — STTM follows the locked pointer, the in-app pointer advances within ~2 s of a new pangti, the raw transcript stops doubling at commit seams, and a stronger retrieval mode is the new default.

**Architecture:** Four targeted edits to `apps/transcribe/`. (1) Auto-push dedup keyed on `(shabadId, verseId)` instead of `shabadId` alone, so STTM re-pushes when the locked pointer moves. (2) New live-mic-only "fast pointer" path that transcribes a 1.5 s tail of audio without carry-over, gated by tighter `min_transcribe_s` / `throttle_s` constants. The pointer-advance and unlock decisions are factored apart so the fast path can advance without affecting unlock-miss bookkeeping. (3) Word-level longest-suffix/prefix merge for `committed` to dedup the 2 s carry-over between adjacent commit windows. (4) `DEFAULT_MODE = MODE_SEMANTIC_LITERAL` in the retriever.

**Tech Stack:** Python 3.10+, Gradio, NumPy, faster-whisper / openai-whisper backends, pytest 8.

**Spec:** [docs/superpowers/specs/2026-04-22-locked-pangti-sync-design.md](../specs/2026-04-22-locked-pangti-sync-design.md)

---

## File map

- **Modify:** `apps/transcribe/retriever.py` — change `DEFAULT_MODE` constant.
- **Modify:** `apps/transcribe/app.py` — add `_merge_committed`, replace four `committed` append sites, rename `last_pushed_sid` → `last_pushed_key` and propagate to `_try_auto_push` / `_on_stream_url` / `on_play_sync` / `_lock` / `_unlock` / `on_clear`, factor `_handle_locked` into `_pointer_advance` + `_check_unlock`, add `fast_pointer_*` fields + `_refresh_pointer` helper, wire fast-pointer branch into `on_stream`.
- **Create:** `tests/transcribe/__init__.py` — empty file so pytest discovers the new package.
- **Create:** `tests/transcribe/test_merge_committed.py` — unit tests for the seam-dedup helper.

No new modules; all changes co-locate with what they serve.

---

## Task 1: Switch default retrieval mode to semantic+literal

Trivial first task — single-line constant change. Validates the dev loop end-to-end before touching real logic.

**Files:**
- Modify: `apps/transcribe/retriever.py:74`

- [ ] **Step 1: Apply the constant change**

In `apps/transcribe/retriever.py`, change line 74:

```python
DEFAULT_MODE = MODE_LITERAL
```

to:

```python
DEFAULT_MODE = MODE_SEMANTIC_LITERAL
```

- [ ] **Step 2: Verify the constant resolves**

Run: `python -c "from apps.transcribe.retriever import DEFAULT_MODE, MODE_SEMANTIC_LITERAL; assert DEFAULT_MODE == MODE_SEMANTIC_LITERAL; print('OK', DEFAULT_MODE)"`

Expected output: `OK semantic+literal`

- [ ] **Step 3: Confirm no hardcoded "literal" string in UI defaults**

Run: `grep -n '"literal"' apps/transcribe/app.py`

Expected: no matches (the dropdown reads `value=DEFAULT_MODE` at app.py:1032, which now picks up the new value automatically).

- [ ] **Step 4: Commit**

```bash
git add apps/transcribe/retriever.py
git commit -m "feat(transcribe): default retrieval mode to semantic+literal

MuRIL + 4-gram rerank handles paraphrases / noisy ASR better than
char-only literal mode at ~50 ms/call, well under the 1.2 s throttle."
```

---

## Task 2: TDD `_merge_committed` helper

Build the seam-dedup primitive as a pure function with full test coverage before wiring it into the live paths. Test file goes under a new `tests/transcribe/` package so pytest discovers it via the existing `testpaths = ["tests"]` config.

**Files:**
- Create: `tests/transcribe/__init__.py`
- Create: `tests/transcribe/test_merge_committed.py`
- Modify: `apps/transcribe/app.py` — add `_merge_committed` next to `_suppress_repeat_hallucination`

- [ ] **Step 1: Create the test package marker**

Run: `mkdir -p tests/transcribe && : > tests/transcribe/__init__.py`

- [ ] **Step 2: Write the failing tests**

Create `tests/transcribe/test_merge_committed.py` with this exact content:

```python
"""Unit tests for the raw-transcript seam dedup helper.

The live-mic path appends every commit-window's Whisper output to
`StreamState.committed`. Because of the 2 s carry-over between adjacent
commit windows, the same audio is transcribed twice and the duplicated
words show up glued together. `_merge_committed` strips that overlap
at the word level so the visible transcript reads each pangti once.
"""

from __future__ import annotations

import pytest

from apps.transcribe.app import _merge_committed


def test_empty_prev_returns_new():
    assert _merge_committed("", "ਏਹੁ ਨੀਸਾਣੁ") == "ਏਹੁ ਨੀਸਾਣੁ"


def test_empty_new_returns_prev():
    assert _merge_committed("ਏਹੁ ਨੀਸਾਣੁ", "") == "ਏਹੁ ਨੀਸਾਣੁ"


def test_both_empty_returns_empty():
    assert _merge_committed("", "") == ""


def test_no_overlap_joins_with_single_space():
    assert _merge_committed("ਏਹੁ ਨੀਸਾਣੁ", "ਸਚੁ ਨਾਮੁ") == "ਏਹੁ ਨੀਸਾਣੁ ਸਚੁ ਨਾਮੁ"


def test_two_word_overlap_dropped():
    prev = "ਏਹੁ ਨੀਸਾਣੁ ਸੇਤੀ ਘਰਿ"
    new = "ਸੇਤੀ ਘਰਿ ਜਾਈਐ ਸਚੁ ਨਾਮੁ"
    assert _merge_committed(prev, new) == "ਏਹੁ ਨੀਸਾਣੁ ਸੇਤੀ ਘਰਿ ਜਾਈਐ ਸਚੁ ਨਾਮੁ"


def test_new_fully_contained_in_prev_suffix_returns_prev():
    prev = "ਏਹੁ ਨੀਸਾਣੁ ਸੇਤੀ ਘਰਿ"
    new = "ਸੇਤੀ ਘਰਿ"
    assert _merge_committed(prev, new) == "ਏਹੁ ਨੀਸਾਣੁ ਸੇਤੀ ਘਰਿ"


def test_overlap_capped_at_max_overlap_words():
    # 14 identical words at the seam — the helper only checks up to
    # max_overlap_words=12 by default, so 12 get dropped and 2 stay
    # (but as part of `new` they show up as the start of the appended
    # tail, joined onto prev).
    words = [f"w{i}" for i in range(14)]
    prev = " ".join(words)
    new = " ".join(words + ["x", "y"])
    out = _merge_committed(prev, new)
    # Default cap is 12: the longest matching suffix/prefix the helper
    # is allowed to find is 12 words. So 2 of the 14 repeated words
    # plus "x y" are appended.
    expected = prev + " " + " ".join(words[12:] + ["x", "y"])
    assert out == expected


def test_full_overlap_within_cap_returns_prev_when_new_is_only_overlap():
    prev = "a b c d"
    new = "a b c d"
    assert _merge_committed(prev, new) == "a b c d"


def test_whitespace_normalised_on_input():
    assert _merge_committed("  a b  ", "  b c  ") == "a b c"


def test_overlap_only_anchored_at_seam_not_anywhere():
    # "a b" appears in the middle of new but not at its prefix —
    # we only dedup prev's suffix vs new's prefix.
    prev = "x y z"
    new = "p q a b"
    assert _merge_committed(prev, new) == "x y z p q a b"


@pytest.mark.parametrize("prev,new,expected", [
    ("a", "a", "a"),
    ("a b", "b", "a b"),
    ("a b", "b c", "a b c"),
    ("a b c", "c d", "a b c d"),
])
def test_short_overlaps(prev, new, expected):
    assert _merge_committed(prev, new) == expected
```

- [ ] **Step 3: Run the tests to confirm they fail**

Run: `pytest tests/transcribe/test_merge_committed.py -v`

Expected: every test fails with `ImportError` / `AttributeError` — `_merge_committed` does not exist yet in `apps/transcribe/app.py`.

- [ ] **Step 4: Implement `_merge_committed`**

Open `apps/transcribe/app.py`. Find the `_suppress_repeat_hallucination` function (around line 196). Insert this function immediately after it (before `_shabad_id_from_hit`):

```python
def _merge_committed(
    prev: str, new: str, *, max_overlap_words: int = 12
) -> str:
    """Append `new` to `prev`, dropping the longest token-level overlap
    between prev's suffix and new's prefix.

    The live-mic commit-and-carry pipeline transcribes ~2 s of audio twice
    (the carry-over between adjacent windows). Whisper is context-sensitive,
    so the two transcriptions differ slightly and naive append produces
    near-duplicates glued together. This helper merges at the word level
    (not char level) so we never split a Gurmukhi grapheme cluster
    mid-word, and caps the search at `max_overlap_words` to keep cost
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
    if k >= len(n):
        return prev
    return (prev + " " + " ".join(n[k:])).strip()
```

- [ ] **Step 5: Run the tests to confirm they pass**

Run: `pytest tests/transcribe/test_merge_committed.py -v`

Expected: all 14 tests PASS (10 named tests + 4 parametrised).

- [ ] **Step 6: Commit**

```bash
git add tests/transcribe/__init__.py tests/transcribe/test_merge_committed.py apps/transcribe/app.py
git commit -m "feat(transcribe): add _merge_committed seam-dedup helper

Word-level longest-suffix/prefix merge with a 12-word cap. Word-level
(not char-level) so grapheme clusters stay intact. Pure function with
unit tests; not wired into live paths yet."
```

---

## Task 3: Wire `_merge_committed` into all commit-append sites

Replace the four naive `committed` append sites with calls to the new helper. After this task, the visible transcript no longer doubles at commit seams.

**Files:**
- Modify: `apps/transcribe/app.py` (four call sites: live mic, URL stream, URL drain, URL final tail, file play-sync — five total occurrences)

- [ ] **Step 1: Find all current append sites**

Run: `grep -n 'st.committed = (st.committed' apps/transcribe/app.py`

Expected: five matches around lines 1576, 1910, 1958, 1966, 2017 (line numbers may have drifted slightly from earlier work — use the grep output as the source of truth).

- [ ] **Step 2: Replace the live-mic commit-and-carry append**

Open `apps/transcribe/app.py`. In `on_stream` (the live-mic streaming handler), find the block:

```python
if st.buffer.size > max_samples:
    commit_slice = st.buffer[:commit_samples]
    if _rms(commit_slice) >= st.vad_threshold:
        txt = _transcribe(st, commit_slice)
        if txt:
            st.committed = (st.committed + " " + txt).strip()
            _refresh_matches(st, smooth=True)
            _try_auto_push(st)
```

Replace the `st.committed = ...` line with:

```python
            st.committed = _merge_committed(st.committed, txt)
```

(Indentation matches the surrounding block — three levels in.)

- [ ] **Step 3: Replace the URL-stream live commit append**

In `_on_stream_url`, find:

```python
                    text = _transcribe(st, window)
                    if text:
                        st.committed = (st.committed + " " + text).strip()
                        _refresh_matches(st, smooth=True)
```

Replace the `st.committed = ...` line with:

```python
                        st.committed = _merge_committed(st.committed, text)
```

- [ ] **Step 4: Replace the URL-stream drain-loop append**

Still in `_on_stream_url`, find the drain block:

```python
                s0 = transcribed_up_to
                s1 = s0 + window_samples
                text = _transcribe(st, buf[s0:s1])
                if text:
                    st.committed = (st.committed + " " + text).strip()
                    _refresh_matches(st, smooth=True)
                transcribed_up_to = s1
```

Replace the `st.committed = ...` line with:

```python
                    st.committed = _merge_committed(st.committed, text)
```

- [ ] **Step 5: Replace the URL-stream final-tail append**

Still in `_on_stream_url`, find:

```python
            if buf.size - transcribed_up_to >= 3 * TARGET_SR:
                tail = buf[transcribed_up_to:]
                text = _transcribe(st, tail)
                if text:
                    st.committed = (st.committed + " " + text).strip()
                    _refresh_matches(st, smooth=True)
```

Replace the `st.committed = ...` line with:

```python
                    st.committed = _merge_committed(st.committed, text)
```

- [ ] **Step 6: Replace the file play-sync append**

In `on_play_sync`, find:

```python
                text = _transcribe(st, chunk)
                if text:
                    st.committed = (st.committed + " " + text).strip()
                    _refresh_matches(st, smooth=True)
```

Replace the `st.committed = ...` line with:

```python
                    st.committed = _merge_committed(st.committed, text)
```

- [ ] **Step 7: Confirm no more naive appends remain**

Run: `grep -n 'st.committed = (st.committed' apps/transcribe/app.py`

Expected: zero matches.

Run: `grep -cn '_merge_committed(st.committed' apps/transcribe/app.py`

Expected: 5.

- [ ] **Step 8: Sanity-check the module still imports**

Run: `python -c "from apps.transcribe.app import _merge_committed; print('import OK')"`

Expected: `import OK`.

- [ ] **Step 9: Commit**

```bash
git add apps/transcribe/app.py
git commit -m "feat(transcribe): use _merge_committed at every commit append site

Live mic, URL stream (live + drain + tail), and file play-sync all now
dedup the 2 s carry-over against the previous commit instead of naively
concatenating. Visible transcript stops doubling at the seam."
```

---

## Task 4: STTM dedup keyed on `(shabadId, verseId)`

Replace the `last_pushed_sid: int` shabad-only dedup with a `(shabadId, verseId)` tuple. After this task, every locked-pointer advance triggers a fresh `/api/bani-control` push and STTM's highlight follows the ragi.

**Files:**
- Modify: `apps/transcribe/app.py` — `StreamState`, `_try_auto_push`, `_on_stream_url`, `on_play_sync`, `_lock`, `_unlock`, `on_clear`

- [ ] **Step 1: Rename the StreamState field**

In `apps/transcribe/app.py`, find the `StreamState` dataclass field:

```python
    # Auto-push dedupe across the whole session (mic + URL + file). Reset on
    # clear / unlock. Lets us push once per shabad transition even though
    # the same shabad is the top match for many consecutive windows.
    last_pushed_sid: int | None = None
```

Replace with:

```python
    # Auto-push dedupe across the whole session (mic + URL + file). Reset
    # on clear / lock / unlock. Keyed on (shabadId, verseId) so a within-
    # shabad pointer advance re-pushes and STTM follows the ragi line by
    # line. (Keying on shabadId alone left STTM stuck on whichever line we
    # pushed at lock time.)
    last_pushed_key: tuple[int, int] | None = None
```

- [ ] **Step 2: Update `_try_auto_push`**

Find the current body:

```python
        def _try_auto_push(st: StreamState) -> None:
            """Push the current top match to STTM if the session looks ready.

            Gates:
              • at least one match above the auto-push threshold
              • STTM reachable (probes once if unknown)
              • we haven't already pushed this shabadId (dedupes per transition)
            Called from every path that refreshes matches (live mic, URL
            stream, file play-sync).
            """
            if not st.matches:
                return
            top = st.matches[0]
            if float(top.get("score", 0.0)) < st.auto_push_threshold:
                return
            _ensure_sttm(st)
            if not (st.sttm_connected and st.sttm_port):
                return
            sid = top.get("shabadId")
            if not sid or sid == st.last_pushed_sid:
                return
            res = push_hit(st.sttm_host, st.sttm_port, top, pin=st.sttm_pin or None)
            if res.ok:
                st.last_pushed_sid = sid
                print(f"[sttm] pushed sid={sid} · score={top.get('score')}")
            else:
                print(f"[sttm] push failed: {res.detail}")
```

Replace with:

```python
        def _try_auto_push(st: StreamState) -> None:
            """Push the current top match to STTM if the session looks ready.

            Gates:
              • at least one match above the auto-push threshold
              • STTM reachable (probes once if unknown)
              • we haven't already pushed this (shabadId, verseId) pair
            Dedup is on the *pair* so locked-pointer advances re-push and
            STTM follows line-by-line. Called from every path that refreshes
            matches (live mic, URL stream, file play-sync).
            """
            if not st.matches:
                return
            top = st.matches[0]
            if float(top.get("score", 0.0)) < st.auto_push_threshold:
                return
            _ensure_sttm(st)
            if not (st.sttm_connected and st.sttm_port):
                return
            sid = top.get("shabadId")
            vid = top.get("verseId") or sid
            if not sid:
                return
            key = (int(sid), int(vid))
            if key == st.last_pushed_key:
                return
            res = push_hit(st.sttm_host, st.sttm_port, top, pin=st.sttm_pin or None)
            if res.ok:
                st.last_pushed_key = key
                print(f"[sttm] pushed sid={sid} verseId={vid} · score={top.get('score')}")
            else:
                print(f"[sttm] push failed: {res.detail}")
```

- [ ] **Step 3: Update `_on_stream_url`'s local dedup variable**

In `_on_stream_url`, find:

```python
            transcribed_up_to = 0
            last_pushed_sid: int | None = None
            t_start = time.time()
```

Replace with:

```python
            transcribed_up_to = 0
            last_pushed_key: tuple[int, int] | None = None
            t_start = time.time()
```

Then find the auto-push block in the live loop:

```python
                        if (auto_push and st.matches
                                and st.sttm_connected and st.sttm_port
                                and float(st.matches[0].get("score", 0.0))
                                    >= st.auto_push_threshold):
                            top = st.matches[0]
                            sid = top.get("shabadId")
                            if sid and sid != last_pushed_sid:
                                push_hit(st.sttm_host, st.sttm_port, top,
                                         pin=st.sttm_pin or None)
                                last_pushed_sid = sid
```

Replace with:

```python
                        if (auto_push and st.matches
                                and st.sttm_connected and st.sttm_port
                                and float(st.matches[0].get("score", 0.0))
                                    >= st.auto_push_threshold):
                            top = st.matches[0]
                            sid = top.get("shabadId")
                            vid = top.get("verseId") or sid
                            if sid:
                                key = (int(sid), int(vid))
                                if key != last_pushed_key:
                                    push_hit(st.sttm_host, st.sttm_port, top,
                                             pin=st.sttm_pin or None)
                                    last_pushed_key = key
```

- [ ] **Step 4: Update `on_play_sync`'s local dedup variable**

In `on_play_sync`, find:

```python
            t_start = time.time()
            i = 0
            last_pushed_sid: int | None = None
```

Replace with:

```python
            t_start = time.time()
            i = 0
            last_pushed_key: tuple[int, int] | None = None
```

Then find the auto-push block:

```python
                    if (auto_push and st.matches and st.sttm_connected and st.sttm_port
                            and float(st.matches[0].get("score", 0.0))
                                >= st.auto_push_threshold):
                        top = st.matches[0]
                        sid = top.get("shabadId")
                        if sid and sid != last_pushed_sid:
                            push_hit(st.sttm_host, st.sttm_port, top,
                                     pin=st.sttm_pin or None)
                            last_pushed_sid = sid
```

Replace with:

```python
                    if (auto_push and st.matches and st.sttm_connected and st.sttm_port
                            and float(st.matches[0].get("score", 0.0))
                                >= st.auto_push_threshold):
                        top = st.matches[0]
                        sid = top.get("shabadId")
                        vid = top.get("verseId") or sid
                        if sid:
                            key = (int(sid), int(vid))
                            if key != last_pushed_key:
                                push_hit(st.sttm_host, st.sttm_port, top,
                                         pin=st.sttm_pin or None)
                                last_pushed_key = key
```

- [ ] **Step 5: Update `_lock` to clear the dedup key**

Find the `_lock` helper:

```python
        def _lock(st: StreamState, sid: str, tuk_row: int | None = None) -> None:
            st.locked_shabad_id = sid
            st.locked_tuk_row = tuk_row
            st.unlock_miss_count = 0
            st.lock_streak_count = 0
            st.last_top_sid_for_lock = sid
            if st.retrieval_ema is not None:
                st.retrieval_ema.reset()
            print(f"[lock] lock · sid={sid} · tuk_row={tuk_row}")
```

Insert one line so a fresh lock allows the immediately-following `_try_auto_push` to fire with the right verseId:

```python
        def _lock(st: StreamState, sid: str, tuk_row: int | None = None) -> None:
            st.locked_shabad_id = sid
            st.locked_tuk_row = tuk_row
            st.unlock_miss_count = 0
            st.lock_streak_count = 0
            st.last_top_sid_for_lock = sid
            st.last_pushed_key = None
            if st.retrieval_ema is not None:
                st.retrieval_ema.reset()
            print(f"[lock] lock · sid={sid} · tuk_row={tuk_row}")
```

- [ ] **Step 6: Update `_unlock` to use the new field name**

Find the `_unlock` helper:

```python
        def _unlock(st: StreamState) -> None:
            st.locked_shabad_id = None
            st.locked_tuk_row = None
            st.unlock_miss_count = 0
            st.lock_streak_count = 0
            st.last_top_sid_for_lock = None
            st.last_search_key = ""  # force fresh retrieval on next tick
            st.last_pushed_sid = None  # allow re-push if ragi returns to same shabad
            if st.retrieval_ema is not None:
                st.retrieval_ema.reset()
```

Replace the `last_pushed_sid = None` line with:

```python
            st.last_pushed_key = None  # allow re-push if ragi returns to same shabad
```

- [ ] **Step 7: Update `on_clear`**

Find `on_clear`:

```python
        def on_clear(st: StreamState):
            # snapshot for undo
            st.undo_committed = st.committed
            st.undo_matches = list(st.matches) if st.matches else []
            st.undo_expires_at = time.time() + UNDO_TTL_S

            st.committed = ""
            st.tentative = ""
            st.buffer = np.zeros(0, dtype=np.float32)
            st.matches = []
            st.last_search_key = ""
            st.last_pushed_sid = None
            if st.retrieval_ema is not None:
                st.retrieval_ema.reset()
```

Replace `st.last_pushed_sid = None` with:

```python
            st.last_pushed_key = None
```

- [ ] **Step 8: Confirm the rename is complete**

Run: `grep -n 'last_pushed_sid' apps/transcribe/app.py`

Expected: zero matches.

Run: `grep -cn 'last_pushed_key' apps/transcribe/app.py`

Expected: 7 (one StreamState field + two locals + four assignments/reads in `_try_auto_push`/`_lock`/`_unlock`/`on_clear`).

- [ ] **Step 9: Sanity-check imports + dataclass**

Run: `python -c "from apps.transcribe.app import StreamState; s = StreamState(); assert s.last_pushed_key is None; print('ok')"`

Expected: `ok`.

- [ ] **Step 10: Commit**

```bash
git add apps/transcribe/app.py
git commit -m "fix(transcribe): dedup STTM auto-push on (shabadId, verseId)

Previously the dedup key was shabadId alone, so STTM stayed on whichever
verseId was pushed at lock-time even after the local pointer advanced.
Now every locked-pointer advance pushes a fresh payload and STTM's
highlight follows the ragi line-by-line."
```

---

## Task 5: Factor `_handle_locked` into pointer + unlock helpers

Split the locked-mode logic so the upcoming fast-pointer path can advance the pointer without touching unlock-miss bookkeeping. After this task, `_handle_locked` is a thin orchestrator that calls the two new helpers.

**Files:**
- Modify: `apps/transcribe/app.py` — `_handle_locked` (around lines 1349-1416)

- [ ] **Step 1: Replace `_handle_locked` with the factored version**

In `apps/transcribe/app.py`, find the entire current `_handle_locked` function (around line 1349) and replace it with these three helpers:

```python
        def _pointer_advance(st: StreamState, query: str) -> bool:
            """Advance `st.locked_tuk_row` to the best within-shabad match
            for `query`. Side-effects: updates `st.locked_tuk_row` and
            `st.matches`. Returns True iff `st.matches` was rewritten.

            No unlock side-effects — see `_check_unlock` for that. Splitting
            the two lets the fast-pointer path run pointer updates on a
            short fresh-tail buffer without burning unlock-misses on what
            is normal sub-tuk noise.
            """
            if not st.locked_shabad_id:
                return False
            prefix_hits = score_within_shabad_prefix(query, st.locked_shabad_id)
            prefix_best = prefix_hits[0] if prefix_hits else None
            prefix_second = (
                prefix_hits[1].overlap if len(prefix_hits) > 1 else 0.0
            )
            overlap_hits = score_within_shabad(query, st.locked_shabad_id)
            overlap_best = overlap_hits[0] if overlap_hits else None

            advance = None
            if (prefix_best is not None
                    and prefix_best.overlap >= 0.8
                    and (prefix_best.overlap - prefix_second) >= 0.15):
                advance = prefix_best
            elif (overlap_best is not None
                    and overlap_best.overlap >= POINTER_MOVE_FLOOR):
                advance = overlap_best

            if advance is None:
                return False

            prev_row = st.locked_tuk_row
            st.locked_tuk_row = advance.tuk_row
            ui_score = 0.5 + 0.5 * min(advance.overlap, 1.0)
            st.matches = [_locked_hit_to_match(advance, ui_score)]
            if prev_row != advance.tuk_row:
                print(
                    f"[lock] pointer row {prev_row} → {advance.tuk_row} "
                    f"(overlap={advance.overlap:.2f})"
                )
            return True

        def _check_unlock(st: StreamState, query: str) -> bool:
            """Run the unlock decision against `query`. Returns True iff
            we just unlocked. Side-effects: bumps or resets
            `st.unlock_miss_count`; calls `_unlock` when threshold hit.

            Driven by overlap (not prefix), so a brief silence or an
            ornament doesn't unlock — only sustained absence of any
            within-shabad 4-gram support does.
            """
            if not st.locked_shabad_id:
                return False
            overlap_hits = score_within_shabad(query, st.locked_shabad_id)
            overlap_best = overlap_hits[0] if overlap_hits else None
            below_unlock_floor = (
                overlap_best is None or overlap_best.overlap < UNLOCK_SCORE_FLOOR
            )
            if not below_unlock_floor:
                st.unlock_miss_count = 0
                return False
            st.unlock_miss_count += 1
            if st.unlock_miss_count < st.unlock_misses_target:
                return False
            print(
                f"[lock] unlock · sid={st.locked_shabad_id} after "
                f"{st.unlock_miss_count} misses "
                f"(best_overlap="
                f"{overlap_best.overlap if overlap_best else 0.0:.2f})"
            )
            _unlock(st)
            return True

        def _handle_locked(st: StreamState, query: str) -> bool:
            """Locked-mode pointer + unlock orchestrator. Returns True if
            we handled the query (either stayed locked or unlocked).

            Pointer-advance and unlock are decoupled by design: we trust
            the lock, so any distinguishable within-shabad overlap is
            enough to move the pointer (POINTER_MOVE_FLOOR), while unlock
            still requires sustained low signal (UNLOCK_SCORE_FLOOR for
            `unlock_misses_target` consecutive windows).
            """
            if not st.locked_shabad_id:
                return False
            if _check_unlock(st, query):
                return False  # fall through to unlocked retrieval
            _pointer_advance(st, query)
            return True
```

- [ ] **Step 2: Confirm the module imports + the orchestrator still resolves**

Run: `python -c "from apps.transcribe.app import build_app; print('import OK')"`

Expected: `import OK`.

(The closure-scoped helpers aren't importable directly, but a successful `build_app` import means the file parses and references resolve.)

- [ ] **Step 3: Confirm there's exactly one `_handle_locked` definition**

Run: `grep -cn 'def _handle_locked' apps/transcribe/app.py`

Expected: 1.

Run: `grep -cn 'def _pointer_advance\|def _check_unlock' apps/transcribe/app.py`

Expected: 2.

- [ ] **Step 4: Commit**

```bash
git add apps/transcribe/app.py
git commit -m "refactor(transcribe): factor _handle_locked into pointer + unlock helpers

_handle_locked is now a thin orchestrator over _pointer_advance and
_check_unlock. Same external behaviour. Sets up the upcoming fast-
pointer path which needs to call _pointer_advance without touching
unlock-miss bookkeeping."
```

---

## Task 6: Add fast-pointer state + `_refresh_pointer` helper

Add the StreamState fields and the helper that the fast-pointer path will call. No behaviour change yet — this is plumbing for Task 7.

**Files:**
- Modify: `apps/transcribe/app.py` — `StreamState`, new `_refresh_pointer` helper

- [ ] **Step 1: Add fast-pointer fields to StreamState**

In `apps/transcribe/app.py`, find the `min_transcribe_s` field in `StreamState`:

```python
    # Whisper hallucinates on very short buffers ("ਵਾਹਿਗੁਰੂ ਵਾਹਿਗੁਰੂ…" loops).
    # Require at least ~3 s of audio before we call the model on live mic.
    min_transcribe_s: float = 3.0
    auto_push_threshold: float = AUTO_PUSH_THRESHOLD_DEFAULT
    hero_threshold: float = HERO_THRESHOLD_DEFAULT
```

Insert three new fields between `min_transcribe_s` and `auto_push_threshold`:

```python
    # Whisper hallucinates on very short buffers ("ਵਾਹਿਗੁਰੂ ਵਾਹਿਗੁਰੂ…" loops).
    # Require at least ~3 s of audio before we call the model on live mic.
    min_transcribe_s: float = 3.0
    # Locked-mode fast-pointer cadence. Tighter than the unlocked path
    # because matches are constrained to the locked shabad's ~30 tuks —
    # hallucinations either fail to match (no advance) or trip the
    # existing unlock floor in _check_unlock. Target: detect a new
    # pangti within ~2 s of the ragi starting it.
    fast_pointer_min_s: float = 1.2
    fast_pointer_throttle_s: float = 0.6
    last_fast_pointer_t: float = 0.0
    auto_push_threshold: float = AUTO_PUSH_THRESHOLD_DEFAULT
    hero_threshold: float = HERO_THRESHOLD_DEFAULT
```

- [ ] **Step 2: Add `_refresh_pointer` helper**

In `apps/transcribe/app.py`, find the `_refresh_matches` function (around line 1465). Insert this helper immediately above it (between the new `_handle_locked` and `_refresh_matches`):

```python
        def _refresh_pointer(st: StreamState, tail_text: str) -> None:
            """Advance the locked pointer using a fresh tail Whisper output.

            Bypasses `_retrieval_query` (no 140-char tail trim — `tail_text`
            already IS the query, transcribed from the last ~1.5 s of audio
            with no carry-over). Skips unlock bookkeeping (the slower
            commit pipeline owns that decision and runs against a longer,
            cleaner buffer).

            No-op if we're not locked or the tail is empty.
            """
            if not st.locked_shabad_id:
                return
            text = (tail_text or "").strip()
            if not text:
                return
            _pointer_advance(st, text)
```

- [ ] **Step 3: Confirm the module still imports**

Run: `python -c "from apps.transcribe.app import StreamState; s = StreamState(); assert s.fast_pointer_min_s == 1.2; assert s.fast_pointer_throttle_s == 0.6; assert s.last_fast_pointer_t == 0.0; print('ok')"`

Expected: `ok`.

- [ ] **Step 4: Commit**

```bash
git add apps/transcribe/app.py
git commit -m "feat(transcribe): add fast-pointer state + _refresh_pointer helper

Plumbing only — fields default to a 1.2 s minimum buffer and 0.6 s
throttle, and the helper is a thin wrapper over _pointer_advance.
Wired into on_stream in the next commit."
```

---

## Task 7: Wire the fast-pointer branch into `on_stream`

Replace the locked-mode tentative branch with the fast-pointer path. Worst-case detection becomes ~1.5–2.1 s. Unlocked-mode behaviour is unchanged.

**Files:**
- Modify: `apps/transcribe/app.py` — `on_stream` tentative section (around lines 1582-1601)

- [ ] **Step 1: Replace the tentative block**

In `apps/transcribe/app.py`, find the tentative block in `on_stream`:

```python
            # tentative
            now = time.time()
            if (st.buffer.size >= min_samples
                    and (now - st.last_call_t) >= st.throttle_s
                    and _rms(st.buffer) >= st.vad_threshold):
                st.last_call_t = now
                t0 = time.time()
                st.tentative = _transcribe(st, st.buffer)
                latency_ms = (time.time() - t0) * 1000.0
                buf_sec = st.buffer.size / TARGET_SR
                rtf = latency_ms / 1000.0 / max(buf_sec, 1e-6)
                st.last_stats_html = _multi_stat([
                    ("buffer", f"{buf_sec:4.1f}s"),
                    ("inference", f"{latency_ms:4.0f} ms"),
                    ("rtf", f"{rtf:4.2f}×"),
                    ("rms", f"{_rms(st.buffer):.4f}"),
                    ("backend", backend_label),
                ])
                _refresh_matches(st, smooth=True)
                _try_auto_push(st)
```

Replace it with this branched version (locked → fast-pointer; unlocked → unchanged tentative):

```python
            # tentative — branched by lock state
            now = time.time()
            if st.locked_shabad_id:
                # Fast-pointer path: tighter cadence + a fresh 1.5 s tail
                # (no carry-over contamination), so a new pangti is
                # detected within ~1.5–2.1 s of the ragi starting it.
                fast_min_samples = int(st.fast_pointer_min_s * TARGET_SR)
                fast_tail_samples = int(1.5 * TARGET_SR)
                if (st.buffer.size >= fast_min_samples
                        and (now - st.last_fast_pointer_t)
                            >= st.fast_pointer_throttle_s
                        and _rms(st.buffer[-fast_min_samples:])
                            >= st.vad_threshold):
                    st.last_fast_pointer_t = now
                    tail = (
                        st.buffer[-fast_tail_samples:]
                        if st.buffer.size > fast_tail_samples
                        else st.buffer
                    )
                    t0 = time.time()
                    tail_text = _transcribe(st, tail)
                    latency_ms = (time.time() - t0) * 1000.0
                    if tail_text:
                        st.tentative = tail_text
                        _refresh_pointer(st, tail_text)
                        _try_auto_push(st)
                    tail_sec = tail.size / TARGET_SR
                    rtf = latency_ms / 1000.0 / max(tail_sec, 1e-6)
                    st.last_stats_html = _multi_stat([
                        ("buffer", f"{st.buffer.size / TARGET_SR:4.1f}s"),
                        ("fast-tail", f"{tail_sec:4.1f}s"),
                        ("inference", f"{latency_ms:4.0f} ms"),
                        ("rtf", f"{rtf:4.2f}×"),
                        ("backend", backend_label),
                    ])
            elif (st.buffer.size >= min_samples
                    and (now - st.last_call_t) >= st.throttle_s
                    and _rms(st.buffer) >= st.vad_threshold):
                st.last_call_t = now
                t0 = time.time()
                st.tentative = _transcribe(st, st.buffer)
                latency_ms = (time.time() - t0) * 1000.0
                buf_sec = st.buffer.size / TARGET_SR
                rtf = latency_ms / 1000.0 / max(buf_sec, 1e-6)
                st.last_stats_html = _multi_stat([
                    ("buffer", f"{buf_sec:4.1f}s"),
                    ("inference", f"{latency_ms:4.0f} ms"),
                    ("rtf", f"{rtf:4.2f}×"),
                    ("rms", f"{_rms(st.buffer):.4f}"),
                    ("backend", backend_label),
                ])
                _refresh_matches(st, smooth=True)
                _try_auto_push(st)
```

- [ ] **Step 2: Confirm the module still imports**

Run: `python -c "from apps.transcribe.app import build_app; print('import OK')"`

Expected: `import OK`.

- [ ] **Step 3: Re-run the merge-committed unit tests as a regression check**

Run: `pytest tests/transcribe/test_merge_committed.py -v`

Expected: all tests still PASS.

- [ ] **Step 4: Commit**

```bash
git add apps/transcribe/app.py
git commit -m "feat(transcribe): fast-pointer branch in on_stream live mic

When LOCKED, replace the regular tentative branch with a fresh-tail
fast-pointer call (1.5 s of audio, no carry-over, 0.6 s throttle, 1.2 s
min buffer). Worst-case new-pangti detection drops from ~3-5 s to
~1.5-2.1 s. Unlocked-mode tentative path is unchanged."
```

---

## Manual verification

After Task 7, run a live-mic UAT pass against a known shabad. This is not a step in the plan (no automated assertion possible), but it's the only way to confirm the design lands.

1. `pytest tests/transcribe/ -v` — all tests pass.
2. `python -c "from apps.transcribe.retriever import DEFAULT_MODE; print(DEFAULT_MODE)"` → `semantic+literal`.
3. Launch the Gradio app, mic on, sing a known shabad.
4. After auto-lock, sing the next pangti. Confirm in-app highlight follows within ~2 s.
5. With STTM open and connected, repeat — confirm STTM's highlighted line moves with the in-app pointer (not just the shabad).
6. Loop back to a Rahao line. Same expectation.
7. Inspect the raw transcript strip — no obvious doubling at 10 s commit boundaries.
8. Reload the app — settings dropdown shows "Semantic + literal …" as the selected retrieval method.

---

## Self-review notes

- **Spec coverage:** Section 1 (STTM) → Task 4. Section 2 (fast-pointer) → Tasks 5–7. Section 3 (seam dedup) → Tasks 2–3. Section 4 (default mode) → Task 1. Verification section → Manual verification block above. No gaps.
- **Type consistency:** `last_pushed_key` is `tuple[int, int] | None` everywhere it appears (StreamState field + two locals + four read/write sites in `_try_auto_push`/`_lock`/`_unlock`/`on_clear`). `_pointer_advance` and `_check_unlock` both take `(st: StreamState, query: str)` and return `bool`. `_refresh_pointer` takes `(st, tail_text: str)` returning `None`.
- **No placeholders:** every code block is the literal text to paste; every `grep` / `pytest` step has explicit expected output.
