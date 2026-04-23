---
phase: quick-2
plan: 1
subsystem: apps/transcribe
tags: [sttm, cdp, playwright, offline-control, bug-fix]
dependency_graph:
  requires: []
  provides:
    - STTMController (apps/transcribe/sttm_controller.py) — offline CDP driver
    - get_controller() module-level singleton (thread-safe)
  affects:
    - apps/transcribe/app.py push handlers (on_push, on_auto_push, on_pick_match)
    - StreamState shape (sttm_host / sttm_port gone; sttm_pin kept for config compat)
tech_stack:
  added:
    - playwright>=1.43  # CDP client (sync API); no browsers downloaded — we attach
  patterns:
    - "Sync Playwright in a Gradio handler — no asyncio glue. sync_playwright().start()
       owned by a long-lived singleton; handlers call get_controller().push_hit(hit)."
    - "Graceful auth-gate: connect() swallows errors and records last_error(); UI
       surfaces the exact `open -a 'SikhiToTheMax' --args --remote-debugging-port=9222`
       command instead of a silent success toast."
    - "Forgiving selector fallback list: tries data-verseid / data-verse-id /
       data-line-index / data-line / data-idx, then keyboard-arrow fallback. Task 3
       will pin down which one matches the running STTM build."
key_files:
  created: []
  modified:
    - apps/transcribe/sttm_controller.py
    - apps/transcribe/requirements.txt
    - apps/transcribe/app.py
decisions:
  - "Sync Playwright, not async. Gradio handlers are synchronous; async-wrapping
    them against sync_playwright is painful and adds zero value for a same-machine
    loopback driver."
  - "Drop sttm_host / sttm_port from StreamState entirely rather than keeping them
    unused. CDP has one address (localhost:9222) and no user-tunable port on our
    side — keeping the fields would invite drift."
  - "Hide (not remove) the PIN textbox. Persisted ~/.surt/config.json carries
    sttm_pin; existing configs should load/save cleanly, just with the field
    unused on the CDP path."
  - "on_push no longer fabricates a push from raw transcript text. Transcript → 
    shabad now flows exclusively through the retriever; pushing a non-shabad
    string was a legacy workaround that doesn't map to CDP's 'open the shabad,
    then advance the line' primitives."
  - "open_shabad uses first-letter search (e.g. 'ਸਪ' for 'ਸਤਿਗੁਰ ਪ੍ਰਸਾਦਿ') and
    clicks the top result. That matches STTM's standard UX; we don't fabricate
    a shabadId-based jump (no public DOM hook for that)."
metrics:
  duration: "~6 min (implementation + verify, excluding Task 3 human verify)"
  completed_date: "2026-04-23"
  tasks_completed: 2
  tasks_total: 3
---

# Quick Task 2: Switch STTM push to real STTM.co control — Summary

## One-liner

Replaced the dead REST-based STTM push (POST-to-a-catch-all-that-silently-drops-
the-payload) with a Playwright sync-API Chrome DevTools Protocol controller that
attaches to STTM Desktop's Electron renderer at `localhost:9222`, so "Push to
STTM" actually moves the highlighted pangti — offline, same-machine, no cloud,
no PIN.

## What changed

### Task 1 — new `STTMController` (commit `efba19e`)

Rewrote `apps/transcribe/sttm_controller.py`:

- **Removed** `push_shabad`, `push_hit`, `push_transcript_as_shabad`, `discover`,
  `CANDIDATE_PORTS`, `_post_json` — the whole REST path that quietly no-op'd on
  STTM's overlay catch-all.
- **Preserved** `search_shabad_topn`, `search_shabad`, `_norm_hit`, `STTMStatus`.
  These hit BaniDB for suggest/autolock and are called from `app.py` and
  `retriever.py`. `STTMStatus(ok, host, port, detail)` shape is unchanged so
  call sites that read `res.ok` / `res.detail` didn't need updating.
- **Added** `class STTMController`:
  - `connect(cdp_port=9222)` — idempotent, never raises, records `last_error()`.
  - `is_connected()`, `disconnect()`.
  - `open_shabad(shabad_id, first_line_gurmukhi)` — first-letter search + click
    top result. Uses a helper `_first_letters()` that extracts Gurmukhi
    consonants (`ਸਤਿਗੁਰ ਪ੍ਰਸਾਦਿ` → `ਸਪ`).
  - `advance_to_verse(verse_id, line_idx, full_rowids, gurmukhi)` — tries
    `[data-verseid=…]`, `[data-verse-id=…]`, `[data-line-index=…]`,
    `[data-line=…]`, `[data-idx=…]` in order, falls back to keyboard arrows
    from the last known line index.
  - `push_hit(hit)` — opens the shabad on new sid, advances on same sid,
    returns STTMStatus.
- **Added** `get_controller()` thread-safe module singleton (protected by
  `threading.Lock`) that lazy-connects on first use.
- **Added** `playwright>=1.43` to `apps/transcribe/requirements.txt`. No
  `playwright install` needed — we `connect_over_cdp()` to an existing
  Chromium, we never launch one.

### Task 2 — `app.py` wiring (commit `878f2bb`)

- Import line reduced to `from apps.transcribe.sttm_controller import (STTMStatus, get_controller)`.
- `StreamState.sttm_host` / `StreamState.sttm_port` deleted. `sttm_pin` kept
  for config round-trip (now hidden in the UI).
- `_sttm_pill` reads `get_controller().is_connected()` at render time; shows
  "STTM · CDP :9222" when attached, "STTM offline · retry" otherwise.
- `on_connect(pin, state)` now attaches the singleton; on failure, the toast
  carries the controller's `last_error()` string (which includes the exact
  `open -a 'SikhiToTheMax' --args --remote-debugging-port=9222` command).
- `_ensure_sttm`, `_try_auto_push`, `on_push`, `on_pick_match` all call
  `get_controller().push_hit(hit)`. Non-ok results render as **error** toasts
  (no more fake "Pushed …" on an HTTP-200-silently-dropped payload).
- `on_push`'s "no matches, push raw transcript" path is gone — the retriever
  is now the only transcript → shabad bridge.
- STTM card simplified: Host/Port textboxes removed (CDP has no user-tunable
  address on our side), short "launch with --remote-debugging-port=9222" hint
  added, PIN textbox hidden.
- `connect_btn.click` and `sttm_pin.change` wiring reduced to
  `inputs=[sttm_pin, state]`; the old `sttm_host.change` / `sttm_port.change`
  bindings are gone.

## Verification

### Task 1 automated verify (passed)

AST check confirms:
- `STTMController` class + `connect`, `open_shabad`, `advance_to_verse`,
  `push_hit`, `disconnect` methods all present.
- `get_controller` exists as a module-level function.
- REST leftovers (`/api/bani-control`, `CANDIDATE_PORTS`, old `push_shabad`)
  are gone.
- `search_shabad_topn`, `search_shabad` preserved.
- Playwright imported.
- `playwright>=1.43` listed in `requirements.txt`.

Additional smoke test:
- `from apps.transcribe.sttm_controller import STTMController, get_controller,
  STTMStatus, search_shabad_topn, search_shabad` imports cleanly.
- `STTMController().is_connected()` is `False` without a running STTM — no
  exception.
- `push_hit(...)` with no CDP running returns `STTMStatus(ok=False, ...)`
  with the "playwright not installed" / "CDP not reachable" message.
- `get_controller()` returns the same instance across calls (singleton works).
- `_first_letters("ਸਤਿਗੁਰ ਪ੍ਰਸਾਦਿ") == "ਸਪ"`.

### Task 2 automated verify (passed)

- `st.sttm_host`, `st.sttm_port`, `CANDIDATE_PORTS` no longer appear in
  `apps/transcribe/app.py`.
- `from .sttm_controller import push_hit` is gone.
- `get_controller` is wired into the push paths.
- `python3 -m py_compile apps/transcribe/app.py` compiles clean.

### Plan-level cross-module verification

- `grep -rn "/api/bani-control|CANDIDATE_PORTS|push_shabad|push_transcript_as_shabad|discover("
  apps/transcribe/` → **no matches**.

## Push call sites — final state in `apps/transcribe/app.py`

| Location | Before | After |
| --- | --- | --- |
| `_try_auto_push` (line ~2180) | `push_hit(st.sttm_host, st.sttm_port, top, pin=st.sttm_pin or None)` | `get_controller().push_hit(top)` |
| `on_push` top-match path | `push_hit(st.sttm_host, st.sttm_port, top, pin=pin)` | `get_controller().push_hit(top)` |
| `on_push` transcript path | `push_transcript_as_shabad(...)` | removed; shows "wait for a shabad match" warn toast |
| `on_pick_match` | `push_hit(st.sttm_host, st.sttm_port, st.matches[idx], pin=…)` | `get_controller().push_hit(st.matches[idx])` |
| `on_connect` | `discover(host=host, ports=CANDIDATE_PORTS)` → sets `st.sttm_host/port/connected` | `get_controller().connect()` → sets `st.sttm_connected` + surfaces `last_error()` |
| `_ensure_sttm` | REST probe on `st.sttm_host` | `get_controller().connect()` if not attached |
| `_sttm_pill` | Read `st.sttm_connected` + `st.sttm_port` | Read `get_controller().is_connected()` at render time; shows "STTM · CDP :9222" |
| `build_app` STTM card | Host/Port/PIN textboxes | Connect button + CDP launch hint; PIN hidden |
| Wiring | `inputs=[sttm_host, sttm_port, sttm_pin, state]` + host/port `.change` | `inputs=[sttm_pin, state]` only |

## Deviations from plan

None material. Two small judgment calls worth flagging:

1. **PIN textbox hidden, not removed.** Plan said "leave sttm_pin alone so
   config round-trips cleanly." I kept the Gradio textbox but set
   `visible=False` so users don't see a dead field, while `saved_pin` still
   loads from `~/.surt/config.json` and `on_connect` still writes updates back.
   No behavior change, just less UI noise.

2. **STTM card UI simplified.** Plan said "update the side-panel STTM status
   HTML to call `get_controller().is_connected()` and render a short string."
   I went further: dropped the Host/Port textboxes entirely (they were
   meaningless on the CDP path) and added a one-line hint explaining the
   `--remote-debugging-port=9222` requirement. This matches the plan's
   intent ("clear actionable error, no fake success") and avoids leaving
   ghost inputs in the DOM.

Neither is a Rule 4 architectural change — both are UX polish that would
have been a Task 3 follow-up anyway.

## Pending

### Task 3 — human verification (blocking)

Task 3 is `checkpoint:human-verify` and is **NOT** executed by the agent.
The user must:

1. `pip install -r apps/transcribe/requirements.txt` (installs Playwright;
   no `playwright install` needed since we attach to existing Chromium).
2. Relaunch STTM with CDP:
   ```bash
   killall SikhiToTheMax || true
   open -a "SikhiToTheMax" --args --remote-debugging-port=9222
   ```
3. Confirm `curl http://127.0.0.1:9222/json/version` returns JSON with
   `Browser` / `webSocketDebuggerUrl`.
4. Restart Gradio; the side-panel pill should read **STTM · CDP :9222**.
5. Turn Wi-Fi off, play a kirtan clip, lock a shabad, click **Push to STTM**.
6. Watch STTM Desktop's main window — the highlighted pangti should jump.

### Selector uncertainty to flag to the user

The selector lists in `open_shabad` and `advance_to_verse` are deliberately
forgiving because we don't have ground truth for STTM Desktop's current DOM.
If Task 3 fails with "CDP reachable but clicks don't land," the user should
open STTM's DevTools (exposed via CDP) and paste back:

- **For search UI:** the tag+class+id of the search input (we currently try
  `input.search-field`, `input#search-field`, `input.search`, `input#search`,
  `input[type='search']`, `input[placeholder*='Search' i]`).
- **For result rows:** the selector of one result row in the dropdown (we try
  `.search-results .result-row`, `.search-result`, `.search-result-row`,
  `li.result`, `[data-shabadid]`).
- **For pangti rows (most important):** the exact attribute name on a pangti
  element — is it `data-verseid`, `data-verse-id`, `data-line-index`,
  `data-line`, `data-idx`, or something else? We currently try all five.

Once the user pastes the real attribute, a follow-up commit can trim the
selector list to just the one that works, which will make the controller
faster (Playwright's `locator().count()` has a small per-call overhead) and
more robust (no ambiguous matches).

## Commits

| Task | Description | Commit |
| --- | --- | --- |
| 1 | Rip dead REST path + add CDP-based STTMController | `efba19e` |
| 2 | Wire app.py push handlers to get_controller() | `878f2bb` |

## Self-Check: PASSED

- `apps/transcribe/sttm_controller.py` exists and imports cleanly — verified.
- `apps/transcribe/requirements.txt` lists `playwright>=1.43` — verified.
- `apps/transcribe/app.py` compiles under `python3 -m py_compile` — verified.
- Commit `efba19e` exists in `git log` — verified.
- Commit `878f2bb` exists in `git log` — verified.
- Cross-module grep for REST leftovers returns no hits — verified.
