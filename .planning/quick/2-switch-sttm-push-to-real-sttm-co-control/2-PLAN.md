---
phase: quick-2
plan: 1
type: execute
wave: 1
depends_on: []
files_modified:
  - apps/transcribe/sttm_controller.py
  - apps/transcribe/app.py
  - apps/transcribe/requirements.txt
autonomous: false
requirements:
  - STTM-01  # STTM push must actually move STTM Desktop's highlight
  - STTM-02  # Must work fully offline (same-machine, no internet)
  - STTM-03  # Stop claiming "pushed ..." when the push didn't land
must_haves:
  truths:
    - "Clicking 'Push to STTM' in the Gradio app advances the highlight on the same-machine STTM Desktop window — verified by eye on the projector/main window."
    - "The push path requires zero internet — confirmed by disconnecting Wi-Fi and seeing the push still land."
    - "When STTM is not reachable (not running, or not launched with --remote-debugging-port), the UI surfaces a clear actionable error and does NOT report a fake 'pushed ...' success."
  artifacts:
    - path: "apps/transcribe/sttm_controller.py"
      provides: "CDP-based STTM controller replacing the dead REST path"
      contains: "class STTMController"
    - path: "apps/transcribe/requirements.txt"
      provides: "playwright dependency"
      contains: "playwright"
  key_links:
    - from: "on_push / on_auto_push handlers in app.py"
      to: "STTMController.push_hit"
      via: "replaces push_hit() import from sttm_controller"
      pattern: "STTMController"
---

<objective>
Replace the dead REST-based STTM push (`POST localhost:8000/api/bani-control`
→ silently no-op) with a **local Chrome-DevTools-Protocol (CDP) controller**
that drives STTM Desktop's Electron renderer directly. Works offline,
same-machine, no sync code, no cloud.

Reference implementation: the user's own `~/.sttm-automate/src/controller/
sttm_playwright.py`. That path is the offline-working one; their
`sttm_http.py` alongside it has an explicit comment that the REST payloads
are unverified guesses — which explains why our port of it has never
worked.
</objective>

<diagnosis>

### Why the current REST path never worked

- `POST http://127.0.0.1:8000/api/bani-control` returns HTTP 200 from
  STTM's overlay socket.io server's catch-all, but **there is no handler**
  for that route — it's a no-op.
- `~/.sttm-automate/src/controller/sttm_http.py:148-153` explicitly
  documents this as an unverified guess — the code we ported was never
  validated to work.
- Local socket.io on :8000 only emits `show-line` / `update-prefs` to
  overlays; emitting `data`/`request-control` to it yields no
  `response-control`. Confirmed via probe 2026-04-23.

### The real offline protocol (same-machine)

- STTM Desktop is an Electron app. When launched with
  `--remote-debugging-port=9222` it exposes Chrome DevTools Protocol
  (CDP) on localhost:9222.
- Playwright connects via `chromium.connect_over_cdp("http://localhost:9222")`,
  grabs the renderer page, and can:
  - Type into the search input (first letters → shabad lookup)
  - Click a specific pangti row (jumps highlight to that line)
  - Send keyboard arrows (line navigation)
- Zero internet. Zero cloud. Zero PIN. All loopback.

### What replaces `push_shabad` / `push_hit`

A new `STTMController` class wrapping a persistent Playwright connection.
Two primary methods:

- `open_shabad(shabad_id, first_line_gurmukhi)` — types first-letters of
  the first line, picks the matching result. Anchors STTM on the shabad.
- `advance_line(verse_id, line_idx, total_lines)` — jumps highlight to a
  specific pangti. Preferred implementation: find the pangti row in the
  renderer DOM (by verseId or index) and click it. Fallback:
  `ArrowDown × (line_idx - current_idx)`.

The controller is **instantiated once**, holds a long-lived CDP connection,
and is shared across pushes (amortizes the ~300 ms CDP handshake).

</diagnosis>

<tasks>

<task type="auto" tdd="false">
  <name>Task 1: Rip dead REST path + add Playwright/CDP-based STTMController</name>
  <files>
    - apps/transcribe/sttm_controller.py
    - apps/transcribe/requirements.txt
  </files>
  <behavior>
    - `sttm_controller.py` no longer contains `push_shabad`, `push_hit`,
      `push_transcript_as_shabad`, or `discover` as REST helpers. Those
      names either go away or become thin shims on top of the new
      controller. (Keep `search_shabad_topn` and `search_shabad` — those
      hit BaniDB and are used for suggest/autolock; unchanged.)
    - New class `STTMController` with:
      - `connect(cdp_port: int = 9222) -> bool` (returns False, does NOT raise, on CDP unreachable)
      - `is_connected() -> bool`
      - `open_shabad(shabad_id: int, first_line_gurmukhi: str) -> bool`
      - `advance_to_verse(verse_id: int, line_idx: int, full_rowids: list[int], gurmukhi: str) -> bool`
      - `push_hit(hit: dict) -> "STTMStatus"` — convenience wrapper that
        decides between open_shabad (first push to a new shabad) and
        advance_to_verse (subsequent pushes within the same shabad).
      - `disconnect() -> None`
    - The controller is **sync-API** (Playwright sync mode via
      `playwright.sync_api.sync_playwright`), because Gradio runs handlers
      synchronously and wrapping async ↔ sync is painful.
    - On CDP unreachable, `STTMStatus.ok = False` with a message like
      "STTM not reachable — launch with `open -a 'SikhiToTheMax' --args
      --remote-debugging-port=9222`". Do NOT print a fake "pushed …"
      success.
    - STTMStatus dataclass stays; message format keeps the old
      `"pushed shabad=... verse=..."` shape so the UI doesn't need to
      change.
    - `requirements.txt` gains `playwright>=1.43`. No `playwright install`
      step is needed for CDP (we attach to an existing Chromium, not
      launch one).

    **Implementation shape for advance_to_verse:**

    Inject a one-line JS into the renderer that finds the pangti by its
    `data-verse-id` attribute (or whatever selector STTM uses) and clicks
    it:

    ```python
    def advance_to_verse(self, verse_id, line_idx, full_rowids, gurmukhi):
        if not self._page:
            return False
        # STTM Desktop marks each pangti with verseId; this is the most
        # robust selector across STTM builds.
        selectors = [
            f"[data-verseid='{verse_id}']",
            f"[data-verse-id='{verse_id}']",
            f"[data-line-index='{line_idx}']",
        ]
        for sel in selectors:
            loc = self._page.locator(sel).first
            try:
                if loc.count() > 0:
                    loc.click(timeout=2000)
                    return True
            except Exception:
                continue
        # Fallback: keyboard arrows from wherever we are.
        try:
            delta = line_idx - self._active_line_idx
            key = "ArrowDown" if delta >= 0 else "ArrowUp"
            for _ in range(abs(delta)):
                self._page.keyboard.press(key)
            return True
        except Exception:
            return False
    ```

    **Implementation shape for open_shabad:**

    Type first-letters of the first pangti into the search field, pick
    the top match. Respects STTM's standard first-letter search UX.

    ```python
    def open_shabad(self, shabad_id, first_line_gurmukhi):
        if not self._page:
            return False
        try:
            q = _first_letters(first_line_gurmukhi)  # Gurmukhi akhars only
            si = self._page.locator(
                "input.search, input#search, input[type='search']").first
            si.click()
            si.fill("")
            si.type(q, delay=40)
            self._page.wait_for_timeout(350)
            # Click the top result row — STTM's result list uses .result-row or similar.
            rr = self._page.locator(".result-row, .search-result-row, li.result").first
            if rr.count() > 0:
                rr.click(timeout=2000)
                self._active_shabad_id = shabad_id
                self._active_line_idx = 0
                return True
        except Exception:
            pass
        return False
    ```

    (These selectors are a best guess — Task 2 is the human-verify step
    that confirms them against the running STTM Desktop DOM. If selectors
    are wrong, we fix them there. We do NOT silently fallback to REST.)

  </behavior>
  <action>
    1. Rewrite apps/transcribe/sttm_controller.py:
       - Drop `push_shabad`, `push_hit`, `push_transcript_as_shabad`, `discover`, `CANDIDATE_PORTS`, `_post_json`.
       - Keep `search_shabad_topn`, `search_shabad`, `_norm_hit`.
       - Add `class STTMController` per the behavior block above.
       - Add module-level singleton `get_controller()` that lazy-creates
         one STTMController instance and returns it. Thread-safe with a lock.
    2. Add `playwright>=1.43` to apps/transcribe/requirements.txt.
    3. Preserve the module's existing imports used elsewhere (search_shabad_topn etc.).
    4. Do NOT modify apps/transcribe/app.py in this task — that's Task 2.
  </action>
  <verify>
    <automated>cd /Users/surindersingh/Developer/Gurbani_ASR_v4 &amp;&amp; python3 -c "
import ast
src = open('apps/transcribe/sttm_controller.py').read()
tree = ast.parse(src)
cls = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
fns = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
assert 'STTMController' in cls, 'STTMController class missing'
for m in ['connect', 'open_shabad', 'advance_to_verse', 'push_hit', 'disconnect']:
    assert m in fns, f'missing method {m}'
assert 'get_controller' in fns
# REST leftovers must be gone
assert 'push_shabad' not in fns or 'push_shabad' in {m.name for m in ast.walk(tree) if isinstance(m, ast.FunctionDef) and m.name=='push_shabad' and len(m.body)==0}, 'REST push_shabad must be removed'
assert '/api/bani-control' not in src, 'REST endpoint string must be removed'
assert 'CANDIDATE_PORTS' not in src, 'REST port list must be removed'
# BaniDB search helpers preserved
assert 'search_shabad_topn' in fns and 'search_shabad' in fns
# Playwright import
assert 'from playwright' in src or 'import playwright' in src, 'playwright not imported'
print('sttm_controller.py shape OK')
"
grep -q '^playwright' apps/transcribe/requirements.txt || (echo 'playwright missing from requirements.txt' &amp;&amp; exit 1)
echo 'requirements.txt OK'</automated>
  </verify>
  <done>
    - New STTMController class present with connect/open_shabad/advance_to_verse/push_hit/disconnect methods.
    - get_controller() singleton exists.
    - REST helpers and port-probe removed from sttm_controller.py.
    - apps/transcribe/requirements.txt lists `playwright>=1.43`.
    - Static checks above pass.
  </done>
</task>

<task type="auto" tdd="false">
  <name>Task 2: Wire app.py push handlers to the new controller</name>
  <files>apps/transcribe/app.py</files>
  <behavior>
    - `on_push`, `on_auto_push`, and any other push call-site routes
      through `get_controller().push_hit(hit)` instead of the old REST
      `push_hit(host, port, hit, pin=…)`.
    - Remove the STTM host/port fields from StreamState (no longer
      used). `st.sttm_pin` stays (kept for any future pin-gated path).
    - Replace the sttm discovery status (the line that used to show
      `STTM reachable on :8000`) with a CDP reachability check:
      "STTM control OK (CDP :9222)" vs "STTM not reachable — launch
      with --remote-debugging-port=9222" (shortened for the side panel).
    - Auto-push ("Push to STTM" button + the locked-auto toggle) still
      prints the existing `[sttm] pushed sid=... verseId=...` log line
      on success. On failure, prints the CDP error clearly — no
      HTTP-200-but-silently-failed pattern.
  </behavior>
  <action>
    1. In app.py, replace the import line
       `from .sttm_controller import (push_hit, push_transcript_as_shabad, ...)`
       with one that imports `get_controller, search_shabad_topn, search_shabad, STTMStatus`.
    2. Update every call site:
       - `push_hit(st.sttm_host, st.sttm_port, top, pin=st.sttm_pin)` →
         `get_controller().push_hit(top)`
       - `push_transcript_as_shabad(...)` → remove (nobody uses it after
         this refactor; transcript-to-shabad logic lives in the retriever path).
    3. Drop `StreamState.sttm_host`, `StreamState.sttm_port`, and the
       code that calls `discover()` at app startup.
    4. Update the side-panel STTM status HTML to call
       `get_controller().is_connected()` and render a short string.
    5. Leave `st.sttm_pin` alone (persisted PIN still loads from
       `~/.surt/config.json`; we just don't use it on the CDP path, but
       keep the field so config round-trips cleanly).
  </action>
  <verify>
    <automated>cd /Users/surindersingh/Developer/Gurbani_ASR_v4 &amp;&amp; python3 -c "
src = open('apps/transcribe/app.py').read()
assert 'push_hit(' in src or 'get_controller' in src
# No more REST args being passed
assert 'st.sttm_host' not in src, 'st.sttm_host still referenced'
assert 'st.sttm_port' not in src, 'st.sttm_port still referenced'
assert 'CANDIDATE_PORTS' not in src
# Old REST import style gone
assert 'from .sttm_controller import push_hit' not in src
assert 'get_controller' in src, 'new controller accessor not wired'
print('app.py wiring OK')
"</automated>
  </verify>
  <done>
    - app.py imports `get_controller` (and not `push_hit`-as-function-from-module).
    - No references to `st.sttm_host`/`st.sttm_port` remain.
    - Static checks pass.
  </done>
</task>

<task type="checkpoint:human-verify" gate="blocking">
  <name>Task 3: Human verification — offline STTM control works end-to-end</name>
  <what-built>
    sttm_controller.py now speaks CDP, app.py push paths call
    `get_controller().push_hit(...)`, and no REST endpoint is contacted.
    Playwright is listed in requirements.
  </what-built>
  <how-to-verify>
    1. `pip install -r apps/transcribe/requirements.txt` (installs playwright;
       no `playwright install` needed since we attach to existing Chromium).
    2. **Relaunch STTM with CDP:**
       ```
       killall SikhiToTheMax || true
       open -a "SikhiToTheMax" --args --remote-debugging-port=9222
       ```
    3. Verify `curl http://127.0.0.1:9222/json/version` returns a JSON
       with `Browser` / `webSocketDebuggerUrl`. If not, STTM didn't honor
       the flag — stop and inspect.
    4. Restart the Gradio app (`python -m apps.transcribe.app`). The side
       panel should read "STTM control OK (CDP :9222)".
    5. Disconnect Wi-Fi (Airplane mode / `networksetup -setairportpower en0 off`).
    6. Play a kirtan clip into the mic (or URL tab). Once a shabad locks
       and the pointer has moved past line 1, click **Push to STTM**.
    7. **Watch STTM Desktop's main window** — the highlighted pangti
       should jump to the line our pointer is on. If it does, the offline
       path works end-to-end.
    8. Let it run for ~30 s on a moving shabad — auto-push should advance
       the highlight line-by-line without internet.
    9. Turn Wi-Fi back on.

    **Failure diagnostics:**
    - If CDP reachable but clicks don't land → the selectors are wrong.
      Open STTM's DevTools (CDP exposes them), inspect a pangti row, find
      the real attribute name (`data-verseid`, `data-line-id`, etc.),
      and paste it back here — I'll patch the selector list.
    - If CDP not reachable after relaunch → STTM may not honor the flag on
      this build; check `ps -ef | grep SikhiToTheMax` for the flag in argv.
  </how-to-verify>
  <resume-signal>
    Type "approved" if the highlight moves on STTM when you click Push
    (with Wi-Fi off). Otherwise paste: (a) the STTM control status line,
    (b) one console line from the Gradio terminal when you clicked Push,
    and (c) the HTML of one pangti row from STTM's DOM inspector.
  </resume-signal>
</task>

</tasks>

<verification>
- `grep -rn "/api/bani-control\|CANDIDATE_PORTS\|push_shabad" apps/transcribe/` returns no hits.
- `python3 -c "from apps.transcribe.sttm_controller import STTMController, get_controller; print('ok')"` succeeds.
- On a fresh launch, the app reports the CDP status honestly (reachable or not), not a fake 2xx.
- Live-mic, file-upload, URL-play, retriever, lock-streak code all untouched.
- requirements.txt contains `playwright>=1.43`.
</verification>

<success_criteria>
- Offline (no internet) push from Gradio app moves STTM Desktop's highlight on the same machine.
- When STTM is not launched with CDP, the app says so clearly (no silent success).
- No REST POSTs to 8000/42424/… in the push path.
- Mic / retriever / lock-streak paths untouched (no regression on the Task 1 parity fix we just shipped).
</success_criteria>

<output>
Append a one-paragraph note to
`.planning/quick/2-switch-sttm-push-to-real-sttm-co-control/2-SUMMARY.md`
describing the switch from dead-REST to offline-CDP, the files touched,
and the pending Task 3 human verification.
</output>
