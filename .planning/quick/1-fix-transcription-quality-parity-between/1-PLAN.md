---
phase: quick-1
plan: 1
type: execute
wave: 1
depends_on: []
files_modified:
  - apps/transcribe/app.py
autonomous: false
requirements:
  - PARITY-01  # File/URL paths use same windowing strategy as mic
  - PARITY-02  # Upload path must not silently truncate >30s audio
must_haves:
  truths:
    - "Same audio file played back via the mic (loopback) and via the 'Play from file/URL' tab produces substantially similar committed transcripts (edit distance within ~10%)."
    - "Uploading an audio file longer than 30 s produces a transcript covering the whole file, not just the first 30 s."
    - "URL/playback windowing keeps carry-over context across window boundaries so mid-phrase cuts don't fragment into hallucinated completions."
    - "No change to live-mic behaviour: Silero-VAD gate, min_transcribe_s=3.0, RMS pre-gate, fast-pointer path, lock/unlock math all untouched."
  artifacts:
    - path: "apps/transcribe/app.py"
      provides: "Shared chunked-decode helper + refactored _on_stream_url / on_play_sync / on_upload / _drain_tail consumers"
      contains: "def _chunked_decode"
  key_links:
    - from: "_on_stream_url"
      to: "_chunked_decode"
      via: "replaces inline 10s non-overlapping window loop at app.py:2203-2228"
      pattern: "_chunked_decode\\("
    - from: "on_play_sync"
      to: "_chunked_decode"
      via: "replaces inline 10s window loop at app.py:2316-2336"
      pattern: "_chunked_decode\\("
    - from: "on_upload"
      to: "_chunked_decode"
      via: "replaces single-shot _transcribe at app.py:1915 for audio >max_window_s"
      pattern: "_chunked_decode\\("
---

<objective>
Bring file-upload and URL/play-sync transcription quality up to live-mic
parity by making all three paths share the mic's commit-and-carry window
strategy. Today's gap is almost entirely a windowing + truncation
problem, not a resample/VAD/decoder problem.

Purpose: user reports worse output from the file / URL paths on the same
content. This is real — the URL path cuts Whisper mid-phrase every 10 s
with zero carry-over, and `on_upload` either one-shots (CT2 can VAD-chunk
internally) or silently truncates to 30 s on the Torch backend.

Output: a single `_chunked_decode` helper wired into all three non-mic
paths, replacing ~3 open-coded 10-s loops.
</objective>

<diagnosis>

Static audit of the two paths against the investigation hints. Ranked by
impact on transcription quality:

### 1. WINDOW SIZE & CARRY-OVER (root cause, large impact)

**Mic path** (`on_stream`, app.py:1806-1902)
- `commit_s = 10.0`, `max_window_s = 12.0`, `carry_over_s = 2.0`
  (StreamState defaults, app.py:137-139).
- When buffer > 12 s: commits first 10 s, **keeps the last 2 s** as carry
  (line 1844 `st.buffer = st.buffer[-carry_samples:].copy()`). Next decode
  sees the tail of the previous phrase, so Whisper has context.
- Per-tick "tentative" decode of the whole buffer (line 1887) — Whisper
  gets up to 12 s of audio with full internal attention.
- `_merge_committed` (app.py:278-308) de-duplicates the carry-over overlap
  at word level when committing.

**URL/play-sync path** (`_on_stream_url` app.py:2203-2228, `on_play_sync`
app.py:2316-2336)
- `window_s = 10.0`, `step_s = 10.0` → **non-overlapping** windows.
- `s0 = transcribed_up_to; s1 = s0 + window_samples; transcribed_up_to = s1`
  (line 2208-2228) — every window starts at a hard sample boundary.
- No carry-over. No merge-dedup. Whisper enters mid-word, emits a
  plausible-but-wrong word completion, then the next window enters mid-
  word on the next phrase. Systematic fragmentation on connected kirtan.
- The 3-second tail drain (line 2268-2273) is called only at end-of-stream.

**Upload path** (`on_upload` app.py:1906-1928)
- `y = _resample_hq(...); text = _transcribe(st, y)` — **one call** on the
  entire audio regardless of length (line 1915).
- CT2 backend (backend.py:75-102) auto-VAD-chunks internally, so uploads
  come out OK-ish on CT2.
- **Torch backend (backend.py:170-181) truncates to 30 s** — the feature
  extractor pads/truncates to Whisper's fixed 30 s window. Uploads longer
  than 30 s lose everything past second 30, silently. This is the
  "uploads are terrible" bug.

### 2. VAD / RMS GATE (minor, asymmetric but defensible)

Mic:
- Backend Silero VAD via `vad_filter=st.vad_filter_enabled` (line 1761,
  default True).
- Additional RMS pre-gate `_rms(slice) >= st.vad_threshold (0.005)` before
  every `_transcribe` call (lines 1838, 1858, 1884).
- `min_transcribe_s = 3.0` guard (lines 1833-1836, 1853-1859, 1882-1884).

URL/upload:
- Backend Silero VAD runs the same way — `_transcribe` at line 1760 is
  shared. **The backend-level VAD is in place on both paths.** Parity is
  preserved there.
- No RMS pre-gate on URL/upload. That's **correct** — mic's 0.005 floor
  is calibrated for browser-mic noise; studio audio has different
  absolute levels and gating on RMS would be wrong. No change.
- No `min_transcribe_s` guard on `on_upload` for tiny clips — low impact,
  CT2/Torch both handle short audio fine. Leave alone.

### 3. RESAMPLE (non-issue)

- Mic: `_resample_hq` (scipy polyphase, anti-aliased) at line 1825.
- Upload: same `_resample_hq` at line 1913 — **identical**.
- URL stream: ffmpeg (`stream_url.py:119`) produces 16 kHz mono directly
  via `-ar 16000 -ac 1`. No resample in Python. ffmpeg's resampler is
  high-quality — not the problem.
- Player file via `player.py::load_audio_16k` (line 39-46) uses
  `_resample_to` — **linear np.interp, no anti-alias**. But `player_audio`
  is only used by `on_play_sync` (app.py:2291), and the URL branch
  dominates real usage. Low priority; can fix later if user hits it on
  a high-sample-rate local file.

### 4. GAIN NORMALIZE (non-issue)

- Mic: `_normalize_gain` opt-in via `st.gain_normalize` (line 1827),
  default False.
- Upload/URL: not applied. Matches mic default. No action.

### 5. MONO DOWNMIX (non-issue)

- Mic/upload: `_to_mono_float32` → `y.mean(axis=1)` (line 185-192).
- URL: ffmpeg `-ac 1` upstream. All paths end up mono the same way.

### 6. DECODER ARGS (non-issue — user memory explicit)

- All paths call the same `_transcribe` → `backend.transcribe` helper
  (app.py:1755-1768). No divergence in temperature, beam size,
  `condition_on_previous_text`, or `no_repeat_ngram_size`. CT2 kwargs
  at backend.py:90-100 are identical for mic and file. Memory file
  `feedback_whisper_gurbani_decoder.md` forbids changing any of this —
  respected.

### Conclusion

The quality gap is almost entirely **#1 (windowing / truncation)**. The
mic's commit-and-carry pipeline is the tuned path per project memory.
Rather than inventing a third path, extract it into `_chunked_decode`
and call it from the URL, play-sync, and upload handlers.

</diagnosis>

<execution_context>
@/Users/surindersingh/.claude/get-shit-done/workflows/execute-plan.md
</execution_context>

<context>
@apps/transcribe/app.py
@apps/transcribe/backend.py
@apps/transcribe/player.py
@apps/transcribe/stream_url.py
@apps/transcribe/RETRIEVER.md

<interfaces>
<!-- Key existing helpers the executor will lean on. -->

From apps/transcribe/app.py (already defined):
- `_to_mono_float32(y: np.ndarray) -> np.ndarray`  (line 185)
- `_resample_hq(y, sr_in, sr_out=16000) -> np.ndarray`  (line 206)
- `_merge_committed(prev: str, new: str, *, max_overlap_words=12) -> str`  (line 278)
- `_suppress_repeat_hallucination(text: str) -> str`  (line 259)
- `_transcribe(st, audio, sr=16000) -> str`  (nested inside build_app, line 1755)
- StreamState defaults: commit_s=10.0, max_window_s=12.0, carry_over_s=2.0, min_transcribe_s=3.0  (line 137-142)

Backend contract (backend.py:75, 114, 170):
- `backend.transcribe(audio: np.ndarray, sr: int, *, vad_filter: bool = True) -> str`
- CT2 handles long audio via internal VAD chunking. Torch TRUNCATES to 30 s.
</interfaces>
</context>

<tasks>

<task type="auto" tdd="false">
  <name>Task 1: Extract _chunked_decode helper and wire URL + play-sync + upload through it</name>
  <files>apps/transcribe/app.py</files>
  <behavior>
    - Uploaded audio &gt; max_window_s (12 s) is transcribed in 10-s commit
      windows with 2-s carry-over, not a single shot. This fixes the
      Torch-backend 30-s truncation bug AND matches mic output quality.
    - URL/playback 10-s windows keep a 2-s tail of the previous window
      as audio context for the next decode; committed text is merged
      with `_merge_committed` so the carry-over doesn't duplicate words.
    - Uploaded audio &lt;= max_window_s still takes the one-shot fast path
      (no change for short clips — avoids regressions on 5-s test clips).
    - Live-mic `on_stream` is UNCHANGED — no edits to lines 1806-1902.
      The mic's inline commit-and-carry stays where it is; the new helper
      is a non-mic consolidation only. (Refactoring the mic into the
      helper is out of scope — the mic has throttle/fast-pointer/
      retrieval-smoothing logic intertwined that we do not want to touch
      per project memory.)
    - The `_transcribe` wrapper (line 1755) and the `backend.transcribe`
      kwargs path are untouched. Silero VAD, min_silence_duration_ms=300,
      no temperature fallback, no no_repeat_ngram_size — all preserved.
  </behavior>
  <action>
    1. Inside `build_app` (before `_transcribe` at app.py:1755), add a new
       nested helper:

       ```python
       def _chunked_decode(
           st: StreamState,
           audio: np.ndarray,
           *,
           on_window=None,
       ) -> str:
           """Mic-parity chunked decode for non-streaming paths.

           Replicates on_stream's commit-and-carry: 10-s commit windows
           with 2-s carry-over merged via _merge_committed so Whisper
           never enters a window at a hard mid-word boundary.

           on_window(text, s0, s1) is called after each committed window
           so _on_stream_url / on_play_sync can run _refresh_matches +
           auto-push per window (their current per-window side effects).

           Returns the fully merged transcript (useful for on_upload).
           """
           if audio is None or audio.size == 0:
               return ""
           commit_samples = int(st.commit_s * TARGET_SR)
           max_samples = int(st.max_window_s * TARGET_SR)
           carry_samples = int(st.carry_over_s * TARGET_SR)
           # Short clip: one-shot is fine (and faster).
           if audio.size <= max_samples:
               text = _transcribe(st, audio) or ""
               if text and on_window is not None:
                   on_window(text, 0, audio.size)
               return text
           merged = ""
           cursor = 0
           while cursor + commit_samples <= audio.size:
               s0 = cursor
               s1 = s0 + commit_samples
               window = audio[s0:s1]
               text = _transcribe(st, window) or ""
               if text:
                   merged = _merge_committed(merged, text)
                   if on_window is not None:
                       on_window(text, s0, s1)
               # Advance by commit - carry so next window sees 2 s overlap.
               cursor = s1 - carry_samples
           # Drain the tail if &gt;= min_transcribe_s of audio remains.
           remainder = audio[cursor:]
           min_samples = int(st.min_transcribe_s * TARGET_SR)
           if remainder.size >= min_samples:
               text = _transcribe(st, remainder) or ""
               if text:
                   merged = _merge_committed(merged, text)
                   if on_window is not None:
                       on_window(text, cursor, audio.size)
           return merged
       ```

    2. **on_upload** (app.py:1906-1928): replace the line
       `text = _transcribe(st, y) or ""` with
       `text = _chunked_decode(st, y)`. Leave everything else (mono,
       resample, stats) as-is.

    3. **on_play_sync** (app.py:2316-2336): replace the inline window loop
       with a per-window callback that appends into `st.committed` via
       `_merge_committed` (NOT the current naive space-concat) and runs
       `_refresh_matches` + auto-push, then calls `_chunked_decode` on
       the whole `audio` once with that callback. Preserve the wall-clock
       pacing by advancing `i` inside the callback and sleeping between
       callbacks — simplest is to inline a small pacing state dict; see
       commit for exact shape.

    4. **_on_stream_url** (app.py:2203-2228): this one is streaming
       (buffer grows over time), so it cannot just call `_chunked_decode`
       on the whole buffer. Instead, change the windowing so when we
       decide to decode a window, we use:

       ```python
       commit_samples = int(st.commit_s * TARGET_SR)
       carry_samples = int(st.carry_over_s * TARGET_SR)
       # ... inside the while loop, when playhead allows:
       s0 = transcribed_up_to
       s1 = s0 + commit_samples
       window = buf[s0:s1]
       text = _transcribe(st, window)
       if text:
           st.committed = _merge_committed(st.committed, text)  # was naive " "+text
           _refresh_matches(st, smooth=True)
           # ... auto-push block unchanged
       transcribed_up_to = s1 - carry_samples  # was: transcribed_up_to = s1
       ```

       Also update the drain loop (app.py:2247-2273) to use the same
       `transcribed_up_to = s1 - carry_samples` advance and
       `_merge_committed` for appends. This replicates mic carry-over
       exactly with O(1) added cost.

    5. Do NOT touch `on_stream` (app.py:1806-1902). Do NOT touch
       `_transcribe` (app.py:1755). Do NOT touch `_refresh_matches`,
       `_handle_locked`, `_maybe_auto_lock`, `_try_auto_push`, or any
       retriever/lock code.

    6. Double-check that `st.committed` after these changes is still a
       plain string used by `_retrieval_query` and `_render_transcript_html` —
       yes, `_merge_committed` returns a string. No schema change.
  </action>
  <verify>
    <automated>cd /Users/surindersingh/Developer/Gurbani_ASR_v4 &amp;&amp; python -c "
import numpy as np
import importlib, sys, types
# Stub heavy deps so we can import app.py without spinning a backend.
for m in ['gradio', 'torch', 'transformers', 'faster_whisper', 'soundfile']:
    sys.modules.setdefault(m, types.ModuleType(m))
sys.path.insert(0, 'apps/transcribe')
# Smoke-test that the helpers we rely on still exist and the module parses.
import ast
src = open('apps/transcribe/app.py').read()
tree = ast.parse(src)
names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
assert '_chunked_decode' in names, '_chunked_decode missing'
assert '_merge_committed' in names, '_merge_committed still needed'
assert 'on_stream' in names, 'on_stream intact'
assert 'on_upload' in names, 'on_upload intact'
# Confirm on_stream body is untouched by checking a signature line survives
assert 'commit_samples = int(st.commit_s * TARGET_SR)' in src
assert 'carry_samples = int(st.carry_over_s * TARGET_SR)' in src
# Confirm URL path now uses _merge_committed (not naive concat) at least once
# in _on_stream_url body.
url_fn_src = src.split('def _on_stream_url')[1].split('def on_play_sync')[0]
assert '_merge_committed' in url_fn_src, 'URL path must use _merge_committed for carry-over dedupe'
# And carry-over advance
assert 'carry_samples' in url_fn_src, 'URL path must advance by commit - carry'
# Upload must route through _chunked_decode
upload_fn_src = src.split('def on_upload')[1].split('def on_clear')[0]
assert '_chunked_decode' in upload_fn_src, 'on_upload must call _chunked_decode'
print('parity static checks OK')
"</automated>
  </verify>
  <done>
    - `_chunked_decode` helper exists inside `build_app` above `_transcribe`.
    - `on_upload` routes through `_chunked_decode` (no direct
      `_transcribe(st, y)` for the full upload).
    - `_on_stream_url` and `on_play_sync` use 2-s carry-over advance
      (`transcribed_up_to = s1 - carry_samples`) and append via
      `_merge_committed`.
    - `on_stream` source range (app.py:1806-1902) byte-identical to
      pre-change (`git diff` shows zero hunks there).
    - `_transcribe` helper (app.py:1755-1768) unchanged.
    - Static smoke assertions above pass.
  </done>
</task>

<task type="checkpoint:human-verify" gate="blocking">
  <name>Task 2: Human A/B parity check — same audio via mic loopback vs URL vs upload</name>
  <what-built>
    Task 1 wired upload / URL / play-sync through a shared chunked-decode
    helper with 10-s commit windows + 2-s carry-over, matching the mic's
    commit-and-carry shape. Backend VAD and decoder args unchanged on
    every path.
  </what-built>
  <how-to-verify>
    Run the app locally and A/B the same audio three ways. The user has
    a known-good kirtan clip handy (pick a 3–5 minute SikhNet / akj URL
    they've already verified on mic).

    1. Start the app:
       ```
       cd /Users/surindersingh/Developer/Gurbani_ASR_v4
       python -m apps.transcribe.app
       ```
    2. **Upload test:** Upload the &gt;30 s clip via the upload tab. Confirm:
       - Transcript length scales with audio duration (no 30-s cliff on
         the Torch backend).
       - Gurmukhi output is connected, not fragmented every 10 s.
    3. **URL test:** Paste the same SikhNet/YouTube URL into Play from
       file / URL. Let it run ~1 minute. Confirm:
       - Committed transcript reads coherently — no half-word splits at
         ~10 s intervals ("ਪ੍ਰੀ" then "ਤਿ ਮੇਰੀ" on the next window).
       - STTM pointer still advances line-by-line when locked (retriever
         code untouched; regression check).
    4. **Mic loopback (gold):** Play the same clip through system audio
       into the mic (or run `ffplay` on another box). Record the live-mic
       transcript. This is the reference.
    5. **Compare** all three transcripts. Upload and URL should now be
       *substantially similar* to mic (word error within ~10%), not
       obviously worse. If one path is still visibly worse than mic,
       note which and stop.

    Also verify no regression on live mic:
    6. Without playing any file, speak into the mic. Confirm Silero VAD
       still gates quiet/silent buffers, `min_transcribe_s=3.0` still
       holds (no hallucinated "ਵਾਹਿਗੁਰੂ" on short buffers), shabad lock
       + pointer advance still work.
  </how-to-verify>
  <resume-signal>
    Type "approved" if all three paths produce comparable quality and mic
    is unregressed. Otherwise describe which path is still worse and
    paste ~10 lines of each transcript for the same audio region.
  </resume-signal>
</task>

</tasks>

<verification>
Phase-level checks:
- `git diff apps/transcribe/app.py` shows NO changes in the range
  `on_stream` (app.py:1806-1902) or `_transcribe` (1755-1768).
- Changes confined to: new `_chunked_decode` helper, and the three
  consumer sites (`on_upload`, `_on_stream_url`, `on_play_sync`).
- Project-memory invariants preserved:
  - No temperature fallback added.
  - No `no_repeat_ngram_size` added.
  - Silero VAD still on by default via `vad_filter=st.vad_filter_enabled`.
  - `min_transcribe_s=3.0` still respected on mic tail paths.
  - Retriever, lock-streak, pointer-advance, STTM controller: untouched.
</verification>

<success_criteria>
- Uploaded audio &gt; 30 s on Torch backend produces a transcript covering
  the whole file (not truncated to the first 30 s).
- URL / play-sync committed transcripts no longer show systematic word
  fragmentation at 10-s window boundaries.
- User confirms in Task 2 that upload and URL output is comparable in
  quality to mic on the same clip.
- Live-mic behaviour (lock/unlock, pointer advance, auto-push, VAD gate)
  is unchanged.
</success_criteria>

<output>
After completion, append a one-paragraph note to
`.planning/quick/1-fix-transcription-quality-parity-between/SUMMARY.md`
describing the fix (shared `_chunked_decode`, carry-over windowing,
Torch-backend 30-s truncation closed) and which paths were touched.
</output>
