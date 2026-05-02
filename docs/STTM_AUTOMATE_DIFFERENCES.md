# Gurbani_ASR_v4 vs sttm-automate (Live Listening + Transcription)

This note captures how our current approach in `Gurbani_ASR_v4` differs from `sttm-automate`, based on code-level comparison.

## 1) Core Matching Strategy

- **Gurbani_ASR_v4 (`apps/transcribe`)**
  - Semantic-first retrieval: MuRIL embeddings + FAISS tuk/shabad indexes.
  - Optional literal rerank (char 4-gram overlap).
  - Locked mode uses in-shabad pointer scoring for fast line movement.
  - Key files:
    - `apps/transcribe/retriever.py`
    - `apps/transcribe/app.py`

- **sttm-automate**
  - First-letter pipeline (Gurmukhi -> first letters -> indexed search).
  - SQLite offline search (`first_letters` prefix/contains) + optional full-word search.
  - SequenceMatcher-style confidence scoring + word-overlap bonuses.
  - Key files:
    - `src/matcher/offline_search.py`
    - `src/matcher/scorer.py`
    - `src/transcription/transliterate.py`

## 2) Pipeline / State Machine Complexity

- **Gurbani_ASR_v4**
  - Simpler flow: stream -> transcribe -> retrieve -> optional lock -> auto-push.
  - Lock/unlock primarily threshold + streak based.
  - Uses EMA smoothing on live retrieval results.
  - Key logic:
    - `apps/transcribe/app.py` (`_refresh_matches`, `_maybe_auto_lock`, `_check_unlock`)
    - `apps/transcribe/ema.py`

- **sttm-automate**
  - Heavier explicit state machine:
    - `SEARCHING`, `CANDIDATE_LOCK`, `LOCKED`, `UNSTABLE_LOCK`.
  - Challenger logic, persistence windows, decaying hypothesis evidence.
  - Separate recovery behaviors when line confidence gets weak.
  - Key logic:
    - `src/matcher/tracker.py`
    - `src/pipeline/orchestrator.py`

## 3) Audio Windowing / Timing Behavior

- **Gurbani_ASR_v4**
  - Fixed commit-and-carry style windows for live flow.
  - Typical settings:
    - `commit_s=10.0`, `max_window_s=12.0`, `carry_over_s=2.0`
    - fast pointer short cadence while locked.
  - Key fields and handlers in `apps/transcribe/app.py`.

- **sttm-automate**
  - Adaptive windowing by context:
    - start-after-break window,
    - locked fast/normal/recovery windows,
    - search fast/normal windows.
  - Also adapts by speech-rate estimate (letters/sec EMA).
  - Key logic:
    - `src/config.py` (window configs)
    - `src/pipeline/orchestrator.py` (`_select_transcription_audio`)

## 4) ASR Decoding Defaults

- **Gurbani_ASR_v4**
  - CT2/faster-whisper path prioritized.
  - Lower-latency defaults (e.g., beam size 1 in CT2 backend).
  - Supports prompt carry and optional VAD filter toggle.
  - Key file:
    - `apps/transcribe/backend.py`

- **sttm-automate**
  - CT2 conversion-on-first-load for HF model.
  - More conservative decode default (`beam_size=5` in config).
  - Extra post-processing filter for garbage/hallucination patterns.
  - Key files:
    - `src/transcription/engine.py`
    - `src/transcription/processor.py`
    - `src/config.py`

## 5) Input/Capture Path

- **Gurbani_ASR_v4**
  - Gradio/browser streaming mic + file + URL/HLS playback integration.
  - Strong UI controls/toggles for live experimentation.
  - Key files:
    - `apps/transcribe/app.py`
    - `apps/transcribe/stream_url.py`

- **sttm-automate**
  - Primary local capture with `sounddevice` and audio device auto-selection
    (BlackHole/aggregate/default), plus remote mic mode via websocket.
  - Key files:
    - `src/audio/capture.py`
    - `src/api/server.py`

## 6) STTM Control Path

- **Gurbani_ASR_v4**
  - CDP Playwright control path is the primary mechanism.
  - Explicit `push_hit()` handling for opening shabad and advancing verse/line.
  - Key file:
    - `apps/transcribe/sttm_controller.py`

- **sttm-automate**
  - HTTP controller is primary, Playwright fallback.
  - HTTP payload strategy relies on STTM local endpoint behavior.
  - Key files:
    - `src/controller/sttm_http.py`
    - `src/controller/sttm_playwright.py`

## 7) Why sttm-automate May Feel Better in Live Tracking

Likely contributors:

- More explicit multi-state lock lifecycle (`CANDIDATE_LOCK`, `UNSTABLE_LOCK`).
- Adaptive windows tied to context (search vs lock vs recovery).
- Challenger/hypothesis persistence logic reduces abrupt wrong switches.
- First-letter + word-overlap heuristics are tightly aligned with STTM-like navigation behavior.

## Quick Summary

- `Gurbani_ASR_v4` is **embedding-retrieval centric** with strong live UI tooling and lock-pointer mechanics.
- `sttm-automate` is **state-machine + first-letter heuristic centric**, with aggressive runtime adaptation and confidence policy controls.
- Both are valid; they optimize for different failure modes.
