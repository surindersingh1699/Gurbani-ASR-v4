---
phase: quick-1
plan: 1
subsystem: apps/transcribe
tags: [transcribe, whisper, windowing, upload, url-stream, play-sync]
status: code-complete-awaiting-human-verify
dependency_graph:
  requires: [_merge_committed, _transcribe, StreamState.commit_s/max_window_s/carry_over_s]
  provides: [_chunked_decode]
  affects: [on_upload, _on_stream_url, on_play_sync]
tech_stack:
  added: []
  patterns: [commit-and-carry windowing, word-level overlap merge]
key_files:
  created: []
  modified: [apps/transcribe/app.py]
decisions:
  - Reuse existing _merge_committed semantics for carry-over dedup across all three non-mic paths instead of inventing a new merge function.
  - Keep on_upload one-shot for clips <= max_window_s (12 s) to avoid regressing short test clips; only chunk when audio is long enough to cross a window boundary.
  - For streaming paths (_on_stream_url, on_play_sync) do not literally call _chunked_decode — they need wall-clock pacing. Instead replicate the same semantics inline: advance transcribed_up_to by `s1 - carry_samples` and merge via _merge_committed.
  - on_stream (live mic) left byte-identical — refactoring it into _chunked_decode would touch the fast-pointer / min_transcribe_s / throttle logic which is explicitly out-of-scope per project memory `feedback_whisper_gurbani_decoder.md` and `feedback_lock_streak_default.md`.
metrics:
  completed_date: 2026-04-22
  task_commit: 43fe190
---

# Quick Task 1: Fix Transcription Quality Parity Between Mic and Non-Mic Paths

One-liner: Upload, URL-stream, and play-sync paths now share the live-mic commit-and-carry semantics (10 s commit windows with 2 s carry-over, word-level `_merge_committed` dedup), closing the Torch-backend 30 s truncation bug and stopping systematic mid-word fragmentation at window boundaries.

## What Changed

- **New helper `_chunked_decode(st, audio, *, on_window=None)`** in `build_app`, placed immediately after `_transcribe`. Walks audio in 10 s commit windows advancing by `commit - carry` samples, merges per-window Whisper output via `_merge_committed`, and drains any `>= min_transcribe_s` tail remainder. Short clips (`audio.size <= max_window_s * SR`) fall through to a one-shot `_transcribe` call for speed.
- **`on_upload`** now routes through `_chunked_decode(st, y)` instead of a direct `_transcribe(st, y)`. This is the fix for the long-standing Torch-backend 30 s truncation bug: `backend.transcribe` feeds the entire audio to Whisper's fixed 30 s mel extractor in one shot, silently dropping everything past second 30. With chunking, long uploads produce a full transcript and also get mic-quality output on CT2.
- **`_on_stream_url`** now advances `transcribed_up_to = s1 - carry_samples` in both the live streaming loop and the post-stream drain loop, giving the next `_transcribe` call 2 s of audio context from the prior window. `st.committed` was already merged via `_merge_committed` (pre-existing working-tree change); the audio handed to the model now overlaps too, matching the mic pipeline end-to-end.
- **`on_play_sync`** (local file, wall-clock-paced) same treatment: `carry_samples = int(st.carry_over_s * sr)` added, `transcribed_up_to = s1 - carry_samples` in the per-played-window decode loop.

## What Was NOT Changed

Verified byte-identical between HEAD and working tree (after my edits):

- `on_stream` (live mic path) — current lines 1896–1992, same 4927-byte body as HEAD. Silero VAD gate, `min_transcribe_s = 3.0`, fast-pointer locked-mode path, RMS pre-gate, throttle, lock/unlock math all untouched.
- `_transcribe` wrapper — current lines 1785–1798, same 694-byte body as HEAD. `backend.transcribe(audio, sr, vad_filter=st.vad_filter_enabled)` contract preserved. No temperature fallback, no `no_repeat_ngram_size`, no beam-size changes (respects `feedback_whisper_gurbani_decoder.md`).

Also untouched: `_refresh_matches`, `_handle_locked`, `_maybe_auto_lock`, `_try_auto_push`, retriever, STTM controller, lock-streak, pointer advance.

## Files Modified

- `apps/transcribe/app.py` — single-file change, commit `43fe190`.

## Commits

- `43fe190` — `feat(transcribe): mic-parity chunked decode for upload / URL / play-sync`

## Deviations from Plan

None that required the user's sign-off. The `<on_play_sync_note>` constraint explicitly said to do the simpler thing if pacing is already handled by the generator — so I kept the existing wall-clock `played_s = time.time() - t_start` loop and only added the carry-over advance + reuse of `_merge_committed` (already present in the working-tree pre-edit state). This is consistent with the plan's goal list (a) 2-s carry-over + (b) `_merge_committed`.

## Auth Gates

None.

## Self-Check

- `_chunked_decode` defined: yes, line 1800.
- `_chunked_decode` called from `on_upload`: yes.
- `_on_stream_url` uses `carry_samples` advance: yes, both main loop (line 2339) and drain loop (line 2390).
- `on_play_sync` uses `carry_samples` advance: yes (line 2468).
- `on_stream` body byte-equal to HEAD: yes (SHA `1a0815d59443b39f`, len 4927).
- `_transcribe` body byte-equal to HEAD: yes (694-byte body through the `_suppress_repeat_hallucination` return).
- Plan's automated AST verification block: passes ("parity static checks OK").
- `python3 -m py_compile apps/transcribe/app.py`: clean.
- Commit `43fe190` contains only `apps/transcribe/app.py`: verified via `git show --stat 43fe190`.

## Self-Check: PASSED

## Task 2 Status

Deferred to human. Task 2 is a blocking `checkpoint:human-verify` A/B test requiring the user to run the Gradio app locally, play the same kirtan clip via (a) upload tab, (b) URL tab, (c) mic loopback, and compare the three transcripts. The agent cannot perform this evaluation. The app is ready to run via `python -m apps.transcribe.app` from the repo root.
