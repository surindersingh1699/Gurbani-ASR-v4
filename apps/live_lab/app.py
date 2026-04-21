"""Surt Live Lab — live mic harness + file-upload tab, with shabad tracking.

Tabs:
    - Live: mic stream, toggleable preprocessing/VAD/segmenter/decoder, shabad panels
    - File: upload any audio file, run long-form transcription, same shabad panels

All heavy objects (backends, Silero VAD, MuRIL + FAISS Tracker) are cached
at module scope so tab switches and re-runs are instant after first use.

Run:
    python -m apps.live_lab.app
"""

from __future__ import annotations

import html
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import gradio as gr
import numpy as np

from apps.live_lab.asr import (
    ASRSettings,
    FasterWhisperBackend,
    TorchFallbackBackend,
    load_faster_whisper,
)
from apps.live_lab.pipeline import (
    Preprocessor,
    PreprocessSettings,
    SR,
    Segmenter,
    SegmenterSettings,
    SileroVAD,
    VADSettings,
    resample_to_16k,
    to_mono_float32,
)
from apps.live_lab.tracker import (
    ShabadCandidate,
    Tracker,
    TrackerResult,
    TrackerSettings,
)

DEFAULT_CT2_DIR = os.environ.get(
    "SURT_CT2_DIR", str(Path.home() / "models" / "surt-small-v3-int8")
)
DEFAULT_HF_REPO = os.environ.get("SURT_MODEL_ID", "surindersinghssj/surt-small-v3")

# ---------------------------------------------------------------------------
# Cached singletons
# ---------------------------------------------------------------------------

_BACKENDS: dict[tuple, object] = {}
_SILERO: Optional[SileroVAD] = None
_TRACKER: Optional[Tracker] = None


def get_backend(kind, model_path, compute_type, device, cpu_threads):
    key = (kind, model_path, compute_type, device, cpu_threads)
    if key in _BACKENDS:
        return _BACKENDS[key]
    if kind == "faster-whisper":
        be = load_faster_whisper(model_path, compute_type=compute_type, device=device,
                                 cpu_threads=cpu_threads)
    elif kind == "transformers":
        be = TorchFallbackBackend(model_path or DEFAULT_HF_REPO)
    else:
        raise ValueError(f"Unknown backend: {kind}")
    _BACKENDS[key] = be
    return be


def get_silero() -> SileroVAD:
    global _SILERO
    if _SILERO is None:
        _SILERO = SileroVAD()
    return _SILERO


def get_tracker() -> Tracker:
    global _TRACKER
    if _TRACKER is None:
        _TRACKER = Tracker()
    return _TRACKER


# ---------------------------------------------------------------------------
# Per-session state (Live tab)
# ---------------------------------------------------------------------------


@dataclass
class SessionState:
    preprocessor: Preprocessor = field(
        default_factory=lambda: Preprocessor(PreprocessSettings())
    )
    segmenter: Optional[Segmenter] = None
    committed: str = ""
    log: list[str] = field(default_factory=list)
    last_rms_dbfs: float = -120.0
    last_peak_dbfs: float = -120.0
    last_vad_prob: float = 0.0
    total_in_s: float = 0.0
    total_inf_s: float = 0.0
    tracker_snapshot: Optional[TrackerResult] = None


def _append_log(state: SessionState, line: str, cap: int = 80) -> None:
    ts = time.strftime("%H:%M:%S")
    state.log.append(f"[{ts}] {line}")
    if len(state.log) > cap:
        state.log = state.log[-cap:]


def _build_settings_from_ui(
    hpf_hz, do_normalize, target_dbfs,
    vad_kind, silero_thr, energy_thr_db,
    seg_mode, max_seg_s, min_seg_s, min_sil_s, pre_roll_s,
    roll_commit_s, roll_carry_s, roll_max_s,
):
    pre = PreprocessSettings(
        highpass_hz=float(hpf_hz),
        normalize=bool(do_normalize),
        target_rms_dbfs=float(target_dbfs),
    )
    vad = VADSettings(
        kind=vad_kind,
        silero_threshold=float(silero_thr),
        energy_threshold_dbfs=float(energy_thr_db),
    )
    seg = SegmenterSettings(
        mode=seg_mode,
        max_segment_s=float(max_seg_s),
        min_segment_s=float(min_seg_s),
        min_silence_s=float(min_sil_s),
        pre_roll_s=float(pre_roll_s),
        rolling_commit_s=float(roll_commit_s),
        rolling_carry_s=float(roll_carry_s),
        rolling_max_window_s=float(roll_max_s),
    )
    return pre, vad, seg


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def _esc(s: str) -> str:
    return html.escape(s or "", quote=False)


def _render_transcript(text: str) -> str:
    if not text.strip():
        return (
            '<div class="placeholder">ਵਾਹਿਗੁਰੂ ਜੀ ਕਾ ਖਾਲਸਾ '
            '<br/><small>Awaiting audio…</small></div>'
        )
    return f'<div class="gurmukhi">{_esc(text.strip())}</div>'


def _render_meters(state: SessionState) -> str:
    rtf = state.total_inf_s / state.total_in_s if state.total_in_s > 0.01 else 0.0
    return (
        '<div id="surt-meters">'
        f'<span>RMS <b>{state.last_rms_dbfs:5.1f} dBFS</b></span>'
        f'<span>Peak <b>{state.last_peak_dbfs:5.1f} dBFS</b></span>'
        f'<span>VAD <b>{state.last_vad_prob:4.2f}</b></span>'
        f'<span>audio <b>{state.total_in_s:4.1f} s</b></span>'
        f'<span>inference <b>{state.total_inf_s:4.1f} s</b></span>'
        f'<span>RTF <b>{rtf:4.2f}×</b></span>'
        '</div>'
    )


def _render_log(state: SessionState) -> str:
    if not state.log:
        return '<div class="log-empty">(no events yet)</div>'
    lines = "\n".join(state.log[::-1])
    return f'<pre class="log">{_esc(lines)}</pre>'


def _render_current_shabad(snapshot: Optional[TrackerResult]) -> str:
    if snapshot is None or snapshot.current_shabad is None:
        return (
            '<div class="shabad-card empty">'
            '<div class="label">current shabad</div>'
            '<div class="none">— still listening —</div>'
            '</div>'
        )
    info = snapshot.current_shabad
    alap_badge = (
        '<span class="badge alap">alap / silence</span>' if snapshot.alap else
        '<span class="badge live">live</span>'
    )
    line_label = (
        f'line {snapshot.current_line_idx + 1} / {len(info.tuks)} '
        f'· score {snapshot.current_line_score:.2f}'
        if snapshot.current_line_idx >= 0 else 'no line locked'
    )
    # tuks list with current-line highlighted
    lines_html: list[str] = []
    for i, tuk in enumerate(info.tuks):
        cls = "tuk current" if i == snapshot.current_line_idx and not snapshot.alap \
            else ("tuk near" if abs(i - snapshot.current_line_idx) == 1 else "tuk")
        lines_html.append(
            f'<div class="{cls}"><span class="n">{i+1}</span>'
            f'<span class="t">{_esc(tuk)}</span></div>'
        )
    return (
        '<div class="shabad-card">'
        f'<div class="label">current shabad {alap_badge}</div>'
        f'<div class="title">{_esc(info.raag)} · {_esc(info.writer)}</div>'
        f'<div class="sub">ang {info.ang} · shabad {_esc(info.shabad_id)} · {line_label}</div>'
        f'<div class="tuks">{"".join(lines_html)}</div>'
        '</div>'
    )


def _render_candidates(snapshot: Optional[TrackerResult]) -> str:
    if snapshot is None or not snapshot.candidates:
        return (
            '<div class="cands-card"><div class="label">candidates</div>'
            '<div class="none">— waiting for matches —</div></div>'
        )
    best = snapshot.candidates[0].score
    rows: list[str] = []
    for i, c in enumerate(snapshot.candidates):
        pct = int(100 * c.score / best) if best > 0 else 0
        rows.append(
            f'<div class="cand-row {"top" if i == 0 else ""}">'
            f'<div class="cand-head">'
            f'<span class="cand-title">{_esc(c.info.raag)} · {_esc(c.info.writer)}</span>'
            f'<span class="cand-score">{c.score:.2f}</span>'
            f'</div>'
            f'<div class="cand-first">{_esc(c.info.first_tuk)}</div>'
            f'<div class="cand-bar"><div class="fill" style="width:{pct}%"></div></div>'
            '</div>'
        )
    return (
        '<div class="cands-card"><div class="label">top candidates</div>'
        + "".join(rows) + '</div>'
    )


def _render_history(snapshot: Optional[TrackerResult], tracker: Optional[Tracker]) -> str:
    if snapshot is None or not snapshot.history:
        return (
            '<div class="hist-card"><div class="label">history</div>'
            '<div class="none">— no committed lines yet —</div></div>'
        )
    shabads = tracker.shabads if tracker is not None else {}
    rows: list[str] = []
    for item in snapshot.history[::-1][:20]:
        info = shabads.get(item.shabad_id)
        tuk_text = info.tuks[item.line_idx] if info and item.line_idx < len(info.tuks) else "?"
        ang = info.ang if info else "?"
        t = time.strftime("%H:%M:%S", time.localtime(item.ts))
        rows.append(
            f'<div class="hist-row">'
            f'<span class="hist-ts">{t}</span>'
            f'<span class="hist-ang">ang {ang}</span>'
            f'<span class="hist-tuk">{_esc(tuk_text)}</span>'
            '</div>'
        )
    return (
        '<div class="hist-card"><div class="label">history</div>'
        + "".join(rows) + '</div>'
    )


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CSS = """
.gradio-container { background: radial-gradient(1200px 600px at 10% 0%, #1e293b 0%, #0b0f14 55%) !important; }

.gurmukhi { font-family:"Gurbani Akhar","Noto Sans Gurmukhi","Nirmala UI",serif;
    font-size:26px; line-height:1.55; color:#f8fafc; }
.placeholder { color:#94a3b8; font-style:italic; font-size:20px; }

#surt-transcript-card, .shabad-card, .cands-card, .hist-card, #surt-file-card {
    background:#111827; border:1px solid rgba(255,255,255,0.06);
    border-radius:14px; padding:14px 18px; }
#surt-transcript-card { min-height:140px; }

.label { color:#94a3b8; font-size:11px; text-transform:uppercase;
    letter-spacing:1px; margin-bottom:8px; display:flex; align-items:center; gap:8px; }
.none { color:#475569; font-style:italic; padding:10px 0; }

.badge { font-size:10px; padding:2px 8px; border-radius:999px; letter-spacing:0.5px; }
.badge.live { background:#064e3b; color:#6ee7b7; }
.badge.alap { background:#431407; color:#fdba74; }

.shabad-card .title { color:#f8fafc; font-size:16px; font-weight:600; }
.shabad-card .sub { color:#94a3b8; font-size:12px; margin-bottom:10px; }
.shabad-card .tuks { max-height:340px; overflow:auto; border-top:1px solid rgba(255,255,255,0.05); padding-top:10px; }
.shabad-card .tuk { display:flex; gap:10px; padding:4px 6px; border-radius:6px;
    font-family:"Gurbani Akhar","Noto Sans Gurmukhi",serif; font-size:17px;
    color:#cbd5e1; transition:background 0.2s; }
.shabad-card .tuk .n { color:#475569; font-family:ui-monospace,monospace;
    font-size:11px; min-width:22px; padding-top:4px; }
.shabad-card .tuk.near { background:rgba(148,163,184,0.04); color:#e2e8f0; }
.shabad-card .tuk.current {
    background:linear-gradient(90deg,rgba(245,158,11,0.20),rgba(245,158,11,0.03));
    color:#fde68a; box-shadow:inset 2px 0 0 #f59e0b; }

.cand-row { padding:8px 0; border-top:1px solid rgba(255,255,255,0.04); }
.cand-row:first-of-type { border-top:none; }
.cand-row.top .cand-title { color:#fde68a; }
.cand-head { display:flex; justify-content:space-between; font-size:12px; color:#e2e8f0; }
.cand-title { color:#cbd5e1; }
.cand-score { color:#f8fafc; font-family:ui-monospace,monospace; }
.cand-first { color:#cbd5e1; font-family:"Gurbani Akhar","Noto Sans Gurmukhi",serif;
    font-size:16px; margin:3px 0 5px; }
.cand-bar { background:rgba(255,255,255,0.06); height:4px; border-radius:2px; overflow:hidden; }
.cand-bar .fill { background:linear-gradient(90deg,#f59e0b,#fbbf24); height:100%; }

.hist-card { max-height:420px; overflow:auto; }
.hist-row { display:flex; gap:10px; padding:4px 0; border-bottom:1px dashed rgba(255,255,255,0.05);
    font-size:13px; color:#cbd5e1; }
.hist-ts { color:#64748b; font-family:ui-monospace,monospace; font-size:11px; min-width:62px; padding-top:2px; }
.hist-ang { color:#94a3b8; font-size:11px; min-width:50px; padding-top:2px; }
.hist-tuk { font-family:"Gurbani Akhar","Noto Sans Gurmukhi",serif; font-size:16px; color:#e2e8f0; }

#surt-meters { color:#94a3b8; font-size:12px; display:flex; gap:16px; flex-wrap:wrap; margin-top:8px; }
#surt-meters b { color:#f8fafc; font-family:ui-monospace,monospace; }

.log { background:#0f172a; color:#cbd5e1; font-size:11px; padding:10px;
    border-radius:8px; max-height:220px; overflow:auto; white-space:pre-wrap; }
.log-empty { color:#475569; font-style:italic; }
footer { display:none !important; }
"""


# ---------------------------------------------------------------------------
# Shared: apply snapshot to outputs
# ---------------------------------------------------------------------------


def _snapshot_outputs(
    transcript_text: str,
    state_like: SessionState,
    snapshot: Optional[TrackerResult],
    tracker: Optional[Tracker],
):
    return (
        f'<div id="surt-transcript-card">{_render_transcript(transcript_text)}</div>',
        _render_meters(state_like),
        _render_current_shabad(snapshot),
        _render_candidates(snapshot),
        _render_history(snapshot, tracker),
        _render_log(state_like),
    )


# ---------------------------------------------------------------------------
# Live stream callback
# ---------------------------------------------------------------------------


def on_stream(
    chunk, st: SessionState,
    be_kind, mpath, ct, dev, threads,
    hpf, do_norm, tgt_db,
    vkind, s_thr, e_thr,
    smode, maxs, mins, minsil, preroll,
    rcommit, rcarry, rmaxw,
    beam, temp, cond_prev, fw_vad, lang, prompt,
    ema_alpha,
):
    def outs():
        return (st, *_snapshot_outputs(st.committed, st, st.tracker_snapshot, _TRACKER))

    if chunk is None:
        return outs()
    sr_in, arr = chunk
    if arr is None or len(arr) == 0:
        return outs()

    pre_cfg, vad_cfg, seg_cfg = _build_settings_from_ui(
        hpf, do_norm, tgt_db,
        vkind, s_thr, e_thr,
        smode, maxs, mins, minsil, preroll,
        rcommit, rcarry, rmaxw,
    )

    st.preprocessor.configure(pre_cfg)
    vad_obj = get_silero() if vad_cfg.kind == "silero" else None
    if (
        st.segmenter is None
        or st.segmenter.seg_cfg != seg_cfg
        or st.segmenter.vad_cfg != vad_cfg
        or (st.segmenter.vad is None) != (vad_obj is None)
    ):
        st.segmenter = Segmenter(seg_cfg, vad_cfg, vad_obj)

    y = resample_to_16k(to_mono_float32(arr), sr_in)
    y, met = st.preprocessor.process(y)
    st.last_rms_dbfs = met["rms_dbfs"]
    st.last_peak_dbfs = met["peak_dbfs"]
    st.total_in_s += len(y) / SR

    events = st.segmenter.push(y)
    st.last_vad_prob = st.segmenter.last_vad_prob

    if events:
        try:
            be = get_backend(be_kind, mpath, ct, dev, int(threads))
        except Exception as e:  # noqa: BLE001
            _append_log(st, f"backend not ready: {e}")
            return outs()

        asr_cfg = ASRSettings(
            beam_size=int(beam),
            temperature=float(temp),
            condition_on_previous_text=bool(cond_prev),
            initial_prompt=str(prompt or ""),
            language=str(lang or "pa"),
            vad_filter=bool(fw_vad),
        )

        tracker = get_tracker()
        # propagate ema alpha live
        tracker.cfg.ema_alpha = float(ema_alpha)

        for ev in events:
            t0 = time.time()
            try:
                text, meta = be.transcribe(ev.audio, SR, asr_cfg)
            except Exception as e:  # noqa: BLE001
                _append_log(st, f"transcribe failed: {e}")
                continue
            dt = time.time() - t0
            st.total_inf_s += dt
            if text:
                st.committed = (st.committed + " " + text).strip()
                st.tracker_snapshot = tracker.update(text, record_history=True)

            vad_str = f" vadmax={ev.vad_max:.2f}" if ev.vad_max is not None else ""
            nsp = meta.get("no_speech_prob")
            nsp_str = f" no_sp={nsp:.2f}" if nsp is not None else ""
            _append_log(
                st,
                f"seg {ev.duration_s:4.1f}s [{ev.reason}]{vad_str}{nsp_str} "
                f"→ {dt*1000:.0f} ms · {text[:60]}"
                + ("…" if len(text) > 60 else ""),
            )

    return outs()


# ---------------------------------------------------------------------------
# File-upload transcription
# ---------------------------------------------------------------------------


def _chunk_audio_for_tracker(segments: Iterable, tracker: Tracker, state_like: SessionState):
    """Feed each faster-whisper segment to the tracker and build committed text."""
    committed = ""
    for seg in segments:
        text = (seg.text or "").strip()
        if not text:
            continue
        committed = (committed + " " + text).strip()
        state_like.committed = committed
        snapshot = tracker.update(text, record_history=True)
        yield committed, snapshot


def on_file_transcribe(
    audio_path, be_kind, mpath, ct, dev, threads,
    beam, temp, cond_prev, fw_vad, lang, prompt, ema_alpha,
):
    state = SessionState()
    tracker = get_tracker()
    tracker.reset()
    tracker.cfg.ema_alpha = float(ema_alpha)

    if not audio_path:
        _append_log(state, "no audio file provided")
        yield _snapshot_outputs("", state, None, tracker)
        return

    try:
        be = get_backend(be_kind, mpath, ct, dev, int(threads))
    except Exception as e:  # noqa: BLE001
        _append_log(state, f"backend not ready: {e}")
        yield _snapshot_outputs("", state, None, tracker)
        return

    if not isinstance(be, FasterWhisperBackend):
        _append_log(state, "file tab requires faster-whisper backend for streaming segments")
        yield _snapshot_outputs("", state, None, tracker)
        return

    _append_log(state, f"transcribing {Path(audio_path).name} …")
    yield _snapshot_outputs("", state, None, tracker)

    t0 = time.time()
    try:
        segments, info = be.model.transcribe(
            audio_path,
            beam_size=int(beam),
            temperature=float(temp),
            language=str(lang or "pa") or None,
            task="transcribe",
            condition_on_previous_text=bool(cond_prev),
            initial_prompt=str(prompt or "") or None,
            vad_filter=bool(fw_vad),
            word_timestamps=False,
        )
    except Exception as e:  # noqa: BLE001
        _append_log(state, f"transcribe failed: {e}")
        yield _snapshot_outputs("", state, None, tracker)
        return

    committed = ""
    last_yield = 0.0
    for seg in segments:
        text = (seg.text or "").strip()
        if not text:
            continue
        committed = (committed + " " + text).strip()
        state.committed = committed
        state.tracker_snapshot = tracker.update(text, record_history=True)
        _append_log(
            state,
            f"[{seg.start:6.1f}–{seg.end:6.1f}s] no_sp={getattr(seg,'no_speech_prob',0):.2f} · {text[:60]}"
            + ("…" if len(text) > 60 else ""),
        )
        # Throttle UI updates to every ~0.4 s wall-clock to avoid flooding
        now = time.time()
        if now - last_yield > 0.4:
            last_yield = now
            yield _snapshot_outputs(committed, state, state.tracker_snapshot, tracker)

    state.total_in_s = float(getattr(info, "duration", 0.0)) or state.total_in_s
    state.total_inf_s = time.time() - t0
    _append_log(state, f"done in {state.total_inf_s:.1f} s")
    yield _snapshot_outputs(committed, state, state.tracker_snapshot, tracker)


# ---------------------------------------------------------------------------
# Controls (shared Accordion groups between tabs)
# ---------------------------------------------------------------------------


def _controls_backend():
    with gr.Accordion("Model / Backend", open=True):
        with gr.Row():
            backend_kind = gr.Radio(
                choices=["faster-whisper", "transformers"],
                value="faster-whisper",
                label="Backend",
            )
            compute_type = gr.Dropdown(
                choices=["int8", "int8_float16", "float16", "float32"],
                value="int8",
                label="Compute type (faster-whisper)",
            )
            device = gr.Dropdown(
                choices=["cpu", "cuda", "auto"],
                value="cpu",
                label="Device",
            )
            cpu_threads = gr.Slider(0, 16, value=0, step=1,
                                    label="CPU threads (0 = auto)")
        model_path = gr.Textbox(
            value=DEFAULT_CT2_DIR,
            label="Model path (CT2 dir for faster-whisper, HF repo for transformers)",
        )
        reload_btn = gr.Button("Load / reload model", variant="primary")
        status = gr.HTML('<span style="color:#94a3b8">No backend loaded yet.</span>')
    return backend_kind, compute_type, device, cpu_threads, model_path, reload_btn, status


def _controls_decoder():
    with gr.Accordion("ASR decoding", open=False):
        with gr.Row():
            beam_size = gr.Slider(1, 5, value=1, step=1, label="Beam size")
            temperature = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="Temperature")
            condition_prev = gr.Checkbox(value=False, label="Condition on previous text")
            fw_vad_filter = gr.Checkbox(value=False, label="faster-whisper built-in VAD filter")
        language = gr.Textbox(value="pa", label="Language code")
        initial_prompt = gr.Textbox(
            value="",
            label="Initial prompt (optional — prime with a known Gurbani line)",
            lines=2,
        )
        ema_alpha = gr.Slider(0.1, 0.9, value=0.35, step=0.05,
                              label="Tracker EMA α (higher = more responsive)")
    return beam_size, temperature, condition_prev, fw_vad_filter, language, initial_prompt, ema_alpha


# ---------------------------------------------------------------------------
# App builder
# ---------------------------------------------------------------------------


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Surt Live Lab") as demo:
        demo._surt_css = CSS
        gr.Markdown("## Surt Live Lab · real-time Gurbani transcription + shabad tracker")

        with gr.Tabs():

            # ------------------- LIVE TAB -------------------
            with gr.Tab("Live mic"):
                with gr.Row():
                    with gr.Column(scale=2):
                        transcript_live = gr.HTML(
                            '<div id="surt-transcript-card">'
                            '<div class="placeholder">waiting for audio…</div></div>'
                        )
                        meters_live = gr.HTML(_render_meters(SessionState()))
                        shabad_live = gr.HTML(_render_current_shabad(None))
                    with gr.Column(scale=1):
                        mic = gr.Audio(
                            sources=["microphone"],
                            streaming=True,
                            type="numpy",
                            label="Microphone",
                        )
                        clear_btn = gr.Button("Clear", variant="secondary")
                        cands_live = gr.HTML(_render_candidates(None))
                        hist_live = gr.HTML(_render_history(None, None))
                with gr.Row():
                    gr.Markdown("#### Event log")
                log_live = gr.HTML(_render_log(SessionState()))

                (backend_kind, compute_type, device, cpu_threads, model_path,
                 reload_btn, backend_status) = _controls_backend()

                with gr.Accordion("Preprocessing", open=False):
                    with gr.Row():
                        hpf_hz = gr.Slider(0, 200, value=80, step=5,
                                           label="High-pass cutoff (Hz, 0 = off)")
                        do_normalize = gr.Checkbox(value=True, label="RMS normalize")
                        target_dbfs = gr.Slider(-30, -10, value=-20, step=1,
                                                label="Target RMS (dBFS)")
                with gr.Accordion("VAD", open=False):
                    with gr.Row():
                        vad_kind = gr.Radio(
                            choices=["off", "silero", "energy"],
                            value="silero", label="VAD mode",
                        )
                        silero_thr = gr.Slider(0.0, 0.95, value=0.5, step=0.05,
                                               label="Silero threshold")
                        energy_thr_db = gr.Slider(-70, -10, value=-40, step=1,
                                                  label="Energy threshold (dBFS)")
                with gr.Accordion("Segmenter", open=False):
                    with gr.Row():
                        seg_mode = gr.Radio(
                            choices=["vad", "fixed", "rolling"],
                            value="rolling", label="Mode",
                        )
                        max_seg_s = gr.Slider(3, 30, value=15, step=1,
                                              label="Max segment (s)")
                        min_seg_s = gr.Slider(0.5, 5, value=1.0, step=0.1,
                                              label="Min segment (s)")
                        min_sil_s = gr.Slider(0.1, 2.0, value=0.6, step=0.05,
                                              label="Min silence to close (s)")
                        pre_roll_s = gr.Slider(0.0, 1.0, value=0.2, step=0.05,
                                               label="Pre-roll (s)")
                    with gr.Row():
                        roll_commit_s = gr.Slider(2, 15, value=8, step=0.5,
                                                  label="Rolling: commit (s)")
                        roll_carry_s = gr.Slider(0.5, 5, value=1.5, step=0.1,
                                                 label="Rolling: carry (s)")
                        roll_max_s = gr.Slider(5, 20, value=11, step=0.5,
                                               label="Rolling: max window (s)")

                (beam_size, temperature, condition_prev, fw_vad_filter,
                 language, initial_prompt, ema_alpha) = _controls_decoder()

                live_state = gr.State(value=SessionState())

                def on_reload(kind, path, ct, dev, threads):
                    try:
                        be = get_backend(kind, path, ct, dev, int(threads))
                        return f'<span style="color:#22c55e">✓ {be.describe()}</span>'
                    except Exception as e:  # noqa: BLE001
                        return f'<span style="color:#ef4444">✗ {e}</span>'

                reload_btn.click(
                    on_reload,
                    inputs=[backend_kind, model_path, compute_type, device, cpu_threads],
                    outputs=[backend_status],
                )

                def on_clear(_st):
                    s = SessionState()
                    if _TRACKER is not None:
                        _TRACKER.reset()
                    return (
                        s,
                        *_snapshot_outputs("", s, None, _TRACKER),
                    )

                clear_btn.click(
                    on_clear,
                    inputs=[live_state],
                    outputs=[live_state, transcript_live, meters_live,
                             shabad_live, cands_live, hist_live, log_live],
                )

                mic.stream(
                    on_stream,
                    inputs=[
                        mic, live_state,
                        backend_kind, model_path, compute_type, device, cpu_threads,
                        hpf_hz, do_normalize, target_dbfs,
                        vad_kind, silero_thr, energy_thr_db,
                        seg_mode, max_seg_s, min_seg_s, min_sil_s, pre_roll_s,
                        roll_commit_s, roll_carry_s, roll_max_s,
                        beam_size, temperature, condition_prev, fw_vad_filter,
                        language, initial_prompt, ema_alpha,
                    ],
                    outputs=[live_state, transcript_live, meters_live,
                             shabad_live, cands_live, hist_live, log_live],
                    stream_every=0.5,
                    concurrency_limit=1,
                    show_progress="hidden",
                )

            # ------------------- FILE TAB -------------------
            with gr.Tab("File upload"):
                with gr.Row():
                    with gr.Column(scale=1):
                        file_audio = gr.Audio(
                            sources=["upload"],
                            type="filepath",
                            label="Audio file (wav / mp3 / m4a)",
                        )
                        file_run = gr.Button("Transcribe & track", variant="primary")
                        meters_file = gr.HTML(_render_meters(SessionState()))
                    with gr.Column(scale=2):
                        transcript_file = gr.HTML(
                            '<div id="surt-file-card">'
                            '<div class="placeholder">Upload an audio file and click <b>Transcribe</b>.</div>'
                            '</div>'
                        )
                with gr.Row():
                    with gr.Column(scale=2):
                        shabad_file = gr.HTML(_render_current_shabad(None))
                    with gr.Column(scale=1):
                        cands_file = gr.HTML(_render_candidates(None))
                        hist_file = gr.HTML(_render_history(None, None))
                gr.Markdown("#### Event log")
                log_file = gr.HTML(_render_log(SessionState()))

                # Re-use the same backend/decoder controls (separate widgets so they
                # don't conflict with the live tab's state)
                (backend_kind_f, compute_type_f, device_f, cpu_threads_f, model_path_f,
                 reload_btn_f, status_f) = _controls_backend()
                (beam_f, temp_f, cond_f, fwvad_f, lang_f, prompt_f, ema_f) = _controls_decoder()

                reload_btn_f.click(
                    on_reload,
                    inputs=[backend_kind_f, model_path_f, compute_type_f, device_f, cpu_threads_f],
                    outputs=[status_f],
                )

                file_run.click(
                    on_file_transcribe,
                    inputs=[
                        file_audio,
                        backend_kind_f, model_path_f, compute_type_f, device_f, cpu_threads_f,
                        beam_f, temp_f, cond_f, fwvad_f, lang_f, prompt_f, ema_f,
                    ],
                    outputs=[transcript_file, meters_file, shabad_file,
                             cands_file, hist_file, log_file],
                )

    return demo


def main() -> None:
    demo = build_app()
    theme = gr.themes.Base(
        primary_hue="amber",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"],
    )
    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        inbrowser=True,
        theme=theme,
        css=getattr(demo, "_surt_css", None),
    )


if __name__ == "__main__":
    main()
