"""Surt v3 — live Gurbani transcription (desktop + tablet web layout).

UX refresh:
- Hero stage promotes matched shabad when top-match confidence >= 70%.
- Primary action bar under the stage: Push to STTM / Clear / Copy / Export.
- Keyboard shortcuts: Space = record toggle, Enter = push top, 1-5 = push Nth,
  E = edit transcript, Esc = clear-with-undo, ? = show help.
- Committed transcript is contenteditable; blur re-runs the BaniDB search.
- Manual shabad search box lives inside the Matches card as a fallback.
- Auto-push has a confidence floor (slider) so bad matches don't project.
- Clear is undo-able for 8 seconds via a toast.
- Dark / high-contrast mode for gurdwara projection rooms.
- Settings + Session collapsed by default to reduce visual clutter.
- Tablet layout (single column) kicks in below ~960px.

Layout:
  Top bar: brand | STTM pill (clickable = retry) | dark-mode | help
  Main grid:
      LEFT  (7/12) — hero stage + primary action bar
      RIGHT (5/12) — Input tabs · Matches (with search) · Settings · Session
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import gradio as gr
import numpy as np

from apps.transcribe.backend import load_backend  # noqa: E402
from apps.transcribe.ema import RetrievalEMA
from apps.transcribe.player import prepare_source
from apps.transcribe.retriever import (
    DEFAULT_MODE,
    LOCK_SCORE_FLOOR,
    MODE_LABELS,
    UNLOCK_SCORE_FLOOR,
    get_retriever,
    score_within_shabad,
    search_shabad_topn,
)
from apps.transcribe.sttm_controller import (
    CANDIDATE_PORTS,
    STTMStatus,
    discover,
    push_hit,
    push_transcript_as_shabad,
)

TARGET_SR = 16_000
HERO_THRESHOLD_DEFAULT = 0.70
AUTO_PUSH_THRESHOLD_DEFAULT = 0.60
UNDO_TTL_S = 8.0

# Lock defaults — users can change streak from the UI (1 = lock on first
# confident window, 2 = require agreement across two windows).
LOCK_STREAK_DEFAULT = 1
UNLOCK_MISSES_DEFAULT = 3

# Persisted user preferences (STTM PIN so users don't retype each launch).
# Stored per-user rather than in the repo; env var lets tests point elsewhere.
CONFIG_PATH = Path(
    os.environ.get("SURT_CONFIG_PATH", str(Path.home() / ".surt" / "config.json"))
)


def _load_config() -> dict:
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as e:
        print(f"[surt] could not read {CONFIG_PATH}: {e}")
        return {}


def _save_config(cfg: dict) -> None:
    try:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with CONFIG_PATH.open("w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, sort_keys=True)
    except OSError as e:
        print(f"[surt] could not write {CONFIG_PATH}: {e}")

# Sliding-window retrieval: live-mic buffers grow unbounded as the ragi sings,
# and sending the whole committed transcript to the retriever both dilutes
# MuRIL's embedding and drags the literal rerank down. We query against a tail
# slice that's roughly one tuk-pair long — enough context to disambiguate
# shabads, short enough to keep the embedding tight.
RETRIEVAL_TAIL_CHARS = 140


@dataclass
class StreamState:
    buffer: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    committed: str = ""
    tentative: str = ""
    last_call_t: float = 0.0
    last_stats_html: str = (
        '<div class="surt-stat-row"><span>status</span><b>waiting…</b></div>'
    )

    # STTM
    sttm_host: str = "127.0.0.1"
    sttm_port: int | None = None
    sttm_pin: str = ""
    sttm_connected: bool = False

    # Smoother for live paths (mic + play-sync); stays None for one-shot paths
    retrieval_ema: RetrievalEMA | None = None
    # Which retriever to use (toggled from the UI dropdown)
    retrieval_mode: str = DEFAULT_MODE

    # Debug: last mic-stream callback timestamp, so we can tell if Gradio is
    # actually firing stream events.
    last_stream_t: float = 0.0
    stream_calls: int = 0

    # Settings
    vad_threshold: float = 0.005
    throttle_s: float = 1.2
    commit_s: float = 10.0
    max_window_s: float = 12.0
    carry_over_s: float = 2.0
    # Whisper hallucinates on very short buffers ("ਵਾਹਿਗੁਰੂ ਵਾਹਿਗੁਰੂ…" loops).
    # Require at least ~3 s of audio before we call the model on live mic.
    min_transcribe_s: float = 3.0
    auto_push_threshold: float = AUTO_PUSH_THRESHOLD_DEFAULT
    hero_threshold: float = HERO_THRESHOLD_DEFAULT

    # Matches
    matches: list[dict] = field(default_factory=list)
    last_search_key: str = ""

    # Shabad lock / pointer state. Once a shabad is locked we score the tail
    # against just that shabad's tuks — instant pointer updates for line
    # advances, with content-based auto-unlock when the ragi moves away.
    locked_shabad_id: str | None = None
    locked_tuk_row: int | None = None  # current pointer row within the shabad
    lock_streak_target: int = LOCK_STREAK_DEFAULT
    lock_streak_count: int = 0
    last_top_sid_for_lock: str | None = None
    unlock_miss_count: int = 0
    unlock_misses_target: int = UNLOCK_MISSES_DEFAULT

    # Kirtan player
    player_audio: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    player_path: str = ""

    # Undo
    undo_committed: str | None = None
    undo_matches: list[dict] | None = None
    undo_expires_at: float = 0.0


def _to_mono_float32(y: np.ndarray) -> np.ndarray:
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1.5:
        y = y / 32768.0
    return y


def _resample(y: np.ndarray, sr_in: int, sr_out: int = TARGET_SR) -> np.ndarray:
    if sr_in == sr_out or y.size == 0:
        return y
    n = int(round(len(y) * sr_out / sr_in))
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, num=len(y), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n, endpoint=False)
    return np.interp(x_new, x_old, y).astype(np.float32)


def _rms(y: np.ndarray) -> float:
    if y.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(y.astype(np.float32) ** 2)))


def _suppress_repeat_hallucination(text: str) -> str:
    """Blank out Whisper's pathological repeat outputs.

    Live-mic buffers that are mostly silence or low-quality audio sometimes
    produce "ਵਾਹਿਗੁਰੂ ਵਾਹਿਗੁਰੂ ਵਾਹਿਗੁਰੂ …" or similar single-token loops even
    after temperature fallback + no_repeat_ngram_size. If every token in the
    output is the same word, it's almost certainly a hallucination rather
    than a ragi actually chanting one word — drop it before it pollutes the
    running committed transcript.
    """
    s = (text or "").strip()
    if not s:
        return ""
    toks = s.split()
    if len(toks) >= 4 and len(set(toks)) == 1:
        return ""
    return s


def _shabad_id_from_hit(hit: dict) -> str | None:
    """Map a UI match dict back to the retriever's internal shabad_id.

    Match dicts carry `sttm_id` only; auto-lock needs the Q-id key used by
    `_tuks_by_shabad`. We look up by the tuk_row when present, otherwise
    reverse-search the sttm_id map.
    """
    tuk_row = hit.get("_tuk_row")
    r = get_retriever()
    if isinstance(tuk_row, int) and tuk_row >= 0:
        meta = r._tuk_meta.get(tuk_row)
        if meta:
            return meta.get("shabad_id")
    sttm = hit.get("shabadId")
    if sttm is None:
        return None
    for sid, sid_sttm in r._sttm_id_of.items():
        if sid_sttm == sttm:
            return sid
    return None


def _retrieval_query(text: str, max_chars: int = RETRIEVAL_TAIL_CHARS) -> str:
    """Tail slice of the committed transcript for live-mic retrieval.

    Avoids diluting MuRIL and the literal rerank with many repetitions of the
    same tuk. Snaps to a word boundary so we don't cut mid-akhar.
    """
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    tail = text[-max_chars:]
    sp = tail.find(" ")
    if 0 < sp < len(tail) - 4:
        tail = tail[sp + 1 :]
    return tail


# --- renderers ---------------------------------------------------------------

def _render_stage(st: StreamState) -> str:
    """Single stage renderer. Shows hero match if confident, else transcript."""
    committed = (st.committed or "").strip()
    tentative = (st.tentative or "").strip()
    top = st.matches[0] if st.matches else None
    hero_mode = bool(top and float(top.get("score", 0.0)) >= st.hero_threshold)

    # Capturing audio but no text yet — show an explicit "listening" state so
    # the user knows the mic is live. Without this, the stage looks identical
    # to idle until the first commit (~12 s) which feels broken.
    if not committed and not tentative and not hero_mode and st.buffer.size > 0:
        dur = st.buffer.size / TARGET_SR
        rms = _rms(st.buffer)
        vad_ok = rms >= st.vad_threshold
        status = "speaking" if vad_ok else "quiet (below VAD)"
        return (
            '<div class="surt-ph listening">'
            '<div class="ik-onkar pulse">🎤</div>'
            f'<div class="ik-sub">Listening… <span class="pulse-dot">●</span></div>'
            f'<div class="ik-hint">{dur:4.1f} s captured · RMS {rms:.4f} · {status} '
            '· first line appears after the commit window</div>'
            '</div>'
        )

    # Idle: neither audio nor text
    if not committed and not tentative and not hero_mode:
        return (
            '<div class="surt-ph">'
            '<div class="ik-onkar">ੴ</div>'
            '<div class="ik-sub">ਸਤਿ ਨਾਮੁ</div>'
            '<div class="ik-hint">Press <kbd>Space</kbd> to start listening · '
            'upload a file · or play from URL</div>'
            '</div>'
        )

    # Hero: matched shabad is the main content, transcript is a footer strip
    if hero_mode and top is not None:
        meta_bits: list[str] = []
        if top.get("writer"):  meta_bits.append(str(top["writer"]))
        if top.get("raag"):    meta_bits.append(f'Raag {top["raag"]}')
        if top.get("source"):  meta_bits.append(str(top["source"]))
        if top.get("ang"):     meta_bits.append(f'Ang {top["ang"]}')
        meta = " · ".join(meta_bits) or "—"
        gur = (top.get("gurmukhi") or "").strip() or "(no text)"
        score_pct = int(round(float(top["score"]) * 100))
        transcript_strip = _transcript_html(committed, tentative, small=True)
        lock_badge = ""
        if st.locked_shabad_id:
            lock_badge = (
                '  <button class="lock-pill" '
                'onclick="document.getElementById(\'surt-unlock-btn\').click()" '
                f'title="Locked on shabad · click to unlock">🔒 locked</button>'
            )
        return (
            '<div class="stage-hero">'
            f'  <div class="hero-label">Matched shabad · {score_pct}% confidence{lock_badge}</div>'
            f'  <div class="hero-gur" contenteditable="false">{gur}</div>'
            f'  <div class="hero-meta">{meta}</div>'
            '</div>'
            f'<div class="stage-transcript-strip">{transcript_strip}</div>'
        )

    # Transcript-only stage (listening, no confident match yet)
    return f'<div class="stage-transcript">{_transcript_html(committed, tentative)}</div>'


def _transcript_html(committed: str, tentative: str, *, small: bool = False) -> str:
    committed = (committed or "").strip()
    tentative = (tentative or "").strip()
    if not committed and not tentative:
        return '<div class="surt-line empty">listening…</div>'
    parts: list[str] = []
    if committed:
        # contenteditable span so user can fix ASR mistakes before pushing
        parts.append(
            f'<span class="committed" contenteditable="true" '
            f'spellcheck="false" data-role="committed">{committed}</span>'
        )
    if tentative and tentative != committed:
        if committed:
            parts.append(" ")
        parts.append(f'<span class="tentative">{tentative}</span>')
    size_cls = " small" if small else ""
    return f'<div class="surt-line{size_cls}">{"".join(parts)}</div>'


def _sttm_pill(st: StreamState) -> str:
    if st.sttm_connected and st.sttm_port:
        return (
            f'<button class="sttm-pill ok" '
            f'onclick="document.getElementById(\'surt-connect-real\').click()" '
            f'title="Re-check STTM connection">'
            f'<span class="dot"></span>STTM · <b>:{st.sttm_port}</b></button>'
        )
    return (
        '<button class="sttm-pill off" '
        'onclick="document.getElementById(\'surt-connect-real\').click()" '
        'title="Retry STTM discovery">'
        '<span class="dot"></span>STTM offline · retry</button>'
    )


def _render_matches(matches: list[dict]) -> str:
    if not matches:
        return (
            '<div class="match-empty">No candidates yet — matches appear '
            'after the first committed line, or use the search box above.</div>'
        )
    rows: list[str] = []
    for i, m in enumerate(matches):
        score_pct = int(round(float(m.get("score", 0.0)) * 100))
        meta_bits: list[str] = []
        if m.get("writer"):  meta_bits.append(str(m["writer"]))
        if m.get("raag"):    meta_bits.append(f'Raag {m["raag"]}')
        if m.get("source"):  meta_bits.append(str(m["source"]))
        if m.get("ang"):     meta_bits.append(f'Ang {m["ang"]}')
        meta = " · ".join(meta_bits) or "—"
        gurmukhi = (m.get("gurmukhi") or "").strip() or "(no text)"
        sid = m.get("shabadId")
        rank_class = "top" if i == 0 else ""
        rows.append(
            f'<div class="match-row {rank_class}" '
            f'data-idx="{i}" onclick="window._surt_pick({i})" '
            f'title="Click to project this shabad (or press {i+1})">'
            f'  <div class="match-rank">#{i+1}</div>'
            f'  <div class="match-body">'
            f'    <div class="match-gur">{gurmukhi}</div>'
            f'    <div class="match-meta">{meta} · shabadId {sid}</div>'
            f'  </div>'
            f'  <div class="match-score">'
            f'    <div class="score-bar"><span style="width:{score_pct}%"></span></div>'
            f'    <div class="score-pct">{score_pct}%</div>'
            f'  </div>'
            f'  <div class="match-push" title="Project this shabad">→</div>'
            f'</div>'
        )
    return '<div id="surt-matches">' + "".join(rows) + '</div>'


def _render_action_bar(st: StreamState) -> str:
    """Full action bar: big Push CTA + Clear + Copy + Export."""
    top = st.matches[0] if st.matches else None
    disabled = not top
    if top:
        score_pct = int(round(float(top["score"]) * 100))
        label = f'↗ Push to STTM · {score_pct}%'
    else:
        label = '↗ Push to STTM'
    cls = "primary-cta" + (" disabled" if disabled else "") + (" ready" if top else "")
    disabled_attr = "disabled" if disabled else ""
    # Clear has its own secondary CTA (not just an icon) because users asked
    # for it to be obvious — "Start over" between kirtan sessions.
    return (
        '<div class="surt-action-bar">'
        f'  <button class="{cls}" {disabled_attr} '
        f'    onclick="document.getElementById(\'surt-push-real\').click()" '
        f'    title="Press Enter to push the top match">{label}</button>'
        f'  <button class="secondary-cta clear-cta" '
        f'    onclick="document.getElementById(\'surt-clear-real\').click()" '
        f'    title="Clear transcript + matches (Esc) — undo-able for 8 s">'
        f'    {ICON_TRASH} <span>Clear</span></button>'
        f'  <button class="action-icon-btn" title="Copy transcript (C)" '
        f'    onclick="window._surt_copy()">{ICON_COPY}</button>'
        f'  <button class="action-icon-btn" title="Export transcript (X)" '
        f'    onclick="window._surt_export()">{ICON_DOWNLOAD}</button>'
        '</div>'
    )


def _render_toast(msg: str, kind: str = "info", *, undo: bool = False) -> str:
    if not msg:
        return ""
    undo_btn = (
        '<button class="toast-undo" '
        'onclick="document.getElementById(\'surt-undo-real\').click()">Undo</button>'
        if undo else ""
    )
    return f'<div class="surt-toast {kind}">{msg}{undo_btn}</div>'


# --- inline SVGs -------------------------------------------------------------
_SVG_ATTRS = ('width="20" height="20" viewBox="0 0 24 24" fill="none" '
              'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
              'stroke-linejoin="round"')
ICON_TRASH = f'<svg {_SVG_ATTRS}><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/><path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/></svg>'
ICON_COPY = f'<svg {_SVG_ATTRS}><rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>'
ICON_DOWNLOAD = f'<svg {_SVG_ATTRS}><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>'
ICON_MOON = f'<svg {_SVG_ATTRS}><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>'
ICON_HELP = f'<svg {_SVG_ATTRS}><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>'


def build_app(backend) -> gr.Blocks:
    user_config = _load_config()
    saved_pin = str(user_config.get("sttm_pin") or "")
    css = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --bg:#faf8f1; --card:#ffffff;
        --ink:#1a2740; --ink-dim:#56627a; --muted:#98a0b2;
        --navy:#2c6aa0; --navy-deep:#1f4e7a;
        --amber:#e6a537; --amber-soft:#f6d38a;
        --rule:#eae3d2; --rule-soft:#f1ecde;
        --stage-bg:#ffffff;
        --shadow-sm:0 1px 2px rgba(31,78,122,0.04), 0 4px 12px -4px rgba(31,78,122,0.08);
        --shadow:0 10px 30px -12px rgba(31,78,122,0.18);
        --ok:#2e8b57; --warn:#b5651d; --danger:#b5391d;
    }
    /* Dark / high-contrast mode (projector rooms) */
    body.surt-dark {
        --bg:#0b1020; --card:#141a2e;
        --ink:#f2ecdc; --ink-dim:#b9c0d4; --muted:#7b87a3;
        --navy:#7aaedd; --navy-deep:#a7c8e8;
        --amber:#f3c56a; --amber-soft:#8a6a22;
        --rule:#29304a; --rule-soft:#1c2238;
        --stage-bg:#141a2e;
        --shadow-sm:0 1px 2px rgba(0,0,0,0.3), 0 4px 12px -4px rgba(0,0,0,0.4);
        --shadow:0 10px 30px -12px rgba(0,0,0,0.6);
    }

    html, body, .gradio-container {
        background: var(--bg) !important; color: var(--ink) !important;
        font-family: 'Inter', ui-sans-serif, system-ui, sans-serif !important;
    }
    .gradio-container {
        max-width: 1520px !important; margin: 0 auto !important;
        padding: 0 28px 28px !important;
    }

    /* top bar */
    #surt-topbar {
        display:flex; align-items:center; justify-content:space-between;
        gap: 24px; padding: 18px 0 16px;
        border-bottom: 1px solid var(--rule); margin-bottom: 22px;
    }
    #surt-topbar .left { display:flex; align-items:baseline; gap: 8px; }
    #surt-topbar .ik {
        font-family: "Gurbani Akhar","Noto Sans Gurmukhi", serif;
        font-size: 40px; line-height: 1; color: var(--amber);
    }
    #surt-topbar .word-blue { font-size: 26px; font-weight: 700; color: var(--navy); letter-spacing: -.3px; }
    #surt-topbar .stack {
        display:inline-flex; flex-direction:column; align-items:center; justify-content:center;
        font-size: 10px; color: var(--navy); line-height: 1.1; margin: 0 2px; font-weight: 500;
    }
    #surt-topbar .word-amber { font-size: 26px; font-weight: 700; color: var(--amber); letter-spacing: -.3px; }
    #surt-topbar .right { display:flex; align-items:center; gap: 10px; }
    #surt-topbar .chip {
        font-size: 10px; letter-spacing:.6px; text-transform: uppercase; font-weight: 600;
        color: var(--amber); border: 1px solid var(--amber-soft);
        background: #fff7e4; padding: 4px 10px; border-radius: 999px;
    }
    body.surt-dark #surt-topbar .chip { background: rgba(243,197,106,0.1); }

    /* icon-button (topbar) */
    .icon-btn {
        width: 34px; height: 34px; border-radius: 10px;
        display:inline-flex; align-items:center; justify-content:center;
        background: transparent; border: 1px solid var(--rule);
        color: var(--ink-dim); cursor: pointer; transition: all .15s ease;
    }
    .icon-btn:hover { background: var(--card); color: var(--navy); border-color: var(--navy); }
    .icon-btn svg { width: 16px; height: 16px; }

    #surt-main { align-items: stretch !important; gap: 22px !important; }

    /* --- stage --- */
    #surt-stage {
        background: var(--stage-bg); border: 1px solid var(--rule);
        border-radius: 18px; box-shadow: var(--shadow);
        padding: 36px 44px; min-height: 58vh;
        display:flex; flex-direction: column; gap: 14px;
        position: relative; overflow: hidden;
    }
    #surt-stage::before, #surt-stage::after {
        content: ""; position:absolute; width:180px; height:180px;
        background: radial-gradient(circle, rgba(230,165,55,0.18) 0%, transparent 60%);
        filter: blur(10px); pointer-events:none;
    }
    body.surt-dark #surt-stage::before, body.surt-dark #surt-stage::after {
        background: radial-gradient(circle, rgba(243,197,106,0.12) 0%, transparent 60%);
    }
    #surt-stage::before { top: -60px; left: -60px; }
    #surt-stage::after  { bottom: -60px; right: -60px; }

    /* hero variant */
    .stage-hero {
        flex: 1; display:flex; flex-direction: column; justify-content: center;
        align-items: center; text-align: center; padding: 10px 0;
    }
    .hero-label {
        font-size: 11px; letter-spacing: .6px; text-transform: uppercase;
        color: var(--amber); font-weight: 700; margin-bottom: 18px;
    }
    .hero-gur {
        font-family: "Gurbani Akhar","Noto Sans Gurmukhi", serif;
        font-size: 48px; line-height: 1.4; color: var(--ink);
        max-width: 900px; margin: 0 auto;
    }
    .hero-meta { font-size: 13px; color: var(--ink-dim); margin-top: 18px; letter-spacing: .3px; }
    .stage-transcript-strip {
        border-top: 1px dashed var(--rule);
        padding-top: 12px; margin-top: 8px;
        max-height: 100px; overflow: auto;
        text-align: center;
    }

    /* transcript variant (no confident match yet) */
    .stage-transcript {
        flex: 1; display:flex; align-items: center; justify-content:center;
    }
    .surt-line {
        font-family: "Gurbani Akhar","Noto Sans Gurmukhi","Nirmala UI", serif;
        font-size: 40px; line-height: 1.5; color: var(--ink);
        letter-spacing: 0.2px; text-align: center;
        max-width: 900px; margin: 0 auto;
    }
    .surt-line.small { font-size: 16px; line-height: 1.4; color: var(--ink-dim); }
    .surt-line.empty { color: var(--muted); font-style: italic; font-size: 16px; }
    .committed { color: var(--ink); outline: none; }
    .committed[contenteditable="true"]:focus {
        background: rgba(230,165,55,0.08);
        border-radius: 6px; padding: 0 6px;
    }
    .tentative { color: var(--amber); opacity: 0.95; }

    /* idle state */
    .surt-ph { text-align:center; color: var(--ink-dim); padding: 6vh 0; }
    .ik-onkar {
        font-family: "Gurbani Akhar","Noto Sans Gurmukhi", serif;
        font-size: 120px; line-height: 1; color: var(--amber);
        text-shadow: 0 0 40px rgba(230,165,55,0.30); margin-bottom: 12px;
    }
    .ik-sub {
        font-family: "Gurbani Akhar","Noto Sans Gurmukhi", serif;
        font-size: 26px; color: #b9a274; margin-bottom: 14px;
    }
    body.surt-dark .ik-sub { color: #8a7a50; }
    .ik-hint { font-size: 12px; color: var(--muted); letter-spacing: 0.4px; }
    .surt-ph.listening .pulse, .pulse-dot {
        animation: pulse-mic 1.2s ease-in-out infinite;
        color: var(--amber); display: inline-block;
    }
    @keyframes pulse-mic { 0%,100% { opacity: 0.5; } 50% { opacity: 1; } }
    .ik-hint kbd {
        background: var(--card); border: 1px solid var(--rule); border-radius: 4px;
        padding: 1px 6px; font-family: ui-monospace, monospace; font-size: 11px;
        color: var(--ink-dim); margin: 0 2px;
    }

    /* --- primary action bar under stage --- */
    #surt-action-mount { width: 100%; margin-top: 14px; }
    .surt-action-bar {
        display:flex; align-items:center; gap: 10px; width: 100%;
    }
    .primary-cta {
        flex: 1; min-height: 54px;
        background: linear-gradient(180deg, var(--amber), #d89622);
        color: #fff; font-weight: 700; font-size: 15px;
        letter-spacing: .3px; border: 0; border-radius: 14px;
        box-shadow: 0 6px 20px -6px rgba(230,165,55,0.6);
        cursor: pointer; transition: all .15s ease;
    }
    .primary-cta:hover:not(.disabled):not([disabled]) {
        transform: translateY(-1px);
        box-shadow: 0 10px 28px -8px rgba(230,165,55,0.75);
    }
    .primary-cta.disabled, .primary-cta[disabled] {
        background: var(--rule); color: var(--muted);
        box-shadow: none; cursor: not-allowed;
    }
    .primary-cta.ready { animation: pulse-ready 2s ease-in-out infinite; }
    @keyframes pulse-ready {
        0%, 100% { box-shadow: 0 6px 20px -6px rgba(230,165,55,0.6); }
        50%      { box-shadow: 0 6px 28px -4px rgba(230,165,55,0.9); }
    }

    .action-icon-btn {
        width: 54px; height: 54px; border-radius: 14px;
        background: var(--card); border: 1px solid var(--rule);
        color: var(--navy); display:inline-flex; align-items:center; justify-content:center;
        cursor: pointer; transition: all .15s ease;
        box-shadow: var(--shadow-sm);
    }
    .action-icon-btn:hover { border-color: var(--navy); color: var(--navy-deep); }
    .action-icon-btn svg { width: 20px; height: 20px; }

    /* Secondary CTA (Clear) — more visible than a plain icon */
    .secondary-cta {
        min-height: 54px; padding: 0 18px;
        background: var(--card); color: var(--navy);
        border: 1px solid var(--rule); border-radius: 14px;
        display:inline-flex; align-items:center; justify-content:center; gap: 8px;
        font-family: inherit; font-size: 14px; font-weight: 600;
        cursor: pointer; transition: all .15s ease;
        box-shadow: var(--shadow-sm);
    }
    .secondary-cta svg { width: 18px; height: 18px; }
    .secondary-cta:hover { border-color: var(--navy); color: var(--navy-deep); }
    .clear-cta:hover { border-color: var(--danger); color: var(--danger); }

    /* --- toast --- */
    #surt-toast-mount { position: relative; }
    .surt-toast {
        position: fixed; left: 50%; bottom: 30px; transform: translateX(-50%);
        background: var(--card); color: var(--ink);
        border: 1px solid var(--rule); border-radius: 12px;
        padding: 10px 16px; box-shadow: var(--shadow);
        display:flex; align-items:center; gap: 14px;
        font-size: 13px; z-index: 9999;
        animation: toast-in .2s ease-out;
    }
    @keyframes toast-in { from { opacity: 0; transform: translate(-50%, 10px); } to { opacity: 1; } }
    .surt-toast.success { border-color: #c9e7d2; }
    .surt-toast.warn    { border-color: var(--amber-soft); }
    .surt-toast.error   { border-color: #e7c9c9; color: var(--danger); }
    .toast-undo {
        background: var(--navy); color: #fff; border: 0; border-radius: 8px;
        padding: 6px 12px; font-weight: 600; font-size: 12px; cursor: pointer;
    }
    .toast-undo:hover { background: var(--navy-deep); }

    /* --- sidebar --- */
    #surt-sidebar { display:flex; flex-direction: column; gap: 16px; }
    .surt-card {
        background: var(--card); border: 1px solid var(--rule);
        border-radius: 14px; box-shadow: var(--shadow-sm);
        padding: 16px 18px;
    }
    .surt-card .card-head {
        display:flex; align-items:center; justify-content:space-between;
        font-size: 11px; letter-spacing: 0.6px; text-transform: uppercase;
        color: var(--ink-dim); font-weight: 600; margin-bottom: 12px;
    }
    .surt-card .card-head .tag {
        font-size: 10px; color: var(--muted); font-weight: 500;
        letter-spacing: 0.4px; text-transform: none;
    }

    /* tabs (input card) */
    .tab-nav { border-bottom: 1px solid var(--rule) !important; }
    .tab-nav button {
        color: var(--ink-dim) !important; font-weight: 500 !important;
        font-size: 12px !important; padding: 8px 12px !important;
    }
    .tab-nav button.selected {
        color: var(--navy) !important; border-bottom-color: var(--navy) !important;
    }
    .gradio-audio { background: transparent !important; border: none !important; }

    /* STTM pill (button) */
    .sttm-pill {
        display:inline-flex; align-items:center; gap: 7px;
        padding: 6px 12px; border-radius: 999px;
        font-size: 11px; font-weight: 500;
        cursor: pointer; transition: all .15s ease;
        font-family: inherit;
    }
    .sttm-pill .dot { width: 7px; height:7px; border-radius:50%; }
    .sttm-pill.ok  { color: var(--ok); background:#eafaf0; border:1px solid #c9e7d2; }
    .sttm-pill.ok  .dot { background: var(--ok); }
    .sttm-pill.off { color: var(--muted); background: var(--card); border:1px solid var(--rule); }
    .sttm-pill.off .dot { background: var(--muted); }
    .sttm-pill:hover { transform: translateY(-1px); }
    body.surt-dark .sttm-pill.ok { background: rgba(46,139,87,0.15); }

    /* --- manual search in Matches card --- */
    .match-search-row {
        display:flex; gap: 8px; align-items: center;
        margin-bottom: 10px;
    }
    .match-search-row input {
        flex: 1; background: var(--card); border: 1px solid var(--rule);
        border-radius: 8px; padding: 7px 10px; font-size: 12px;
        color: var(--ink); outline: none;
        font-family: "Gurbani Akhar","Noto Sans Gurmukhi", "Inter", serif;
    }
    .match-search-row input:focus { border-color: var(--navy); }
    .match-search-row button {
        background: var(--navy); color: #fff; border: 0; border-radius: 8px;
        padding: 7px 12px; font-size: 11px; font-weight: 600; cursor: pointer;
    }
    .match-search-row button:hover { background: var(--navy-deep); }

    /* Matches */
    #surt-matches { display:flex; flex-direction: column; gap: 8px; }
    .match-row {
        display:flex; align-items: stretch; gap: 10px;
        padding: 10px 10px; border-radius: 10px;
        background: #fcfaf2; border: 1px solid var(--rule-soft);
        cursor: pointer; transition: border-color .15s, background .15s, transform .1s;
        position: relative;
    }
    body.surt-dark .match-row { background: rgba(243,197,106,0.04); }
    .match-row:hover { border-color: var(--navy); background: var(--card); transform: translateX(2px); }
    .match-row.top { background: #fff7e4; border-color: var(--amber-soft); }
    body.surt-dark .match-row.top { background: rgba(243,197,106,0.15); }
    .match-rank {
        width: 28px; flex: none; font-weight: 700; color: var(--navy);
        font-size: 13px; display:flex; align-items:center; justify-content:center;
    }
    .match-row.top .match-rank { color: var(--amber); }
    .match-body { flex: 1; min-width: 0; }
    .match-gur {
        font-family: "Gurbani Akhar","Noto Sans Gurmukhi", serif;
        font-size: 15px; color: var(--ink); line-height: 1.4;
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .match-meta { font-size: 10px; color: var(--muted); margin-top: 3px; letter-spacing: .2px; }
    .match-score { width: 66px; flex: none; display:flex; flex-direction:column;
        align-items: flex-end; justify-content: center; gap: 3px; }
    .score-bar {
        width: 56px; height: 5px; background: var(--rule); border-radius: 999px; overflow: hidden;
    }
    .score-bar span {
        display:block; height: 100%; background: var(--navy); border-radius: 999px;
    }
    .match-row.top .score-bar span { background: var(--amber); }
    .score-pct { font-size: 10px; color: var(--ink-dim); font-weight: 600; }
    .match-push {
        flex: none; width: 26px;
        display:flex; align-items:center; justify-content:center;
        color: var(--muted); font-size: 18px; font-weight: 700;
        transition: color .15s ease;
    }
    .match-row:hover .match-push { color: var(--navy); }
    .match-row.top .match-push { color: var(--amber); }
    .match-empty { font-size: 11px; color: var(--muted); text-align:center; padding: 6px 0; }

    /* Session */
    .surt-stat-row { display:flex; justify-content:space-between; gap: 10px;
        padding: 5px 0; font-size: 12px; color: var(--ink-dim);
        border-bottom: 1px dashed var(--rule-soft); }
    .surt-stat-row:last-child { border-bottom: none; }
    .surt-stat-row b { color: var(--navy); font-weight: 600; }

    #surt-hidden-btns { display:none !important; }

    /* help overlay */
    #surt-help {
        display: none;
        position: fixed; inset: 0; background: rgba(10,14,28,0.55);
        z-index: 10000; align-items: center; justify-content: center;
    }
    #surt-help.show { display: flex; }
    #surt-help .panel {
        background: var(--card); color: var(--ink);
        border-radius: 16px; padding: 24px 28px; max-width: 460px; width: 92%;
        box-shadow: 0 30px 80px -20px rgba(0,0,0,0.5);
    }
    #surt-help h2 { font-size: 15px; margin: 0 0 14px; color: var(--navy); letter-spacing: .5px; }
    #surt-help dl { display:grid; grid-template-columns: auto 1fr; gap: 8px 18px; font-size: 13px; }
    #surt-help dt { font-family: ui-monospace, monospace; color: var(--amber); font-weight: 700; }
    #surt-help dd { color: var(--ink-dim); margin: 0; }
    #surt-help .close {
        margin-top: 18px; background: var(--navy); color: #fff; border: 0;
        border-radius: 8px; padding: 8px 16px; font-weight: 600; cursor: pointer;
    }

    #surt-footer {
        margin-top: 26px; padding-top: 14px;
        border-top: 1px solid var(--rule);
        display:flex; justify-content:space-between; align-items:center;
        color: var(--muted); font-size: 11px; letter-spacing: .3px;
    }
    #surt-footer b { color: var(--ink-dim); font-weight: 600; }

    footer, .built-with { display:none !important; }

    /* tablet layout */
    @media (max-width: 960px) {
        .gradio-container { padding: 0 14px 14px !important; }
        #surt-main { flex-direction: column !important; }
        #surt-stage { padding: 24px 20px; min-height: 40vh; }
        .hero-gur { font-size: 32px; }
        .surt-line { font-size: 28px; }
    }
    """

    with gr.Blocks(title="Surt — Live Gurbani") as demo:
        demo._surt_css = css
        backend_label = getattr(backend, "name", type(backend).__name__)

        gr.HTML(
            """
            <div id="surt-topbar">
              <div class="left">
                <span class="ik">ੴ</span>
                <span class="word-blue">Surt</span>
                <span class="stack"><span>To</span><span>The</span></span>
                <span class="word-amber">Max</span>
                <span class="chip" style="margin-left:14px">Testing mode</span>
              </div>
              <div class="right">
                <div id="surt-sttm-mount"></div>
                <button class="icon-btn" title="Toggle dark mode (D)"
                    onclick="window._surt_toggleDark()">
                """ + ICON_MOON + """
                </button>
                <button class="icon-btn" title="Keyboard shortcuts (?)"
                    onclick="document.getElementById('surt-help').classList.add('show')">
                """ + ICON_HELP + """
                </button>
              </div>
            </div>
            """
        )

        state = gr.State(value=StreamState(sttm_pin=saved_pin))

        # ---------- Main grid ----------
        with gr.Row(elem_id="surt-main"):
            # LEFT: stage + action bar
            with gr.Column(scale=7, min_width=520):
                with gr.Column(elem_id="surt-stage"):
                    stage = gr.HTML(_render_stage(StreamState()))

                # primary action bar — single HTML so flex layout isn't split by
                # Gradio's per-component wrappers.
                primary_cta = gr.HTML(
                    _render_action_bar(StreamState()), elem_id="surt-action-mount",
                )

                toast_mount = gr.HTML("", elem_id="surt-toast-mount")

            # RIGHT: sidebar
            with gr.Column(scale=5, min_width=380, elem_id="surt-sidebar"):

                # -- Input card --
                with gr.Column(elem_classes=["surt-card"]):
                    gr.HTML(
                        '<div class="card-head"><span>Input</span>'
                        f'<span class="tag">{backend_label}</span></div>'
                    )
                    with gr.Tabs():
                        with gr.TabItem("Live mic"):
                            mic = gr.Audio(
                                sources=["microphone"], streaming=True, type="numpy",
                                label="Stream from microphone", show_label=False,
                            )
                        with gr.TabItem("Upload file"):
                            upload = gr.Audio(
                                sources=["upload"], streaming=False, type="numpy",
                                label="Audio file (wav/mp3/m4a)", show_label=False,
                            )
                            upload_btn = gr.Button(
                                "Transcribe file", variant="primary", size="sm",
                            )
                        with gr.TabItem("Play from file / URL"):
                            player_url = gr.Textbox(
                                value="", label="YouTube URL (optional)",
                                placeholder="https://youtube.com/watch?v=…",
                            )
                            player_file = gr.Audio(
                                sources=["upload"], streaming=False, type="filepath",
                                label="…or upload a kirtan file",
                            )
                            load_btn = gr.Button(
                                "Load source", variant="secondary", size="sm",
                            )
                            player_preview = gr.Audio(
                                label="Playback (use browser controls)",
                                interactive=False, type="filepath",
                                elem_id="surt-player-preview",
                            )
                            with gr.Row():
                                play_sync_btn = gr.Button(
                                    "▶ Stream + sync to STTM",
                                    variant="primary", size="sm",
                                )
                                auto_push_toggle = gr.Checkbox(
                                    value=True, label="Auto-push top match",
                                )
                            gr.HTML(
                                '<div class="surt-stat-row" style="padding-top:0">'
                                '<span>URL path streams from yt-dlp → ffmpeg; '
                                'file path uses the loaded audio.</span></div>'
                            )
                            player_msg = gr.HTML(
                                '<div class="surt-stat-row"><span>player</span>'
                                '<b>no source</b></div>'
                            )

                # -- Matches card (with manual search) --
                with gr.Column(elem_classes=["surt-card"]):
                    gr.HTML('<div class="card-head"><span>Shabad matches</span>'
                            '<span class="tag">top 5 · local FAISS</span></div>')
                    with gr.Row(elem_classes=["match-search-row"]):
                        manual_query = gr.Textbox(
                            value="", show_label=False, placeholder="Search a pangti…",
                            scale=4, container=False,
                        )
                        manual_search_btn = gr.Button("Search", size="sm", scale=1)
                    matches_html = gr.HTML(_render_matches([]))

                # -- STTM card --
                with gr.Column(elem_classes=["surt-card"]):
                    gr.HTML('<div class="card-head"><span>STTM Projection</span>'
                            '<span class="tag">optional</span></div>')
                    sttm_status = gr.HTML(_sttm_pill(StreamState()))
                    with gr.Row():
                        sttm_host = gr.Textbox(
                            value="127.0.0.1", label="Host", placeholder="127.0.0.1", scale=2,
                        )
                        sttm_port = gr.Textbox(
                            value="", label="Port",
                            placeholder=f"auto ({CANDIDATE_PORTS[0]})", scale=1,
                        )
                        sttm_pin = gr.Textbox(
                            value=saved_pin, label="PIN",
                            placeholder="e.g. 3563",
                            info="Bani Controller PIN — saved locally after connect",
                            scale=1,
                        )

                # -- Settings (collapsed) --
                with gr.Accordion("Settings · retrieval + streaming", open=False):
                    retrieval_mode_dd = gr.Dropdown(
                        choices=[(v, k) for k, v in MODE_LABELS.items()],
                        value=DEFAULT_MODE,
                        label="Retrieval method",
                        info=("Applies to live mic, play-from-URL, file upload, "
                              "and manual search. EMA resets on change."),
                    )
                    with gr.Row():
                        lock_streak_radio = gr.Radio(
                            choices=[("1 window (fast lock)", 1),
                                     ("2 windows (debounced)", 2)],
                            value=LOCK_STREAK_DEFAULT,
                            label="Shabad lock threshold",
                            info=("How many confident windows before we lock. "
                                  "Unlocks automatically on 3 consecutive misses."),
                            scale=3,
                        )
                        unlock_btn = gr.Button(
                            "🔓 Unlock shabad", size="sm", scale=1,
                            elem_id="surt-unlock-btn",
                        )
                    with gr.Row():
                        auto_push_slider = gr.Slider(
                            minimum=0.0, maximum=1.0,
                            value=AUTO_PUSH_THRESHOLD_DEFAULT, step=0.05,
                            label="Auto-push confidence floor",
                        )
                        hero_slider = gr.Slider(
                            minimum=0.3, maximum=1.0,
                            value=HERO_THRESHOLD_DEFAULT, step=0.05,
                            label="Hero-shabad confidence floor",
                        )
                    with gr.Row():
                        vad_slider = gr.Slider(
                            minimum=0.0, maximum=0.05, value=0.005, step=0.001,
                            label="VAD threshold (RMS)",
                        )
                        throttle_slider = gr.Slider(
                            minimum=0.5, maximum=3.0, value=1.2, step=0.1,
                            label="Throttle (s)",
                        )
                    with gr.Row():
                        commit_slider = gr.Slider(
                            minimum=4.0, maximum=20.0, value=10.0, step=0.5,
                            label="Commit window (s)",
                        )
                        max_slider = gr.Slider(
                            minimum=6.0, maximum=24.0, value=12.0, step=0.5,
                            label="Max window (s)",
                        )
                    carry_slider = gr.Slider(
                        minimum=0.5, maximum=5.0, value=2.0, step=0.5,
                        label="Carry-over (s)",
                    )

                # -- Session (collapsed) --
                with gr.Accordion("Session · live stats", open=False):
                    stats = gr.HTML(
                        '<div class="surt-stat-row"><span>status</span>'
                        '<b>waiting for audio…</b></div>'
                    )

        # ---------- Footer ----------
        gr.HTML(
            f"""
            <div id="surt-footer">
              <span><b>Surt v3</b> · live Gurbani transcription</span>
              <span>backend <b>{backend_label}</b> · inspired by SikhiToTheMax</span>
            </div>
            """
        )

        # ---------- Help overlay + JS glue + hls.js ----------
        gr.HTML(
            """
            <script src="https://cdn.jsdelivr.net/npm/hls.js@1.5.13/dist/hls.min.js"></script>
            <div id="surt-help" onclick="if(event.target===this)this.classList.remove('show')">
              <div class="panel">
                <h2>Keyboard shortcuts</h2>
                <dl>
                  <dt>Space</dt> <dd>Toggle mic recording</dd>
                  <dt>Enter</dt> <dd>Push top match to STTM</dd>
                  <dt>1–5</dt>   <dd>Push Nth match</dd>
                  <dt>C</dt>     <dd>Copy transcript</dd>
                  <dt>X</dt>     <dd>Export transcript</dd>
                  <dt>D</dt>     <dd>Toggle dark mode</dd>
                  <dt>Esc</dt>   <dd>Clear (undo-able)</dd>
                  <dt>?</dt>     <dd>Show this help</dd>
                </dl>
                <button class="close"
                    onclick="document.getElementById('surt-help').classList.remove('show')">
                  Close
                </button>
              </div>
            </div>
            <script>
            (function(){
              // --- HLS playback glue ---
              // When `player_preview` gets a file path ending in .m3u8, attach
              // hls.js and start playing. Server paces transcription to wall-
              // clock so STTM pushes match the browser playhead.
              let _surtHls = null;
              let _surtHlsLastSrc = null;
              function attachHlsIfNeeded() {
                const audio = document.querySelector(
                    '#surt-player-preview audio');
                if (!audio) return;
                // Gradio sets audio.src to something like `/file=/tmp/surt_hls/<id>/playlist.m3u8`
                // or `/gradio_api/file=...`. Extract and re-route through hls.js
                // when the src looks like an HLS manifest.
                const src = audio.currentSrc || audio.src || '';
                if (!src || src === _surtHlsLastSrc) return;
                if (src.indexOf('.m3u8') === -1) return;
                _surtHlsLastSrc = src;
                if (_surtHls) { try { _surtHls.destroy(); } catch(e){} _surtHls = null; }
                if (window.Hls && window.Hls.isSupported()) {
                    _surtHls = new window.Hls({ lowLatencyMode: true, liveSyncDuration: 2 });
                    _surtHls.loadSource(src);
                    _surtHls.attachMedia(audio);
                    _surtHls.on(window.Hls.Events.MANIFEST_PARSED, () => {
                        audio.play().catch(() => {});
                    });
                } else if (audio.canPlayType('application/vnd.apple.mpegurl')) {
                    // Safari — native HLS
                    audio.src = src;
                    audio.play().catch(() => {});
                }
              }
              setInterval(attachHlsIfNeeded, 500);

              // Mirror the STTM pill from the sidebar into the topbar via cloneNode
              // (safer than innerHTML; the source is trusted Gradio-rendered output).
              function mirrorPill() {
                const src = document.querySelector('#surt-sidebar .sttm-pill');
                const dst = document.getElementById('surt-sttm-mount');
                if (!src || !dst) return;
                const clone = src.cloneNode(true);
                dst.replaceChildren(clone);
              }
              setInterval(mirrorPill, 600);
              setTimeout(mirrorPill, 300);

              // Dark mode
              window._surt_toggleDark = function(){
                document.body.classList.toggle('surt-dark');
                try {
                  localStorage.setItem('surt-dark',
                      document.body.classList.contains('surt-dark') ? '1' : '0');
                } catch(e) {}
              };
              try {
                if (localStorage.getItem('surt-dark') === '1') {
                  document.body.classList.add('surt-dark');
                }
              } catch(e) {}

              // Copy transcript
              window._surt_copy = function(){
                const el = document.querySelector('#surt-stage .committed') ||
                           document.querySelector('#surt-stage .surt-line');
                const txt = el ? (el.innerText || '').trim() : '';
                if (!txt) return;
                navigator.clipboard.writeText(txt);
              };

              // Export transcript as .txt
              window._surt_export = function(){
                const el = document.querySelector('#surt-stage .committed') ||
                           document.querySelector('#surt-stage .surt-line');
                const txt = el ? (el.innerText || '').trim() : '';
                if (!txt) return;
                const heroEl = document.querySelector('#surt-stage .hero-gur');
                const meta   = document.querySelector('#surt-stage .hero-meta');
                const blob = new Blob([
                  'Surt — session transcript\\n',
                  'Exported: ' + new Date().toISOString() + '\\n\\n',
                  (heroEl ? 'Matched shabad:\\n' + heroEl.innerText.trim() + '\\n' +
                    (meta ? meta.innerText.trim() + '\\n' : '') + '\\n' : ''),
                  'Transcript:\\n', txt, '\\n'
                ], {type: 'text/plain;charset=utf-8'});
                const a = document.createElement('a');
                a.href = URL.createObjectURL(blob);
                a.download = 'surt-session-' + Date.now() + '.txt';
                document.body.appendChild(a); a.click(); a.remove();
                setTimeout(() => URL.revokeObjectURL(a.href), 1000);
              };

              // Match row picker (bridge to hidden textbox)
              window._surt_pick = function(i){
                const el = document.getElementById('surt-selected-idx');
                if(!el) return;
                const tb = el.querySelector('textarea, input');
                if(!tb) return;
                const setter = Object.getOwnPropertyDescriptor(
                  Object.getPrototypeOf(tb), 'value').set;
                setter.call(tb, String(i));
                tb.dispatchEvent(new Event('input', {bubbles:true}));
              };

              // Commit edit bridge (contenteditable span)
              window._surt_commit_edit = function(){
                const el = document.querySelector('#surt-stage .committed');
                if (!el) return;
                const txt = (el.innerText || '').trim();
                const b = document.getElementById('surt-edit-bridge');
                if (!b) return;
                const tb = b.querySelector('textarea, input');
                if (!tb) return;
                const setter = Object.getOwnPropertyDescriptor(
                  Object.getPrototypeOf(tb), 'value').set;
                setter.call(tb, txt);
                tb.dispatchEvent(new Event('input', {bubbles:true}));
              };
              document.addEventListener('blur', function(e){
                if (e.target && e.target.classList &&
                    e.target.classList.contains('committed')) {
                  window._surt_commit_edit();
                }
              }, true);

              // Keyboard shortcuts
              function isTyping(el){
                if (!el) return false;
                if (el.isContentEditable) return true;
                const tag = (el.tagName || '').toLowerCase();
                return tag === 'input' || tag === 'textarea';
              }
              document.addEventListener('keydown', function(e){
                if (isTyping(document.activeElement)) return;
                if (e.key === '?') {
                  document.getElementById('surt-help').classList.toggle('show');
                  e.preventDefault();
                } else if (e.key === 'Escape') {
                  const btn = document.getElementById('surt-clear-real');
                  if (btn) btn.click();
                } else if (e.key === 'Enter') {
                  const btn = document.getElementById('surt-push-real');
                  if (btn) btn.click();
                } else if (e.key === ' ') {
                  const rec = document.querySelector('.gradio-container button.record-button, ' +
                    '.gradio-container [aria-label*="record" i]');
                  if (rec) { rec.click(); e.preventDefault(); }
                } else if (e.key >= '1' && e.key <= '5') {
                  window._surt_pick(parseInt(e.key, 10) - 1);
                } else if (e.key === 'c' || e.key === 'C') {
                  window._surt_copy();
                } else if (e.key === 'x' || e.key === 'X') {
                  window._surt_export();
                } else if (e.key === 'd' || e.key === 'D') {
                  window._surt_toggleDark();
                }
              });
            })();
            </script>
            """
        )

        # hidden bridge buttons (actions triggered from HTML buttons / JS)
        with gr.Row(elem_id="surt-hidden-btns"):
            clear_btn   = gr.Button("Clear",   elem_id="surt-clear-real")
            connect_btn = gr.Button("Connect", elem_id="surt-connect-real")
            push_btn    = gr.Button("Push",    elem_id="surt-push-real")
            undo_btn    = gr.Button("Undo",    elem_id="surt-undo-real")
            selected_idx = gr.Textbox(
                value="", elem_id="surt-selected-idx", label="pick",
            )
            edit_bridge = gr.Textbox(
                value="", elem_id="surt-edit-bridge", label="edit",
            )

        # ---------- helpers ----------

        def _stat(label: str, value: str) -> str:
            return f'<div class="surt-stat-row"><span>{label}</span><b>{value}</b></div>'

        def _multi_stat(rows: list[tuple[str, str]]) -> str:
            return "".join(_stat(k, v) for k, v in rows)

        def _locked_hit_to_match(h, ui_score: float) -> dict:
            """Convert a LockedHit into the match-dict shape the UI expects."""
            return {
                "shabadId": h.sttm_id,
                "verseId": h.sttm_id,
                "gurmukhi": h.tuk_text,
                "writer": h.writer,
                "raag": h.raag,
                "source": "SGGS",
                "ang": h.ang,
                "score": round(float(ui_score), 3),
                "_raw_score": round(float(h.overlap), 4),
                "_tuk_score": round(float(h.overlap), 4),
                "_literal": round(float(h.overlap), 4),
                "_confirmed": True,
                "_locked": True,
                "_tuk_row": h.tuk_row,
            }

        def _handle_locked(st: StreamState, query: str) -> bool:
            """Locked-mode pointer update. Returns True if we handled the query
            (either stayed locked or unlocked on this window)."""
            if not st.locked_shabad_id:
                return False
            hits = score_within_shabad(query, st.locked_shabad_id)
            best = hits[0] if hits else None
            if best is None or best.overlap < UNLOCK_SCORE_FLOOR:
                st.unlock_miss_count += 1
                if st.unlock_miss_count >= st.unlock_misses_target:
                    print(
                        f"[lock] unlock · sid={st.locked_shabad_id} after "
                        f"{st.unlock_miss_count} misses (best={best.overlap if best else 0.0:.2f})"
                    )
                    _unlock(st)
                    return False  # fall through to unlocked retrieval
                # Still within the miss budget — keep the old pointer on screen.
                return True
            # Solid in-shabad hit — advance pointer, reset miss counter.
            st.unlock_miss_count = 0
            st.locked_tuk_row = best.tuk_row
            ui_score = 0.5 + 0.5 * min(best.overlap, 1.0)
            st.matches = [_locked_hit_to_match(best, ui_score)]
            return True

        def _lock(st: StreamState, sid: str, tuk_row: int | None = None) -> None:
            st.locked_shabad_id = sid
            st.locked_tuk_row = tuk_row
            st.unlock_miss_count = 0
            st.lock_streak_count = 0
            st.last_top_sid_for_lock = sid
            if st.retrieval_ema is not None:
                st.retrieval_ema.reset()
            print(f"[lock] lock · sid={sid} · tuk_row={tuk_row}")

        def _unlock(st: StreamState) -> None:
            st.locked_shabad_id = None
            st.locked_tuk_row = None
            st.unlock_miss_count = 0
            st.lock_streak_count = 0
            st.last_top_sid_for_lock = None
            st.last_search_key = ""  # force fresh retrieval on next tick
            if st.retrieval_ema is not None:
                st.retrieval_ema.reset()

        def _maybe_auto_lock(st: StreamState, hits: list[dict]) -> None:
            """Promote UNLOCKED → LOCKED when the top match is confident enough
            for `lock_streak_target` consecutive windows."""
            if not hits:
                st.lock_streak_count = 0
                st.last_top_sid_for_lock = None
                return
            top = hits[0]
            lit = float(top.get("_literal", 0.0))
            if lit < LOCK_SCORE_FLOOR:
                st.lock_streak_count = 0
                st.last_top_sid_for_lock = None
                return
            # Prefer the internal shabad_id (Q-id) for lock keying. The match
            # dict only carries sttm_id; map back via the retriever.
            sid = _shabad_id_from_hit(top)
            if sid is None:
                return
            if sid == st.last_top_sid_for_lock:
                st.lock_streak_count += 1
            else:
                st.last_top_sid_for_lock = sid
                st.lock_streak_count = 1
            if st.lock_streak_count >= max(1, int(st.lock_streak_target)):
                _lock(st, sid, tuk_row=top.get("_tuk_row"))

        def _refresh_matches(st: StreamState, *, smooth: bool = False) -> None:
            """Refresh matches.

            When LOCKED: cheap within-shabad pointer update; no MuRIL, no FAISS.
            When UNLOCKED: full retrieval, then consider auto-locking if the
            top match is confident enough (streak-gated).

            smooth=True → fold through RetrievalEMA so live windows don't
            bounce between shabads (ignored while locked).
            """
            text = (st.committed or st.tentative or "").strip()
            query = _retrieval_query(text)
            if not query or query == st.last_search_key:
                return
            st.last_search_key = query

            if _handle_locked(st, query):
                return

            hits = search_shabad_topn(query, n=5, mode=st.retrieval_mode)
            if smooth:
                if st.retrieval_ema is None:
                    st.retrieval_ema = RetrievalEMA()
                st.matches = st.retrieval_ema.update(hits, top_n=5)
            else:
                st.matches = hits
            _maybe_auto_lock(st, st.matches)

        def _transcribe(st: StreamState, audio: np.ndarray, sr: int = TARGET_SR) -> str:
            try:
                text = backend.transcribe(audio, sr) or ""
            except Exception as e:  # noqa: BLE001
                print(f"[surt] transcribe failed: {e}")
                return ""
            return _suppress_repeat_hallucination(text)

        def _full_stage_outputs(st: StreamState, toast_html: str = ""):
            return (
                st,
                _render_stage(st),
                _render_action_bar(st),
                _render_matches(st.matches),
                toast_html,
            )

        # ---------- settings ----------

        def on_settings_change(vad, throttle, commit, maxw, carry,
                               auto_push_th, hero_th, st: StreamState):
            st.vad_threshold = float(vad)
            st.throttle_s = float(throttle)
            st.commit_s = float(commit)
            st.max_window_s = float(maxw)
            st.carry_over_s = float(carry)
            st.auto_push_threshold = float(auto_push_th)
            st.hero_threshold = float(hero_th)
            return st, _render_stage(st)

        # ---------- mic streaming ----------

        def on_stream(chunk, st: StreamState):
            st.last_stream_t = time.time()
            st.stream_calls += 1
            if st.stream_calls <= 3 or st.stream_calls % 20 == 0:
                shape = None
                if isinstance(chunk, tuple) and len(chunk) == 2:
                    _, a = chunk
                    shape = (a.shape if hasattr(a, "shape") else type(a).__name__)
                print(f"[mic] call #{st.stream_calls} chunk={shape} "
                      f"buffer={st.buffer.size/TARGET_SR:.1f}s")
            if chunk is None or not isinstance(chunk, tuple):
                return (st, _render_stage(st), _render_action_bar(st),
                        _render_matches(st.matches), st.last_stats_html)
            sr_in, arr = chunk
            if arr is None or len(arr) == 0:
                return (st, _render_stage(st), _render_action_bar(st),
                        _render_matches(st.matches), st.last_stats_html)

            y = _resample(_to_mono_float32(arr), sr_in)
            st.buffer = np.concatenate([st.buffer, y]) if st.buffer.size else y

            commit_samples = int(st.commit_s * TARGET_SR)
            max_samples = int(st.max_window_s * TARGET_SR)
            carry_samples = int(st.carry_over_s * TARGET_SR)
            min_samples = int(st.min_transcribe_s * TARGET_SR)

            # commit-and-carry
            if st.buffer.size > max_samples:
                commit_slice = st.buffer[:commit_samples]
                if _rms(commit_slice) >= st.vad_threshold:
                    txt = _transcribe(st, commit_slice)
                    if txt:
                        st.committed = (st.committed + " " + txt).strip()
                        _refresh_matches(st, smooth=True)
                st.buffer = st.buffer[-carry_samples:].copy()
                st.tentative = ""

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

            return (st, _render_stage(st), _render_action_bar(st),
                    _render_matches(st.matches), st.last_stats_html)

        # ---------- upload ----------

        def on_upload(audio, st: StreamState):
            if audio is None:
                return (*_full_stage_outputs(st), _stat("upload", "no file"))
            sr_in, arr = audio
            if arr is None or len(arr) == 0:
                return (*_full_stage_outputs(st), _stat("upload", "empty audio"))
            y = _resample(_to_mono_float32(arr), sr_in)
            t0 = time.time()
            text = _transcribe(st, y) or ""
            latency_ms = (time.time() - t0) * 1000.0
            st.committed = text
            st.tentative = ""
            _refresh_matches(st)
            dur = y.size / TARGET_SR
            stats_html = _multi_stat([
                ("source", "upload"),
                ("duration", f"{dur:4.1f}s"),
                ("inference", f"{latency_ms:4.0f} ms"),
                ("rtf", f"{latency_ms/1000.0/max(dur,1e-6):4.2f}×"),
                ("backend", backend_label),
            ])
            return (*_full_stage_outputs(st), stats_html)

        # ---------- clear + undo ----------

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
            if st.retrieval_ema is not None:
                st.retrieval_ema.reset()
            toast = _render_toast("Transcript cleared.", "warn", undo=True)
            return (*_full_stage_outputs(st, toast), _stat("status", "cleared"))

        def on_undo(st: StreamState):
            if st.undo_committed is None or time.time() > st.undo_expires_at:
                return (*_full_stage_outputs(st, _render_toast(
                    "Nothing to undo.", "error")), _stat("status", "no undo"))
            st.committed = st.undo_committed or ""
            st.matches = st.undo_matches or []
            st.last_search_key = st.committed
            st.undo_committed = None
            st.undo_matches = None
            st.undo_expires_at = 0.0
            return (*_full_stage_outputs(st, _render_toast(
                "Restored.", "success")), _stat("status", "restored"))

        # ---------- STTM ----------

        def on_connect(host: str, port_str: str, pin: str, st: StreamState):
            host = (host or "127.0.0.1").strip()
            new_pin = (pin or "").strip()
            if new_pin != st.sttm_pin:
                st.sttm_pin = new_pin
                cfg = _load_config()
                cfg["sttm_pin"] = new_pin
                _save_config(cfg)
            if port_str and port_str.strip().isdigit():
                ports: tuple[int, ...] = (int(port_str.strip()),)
            else:
                ports = CANDIDATE_PORTS
            status: STTMStatus = discover(host=host, ports=ports)
            st.sttm_host = status.host
            st.sttm_port = status.port
            st.sttm_connected = status.ok
            pin_suffix = f" · PIN {st.sttm_pin}" if st.sttm_pin else ""
            toast = _render_toast(
                f"STTM connected on :{status.port}{pin_suffix}" if status.ok
                else "STTM not reachable — open Bani Controller",
                "success" if status.ok else "error",
            )
            return st, _sttm_pill(st), toast

        def _ensure_sttm(st: StreamState) -> None:
            if not st.sttm_connected:
                s = discover(host=st.sttm_host)
                st.sttm_host, st.sttm_port, st.sttm_connected = s.host, s.port, s.ok

        def on_push(st: StreamState):
            _ensure_sttm(st)
            if not (st.sttm_connected and st.sttm_port):
                toast = _render_toast(
                    "STTM not reachable — open Bani Controller.", "error")
                return st, toast, _sttm_pill(st), _stat("STTM", "offline")
            pin = st.sttm_pin or None
            if st.matches:
                top = st.matches[0]
                res = push_hit(st.sttm_host, st.sttm_port, top, pin=pin)
                toast = _render_toast(f"Pushed: {res.detail}", "success")
                return st, toast, _sttm_pill(st), _stat("STTM", res.detail)
            text = (st.committed or st.tentative or "").strip()
            if not text:
                toast = _render_toast("Nothing to push yet.", "warn")
                return st, toast, _sttm_pill(st), _stat("STTM", "nothing")
            res = push_transcript_as_shabad(st.sttm_host, st.sttm_port, text, pin=pin)
            toast = _render_toast(f"Pushed raw text: {res.detail}", "success")
            return st, toast, _sttm_pill(st), _stat("STTM", res.detail)

        def on_pick_match(idx_str: str, st: StreamState):
            try:
                idx = int(idx_str)
            except (TypeError, ValueError):
                return st, _render_toast("Bad match index.", "error"), _sttm_pill(st)
            if idx < 0 or idx >= len(st.matches):
                return st, _render_toast("Match out of range.", "error"), _sttm_pill(st)
            _ensure_sttm(st)
            if not (st.sttm_connected and st.sttm_port):
                return (st, _render_toast(
                    "STTM not reachable.", "error"), _sttm_pill(st))
            res = push_hit(st.sttm_host, st.sttm_port, st.matches[idx],
                           pin=st.sttm_pin or None)
            return st, _render_toast(f"Pushed #{idx+1}: {res.detail}", "success"), _sttm_pill(st)

        # ---------- manual shabad search ----------

        def on_manual_search(q: str, st: StreamState):
            q = (q or "").strip()
            if not q:
                return st, _render_matches(st.matches), _render_stage(st)
            hits = search_shabad_topn(q, n=5, mode=st.retrieval_mode)
            st.matches = hits
            st.last_search_key = q
            return st, _render_matches(hits), _render_stage(st)

        def on_mode_change(mode: str, st: StreamState):
            """Switch retrieval method; EMA resets so old-mode scores don't bleed."""
            st.retrieval_mode = mode or DEFAULT_MODE
            if st.retrieval_ema is not None:
                st.retrieval_ema.reset()
            # Re-run a search if we have text so the user sees the mode's effect.
            st.last_search_key = ""
            _refresh_matches(st, smooth=False)
            return (st, _render_stage(st), _render_action_bar(st),
                    _render_matches(st.matches),
                    _render_toast(f"Retrieval: {mode}", "info"))

        def on_lock_streak_change(streak, st: StreamState):
            try:
                st.lock_streak_target = max(1, int(streak))
            except (TypeError, ValueError):
                st.lock_streak_target = LOCK_STREAK_DEFAULT
            # Changing the streak doesn't touch existing lock — just affects
            # future promotions.
            return st

        def on_unlock_click(st: StreamState):
            if not st.locked_shabad_id:
                return (st, _render_stage(st), _render_action_bar(st),
                        _render_matches(st.matches),
                        _render_toast("Nothing to unlock.", "info"))
            _unlock(st)
            st.last_search_key = ""
            _refresh_matches(st, smooth=False)
            return (st, _render_stage(st), _render_action_bar(st),
                    _render_matches(st.matches),
                    _render_toast("Shabad unlocked.", "info"))

        # ---------- transcript edit ----------

        def on_edit_commit(new_text: str, st: StreamState):
            new_text = (new_text or "").strip()
            if not new_text or new_text == st.committed:
                return st, _render_stage(st), _render_matches(st.matches)
            st.committed = new_text
            _refresh_matches(st)
            return st, _render_stage(st), _render_matches(st.matches)

        # ---------- kirtan player ----------

        def on_load_source(url: str, file_path: str, st: StreamState):
            try:
                path, audio = prepare_source(url, file_path)
            except Exception as e:  # noqa: BLE001
                return (st, gr.update(), gr.update(),
                        _stat("player", f"load failed — {e}"))
            st.player_path = str(path)
            st.player_audio = audio
            dur = audio.size / TARGET_SR
            return (st,
                    gr.update(value=str(path)),
                    gr.update(value=str(path)),
                    _stat("player", f"loaded · {dur:4.1f}s"))

        def _on_stream_url(url: str, auto_push: bool, st: StreamState):
            """Generator: yt-dlp | ffmpeg -> {PCM for ASR, HLS for browser}.

            Browser plays HLS via hls.js; server transcribes each 10-s window
            and paces itself to wall-clock so STTM pushes follow the playhead.
            """
            import uuid
            from apps.transcribe.stream_url import (
                stream_audio_16k, parse_url_time_offset,
            )

            st.committed = ""
            st.tentative = ""
            st.matches = []
            st.last_search_key = ""
            if st.retrieval_ema is not None:
                st.retrieval_ema.reset()
            if auto_push:
                _ensure_sttm(st)

            t_off = parse_url_time_offset(url)
            msg_prefix = f"stream · from {t_off:d}s" if t_off else "stream"

            # Fresh HLS dir per run — Gradio serves it via allowed_paths.
            hls_dir = f"/tmp/surt_hls/{uuid.uuid4().hex[:8]}"
            os.makedirs(hls_dir, exist_ok=True)
            playlist = f"{hls_dir}/playlist.m3u8"

            # Initial yield — spin up the pipeline. hls_bridge stays empty until
            # the first segment appears on disk, then gets the playlist path
            # (our JS glue picks it up and starts hls.js).
            yield (st, _render_stage(st), _render_action_bar(st),
                   _render_matches(st.matches),
                   _stat("player", f"{msg_prefix} · starting yt-dlp + ffmpeg…"),
                   _sttm_pill(st),
                   gr.update())

            window_s = 10.0
            window_samples = int(window_s * TARGET_SR)
            buf = np.zeros(0, dtype=np.float32)
            transcribed_up_to = 0
            last_pushed_sid: int | None = None
            t_start = time.time()
            playlist_emitted = False
            # Browser playback only starts ~2-3 s after we signal it (first
            # segment must land, hls.js must attach), so offset the "playhead"
            # a little so the server doesn't race ahead of real playback.
            PLAYHEAD_LAG_S = 3.0

            for chunk, meta in stream_audio_16k(url, hls_dir=hls_dir):
                if meta.error:
                    yield (st, _render_stage(st), _render_action_bar(st),
                           _render_matches(st.matches),
                           _stat("player", f"stream error — {meta.error}"),
                           _sttm_pill(st), gr.update())
                    return
                if chunk.size > 0:
                    buf = np.concatenate([buf, chunk])

                # Emit the playlist path to the browser as soon as a .ts segment
                # exists (hls.js errors if the manifest references no segments).
                preview_update = gr.update()
                if not playlist_emitted and os.path.exists(playlist):
                    segs = [f for f in os.listdir(hls_dir) if f.endswith(".ts")]
                    if segs:
                        preview_update = gr.update(value=playlist)
                        playlist_emitted = True
                        t_start = time.time()  # restart playhead clock on attach

                # Wall-clock gate: only transcribe windows whose end-time has
                # been "played" by the browser. This is the sync invariant —
                # STTM push is never ahead of the browser playhead.
                playhead_s = max(0.0, time.time() - t_start - PLAYHEAD_LAG_S)
                while (
                    buf.size - transcribed_up_to >= window_samples
                    and (transcribed_up_to + window_samples) / TARGET_SR
                        <= playhead_s
                ):
                    s0 = transcribed_up_to
                    s1 = s0 + window_samples
                    window = buf[s0:s1]
                    text = _transcribe(st, window)
                    if text:
                        st.committed = (st.committed + " " + text).strip()
                        _refresh_matches(st, smooth=True)
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
                    transcribed_up_to = s1

                pos_s = buf.size / TARGET_SR
                yield (st, _render_stage(st), _render_action_bar(st),
                       _render_matches(st.matches),
                       _stat("player",
                             f"{msg_prefix} · buffered {pos_s:4.1f}s · "
                             f"playhead ≈ {playhead_s:4.1f}s"),
                       _sttm_pill(st), preview_update)

                # If download is ahead of playhead, slow our yield rate so
                # we don't spin. Otherwise stay responsive.
                if buf.size / TARGET_SR > playhead_s + window_s:
                    time.sleep(0.5)
                if meta.done:
                    break

            # Drain: stream finished, but browser may still be playing.
            # Keep transcribing paced to playhead until we're caught up.
            while buf.size - transcribed_up_to >= window_samples:
                playhead_s = time.time() - t_start - PLAYHEAD_LAG_S
                target_end = (transcribed_up_to + window_samples) / TARGET_SR
                wait = target_end - playhead_s
                if wait > 0:
                    time.sleep(min(wait, 1.0))
                    yield (st, _render_stage(st), _render_action_bar(st),
                           _render_matches(st.matches),
                           _stat("player",
                                 f"draining · playhead {playhead_s:4.1f}s / "
                                 f"buf {buf.size/TARGET_SR:4.1f}s"),
                           _sttm_pill(st), gr.update())
                    continue
                s0 = transcribed_up_to
                s1 = s0 + window_samples
                text = _transcribe(st, buf[s0:s1])
                if text:
                    st.committed = (st.committed + " " + text).strip()
                    _refresh_matches(st, smooth=True)
                transcribed_up_to = s1

            if buf.size - transcribed_up_to >= 3 * TARGET_SR:
                tail = buf[transcribed_up_to:]
                text = _transcribe(st, tail)
                if text:
                    st.committed = (st.committed + " " + text).strip()
                    _refresh_matches(st, smooth=True)

            yield (st, _render_stage(st), _render_action_bar(st),
                   _render_matches(st.matches),
                   _stat("player", "stream finished"),
                   _sttm_pill(st), gr.update())

        def on_play_sync(url: str, auto_push: bool, st: StreamState):
            """Sync transcription.

            If a URL is filled in, stream from yt-dlp|ffmpeg and transcribe as
            audio lands (no wait for the full download). Otherwise use the
            loaded file's in-memory audio.
            """
            if (url or "").strip():
                yield from _on_stream_url(url.strip(), auto_push, st)
                return

            audio = st.player_audio
            if audio is None or audio.size == 0:
                yield (st, _render_stage(st), _render_action_bar(st),
                       _render_matches(st.matches),
                       _stat("player", "no audio loaded — paste a URL or upload a file"),
                       _sttm_pill(st), gr.update())
                return
            if auto_push:
                _ensure_sttm(st)

            window_s = 10.0
            step_s = 10.0
            sr = TARGET_SR
            total = audio.size
            t_start = time.time()
            i = 0
            last_pushed_sid: int | None = None

            st.committed = ""
            st.tentative = ""
            st.matches = []
            st.last_search_key = ""
            if st.retrieval_ema is not None:
                st.retrieval_ema.reset()

            while i * step_s * sr < total:
                s0 = int(i * step_s * sr)
                s1 = min(total, s0 + int(window_s * sr))
                chunk = audio[s0:s1]

                text = _transcribe(st, chunk)
                if text:
                    st.committed = (st.committed + " " + text).strip()
                    _refresh_matches(st, smooth=True)
                    if (auto_push and st.matches and st.sttm_connected and st.sttm_port
                            and float(st.matches[0].get("score", 0.0))
                                >= st.auto_push_threshold):
                        top = st.matches[0]
                        sid = top.get("shabadId")
                        if sid and sid != last_pushed_sid:
                            push_hit(st.sttm_host, st.sttm_port, top,
                                     pin=st.sttm_pin or None)
                            last_pushed_sid = sid

                pos_s = (i + 1) * step_s
                dur_s = total / sr
                msg = _stat("player", f"sync {pos_s:4.1f}s / {dur_s:4.1f}s")
                yield (st, _render_stage(st), _render_action_bar(st),
                       _render_matches(st.matches), msg, _sttm_pill(st),
                       gr.update())

                elapsed = time.time() - t_start
                target = (i + 1) * step_s
                sleep_for = target - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)
                i += 1

            yield (st, _render_stage(st), _render_action_bar(st),
                   _render_matches(st.matches),
                   _stat("player", "sync finished"), _sttm_pill(st),
                   gr.update())

        # ---------- wiring ----------

        mic.stream(
            on_stream, inputs=[mic, state],
            outputs=[state, stage, primary_cta, matches_html, stats],
            stream_every=0.5, concurrency_limit=1, show_progress="hidden",
        )
        upload_btn.click(
            on_upload, inputs=[upload, state],
            outputs=[state, stage, primary_cta, matches_html, toast_mount, stats],
        )
        clear_btn.click(
            on_clear, inputs=[state],
            outputs=[state, stage, primary_cta, matches_html, toast_mount, stats],
        )
        undo_btn.click(
            on_undo, inputs=[state],
            outputs=[state, stage, primary_cta, matches_html, toast_mount, stats],
        )
        connect_btn.click(on_connect, inputs=[sttm_host, sttm_port, sttm_pin, state],
                          outputs=[state, sttm_status, toast_mount])
        sttm_host.change(on_connect, inputs=[sttm_host, sttm_port, sttm_pin, state],
                         outputs=[state, sttm_status, toast_mount])
        sttm_port.change(on_connect, inputs=[sttm_host, sttm_port, sttm_pin, state],
                         outputs=[state, sttm_status, toast_mount])
        sttm_pin.change(on_connect, inputs=[sttm_host, sttm_port, sttm_pin, state],
                        outputs=[state, sttm_status, toast_mount])
        push_btn.click(on_push, inputs=[state],
                       outputs=[state, toast_mount, sttm_status, stats])
        selected_idx.change(on_pick_match, inputs=[selected_idx, state],
                            outputs=[state, toast_mount, sttm_status])

        manual_search_btn.click(
            on_manual_search, inputs=[manual_query, state],
            outputs=[state, matches_html, stage],
        )
        manual_query.submit(
            on_manual_search, inputs=[manual_query, state],
            outputs=[state, matches_html, stage],
        )

        lock_streak_radio.change(
            on_lock_streak_change, inputs=[lock_streak_radio, state],
            outputs=[state], show_progress=False,
        )
        unlock_btn.click(
            on_unlock_click, inputs=[state],
            outputs=[state, stage, primary_cta, matches_html, toast_mount],
            show_progress=False,
        )
        retrieval_mode_dd.change(
            on_mode_change, inputs=[retrieval_mode_dd, state],
            outputs=[state, stage, primary_cta, matches_html, toast_mount],
        )

        edit_bridge.change(
            on_edit_commit, inputs=[edit_bridge, state],
            outputs=[state, stage, matches_html],
        )

        # Kirtan player wiring
        load_btn.click(
            on_load_source,
            inputs=[player_url, player_file, state],
            outputs=[state, player_preview, player_file, player_msg],
        )
        play_sync_btn.click(
            None, None, None,
            js=(
                "() => {"
                "  const el = document.querySelector('#surt-player-preview audio');"
                "  if (el) { try { el.currentTime = 0; el.play(); } catch(e) {} }"
                "}"
            ),
        )
        play_sync_btn.click(
            on_play_sync,
            inputs=[player_url, auto_push_toggle, state],
            outputs=[state, stage, primary_cta, matches_html, player_msg,
                     sttm_status, player_preview],
        )

        for ctrl in (vad_slider, throttle_slider, commit_slider,
                     max_slider, carry_slider, auto_push_slider, hero_slider):
            ctrl.change(
                on_settings_change,
                inputs=[vad_slider, throttle_slider, commit_slider,
                        max_slider, carry_slider, auto_push_slider,
                        hero_slider, state],
                outputs=[state, stage],
            )

    return demo


def main() -> None:
    backend = load_backend()
    # Pre-warm the local retriever so the first live-mic window isn't a 5-10 s
    # MuRIL load. Blocks startup by the same amount, but predictable.
    try:
        get_retriever()
    except Exception as e:  # noqa: BLE001
        print(f"[surt] retriever pre-warm failed: {e}")
    demo = build_app(backend)
    theme = gr.themes.Base(
        primary_hue="blue", neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"],
    )
    # /tmp/surt_hls is where the URL-stream path writes HLS segments; Gradio
    # needs explicit permission to serve files from there.
    os.makedirs("/tmp/surt_hls", exist_ok=True)
    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1", inbrowser=True, theme=theme,
        css=getattr(demo, "_surt_css", None),
        allowed_paths=["/tmp/surt_hls"],
    )


if __name__ == "__main__":
    main()
