"""Audio preprocessing, VAD and segmentation for the live_lab app.

Every block is toggleable from the UI so we can A/B combinations on kirtan,
where a standard speech VAD tends to reject sung voice over harmonium.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

SR = 16_000
SILERO_FRAME = 512  # Silero expects 512 samples at 16 kHz (= 32 ms)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass
class PreprocessSettings:
    highpass_hz: float = 80.0        # 0 disables the filter
    normalize: bool = True
    target_rms_dbfs: float = -20.0
    max_gain_db: float = 20.0


@dataclass
class VADSettings:
    # "off"    — no gating, every frame treated as speech
    # "silero" — Silero VAD probability >= threshold
    # "energy" — frame RMS in dBFS >= threshold
    kind: Literal["off", "silero", "energy"] = "silero"
    silero_threshold: float = 0.5
    energy_threshold_dbfs: float = -40.0


@dataclass
class SegmenterSettings:
    # "vad"     — close segment after min_silence_s of non-speech (or max_segment_s)
    # "fixed"   — emit every max_segment_s regardless of content
    # "rolling" — commit-and-carry (lower latency, reprocesses boundary)
    mode: Literal["vad", "fixed", "rolling"] = "vad"
    max_segment_s: float = 15.0
    min_segment_s: float = 1.0
    min_silence_s: float = 0.6
    pre_roll_s: float = 0.2
    # rolling-mode only
    rolling_commit_s: float = 8.0
    rolling_carry_s: float = 1.5
    rolling_max_window_s: float = 11.0


# ---------------------------------------------------------------------------
# Preprocessing: high-pass + RMS normalize
# ---------------------------------------------------------------------------


class HighPassFilter:
    """2nd-order Butterworth HPF with persistent filter state across chunks."""

    def __init__(self, cutoff_hz: float):
        self._sos = None
        self._zi = None
        self.configure(cutoff_hz)

    def configure(self, cutoff_hz: float) -> None:
        from scipy.signal import butter, sosfilt_zi

        self.cutoff = float(cutoff_hz)
        if self.cutoff > 0 and self.cutoff < SR / 2:
            self._sos = butter(2, self.cutoff, btype="highpass", fs=SR, output="sos")
            self._zi = sosfilt_zi(self._sos) * 0.0
        else:
            self._sos = None
            self._zi = None

    def process(self, x: np.ndarray) -> np.ndarray:
        if self._sos is None or x.size == 0:
            return x
        from scipy.signal import sosfilt

        y, self._zi = sosfilt(self._sos, x, zi=self._zi)
        return y.astype(np.float32)


class Preprocessor:
    def __init__(self, settings: PreprocessSettings):
        self.cfg = settings
        self.hpf = HighPassFilter(settings.highpass_hz)

    def configure(self, settings: PreprocessSettings) -> None:
        if settings.highpass_hz != self.cfg.highpass_hz:
            self.hpf.configure(settings.highpass_hz)
        self.cfg = settings

    def process(self, x: np.ndarray) -> tuple[np.ndarray, dict]:
        if x.size == 0:
            return x, {"rms_dbfs": -120.0, "peak_dbfs": -120.0, "gain_db": 0.0}
        x = x.astype(np.float32, copy=False)
        x = self.hpf.process(x)

        rms = float(np.sqrt(np.mean(x * x) + 1e-12))
        peak = float(np.max(np.abs(x)) + 1e-12)
        rms_dbfs = 20.0 * np.log10(rms + 1e-12)
        peak_dbfs = 20.0 * np.log10(peak)
        gain_db = 0.0

        if self.cfg.normalize and rms > 1e-6:
            gain_db = min(self.cfg.target_rms_dbfs - rms_dbfs, self.cfg.max_gain_db)
            gain = 10.0 ** (gain_db / 20.0)
            x = x * gain
            p = float(np.max(np.abs(x)))
            if p > 0.95:
                x = x * (0.95 / p)

        return x, {"rms_dbfs": rms_dbfs, "peak_dbfs": peak_dbfs, "gain_db": gain_db}


# ---------------------------------------------------------------------------
# VAD
# ---------------------------------------------------------------------------


class SileroVAD:
    """Cached Silero VAD — shared model, per-instance RNN state."""

    _shared_model = None
    _torch = None

    def __init__(self):
        if SileroVAD._shared_model is None:
            import torch  # type: ignore

            model, _ = torch.hub.load(
                "snakers4/silero-vad", "silero_vad", trust_repo=True
            )
            model.train(False)  # equivalent to .eval() — inference-only
            SileroVAD._shared_model = model
            SileroVAD._torch = torch
        self.model = SileroVAD._shared_model
        self.torch = SileroVAD._torch
        self.reset()

    def reset(self) -> None:
        if hasattr(self.model, "reset_states"):
            self.model.reset_states()

    def probability(self, frame_512: np.ndarray) -> float:
        with self.torch.no_grad():
            t = self.torch.from_numpy(frame_512.astype(np.float32))
            return float(self.model(t, SR).item())


def energy_dbfs(frame: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(frame * frame) + 1e-12))
    return 20.0 * np.log10(rms + 1e-12)


# ---------------------------------------------------------------------------
# Segmenter
# ---------------------------------------------------------------------------


@dataclass
class SegmentEvent:
    audio: np.ndarray
    reason: str
    vad_max: Optional[float]
    duration_s: float


class Segmenter:
    """Accumulates audio and emits closed segments per the configured mode.

    Holds a small pre-roll ring so segments don't clip leading consonants.
    """

    def __init__(
        self,
        seg_cfg: SegmenterSettings,
        vad_cfg: VADSettings,
        vad: Optional[SileroVAD],
    ):
        self.seg_cfg = seg_cfg
        self.vad_cfg = vad_cfg
        self.vad = vad
        self.buffer = np.zeros(0, dtype=np.float32)
        self.preroll = np.zeros(0, dtype=np.float32)
        self._silence_samples = 0
        self._last_vad_prob = 0.0
        self._max_vad_in_segment = 0.0
        self._in_speech = False

    def reset(self) -> None:
        self.buffer = np.zeros(0, dtype=np.float32)
        self.preroll = np.zeros(0, dtype=np.float32)
        self._silence_samples = 0
        self._last_vad_prob = 0.0
        self._max_vad_in_segment = 0.0
        self._in_speech = False
        if self.vad is not None:
            self.vad.reset()

    def _frame_is_speech(self, frame: np.ndarray) -> tuple[bool, float]:
        """Returns (is_speech, probability_or_db)."""
        if self.vad_cfg.kind == "off":
            return True, 1.0
        if self.vad_cfg.kind == "silero" and self.vad is not None:
            p = self.vad.probability(frame)
            return p >= self.vad_cfg.silero_threshold, p
        if self.vad_cfg.kind == "energy":
            db = energy_dbfs(frame)
            return db >= self.vad_cfg.energy_threshold_dbfs, db
        return True, 1.0

    def _roll_preroll(self, chunk: np.ndarray) -> None:
        pre_n = int(self.seg_cfg.pre_roll_s * SR)
        if pre_n <= 0:
            self.preroll = np.zeros(0, dtype=np.float32)
            return
        combined = (
            np.concatenate([self.preroll, chunk]) if self.preroll.size else chunk
        )
        if combined.size > pre_n:
            self.preroll = combined[-pre_n:].copy()
        else:
            self.preroll = combined.copy()

    def push(self, chunk: np.ndarray) -> list[SegmentEvent]:
        events: list[SegmentEvent] = []
        if chunk.size == 0:
            return events

        mode = self.seg_cfg.mode
        max_samples = int(self.seg_cfg.max_segment_s * SR)
        min_samples = int(self.seg_cfg.min_segment_s * SR)
        min_silence_samples = int(self.seg_cfg.min_silence_s * SR)

        if mode == "fixed":
            self.buffer = (
                np.concatenate([self.buffer, chunk]) if self.buffer.size else chunk
            )
            while self.buffer.size >= max_samples:
                seg = self.buffer[:max_samples].copy()
                self.buffer = self.buffer[max_samples:]
                events.append(SegmentEvent(seg, "fixed-window", None, seg.size / SR))
            return events

        if mode == "rolling":
            self.buffer = (
                np.concatenate([self.buffer, chunk]) if self.buffer.size else chunk
            )
            cap = int(self.seg_cfg.rolling_max_window_s * SR)
            commit = int(self.seg_cfg.rolling_commit_s * SR)
            carry = int(self.seg_cfg.rolling_carry_s * SR)
            if self.buffer.size > cap:
                seg = self.buffer[:commit].copy()
                self.buffer = self.buffer[-carry:].copy()
                events.append(SegmentEvent(seg, "rolling-commit", None, seg.size / SR))
            return events

        # --- VAD mode ---
        frame = SILERO_FRAME
        n_full = (chunk.size // frame) * frame
        tail = chunk[n_full:] if n_full < chunk.size else np.zeros(0, dtype=np.float32)

        frame_probs: list[float] = []
        for i in range(0, n_full, frame):
            f = chunk[i : i + frame]
            is_speech, prob = self._frame_is_speech(f)
            frame_probs.append(prob)

            if not self._in_speech:
                self._roll_preroll(f)
                if is_speech:
                    self._in_speech = True
                    self._silence_samples = 0
                    self._max_vad_in_segment = prob
                    self.buffer = (
                        np.concatenate([self.preroll, f])
                        if self.preroll.size
                        else f.copy()
                    )
                continue

            self.buffer = np.concatenate([self.buffer, f])
            self._max_vad_in_segment = max(self._max_vad_in_segment, prob)
            if is_speech:
                self._silence_samples = 0
            else:
                self._silence_samples += frame

            close_on_silence = (
                self.buffer.size >= min_samples
                and self._silence_samples >= min_silence_samples
            )
            close_on_length = self.buffer.size >= max_samples
            if close_on_silence or close_on_length:
                seg = self.buffer.copy()
                reason = "silence-closed" if close_on_silence else "max-length"
                events.append(
                    SegmentEvent(seg, reason, self._max_vad_in_segment, seg.size / SR)
                )
                self.buffer = np.zeros(0, dtype=np.float32)
                self._silence_samples = 0
                self._max_vad_in_segment = 0.0
                self._in_speech = False
                self._roll_preroll(f)

        if tail.size:
            if self._in_speech:
                self.buffer = np.concatenate([self.buffer, tail])
            else:
                self._roll_preroll(tail)

        if frame_probs:
            self._last_vad_prob = float(frame_probs[-1])
        return events

    @property
    def active_seconds(self) -> float:
        return self.buffer.size / SR

    @property
    def last_vad_prob(self) -> float:
        return self._last_vad_prob


# ---------------------------------------------------------------------------
# Input normalization
# ---------------------------------------------------------------------------


def to_mono_float32(y: np.ndarray) -> np.ndarray:
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32, copy=False)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1.5:  # Gradio int16 container
        y = y / 32768.0
    return y


def resample_to_16k(y: np.ndarray, sr_in: int) -> np.ndarray:
    if sr_in == SR or y.size == 0:
        return y
    n = int(round(len(y) * SR / sr_in))
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, num=len(y), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n, endpoint=False)
    return np.interp(x_new, x_old, y).astype(np.float32)
