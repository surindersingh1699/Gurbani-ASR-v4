"""ASR backends for live_lab.

Primary: faster-whisper (CTranslate2) with INT8 on CPU.
Fallback: transformers (reuses the existing TorchBackend from apps.transcribe).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class ASRSettings:
    beam_size: int = 1
    temperature: float = 0.0
    no_speech_threshold: float = 0.6
    condition_on_previous_text: bool = False
    initial_prompt: str = ""
    language: str = "pa"
    vad_filter: bool = False             # faster-whisper's built-in VAD (separate from our Silero)


class FasterWhisperBackend:
    name = "faster-whisper"

    def __init__(
        self,
        model_path: str,
        compute_type: str = "int8",
        device: str = "cpu",
        cpu_threads: int = 0,
        num_workers: int = 1,
    ):
        from faster_whisper import WhisperModel  # type: ignore

        self.model_path = model_path
        self.compute_type = compute_type
        self.device = device
        self.model = WhisperModel(
            model_path,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            num_workers=num_workers,
        )

    def describe(self) -> str:
        return (
            f"faster-whisper · {self.device} · {self.compute_type} · "
            f"{Path(self.model_path).name}"
        )

    def transcribe(
        self, audio: np.ndarray, sr: int, cfg: ASRSettings
    ) -> tuple[str, dict]:
        if sr != 16000:
            from apps.live_lab.pipeline import resample_to_16k

            audio = resample_to_16k(audio, sr)
        segments, info = self.model.transcribe(
            audio.astype(np.float32),
            beam_size=cfg.beam_size,
            temperature=cfg.temperature,
            language=cfg.language or None,
            task="transcribe",
            condition_on_previous_text=cfg.condition_on_previous_text,
            initial_prompt=cfg.initial_prompt or None,
            vad_filter=cfg.vad_filter,
            no_speech_threshold=cfg.no_speech_threshold,
        )
        parts: list[str] = []
        avg_lp: list[float] = []
        no_speech: list[float] = []
        for seg in segments:
            parts.append(seg.text)
            avg_lp.append(seg.avg_logprob)
            no_speech.append(seg.no_speech_prob)
        text = "".join(parts).strip()
        meta = {
            "avg_logprob": float(np.mean(avg_lp)) if avg_lp else None,
            "no_speech_prob": float(np.mean(no_speech)) if no_speech else None,
            "language_probability": float(getattr(info, "language_probability", 0.0)),
        }
        return text, meta


class TorchFallbackBackend:
    """Stock transformers backend — used when no CT2 model is available.

    Reuses apps.transcribe.backend.TorchBackend to avoid duplicating code.
    """

    name = "transformers-torch"

    def __init__(self, model_id: str, processor_id: str = "openai/whisper-small"):
        from apps.transcribe.backend import TorchBackend  # type: ignore

        self._inner = TorchBackend(model_id, processor_id=processor_id)

    def describe(self) -> str:
        inner = self._inner
        return f"transformers · {inner.device} · {inner.dtype}"

    def transcribe(
        self, audio: np.ndarray, sr: int, cfg: ASRSettings
    ) -> tuple[str, dict]:
        text = self._inner.transcribe(audio, sr)
        return text, {"avg_logprob": None, "no_speech_prob": None,
                      "language_probability": None}


def load_faster_whisper(
    model_path: str,
    compute_type: str = "int8",
    device: str = "cpu",
    cpu_threads: int = 0,
) -> FasterWhisperBackend:
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(
            f"CT2 model directory not found: {model_path}\n\n"
            "Convert your HF checkpoint first:\n"
            "  python -m apps.live_lab.convert_to_ct2 \\\n"
            "      surindersinghssj/surt-small-v3 \\\n"
            f"      {model_path} --quantization int8"
        )
    return FasterWhisperBackend(
        str(p), compute_type=compute_type, device=device, cpu_threads=cpu_threads
    )
