"""Pluggable ASR backends for the Surt v3 live-transcribe UI.

Backend priority (overridable via `SURT_BACKEND=ct2|torch|mlx|auto`):

    auto → CT2 (faster-whisper INT8 on CPU) if model dir exists
         → else MLX if SURT_MLX_DIR points at a converted repo
         → else Torch (transformers on MPS/CUDA/CPU)

- `CT2Backend`   — faster-whisper (CTranslate2) INT8 on CPU. Best on Apple
  Silicon: typically 2-3× faster than torch MPS for Whisper-small because
  MPS has limited op coverage. Model built by
  `python -m apps.live_lab.convert_to_ct2`.
- `MLXBackend`   — optional, only usable when a pre-converted MLX repo is
  provided via `SURT_MLX_DIR`.
- `TorchBackend` — stock `transformers` on Apple-Silicon `mps` (or
  `cuda`/`cpu`). Fallback.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MODEL_ID = os.environ.get("SURT_MODEL_ID", "surindersinghssj/surt-small-v3")
PROCESSOR_ID = os.environ.get("SURT_PROCESSOR_ID", "openai/whisper-small")
MLX_DIR_ENV = os.environ.get("SURT_MLX_DIR")  # only used if set to a real path
# CT2 model dir — default to the path live_lab's convert_to_ct2 writes to.
CT2_DIR_ENV = os.environ.get(
    "SURT_CT2_DIR", str(Path.home() / "models" / "surt-small-v3-int8")
)
CT2_COMPUTE = os.environ.get("SURT_CT2_COMPUTE", "int8")  # int8 | int8_float16 | float16
BACKEND_PREF = os.environ.get("SURT_BACKEND", "auto").lower()  # auto | ct2 | mlx | torch


@dataclass
class Transcription:
    text: str
    backend: str
    latency_ms: float


class CT2Backend:
    """faster-whisper (CTranslate2) INT8 on CPU. Fastest path on Apple Silicon."""
    name = "ct2"

    def __init__(
        self,
        model_path: str,
        compute_type: str = "int8",
        device: str = "cpu",
        cpu_threads: int = 0,
        num_workers: int = 1,
        language: str = "pa",
        beam_size: int = 1,
    ):
        from faster_whisper import WhisperModel  # type: ignore

        self.model_path = model_path
        self.compute_type = compute_type
        self.device = device
        self.language = language
        self.beam_size = int(beam_size)
        self.model = WhisperModel(
            model_path,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            num_workers=num_workers,
        )
        self.dtype = compute_type  # for display parity with TorchBackend

    def transcribe(self, audio: np.ndarray, sr: int) -> str:
        if sr != 16000:
            audio = _resample_to_16k(audio, sr)
        segments, _ = self.model.transcribe(
            audio.astype(np.float32),
            beam_size=self.beam_size,
            temperature=0.0,
            language=self.language,
            task="transcribe",
            condition_on_previous_text=False,
            # Silero VAD — ships with faster-whisper. Skips non-speech regions
            # so short/silent live-mic buffers don't get hallucinated text
            # ("ਵਾਹਿਗੁਰੂ ਵਾਹਿਗੁਰੂ …" loops on quiet browser mic audio).
            # We DON'T use temperature fallback or no_repeat_ngram_size here —
            # both harm legitimate Gurbani repetition ("ਮੇਰੀ ਪ੍ਰੀਤਿ ਮੇਰੀ ਪ੍ਰੀਤਿ"
            # is canonical, not a bug), and higher temperatures degrade script
            # into mixed-script garbage on this small model.
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
            no_speech_threshold=0.6,
        )
        return "".join(seg.text for seg in segments).strip()


class MLXBackend:
    name = "mlx"

    def __init__(self, mlx_dir: Path):
        import mlx_whisper  # type: ignore

        self._mlx_whisper = mlx_whisper
        self._path = str(mlx_dir)

    def transcribe(self, audio: np.ndarray, sr: int) -> str:
        if sr != 16000:
            audio = _resample_to_16k(audio, sr)
        result = self._mlx_whisper.transcribe(
            audio.astype(np.float32),
            path_or_hf_repo=self._path,
            language="pa",
            task="transcribe",
            temperature=0.0,
            condition_on_previous_text=False,
            fp16=True,
            verbose=None,
        )
        return (result.get("text") or "").strip()


class TorchBackend:
    name = "torch"

    def __init__(self, model_id: str, processor_id: str = PROCESSOR_ID):
        import torch  # type: ignore
        from transformers import (  # type: ignore
            WhisperForConditionalGeneration,
            WhisperProcessor,
        )

        if torch.backends.mps.is_available():
            self.device = "mps"
            self.dtype = torch.float16
        elif torch.cuda.is_available():
            self.device = "cuda"
            self.dtype = torch.float16
        else:
            self.device = "cpu"
            self.dtype = torch.float32

        self._torch = torch
        # The v3 repo ships tokenizer.json only (no vocab.json/merges.txt), so
        # the non-fast WhisperTokenizer can't load from it. Load processor from
        # the base model — tokenizer is identical, just needs the vocab files.
        self.processor = self._load_processor(WhisperProcessor, model_id, processor_id)
        model = WhisperForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=self.dtype
        ).to(self.device)
        model.generation_config.language = "punjabi"
        model.generation_config.task = "transcribe"
        model.generation_config.forced_decoder_ids = None
        model.generation_config.max_length = 225
        model.eval()
        self.model = model

    def transcribe(self, audio: np.ndarray, sr: int) -> str:
        feats = self.processor(
            audio.astype(np.float32), sampling_rate=sr, return_tensors="pt"
        ).input_features.to(self.device, self.dtype)
        with self._torch.inference_mode():
            ids = self.model.generate(feats, max_length=225, num_beams=1)
        return self.processor.tokenizer.batch_decode(ids, skip_special_tokens=True)[0].strip()

    @staticmethod
    def _load_processor(WhisperProcessor, model_id: str, processor_id: str):
        last_err = None
        for repo, use_fast in ((model_id, True), (processor_id, True), (processor_id, False)):
            try:
                return WhisperProcessor.from_pretrained(
                    repo, language="punjabi", task="transcribe", use_fast=use_fast
                )
            except Exception as e:  # noqa: BLE001
                last_err = e
                print(f"[surt] processor load failed from {repo} (use_fast={use_fast}): {e}")
        raise RuntimeError(f"Could not load WhisperProcessor: {last_err}")


def _resample_to_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    if sr == 16000:
        return audio
    ratio = 16000 / sr
    n = int(round(len(audio) * ratio))
    x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def load_backend() -> object:
    # 1) CT2 (faster-whisper INT8) — preferred when the converted model exists.
    if BACKEND_PREF in ("auto", "ct2"):
        ct2_dir = Path(CT2_DIR_ENV)
        if ct2_dir.exists():
            try:
                backend = CT2Backend(str(ct2_dir), compute_type=CT2_COMPUTE)
                print(
                    f"[surt] Using CT2 backend ({ct2_dir.name}) · "
                    f"device={backend.device} · compute={backend.compute_type}"
                )
                return backend
            except Exception as e:  # noqa: BLE001
                if BACKEND_PREF == "ct2":
                    raise
                print(f"[surt] CT2 backend failed ({e}); trying next option.")
        elif BACKEND_PREF == "ct2":
            raise RuntimeError(
                f"SURT_CT2_DIR does not exist: {ct2_dir}\n"
                "Convert your HF checkpoint first:\n"
                "  python -m apps.live_lab.convert_to_ct2 \\\n"
                f"      {MODEL_ID} {ct2_dir} --quantization int8"
            )

    # 2) MLX — opt-in via SURT_MLX_DIR
    if BACKEND_PREF in ("auto", "mlx") and MLX_DIR_ENV:
        mlx_dir = Path(MLX_DIR_ENV)
        if mlx_dir.exists():
            try:
                print(f"[surt] Using MLX backend ({mlx_dir})")
                return MLXBackend(mlx_dir)
            except Exception as e:  # noqa: BLE001
                if BACKEND_PREF == "mlx":
                    raise
                print(f"[surt] MLX backend failed ({e}); falling back to transformers.")
        elif BACKEND_PREF == "mlx":
            raise RuntimeError(f"SURT_MLX_DIR does not exist: {mlx_dir}")

    # 3) Torch (transformers on MPS/CUDA/CPU) — always-available fallback.
    backend = TorchBackend(MODEL_ID)
    print(f"[surt] Using torch backend on device={backend.device} dtype={backend.dtype}")
    return backend
