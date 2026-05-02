"""Benchmark mobile-deployable ASR alternatives vs surt-small-v3 baseline.

Compares on the canonical-corrected eval slices:
    surindersinghssj/gurbani-sehajpath-yt-captions-eval-canonical
    surindersinghssj/gurbani-kirtan-yt-captions-eval-canonical

Filter applied: only models with a realistic Mac/Windows/iOS/Android path
(whisper.cpp / WhisperKit / parakeet.cpp / sherpa-onnx / ONNX runtime).

Default backends:
    baseline       — surt-small-v3 via faster-whisper (current production target)
    indicconformer — ai4bharat/indicconformer_stt_pa_hybrid_ctc_rnnt_large
                     (120M Conformer, native Gurmukhi, ONNX→sherpa-onnx mobile path)
    parakeet       — nvidia/parakeet-tdt-0.6b-v3
                     (600M TDT, parakeet.cpp + sherpa-onnx mobile path, fastest CPU)

Opt-in (NOT mobile-deployable, kept for completeness via --only):
    mms            — facebook/mms-1b-all (1B params → too big for iOS app)
    qwen3          — Qwen/Qwen3-ASR-0.6B (LLM-ASR — no mobile inference runtime today)

Per (model, dataset) reports: WER, CER, RTF (CPU), peak RAM, first-decode latency.

The bench is decision-support, not a winner-picker:
- IndicConformer is the only model with Punjabi out-of-box → real WER number.
- Qwen3 outputs Devanagari (Hindi) — we transliterate Devanagari→Gurmukhi
  before WER so the comparison is fair-ish; expect 60-90% WER, that's normal.
- Parakeet has zero Punjabi training — expect ~100% WER, the speed number
  is what matters (it tells us the upper bound on raw CPU throughput).

Usage on RunPod (eval data already cached on NFS):
    export HF_HOME=/workspace/.cache/huggingface
    export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
    python scripts/bench_asr_alternatives.py --device cpu --threads 4

    # Subset of backends if NeMo install fails:
    python scripts/bench_asr_alternatives.py --only baseline,qwen3
"""
from __future__ import annotations

import argparse
import csv
import gc
import os
import re
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from surt.data import normalize_gurbani_text  # noqa: E402

# ──────────────────────────────────────────────────────────────────────────
# Datasets
# ──────────────────────────────────────────────────────────────────────────

EVAL_DATASETS = [
    {
        "name": "sehajpath",
        "hf_id": "surindersinghssj/gurbani-sehajpath-yt-captions-eval-canonical",
        "split": "train",
        "text_col": "final_text",
    },
    {
        "name": "kirtan",
        "hf_id": "surindersinghssj/gurbani-kirtan-yt-captions-eval-canonical",
        "split": "train",
        "text_col": "final_text",
    },
]

TARGET_SR = 16000


# ──────────────────────────────────────────────────────────────────────────
# Devanagari → Gurmukhi transliteration (for Qwen3 output)
# ──────────────────────────────────────────────────────────────────────────
# Code-point parallel mapping. Devanagari U+0900-U+097F maps almost
# 1:1 onto Gurmukhi U+0A00-U+0A7F at the same low byte for shared
# phonemes. Letters that don't exist in Gurmukhi are dropped.
# This is a *crude* romanization-equivalent; good enough for WER ranking
# but NOT for ground-truth canonical comparison.

_DEVA_TO_GURMU = {
    # Vowels (independent forms)
    'अ': 'ਅ', 'आ': 'ਆ', 'इ': 'ਇ', 'ई': 'ਈ', 'उ': 'ਉ', 'ऊ': 'ਊ',
    'ए': 'ਏ', 'ऐ': 'ਐ', 'ओ': 'ਓ', 'औ': 'ਔ',
    # Consonants
    'क': 'ਕ', 'ख': 'ਖ', 'ग': 'ਗ', 'घ': 'ਘ', 'ङ': 'ਙ',
    'च': 'ਚ', 'छ': 'ਛ', 'ज': 'ਜ', 'झ': 'ਝ', 'ञ': 'ਞ',
    'ट': 'ਟ', 'ठ': 'ਠ', 'ड': 'ਡ', 'ढ': 'ਢ', 'ण': 'ਣ',
    'त': 'ਤ', 'थ': 'ਥ', 'द': 'ਦ', 'ध': 'ਧ', 'न': 'ਨ',
    'प': 'ਪ', 'फ': 'ਫ', 'ब': 'ਬ', 'भ': 'ਭ', 'म': 'ਮ',
    'य': 'ਯ', 'र': 'ਰ', 'ल': 'ਲ', 'व': 'ਵ',
    'श': 'ਸ਼', 'ष': 'ਸ਼', 'स': 'ਸ', 'ह': 'ਹ',
    # Vowel signs (matras)
    'ा': 'ਾ', 'ि': 'ਿ', 'ी': 'ੀ', 'ु': 'ੁ', 'ू': 'ੂ',
    'े': 'ੇ', 'ै': 'ੈ', 'ो': 'ੋ', 'ौ': 'ੌ',
    # Diacritics
    '्': '੍',  # virama / halant
    'ं': 'ਂ',  # anusvara → bindi
    'ँ': 'ਁ',  # candrabindu
    'ः': '',    # visarga — no equivalent, drop
    # Nukta variants
    'क़': 'ਕ਼', 'ख़': 'ਖ਼', 'ग़': 'ਗ਼', 'ज़': 'ਜ਼', 'ड़': 'ੜ', 'ढ़': 'ੜ੍ਹ', 'फ़': 'ਫ਼',
    # Digits
    '०': '੦', '१': '੧', '२': '੨', '३': '੩', '४': '੪',
    '५': '੫', '६': '੬', '७': '੭', '८': '੮', '९': '੯',
    # Punctuation
    '।': '।', '॥': '॥',
}


def deva_to_gurmukhi(text: str) -> str:
    """Crude Devanagari → Gurmukhi script-only translit. Not linguistically perfect."""
    if not text:
        return text
    return "".join(_DEVA_TO_GURMU.get(ch, ch) for ch in text)


def looks_like_devanagari(text: str) -> bool:
    if not text:
        return False
    deva = sum(1 for c in text if 0x0900 <= ord(c) <= 0x097F)
    return deva > len(text) * 0.3


# ──────────────────────────────────────────────────────────────────────────
# Backend interface
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class BackendResult:
    text: str
    elapsed_s: float


class Backend:
    name: str = "abstract"
    output_script: str = "gurmukhi"  # or "devanagari" or "latin"

    def transcribe(self, audio: np.ndarray, sr: int) -> BackendResult:
        raise NotImplementedError

    def cleanup(self) -> None:
        pass


# ─── Faster-Whisper baseline ────────────────────────────────────────────

class FasterWhisperBackend(Backend):
    """Mirrors apps/transcribe/backend.py::CT2Backend exactly so the baseline
    number reflects production speed/quality, not a one-off bench config."""
    name = "baseline-faster-whisper"
    output_script = "gurmukhi"

    def __init__(
        self, model_id: str, device: str, threads: int,
        beam_size: int = 5, vad_filter: bool = False,
    ):
        from faster_whisper import WhisperModel  # lazy
        # CT2 doesn't support "mps" — fall back to CPU for it.
        ct2_device = "cuda" if device == "cuda" else "cpu"
        compute_type = "int8" if ct2_device == "cpu" else "float16"
        self.beam_size = int(beam_size)
        self.vad_filter = bool(vad_filter)
        self.model = WhisperModel(
            model_id,
            device=ct2_device,
            compute_type=compute_type,
            cpu_threads=threads,
            num_workers=1,
        )

    def transcribe(self, audio: np.ndarray, sr: int) -> BackendResult:
        assert sr == TARGET_SR, f"expected 16k, got {sr}"
        # VAD off by default for the bench — eval clips are already
        # caption-aligned speech, so VAD is pure overhead AND gives faster-
        # whisper an unfair speed advantage no other backend has. Production
        # (apps/transcribe) keeps it on for live-mic; that's a different goal.
        kwargs: dict = dict(
            beam_size=self.beam_size,
            temperature=0.0,
            language="pa",
            task="transcribe",
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
        )
        if self.vad_filter:
            kwargs["vad_filter"] = True
            kwargs["vad_parameters"] = {"min_silence_duration_ms": 300}
        t0 = time.perf_counter()
        segments, _ = self.model.transcribe(audio.astype(np.float32), **kwargs)
        text = "".join(seg.text for seg in segments)
        return BackendResult(text=text.strip(), elapsed_s=time.perf_counter() - t0)


# ─── AI4Bharat IndicConformer-pa ────────────────────────────────────────

class IndicConformerBackend(Backend):
    name = "ai4bharat-indicconformer-pa"
    output_script = "gurmukhi"

    def __init__(self, device: str, decoder: str = "rnnt", local_path: str | None = None):
        import nemo.collections.asr as nemo_asr  # lazy
        import torch  # lazy
        self._torch = torch
        if local_path:
            # Load from a manually-downloaded .nemo file (gated repos).
            self.model = nemo_asr.models.ASRModel.restore_from(local_path)
        else:
            self.model = nemo_asr.models.ASRModel.from_pretrained(
                "ai4bharat/indicconformer_stt_pa_hybrid_ctc_rnnt_large"
            )
        self.model.freeze()
        self.model = self.model.to(torch.device(device))
        # The "_hybrid_rnnt_" .nemo is RNNT-only; the "_hybrid_ctc_rnnt_" has both.
        # Try requested decoder, fall back to whatever the checkpoint actually has.
        try:
            self.model.cur_decoder = decoder
            self.decoder = decoder
        except Exception:
            print(f"  [info] decoder={decoder!r} unsupported in this checkpoint; using default")
            self.decoder = getattr(self.model, "cur_decoder", "rnnt")

    def transcribe(self, audio: np.ndarray, sr: int) -> BackendResult:
        assert sr == TARGET_SR
        # NeMo ASRModel.transcribe wants file paths.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        try:
            import soundfile as sf
            sf.write(tmp, audio.astype(np.float32), sr, subtype="PCM_16")
            t0 = time.perf_counter()
            kwargs: dict = dict(batch_size=1, language_id="pa")
            if self.decoder == "ctc":
                kwargs["logprobs"] = False
            out = self.model.transcribe([tmp], **kwargs)
            elapsed = time.perf_counter() - t0
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        # NeMo returns shapes that vary by version+decoder:
        #   CTC (older): list[str]
        #   RNNT (older): list[Hypothesis]
        #   RNNT (newer): tuple(list[str], list[str])  — (hypotheses, alignments)
        #   RNNT (newer batch): tuple(list[list[Hypothesis]], list[...])
        # Unwrap to a single str.
        result = out
        if isinstance(result, tuple):
            result = result[0]  # take hypotheses, drop alignments
        if isinstance(result, list) and result:
            result = result[0]
        if isinstance(result, list) and result:
            result = result[0]
        text = result if isinstance(result, str) else getattr(result, "text", str(result))
        return BackendResult(text=text.strip(), elapsed_s=elapsed)


# ─── Qwen3-ASR-0.6B ─────────────────────────────────────────────────────

class Qwen3ASRBackend(Backend):
    name = "qwen3-asr-0.6b"
    output_script = "devanagari"  # forces hi → Devanagari

    def __init__(self, device: str):
        from qwen_asr import Qwen3ASRModel  # lazy
        import torch
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        device_map = device if device.startswith("cuda") else "cpu"
        self.model = Qwen3ASRModel.from_pretrained(
            "Qwen/Qwen3-ASR-0.6B",
            dtype=dtype,
            device_map=device_map,
            max_inference_batch_size=1,
            max_new_tokens=256,
        )

    def transcribe(self, audio: np.ndarray, sr: int) -> BackendResult:
        assert sr == TARGET_SR
        t0 = time.perf_counter()
        # Qwen3-ASR accepts (np.ndarray, sr) tuples directly.
        results = self.model.transcribe(
            audio=(audio.astype(np.float32), sr),
            language="Hindi",  # closest covered language → Devanagari output
        )
        elapsed = time.perf_counter() - t0
        text = results[0].text if results else ""
        return BackendResult(text=text.strip(), elapsed_s=elapsed)


# ─── Facebook MMS-1B-all (wav2vec2-CTC, has Punjabi `pa`) ───────────────

class MMSBackend(Backend):
    name = "facebook-mms-1b-all"
    output_script = "gurmukhi"

    def __init__(self, device: str):
        from transformers import Wav2Vec2ForCTC, AutoProcessor  # lazy
        import torch
        self._torch = torch
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained("facebook/mms-1b-all")
        model = Wav2Vec2ForCTC.from_pretrained(
            "facebook/mms-1b-all",
            target_lang="pan",  # MMS uses ISO-639-3: Punjabi = `pan`
            ignore_mismatched_sizes=True,
            torch_dtype=torch.float32,  # CPU: fp32 is faster than bf16/fp16
        ).to(self.device)
        # Switch the LM head + tokenizer adapter to Punjabi.
        self.processor.tokenizer.set_target_lang("pan")
        model.load_adapter("pan")
        model.requires_grad_(False)
        self.model = model.eval()  # noqa: PT028 — torch eval-mode, not python eval

    def transcribe(self, audio: np.ndarray, sr: int) -> BackendResult:
        assert sr == TARGET_SR
        torch = self._torch
        inputs = self.processor(
            audio.astype(np.float32), sampling_rate=sr, return_tensors="pt"
        )
        input_values = inputs.input_values.to(self.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = self.model(input_values).logits
        pred_ids = torch.argmax(logits, dim=-1)
        text = self.processor.batch_decode(pred_ids)[0]
        elapsed = time.perf_counter() - t0
        return BackendResult(text=text.strip(), elapsed_s=elapsed)


# ─── NVIDIA Parakeet-TDT-0.6B-v3 ────────────────────────────────────────

class ParakeetBackend(Backend):
    """sherpa-onnx INT8 Parakeet — the actual mobile-deploy path (vs the NeMo
    Python loader, which conflicts with AI4Bharat NeMo fork). Loads pre-converted
    encoder/decoder/joiner ONNX files from the sherpa-onnx release tarball:
        https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/
        sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2
    """
    name = "parakeet-tdt-0.6b-v3-onnx"
    output_script = "latin"  # English/EU langs only

    def __init__(self, device: str, model_dir: str | None = None, threads: int = 4):
        import sherpa_onnx  # lazy
        if model_dir is None:
            model_dir = "/tmp/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=f"{model_dir}/encoder.int8.onnx",
            decoder=f"{model_dir}/decoder.int8.onnx",
            joiner=f"{model_dir}/joiner.int8.onnx",
            tokens=f"{model_dir}/tokens.txt",
            num_threads=threads,
            sample_rate=TARGET_SR,
            feature_dim=80,
            decoding_method="greedy_search",
            model_type="nemo_transducer",
        )

    def transcribe(self, audio: np.ndarray, sr: int) -> BackendResult:
        assert sr == TARGET_SR
        t0 = time.perf_counter()
        stream = self.recognizer.create_stream()
        stream.accept_waveform(sr, audio.astype(np.float32))
        self.recognizer.decode_streams([stream])
        text = stream.result.text
        elapsed = time.perf_counter() - t0
        return BackendResult(text=text.strip(), elapsed_s=elapsed)


# ──────────────────────────────────────────────────────────────────────────
# Bench harness
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class RunRow:
    backend: str
    dataset: str
    n_samples: int
    audio_total_s: float
    transcribe_total_s: float
    rtf: float                     # transcribe_total / audio_total
    wer: float
    cer: float
    first_decode_latency_s: float
    peak_rss_mb: float
    output_script: str
    notes: str = ""


def _peak_rss_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def _resample(audio: np.ndarray, sr: int, target: int) -> np.ndarray:
    if sr == target:
        return audio.astype(np.float32)
    import librosa
    return librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=target)


def _normalize_for_wer(text: str, output_script: str) -> str:
    if output_script == "devanagari":
        text = deva_to_gurmukhi(text)
    elif output_script == "latin":
        # Parakeet output won't translit cleanly. Keep raw — WER will hit ~100%.
        pass
    return normalize_gurbani_text(text)


def evaluate(
    backend: Backend,
    dataset_cfg: dict,
    max_samples: int,
    verbose_examples: int = 3,
) -> RunRow:
    from datasets import Audio  # lazy
    from surt.data import _load_dataset_with_retry  # honors SURT_DOWNLOAD_WORKERS
    from jiwer import wer as jwer, cer as jcer

    print(f"\n[{backend.name} × {dataset_cfg['name']}] loading dataset…")
    ds = _load_dataset_with_retry(dataset_cfg["hf_id"], split=dataset_cfg["split"], streaming=False)
    ds = ds.cast_column("audio", Audio(sampling_rate=TARGET_SR))
    if max_samples and max_samples < len(ds):
        ds = ds.select(range(max_samples))
    n = len(ds)
    print(f"  → {n} samples")

    refs: list[str] = []
    hyps: list[str] = []
    audio_total = 0.0
    transcribe_total = 0.0
    first_latency = float("nan")
    started = False

    for i, ex in enumerate(ds):
        ref_raw = ex.get(dataset_cfg["text_col"]) or ""
        if not ref_raw.strip():
            continue
        wav = ex["audio"]["array"]
        sr = ex["audio"]["sampling_rate"]
        if sr != TARGET_SR:
            wav = _resample(np.asarray(wav), sr, TARGET_SR)
            sr = TARGET_SR
        wav = np.asarray(wav, dtype=np.float32)
        audio_total += wav.size / sr

        try:
            res = backend.transcribe(wav, sr)
        except Exception as e:
            print(f"    [warn] {backend.name} failed on sample {i}: {e}")
            continue
        if not started:
            first_latency = res.elapsed_s
            started = True
        transcribe_total += res.elapsed_s

        refs.append(normalize_gurbani_text(ref_raw))
        hyps.append(_normalize_for_wer(res.text, backend.output_script))

        if i < verbose_examples:
            print(f"    REF: {refs[-1][:80]}")
            print(f"    HYP: {hyps[-1][:80]}")
            print(f"    [{res.elapsed_s:.2f}s for {wav.size/sr:.1f}s audio]")
        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{n} done — running RTF={transcribe_total/max(audio_total,1e-6):.2f}×")

    if not refs:
        print(f"  [error] no valid samples for {backend.name} on {dataset_cfg['name']}")
        return RunRow(
            backend=backend.name, dataset=dataset_cfg["name"],
            n_samples=0, audio_total_s=0, transcribe_total_s=0,
            rtf=float("nan"), wer=float("nan"), cer=float("nan"),
            first_decode_latency_s=float("nan"), peak_rss_mb=_peak_rss_mb(),
            output_script=backend.output_script, notes="no_valid_samples",
        )

    w = jwer(refs, hyps) * 100
    c = jcer(refs, hyps) * 100
    rtf = transcribe_total / max(audio_total, 1e-6)
    print(f"  → WER={w:.2f}%  CER={c:.2f}%  RTF={rtf:.3f}×  "
          f"first_latency={first_latency:.2f}s  RSS={_peak_rss_mb():.0f}MB")

    return RunRow(
        backend=backend.name, dataset=dataset_cfg["name"],
        n_samples=len(refs),
        audio_total_s=audio_total,
        transcribe_total_s=transcribe_total,
        rtf=rtf, wer=w, cer=c,
        first_decode_latency_s=first_latency,
        peak_rss_mb=_peak_rss_mb(),
        output_script=backend.output_script,
    )


# ──────────────────────────────────────────────────────────────────────────
# Backend factory
# ──────────────────────────────────────────────────────────────────────────

def build_backend(
    name: str, device: str, threads: int, baseline_id: str,
    beam_size: int = 5, vad_filter: bool = False,
    indicconformer_path: str | None = None,
    indicconformer_decoder: str = "rnnt",
    parakeet_dir: str | None = None,
) -> Backend:
    # Most backends use torch device strings; faster-whisper uses CT2 device.
    # On Apple Silicon (M-series) torch supports "mps" but NeMo does not,
    # so we silently fall back to "cpu" for NeMo backends.
    nemo_device = "cpu" if device == "mps" else device
    torch_device = device  # transformers (MMS, Qwen3) DO support mps
    if name == "baseline":
        return FasterWhisperBackend(
            model_id=baseline_id, device=device, threads=threads,
            beam_size=beam_size, vad_filter=vad_filter,
        )
    if name == "indicconformer":
        return IndicConformerBackend(
            device=nemo_device, decoder=indicconformer_decoder,
            local_path=indicconformer_path,
        )
    if name == "qwen3":
        return Qwen3ASRBackend(device=torch_device)
    if name == "parakeet":
        return ParakeetBackend(device=nemo_device, model_dir=parakeet_dir, threads=threads)
    if name == "mms":
        return MMSBackend(device=torch_device)
    raise ValueError(f"unknown backend: {name}")


# Default set = the two candidate models the user picked: IndicConformer-pa
# (native Gurmukhi, 120M, ONNX→sherpa-onnx mobile path) and Parakeet-TDT
# (600M, parakeet.cpp + sherpa-onnx mobile path, fastest CPU).
# Baseline is opt-in via `--only baseline,indicconformer,parakeet` — production
# numbers are already known.
DEFAULT_BACKENDS = ["indicconformer", "parakeet"]
ALL_BACKENDS = ["baseline", "indicconformer", "parakeet", "mms", "qwen3"]


# ──────────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────────

def write_csv(rows: list[RunRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "backend", "dataset", "n_samples", "audio_total_s",
            "transcribe_total_s", "rtf", "wer_pct", "cer_pct",
            "first_decode_latency_s", "peak_rss_mb", "output_script", "notes",
        ])
        for r in rows:
            w.writerow([
                r.backend, r.dataset, r.n_samples, f"{r.audio_total_s:.2f}",
                f"{r.transcribe_total_s:.2f}", f"{r.rtf:.4f}",
                f"{r.wer:.2f}", f"{r.cer:.2f}",
                f"{r.first_decode_latency_s:.3f}", f"{r.peak_rss_mb:.0f}",
                r.output_script, r.notes,
            ])


def write_markdown(rows: list[RunRow], path: Path, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# ASR Alternatives Benchmark")
    lines.append("")
    lines.append(f"- device: `{args.device}`  threads: `{args.threads}`  "
                 f"max_samples: `{args.max_samples}`")
    lines.append(f"- baseline model: `{args.baseline_id}`  beam: `{args.beam}`  "
                 f"vad: `{'on' if args.vad else 'off'}`")
    lines.append("")
    lines.append("| Backend | Dataset | N | WER% | CER% | RTF (×realtime) | "
                 "First-decode (s) | Peak RAM (MB) | Notes |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|")
    for r in rows:
        lines.append(
            f"| `{r.backend}` | {r.dataset} | {r.n_samples} | "
            f"{r.wer:.1f} | {r.cer:.1f} | {r.rtf:.3f} | "
            f"{r.first_decode_latency_s:.2f} | {r.peak_rss_mb:.0f} | {r.notes} |"
        )
    lines.append("")
    lines.append("**How to read:**")
    lines.append("")
    lines.append("Filter applied: only models with a real Mac/Win/iOS/Android deploy path are in the default set.")
    lines.append("")
    lines.append("- **baseline (faster-whisper surt-small-v3)** — production target. Other rows compete with this. Mobile path: whisper.cpp / WhisperKit / sherpa-onnx (already shipping).")
    lines.append("- **indicconformer-pa** — native Gurmukhi, 120M Conformer CTC. Mobile path: NeMo→ONNX→sherpa-onnx (proven for Conformer).")
    lines.append("- **parakeet-tdt** — zero Punjabi training, WER will be ≈100% — **only the RTF / latency columns matter for this row**. Mobile path: parakeet.cpp / sherpa-onnx (exists).")
    lines.append("- **mms** (opt-in) — has Punjabi but 1GB+ quantized → too big for an iOS app.")
    lines.append("- **qwen3** (opt-in) — LLM-based ASR, no mobile inference runtime today.")
    lines.append("")
    lines.append("**Decision rule:**")
    lines.append("")
    lines.append("- If `indicconformer` WER < `baseline` WER on kirtan → switch architectures (and fine-tune indicconformer on v3).")
    lines.append("- Else if `qwen3` RTF < `baseline` RTF AND WER < ~85% → fine-tune qwen3 on v3.")
    lines.append("- Else if `parakeet` RTF << `baseline` RTF → invest in tokenizer-extension work for Gurmukhi.")
    lines.append("- Else → stay on Whisper-small architecture, focus on data scale (the v3 plan).")
    path.write_text("\n".join(lines) + "\n")


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"],
                        help="cpu (default) is the bench target. mps = Apple Silicon "
                             "(M-series) GPU for transformers backends; NeMo backends "
                             "auto-fall-back to cpu when device=mps.")
    parser.add_argument("--threads", type=int, default=4,
                        help="CPU threads for faster-whisper / torch. M5 Pro: try 8.")
    parser.add_argument("--max-samples", type=int, default=100,
                        help="Per-dataset cap. 100 ranks reliably; raise for tighter CIs.")
    parser.add_argument("--only", default="",
                        help=f"Comma list, subset of {ALL_BACKENDS}. "
                             f"Empty = mobile-deployable defaults: {DEFAULT_BACKENDS}. "
                             "Use --only all to include MMS+Qwen3 (not mobile-deployable).")
    parser.add_argument("--baseline-id", default="surindersinghssj/surt-small-v3",
                        help="HF repo id for the faster-whisper baseline. "
                             "Default matches apps/transcribe/backend.py production.")
    parser.add_argument("--beam", type=int, default=5,
                        help="Beam size for the faster-whisper baseline. "
                             "CTC/TDT backends are greedy by design — flag has no effect on them.")
    parser.add_argument("--vad", action="store_true",
                        help="Enable Silero VAD on faster-whisper. OFF by default — "
                             "the eval clips are pre-trimmed caption-aligned speech, and "
                             "VAD is a faster-whisper-only feature so leaving it on would "
                             "give an unfair speed advantage.")
    parser.add_argument("--out-dir", default="bench_results",
                        help="Output directory for CSV + markdown.")
    parser.add_argument("--indicconformer-path", default=None,
                        help="Path to a manually-downloaded .nemo file (for gated repos). "
                             "If unset, loads from HF hub (requires `huggingface-cli login`).")
    parser.add_argument("--indicconformer-decoder", default="rnnt", choices=["rnnt", "ctc"],
                        help="Which decoder head to use. RNNT is more accurate but slower; "
                             "CTC is faster. The hybrid_rnnt-only .nemo file only has RNNT.")
    parser.add_argument("--parakeet-dir", default=None,
                        help="Path to extracted sherpa-onnx Parakeet INT8 directory "
                             "(contains encoder.int8.onnx, decoder.int8.onnx, joiner.int8.onnx, "
                             "tokens.txt). Default: /tmp/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8")
    args = parser.parse_args()

    if args.device == "cpu":
        # CPU optimization knobs — set BEFORE any heavy import so libraries
        # pick them up at thread-pool initialization.
        os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(args.threads))
        os.environ.setdefault("NUMEXPR_NUM_THREADS", str(args.threads))
        os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
        os.environ.setdefault("KMP_BLOCKTIME", "0")
        try:
            import torch
            torch.set_num_threads(args.threads)
            torch.set_num_interop_threads(max(1, args.threads // 2))
            torch.set_grad_enabled(False)
            # fbgemm = x86 default; qnnpack on ARM macs.
            try:
                torch.backends.quantized.engine = (
                    "qnnpack" if sys.platform == "darwin" else "fbgemm"
                )
            except Exception:
                pass
        except ImportError:
            pass

    if args.only.strip().lower() == "all":
        selected = ALL_BACKENDS
    else:
        selected = [b.strip() for b in args.only.split(",") if b.strip()] or DEFAULT_BACKENDS
    unknown = [b for b in selected if b not in ALL_BACKENDS]
    if unknown:
        print(f"unknown backend(s): {unknown}. valid: {ALL_BACKENDS}")
        return 2

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir) / timestamp
    print(f"[bench] writing results to {out_dir}")
    print(f"[bench] backends: {selected}  device={args.device}  threads={args.threads}")

    rows: list[RunRow] = []

    for backend_name in selected:
        print(f"\n{'='*70}\n[bench] BACKEND: {backend_name}\n{'='*70}")
        try:
            backend = build_backend(
                backend_name, args.device, args.threads, args.baseline_id,
                beam_size=args.beam, vad_filter=args.vad,
                indicconformer_path=args.indicconformer_path,
                indicconformer_decoder=args.indicconformer_decoder,
                parakeet_dir=args.parakeet_dir,
            )
        except Exception as e:
            print(f"  [skip] failed to load {backend_name}: {e}")
            traceback.print_exc()
            for ds_cfg in EVAL_DATASETS:
                rows.append(RunRow(
                    backend=backend_name, dataset=ds_cfg["name"], n_samples=0,
                    audio_total_s=0, transcribe_total_s=0,
                    rtf=float("nan"), wer=float("nan"), cer=float("nan"),
                    first_decode_latency_s=float("nan"),
                    peak_rss_mb=_peak_rss_mb(),
                    output_script="?", notes=f"load_failed: {type(e).__name__}",
                ))
            continue

        for ds_cfg in EVAL_DATASETS:
            try:
                rows.append(evaluate(backend, ds_cfg, args.max_samples))
            except Exception as e:
                print(f"  [error] eval failed: {e}")
                traceback.print_exc()
                rows.append(RunRow(
                    backend=backend_name, dataset=ds_cfg["name"], n_samples=0,
                    audio_total_s=0, transcribe_total_s=0,
                    rtf=float("nan"), wer=float("nan"), cer=float("nan"),
                    first_decode_latency_s=float("nan"),
                    peak_rss_mb=_peak_rss_mb(),
                    output_script=backend.output_script,
                    notes=f"eval_failed: {type(e).__name__}",
                ))

        backend.cleanup()
        del backend
        gc.collect()
        try:
            import torch
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
        except ImportError:
            pass

    csv_path = out_dir / "results.csv"
    md_path = out_dir / "summary.md"
    write_csv(rows, csv_path)
    write_markdown(rows, md_path, args)
    print(f"\n[bench] DONE. CSV → {csv_path}\n[bench] DONE. MD  → {md_path}")
    print("\n" + md_path.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
