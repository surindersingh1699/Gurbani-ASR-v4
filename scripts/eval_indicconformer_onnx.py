"""Evaluate a Punjabi IndicConformer CTC ONNX model on JSONL manifests."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
LOG_EPS = 2**-24
_MATRAS = "".join(
    chr(codepoint)
    for codepoint in [
        0x0A3C,
        0x0A41,
        0x0A42,
        0x0A47,
        0x0A48,
        0x0A4B,
        0x0A4C,
        0x0A4D,
        0x0A3E,
        0x0A3F,
        0x0A40,
        0x0A70,
        0x0A71,
        0x0A02,
        0x0A01,
    ]
)
_MATRA_RE = re.compile(f"[{re.escape(_MATRAS)}]")


def normalize_gurbani_text(text: str) -> str:
    text = re.sub(r"॥[੦-੯]+॥", "", text or "")
    text = text.replace("॥", "")
    return re.sub(r"\s+", " ", text).strip()


def strip_matras(text: str) -> str:
    return _MATRA_RE.sub("", text)


def load_tokens(path: Path) -> list[str]:
    tokens = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.rsplit(maxsplit=1)
        tokens.append(parts[0] if len(parts) == 2 else line)
    return tokens


def greedy_decode(log_probs: np.ndarray, tokens: list[str], blank_id: int) -> str:
    decoded = []
    previous = -1
    for token_id in log_probs.argmax(axis=-1).tolist():
        if token_id != previous and token_id != blank_id:
            decoded.append(token_id)
        previous = token_id
    pieces = [tokens[token_id] for token_id in decoded if 0 <= token_id < len(tokens)]
    return normalize_gurbani_text("".join(pieces).replace("\u2581", " "))


def build_mel_extractor():
    import torch
    import torchaudio.transforms as transforms

    return transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=512,
        win_length=400,
        hop_length=160,
        n_mels=80,
        window_fn=torch.hann_window,
        power=2.0,
        mel_scale="slaney",
        norm="slaney",
        center=True,
    )


def load_audio(path: str) -> np.ndarray:
    import soundfile as sf
    import torch
    import torchaudio.functional as audio_functional

    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        audio = audio_functional.resample(
            torch.from_numpy(audio), sample_rate, SAMPLE_RATE
        ).numpy()
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    return audio.astype(np.float32)


def extract_features(audio: np.ndarray, mel_extractor) -> tuple[np.ndarray, np.ndarray]:
    import torch

    mel = mel_extractor(torch.from_numpy(audio))
    log_mel = torch.log(mel + LOG_EPS)
    mean = log_mel.mean(dim=-1, keepdim=True)
    std = log_mel.std(dim=-1, keepdim=True).clamp_min(1e-5)
    features = ((log_mel - mean) / std).unsqueeze(0).numpy().astype(np.float32)
    return features, np.array([features.shape[-1]], dtype=np.int64)


def evaluate_manifest(session, mel_extractor, tokens, manifest: Path, limit: int | None):
    from jiwer import cer, wer

    rows = [json.loads(line) for line in manifest.open() if line.strip()]
    if limit:
        rows = rows[:limit]

    references = []
    hypotheses = []
    audio_seconds = 0.0
    started = time.perf_counter()
    for index, row in enumerate(rows, start=1):
        reference = normalize_gurbani_text(row["text"])
        if not reference:
            continue
        audio = load_audio(row["audio_filepath"])
        features, length = extract_features(audio, mel_extractor)
        log_probs = session.run(
            ["logprobs"], {"audio_signal": features, "length": length}
        )[0][0]
        references.append(reference)
        hypotheses.append(greedy_decode(log_probs, tokens, len(tokens) - 1))
        audio_seconds += len(audio) / SAMPLE_RATE
        if index % 100 == 0:
            print(f"[{manifest.stem}] {index}/{len(rows)}", flush=True)

    elapsed = time.perf_counter() - started
    return {
        "manifest": str(manifest),
        "clips": len(references),
        "audio_hours": round(audio_seconds / 3600, 4),
        "elapsed_seconds": round(elapsed, 3),
        "rtf": round(elapsed / audio_seconds, 5) if audio_seconds else None,
        "wer": round(wer(references, hypotheses) * 100, 3),
        "cer": round(cer(references, hypotheses) * 100, 3),
        "cer_matra_normalized": round(
            cer(
                [strip_matras(text) for text in references],
                [strip_matras(text) for text in hypotheses],
            )
            * 100,
            3,
        ),
        "examples": [
            {"reference": reference, "hypothesis": hypothesis}
            for reference, hypothesis in list(zip(references, hypotheses))[:5]
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--tokens", required=True)
    parser.add_argument(
        "--manifest",
        action="append",
        required=True,
        help="NAME=/path/to/manifest.jsonl",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = max(1, args.threads)
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        args.onnx, sess_options=options, providers=["CPUExecutionProvider"]
    )
    tokens = load_tokens(Path(args.tokens))
    output_shape = session.get_outputs()[0].shape
    if output_shape[-1] != len(tokens):
        raise RuntimeError(
            f"ONNX output has {output_shape[-1]} classes but token file has "
            f"{len(tokens)} entries"
        )

    results = {
        "model": args.onnx,
        "precision": "int8" if "int8" in args.onnx.lower() else "fp32",
        "tokens": args.tokens,
        "threads": args.threads,
        "datasets": {},
    }
    mel_extractor = build_mel_extractor()
    for value in args.manifest:
        name, manifest_path = value.split("=", 1)
        metrics = evaluate_manifest(
            session, mel_extractor, tokens, Path(manifest_path), args.limit
        )
        results["datasets"][name] = metrics
        print(
            f"{name}: WER={metrics['wer']:.3f}% CER={metrics['cer']:.3f}% "
            f"RTF={metrics['rtf']:.5f}",
            flush=True,
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
