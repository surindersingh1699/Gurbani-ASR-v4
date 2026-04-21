"""Convert a Hugging Face Whisper checkpoint to CTranslate2 for faster-whisper.

Usage:
    python -m apps.live_lab.convert_to_ct2 \
        surindersinghssj/surt-small-v3 \
        ./models/surt-small-v3-int8 \
        --quantization int8

Quantization choices:
    int8            — best for CPU (what we want for the laptop)
    int8_float16    — GPU mixed-precision
    float16         — GPU fp16
    float32         — reference / debugging

If your fine-tuned repo is missing tokenizer files (Surt v3 ships tokenizer.json
only), we fall back to copying them from openai/whisper-small so faster-whisper
can tokenize.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "normalizer.json",
    "added_tokens.json",
    "special_tokens_map.json",
    "preprocessor_config.json",
    "generation_config.json",
]


def _ensure_tokenizer_files(dst: Path, fallback_repo: str) -> None:
    """If CT2 output dir is missing tokenizer files, copy from fallback HF repo."""
    from huggingface_hub import hf_hub_download  # type: ignore

    for fname in TOKENIZER_FILES:
        if (dst / fname).exists():
            continue
        try:
            local = hf_hub_download(repo_id=fallback_repo, filename=fname)
            shutil.copy(local, dst / fname)
            print(f"  + copied {fname} from {fallback_repo}")
        except Exception as e:
            if fname in ("tokenizer.json",):
                print(f"  ! MISSING {fname} (required): {e}", file=sys.stderr)
            else:
                print(f"  . skipped {fname}: {e}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("src", help="HF repo id or local HF checkpoint dir")
    p.add_argument("dst", help="Output directory for the CT2 model")
    p.add_argument(
        "--quantization",
        default="int8",
        choices=["int8", "int8_float16", "float16", "float32"],
    )
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--tokenizer-fallback",
        default="openai/whisper-small",
        help="HF repo to copy tokenizer files from if the source lacks them",
    )
    args = p.parse_args()

    try:
        from ctranslate2.converters import TransformersConverter  # type: ignore
    except ImportError:
        print(
            "ctranslate2 not installed. Run: pip install ctranslate2",
            file=sys.stderr,
        )
        return 1

    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    print(f"Converting {args.src} -> {dst} ({args.quantization}) ...")
    # Drop `copy_files` here — the converter fails if any listed file is
    # missing from the source repo, and we do our own tokenizer backfill
    # below which is more robust.
    conv = TransformersConverter(args.src)
    conv.convert(str(dst), quantization=args.quantization, force=args.force)

    _ensure_tokenizer_files(dst, args.tokenizer_fallback)

    print(f"Done. Point live_lab at: {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
