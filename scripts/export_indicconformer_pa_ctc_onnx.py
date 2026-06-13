"""Export the Punjabi CTC head from a multilingual IndicConformer checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nemo", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--language-id", default="pa")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    import torch
    from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
    from nemo.collections.asr.modules.conv_asr import ConvASRDecoder

    output_dir = Path(args.output_dir)
    fp32_dir = output_dir / "fp32"
    fp32_dir.mkdir(parents=True, exist_ok=True)

    model = EncDecHybridRNNTCTCBPEModel.restore_from(args.nemo, map_location="cpu")
    model.change_decoding_strategy(decoder_type="ctc", lang_id=args.language_id)

    multilingual_decoder = model.ctc_decoder
    language_mask = multilingual_decoder.language_masks[args.language_id]
    language_indices = torch.tensor(
        [index for index, enabled in enumerate(language_mask) if enabled],
        dtype=torch.long,
    )
    tokenizer = model.tokenizer.tokenizers_dict[args.language_id]
    expected_classes = tokenizer.vocab_size + 1
    if language_indices.numel() != expected_classes:
        raise RuntimeError(
            f"Expected {expected_classes} masked classes, got "
            f"{language_indices.numel()}"
        )

    punjabi_decoder = ConvASRDecoder(
        feat_in=multilingual_decoder._feat_in,
        num_classes=tokenizer.vocab_size,
        vocabulary=list(tokenizer.vocab),
        multisoftmax=False,
    )
    with torch.no_grad():
        source_layer = multilingual_decoder.decoder_layers[0]
        target_layer = punjabi_decoder.decoder_layers[0]
        target_layer.weight.copy_(source_layer.weight.index_select(0, language_indices))
        target_layer.bias.copy_(source_layer.bias.index_select(0, language_indices))

    # Prove that the compact head is numerically identical to NeMo's language mask.
    sample = torch.randn(2, multilingual_decoder._feat_in, 19)
    with torch.no_grad():
        expected = multilingual_decoder(
            encoder_output=sample,
            language_ids=[args.language_id, args.language_id],
        )
        actual = punjabi_decoder(encoder_output=sample)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    model.ctc_decoder = punjabi_decoder
    model.eval()
    onnx_path = fp32_dir / "indicconformer-pa-ctc.onnx"
    model.export(
        str(onnx_path),
        onnx_opset_version=args.opset,
        check_trace=False,
    )

    tokens_path = output_dir / "tokens.txt"
    with tokens_path.open("w", encoding="utf-8") as handle:
        for token_id, token in enumerate(tokenizer.vocab):
            handle.write(f"{token} {token_id}\n")
        handle.write(f"<blank> {tokenizer.vocab_size}\n")

    print(f"Exported {onnx_path}")
    print(f"Wrote {tokens_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
