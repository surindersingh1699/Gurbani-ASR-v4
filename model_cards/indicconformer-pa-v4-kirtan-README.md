---
language:
- pa
license: cc-by-4.0
library_name: nemo
pipeline_tag: automatic-speech-recognition
tags:
- automatic-speech-recognition
- speech
- audio
- conformer
- rnnt
- ctc
- onnx
- int8
- gurbani
- kirtan
- gurmukhi
- punjabi
- nemo
- indicconformer
base_model: ai4bharat/indicconformer_stt_pa_hybrid_ctc_rnnt_large
metrics:
- cer
- wer
---

# IndicConformer-pa v4 - Gurbani / Kirtan ASR

Fine-tuned IndicConformer-pa hybrid CTC/RNNT model for Punjabi Gurbani and
kirtan transcription in Gurmukhi. **CTC is the recommended decoder** for this
release because it provides the best character accuracy and has portable FP32
and INT8 ONNX exports.

## Files

| Artifact | Path | Size | Use |
|---|---|---:|---|
| NeMo hybrid CTC/RNNT | `indicconformer-pa-v4-kirtan.nemo` | 499 MB | GPU/NeMo inference and further training |
| Punjabi CTC ONNX FP32 | `onnx-pa-only/fp32/indicconformer-pa-ctc.onnx` | 460 MB | Maximum ONNX fidelity |
| Punjabi CTC ONNX INT8 | `onnx-pa-only/int8/indicconformer-pa-ctc.onnx` | 178 MB | Recommended CPU deployment |
| Punjabi BPE vocabulary | `onnx-pa-only/tokens.txt` | 257 tokens | Greedy CTC decoding |
| Checksums | `onnx-pa-only/SHA256SUMS` | - | Artifact verification |

The ONNX graphs contain only the Punjabi 256-piece CTC head plus the blank
token. The INT8 graph dynamically quantizes 154 `MatMul` operations while
keeping convolutions in FP32 for broad ONNX Runtime compatibility.

## Evaluation

Text is normalized by removing Gurbani verse-number markers and collapsing
whitespace. Results use greedy CTC decoding without an external language model.

| Evaluation set | Clips | Audio | FP32 CER | INT8 CER | FP32 WER | INT8 WER |
|---|---:|---:|---:|---:|---:|---:|
| Kirtan held-out, video-level leak-free | 1,560 | 2.70 h | **12.98%** | **13.01%** | 32.68% | 32.87% |
| YouTube kirtan captions | 573 | 1.43 h | 34.77% | 34.89% | 72.24% | 72.14% |
| Sehaj Path recitation | 444 | 0.96 h | 11.12% | **11.09%** | 38.10% | **37.94%** |

The INT8 model reduces ONNX size by about 61% with no meaningful accuracy
loss. The YouTube-caption set is substantially noisier and contains less exact
references than the held-out kirtan and Sehaj Path sets, so it should not be
compared directly with the leak-free kirtan headline.

### NeMo decoder comparison

On the 1,560-clip held-out kirtan set:

| Decoder | CER | WER |
|---|---:|---:|
| CTC | **12.98%** | 32.83% |
| RNNT | 13.74% | **26.86%** |

CTC is preferred when character accuracy and ONNX deployment matter. RNNT
remains useful when word-level accuracy is the priority.

Full machine-readable reports are available in:

- `onnx-pa-only/fp32_eval.json`
- `onnx-pa-only/int8_eval.json`
- [training run logs](https://huggingface.co/datasets/surindersinghssj/indicconformer-pa-v4-kirtan-runlogs)

## Training

- Base: `ai4bharat/indicconformer_stt_pa_hybrid_ctc_rnnt_large`
- Objective: hybrid RNNT 0.7 + CTC 0.3
- Data: approximately 623,000 Punjabi clips from kirtan and bani recitation
- Augmentation: SpecAugment, speed, gain, white noise, additive noise, and RIR
- Hardware: one NVIDIA A40
- Early stopping: global step 7,500
- Best monitored validation WER: 0.14225

## NeMo usage

```python
import nemo.collections.asr as nemo_asr

model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(
    "indicconformer-pa-v4-kirtan.nemo"
)
model.change_decoding_strategy(decoder_type="ctc", lang_id="pa")
hypotheses = model.transcribe(
    ["clip.wav"],
    batch_size=1,
    language_id="pa",
)
print(hypotheses[0])
```

Use 16 kHz mono audio. The source NeMo checkpoint retains its multilingual
aggregate tokenizer, so Punjabi language ID `pa` must be supplied.

## ONNX usage

The ONNX input is a NeMo-compatible, per-feature normalized log-mel tensor:

- `audio_signal`: float32 `[batch, 80, time]`
- `length`: int64 `[batch]`
- `logprobs`: float32 `[batch, encoded_time, 257]`

Decode by taking `argmax`, collapsing repeated IDs, removing blank ID 256,
joining token pieces, and replacing SentencePiece `▁` with spaces.

## Limitations

- Optimized for Gurbani kirtan and recitation, not general conversational
  Punjabi.
- Sung audio, accompaniment, reverberation, and imperfect caption references
  can materially affect WER.
- The model outputs Gurmukhi text and does not provide punctuation restoration
  or speaker diarization.
