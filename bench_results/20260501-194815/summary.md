# ASR Alternatives Benchmark

- device: `cpu`  threads: `8`  max_samples: `50`
- baseline model: `surindersinghssj/surt-small-v3`  beam: `5`  vad: `off`

| Backend | Dataset | N | WER% | CER% | RTF (×realtime) | First-decode (s) | Peak RAM (MB) | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `indicconformer` | sehajpath | 0 | nan | nan | nan | nan | 1865 | eval_failed: ImportError |
| `indicconformer` | kirtan | 0 | nan | nan | nan | nan | 1917 | eval_failed: ImportError |

**How to read:**

Filter applied: only models with a real Mac/Win/iOS/Android deploy path are in the default set.

- **baseline (faster-whisper surt-small-v3)** — production target. Other rows compete with this. Mobile path: whisper.cpp / WhisperKit / sherpa-onnx (already shipping).
- **indicconformer-pa** — native Gurmukhi, 120M Conformer CTC. Mobile path: NeMo→ONNX→sherpa-onnx (proven for Conformer).
- **parakeet-tdt** — zero Punjabi training, WER will be ≈100% — **only the RTF / latency columns matter for this row**. Mobile path: parakeet.cpp / sherpa-onnx (exists).
- **mms** (opt-in) — has Punjabi but 1GB+ quantized → too big for an iOS app.
- **qwen3** (opt-in) — LLM-based ASR, no mobile inference runtime today.

**Decision rule:**

- If `indicconformer` WER < `baseline` WER on kirtan → switch architectures (and fine-tune indicconformer on v3).
- Else if `qwen3` RTF < `baseline` RTF AND WER < ~85% → fine-tune qwen3 on v3.
- Else if `parakeet` RTF << `baseline` RTF → invest in tokenizer-extension work for Gurmukhi.
- Else → stay on Whisper-small architecture, focus on data scale (the v3 plan).
