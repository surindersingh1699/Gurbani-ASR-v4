# ASR Alternatives Bench — Mac M5 Pro CPU

**Date:** 2026-05-01
**Hardware:** Apple M5 Pro, 15 cores, 48 GB RAM, macOS 26.4
**Bench config:** `--device cpu --threads 8 --max-samples 50` per dataset
**Eval datasets (audio totals over the 50-clip slice):**
- `surindersinghssj/gurbani-sehajpath-yt-captions-eval-canonical` — 403.3 s of audio
- `surindersinghssj/gurbani-kirtan-yt-captions-eval-canonical` — 409.6 s of audio

WER/CER computed after `surt.data.normalize_gurbani_text` (strips `॥` and verse numbers).
Baseline runs with `beam=5`, `vad_filter=off`, `language=pa`, `temperature=0.0`.

---

## Headline numbers

| Backend | Dataset | WER% | CER% | RTF (× realtime) | First-decode (s) | Peak RAM (MB) |
|---|---|---:|---:|---:|---:|---:|
| **surt-small-v3** (faster-whisper int8 CT2, beam=5) | **Sehajpath** | **13.87%** | **4.25%** | 0.575× | 2.87 | 2244 |
| **surt-small-v3** (faster-whisper int8 CT2, beam=5) | **Kirtan** | **57.28%** | **30.69%** | 0.305× | 1.54 | 2301 |
| **IndicConformer-pa-large** (RNNT, NeMo) | **Sehajpath** | 45.15% | 13.40% | **0.055×** | 0.51 | 2135 |
| **IndicConformer-pa-large** (RNNT, NeMo) | **Kirtan** | 98.45% | 73.70% | **0.054×** | 0.39 | 2569 |
| **Parakeet-TDT-0.6B-v3** (sherpa-onnx INT8) | **Sehajpath** | 100.00% | 95.32% | **0.018×** | **0.16** | 2496 |
| **Parakeet-TDT-0.6B-v3** (sherpa-onnx INT8) | **Kirtan** | 102.79% | 98.69% | **0.018×** | **0.09** | 2571 |

> RTF = transcribe_seconds / audio_seconds. Lower is better. Parakeet 0.018× = **55× realtime**; IndicConformer 0.054× = **18× realtime**; baseline 0.305–0.575× = **roughly real-time** at beam=5.

---

## What each row says

### `surt-small-v3` baseline (production, beam=5)
- **Dominant on accuracy.** Sehajpath WER 13.87% / CER 4.25% is essentially noise-level — REF and HYP differ by 1–2 chars per clip. Kirtan WER 57.28% is the genuine production weakness.
- **Slowest of the three by 5–30×** at beam=5. RTF 0.575× on sehajpath means real-time on M5 Pro but no speed headroom for live streaming.
- This is the bar to beat.

### IndicConformer-pa-large
- **Good on sehajpath** out-of-box (CER 13.40%) — proves Gurmukhi tokenizer + acoustic encoder are sound for spoken paath.
- **Broken on kirtan** out-of-box (CER 73.70%) — pure acoustic-domain gap (trained on spoken Punjabi news/conversation, never sang Gurbani with harmonium+tabla+reverb).
- **10× faster than baseline.** RTF 0.054× = 18× realtime.
- Native Gurmukhi tokenizer means **zero vocabulary surgery** to fine-tune. Strongest fine-tune candidate.

### Parakeet-TDT-0.6B-v3 (sherpa-onnx INT8)
- **WER ~100% on both — meaningless** (no Punjabi/Gurmukhi training; outputs romanized garbage).
- **30× faster than baseline.** RTF 0.018× = 55× realtime, first-decode latency 0.085s — the lowest in this bench.
- Fine-tunable but expensive: SentencePiece tokenizer has zero Gurmukhi coverage → vocab extension + much longer training.
- Mobile-deploy path is proven (this bench used the actual `sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8` tarball that ships to iOS/Android).

---

## Decision matrix

| Question | Answer |
|---|---|
| Is the current baseline good enough? | **Sehajpath: yes** (WER 14% / CER 4%). **Kirtan: no** (WER 57% — 1 in 2 words wrong). |
| Can we drop Whisper-small for IndicConformer-pa today? | **No.** Out-of-box IndicConformer loses on both datasets. Needs fine-tune. |
| Is IndicConformer worth fine-tuning? | **Yes — strongest bet.** Native Gurmukhi tokenizer + 10× faster CPU + only acoustic gap to close. |
| Is Parakeet worth fine-tuning? | **Maybe — high cost / high reward.** 3× faster again than IndicConformer, but tokenizer extension is hard cross-script work. Defer until after IndicConformer fine-tune lands. |
| Mobile feasibility? | All three. baseline (whisper.cpp), IndicConformer (sherpa-onnx Conformer), Parakeet (sherpa-onnx Parakeet — what this bench used). |

---

## Recommended next steps

1. **Fine-tune IndicConformer-pa on the v3 corpus first.** Goal: beat baseline kirtan WER (57.28%) by ≥5 pp. Estimated 14–20h on a single A100, ~$25–45 RunPod spot. Sehajpath number proves the model handles Gurmukhi text; the kirtan failure is purely acoustic-distribution which 300h of in-domain audio should fix.
2. **If A wins on kirtan, ship.** Speed bonus is huge (10× faster CPU) and on-device latency improves dramatically.
3. **Run Parakeet fine-tune in parallel only if budget allows ($200–350, 3–5 days H100).** Vocab-extension cross-script transfer is fragile — set a 30k-step early-kill if it hasn't crossed below baseline kirtan WER by then.

---

## Files of record

- baseline CSV: `bench_results/20260501-195548/results.csv`
- IndicConformer CSV: `bench_results/20260501-195211/results.csv`
- Parakeet CSV: `bench_results/20260501-195405/results.csv`
- Bench script: `scripts/bench_asr_alternatives.py`
- Checkpoint plan: `.planning/quick/3-bench-asr-alternatives-3-models/3-PLAN.md`

## Notes on the bench process (for posterity)

**What worked:** sherpa-onnx Parakeet (the actual mobile inference path) loaded in 0.9 s and ran cleanly. AI4Bharat's IndicConformer .nemo file loaded after patching one line in NeMo's tokenizer mixin to handle `type: multilingual` (the file uses an aggregate tokenizer with 22 sub-tokenizers, one per Indic language).

**What was painful:** mainline NVIDIA NeMo can't parse AI4Bharat's multilingual tokenizer config. Had to install AI4Bharat's NeMo fork (`v1.23.0rc0`) without its CUDA-only `triton` dep (`uv pip install --no-deps`), then pin `numpy<2`, `datasets<4`, `huggingface_hub<1.0` to keep it working alongside transformers. The two NeMo backends (IndicConformer vs Parakeet via NeMo) don't coexist in one venv — solved by using sherpa-onnx for Parakeet, which is the right answer for mobile anyway.

**Decoder-output parsing:** newer NeMo RNNT `model.transcribe()` returns `(hypotheses_list, alignments_list)` instead of just `[str]`. The bench script's IndicConformer adapter unwraps this correctly now (commit in `scripts/bench_asr_alternatives.py:IndicConformerBackend.transcribe`).
