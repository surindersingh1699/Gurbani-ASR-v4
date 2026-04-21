# Surt Live Lab

CPU-first live-mic harness and file-upload sandbox for the Surt Gurbani ASR
model, with a built-in shabad tracker that matches live transcript text
against the SGGS index.

Every stage of the pipeline (high-pass, normalize, VAD, segmenter, decoder,
tracker) is **toggleable from the UI** so you can A/B what actually works on
kirtan — where standard speech-VADs tend to reject sung voice over harmonium.

![Two tabs: Live mic + File upload, with preprocess/VAD/segmenter/ASR/tracker accordions and a live shabad panel.]()

## What it does

```
mic / file  ─►  resample 16k ─►  HPF ─►  RMS normalize
                                   │
                                   ▼
                        Silero / energy / off VAD
                                   │
                                   ▼
                 Segmenter (vad-gated / fixed / rolling)
                                   │
                                   ▼
            faster-whisper INT8 (CPU)  or  transformers (fallback)
                                   │
                                   ▼
                 MuRIL-base + FAISS EMA shabad tracker
                                   │
                                   ▼
    transcript · current shabad + highlighted tuk · top-K candidates · history
```

## Layout

| File | What it owns |
|---|---|
| `pipeline.py` | HPF, Silero/energy VAD, segmenter (3 modes), resampler |
| `asr.py` | `FasterWhisperBackend` (INT8 CPU) + `TorchFallbackBackend` |
| `convert_to_ct2.py` | HF → CTranslate2 INT8 converter with tokenizer backfill |
| `tracker.py` | MuRIL + FAISS EMA shabad tracker with alaap-floor freeze |
| `app.py` | Gradio UI: Live tab, File tab, shared accordions |
| `requirements.txt` | Runtime deps on top of the repo base |

## Prereqs

You need the SGGS index artifacts already built (these come from
`scripts/01_build_tuk_index.py` and `scripts/02_build_shabad_index.py`):

```
index/sggs_tuk.faiss      (56,152 × 768 MuRIL, IP)
index/sggs_shabad.faiss   (5,544 × 768 MuRIL, IP)
index/tuk_meta.pkl
index/shabad_meta.pkl
data/processed/tuks.json
```

If they're missing, run those two scripts first.

## Install

```bash
pip install -r apps/live_lab/requirements.txt
```

## One-off: convert your fine-tuned Whisper to CT2 INT8

```bash
python -m apps.live_lab.convert_to_ct2 \
    surindersinghssj/surt-small-v3 \
    ~/models/surt-small-v3-int8 \
    --quantization int8
```

Output is ~250 MB (INT8 weights + tokenizer). The converter automatically
snapshots the source repo if needed and backfills any missing tokenizer
files from `openai/whisper-small`.

### If conversion fails with `'list' object has no attribute 'keys'`

This is a known transformers+CT2 incompatibility with some fine-tuned
Whisper repos whose `tokenizer_config.json` stores `extra_special_tokens`
as a list. Workaround:

```bash
# 1. snapshot the repo locally
python -c "from huggingface_hub import snapshot_download; \
           snapshot_download('surindersinghssj/surt-small-v3', \
                             local_dir='/tmp/surt-src')"

# 2. patch tokenizer_config.json (list -> {})
python -c "import json, pathlib; p=pathlib.Path('/tmp/surt-src/tokenizer_config.json'); \
           d=json.loads(p.read_text()); \
           d['extra_special_tokens']={} if isinstance(d.get('extra_special_tokens'), list) else d.get('extra_special_tokens', {}); \
           p.write_text(json.dumps(d, indent=2))"

# 3. convert from the patched local dir
python -m apps.live_lab.convert_to_ct2 /tmp/surt-src ~/models/surt-small-v3-int8
```

## Run

```bash
python -m apps.live_lab.app
# opens http://127.0.0.1:7860
```

First-time: click **Load / reload model** — status flips to green once the
INT8 backend is resident. The MuRIL + FAISS tracker loads lazily on the
first decoded segment (~15 s one-time cost).

## Compute-type cheat sheet

| Mode | Weights | Compute | Notes |
|---|---|---|---|
| **int8** | int8 | int8→fp32 accum | **CPU default** — ~4× smaller, fastest |
| int8_float16 | int8 | fp16 | GPU only, no CPU benefit |
| float16 | fp16 | fp16 | GPU only |
| float32 | fp32 | fp32 | Accuracy debugging only |

## Toggles that actually matter for kirtan

### VAD mode
- `off` — no gating, segmenter emits at `max_segment_s` boundaries. **Start here** for kirtan.
- `silero` — speech-trained; can over-reject sung voice over harmonium. Try threshold 0.2 before giving up.
- `energy` — pure RMS gate; robust to music as long as the reciter is louder than the drone.

### Segmenter mode
- `vad` — closes when VAD says silence or on max length.
- `fixed` — emits every `max_segment_s` regardless of content. Best for continuous kirtan.
- `rolling` — commit-and-carry; overlapping windows, lower perceived latency.

### ASR decoding
- **Beam size 1** (greedy) — fastest, fine for most kirtan.
- **Condition on previous text = OFF** — set to ON only if the current shabad context should bias decoding (risk: repetition loops).
- **`faster-whisper VAD filter`** — independent of our Silero toggle, applied after our preprocessing. Off by default; turn on if silence segments are still bleeding into decodes.

## Shabad tracker

Every decoded segment is embedded with MuRIL-base, searched against the
56k-tuk FAISS index, and the winning tuk's `shabad_id` is EMA-smoothed into
the running belief. The tracker exposes:

- **Current shabad** — top-EMA shabad, with the currently-matched tuk highlighted.
- **Alaap badge** — lights up when the top cosine drops below `alap_floor`; the pointer freezes so short instrumental passages don't erase context.
- **Top-K candidates** — with normalized score bars.
- **History** — appended only when the top cosine is ≥ floor (rejects alaap).

### Tuning knobs (Shabad tracker accordion)
- `EMA α` — higher = more responsive, noisier. 0.35 is the default.
- `Alaap floor` — cosine below which the pointer freezes. 0.45 is about right for Surt-small; lower it (0.30) if your transcript quality is weaker.
- `Top-K` — how many candidates the side panel shows.

## End-to-end smoke test (from the CLI)

```bash
# 10 min of kirtan → expected RTF ~0.2× on Apple Silicon laptop CPU
python -c "
import sys; sys.path.insert(0, '.')
from faster_whisper import WhisperModel
from apps.live_lab.tracker import Tracker, TrackerSettings

m = WhisperModel('~/models/surt-small-v3-int8', device='cpu', compute_type='int8')
trk = Tracker(TrackerSettings())
segs, info = m.transcribe('path/to/kirtan.wav', beam_size=1, language='pa')
for seg in segs:
    res = trk.update(seg.text, record_history=True)
    cur = res.current_shabad
    if cur:
        print(f'[{seg.start:6.1f}s] ang {cur.ang} line {res.current_line_idx+1} · {seg.text[:60]}')
"
```

## Environment variables

| Var | Default | Purpose |
|---|---|---|
| `SURT_CT2_DIR` | `~/models/surt-small-v3-int8` | Default CT2 dir the UI points at |
| `SURT_MODEL_ID` | `surindersinghssj/surt-small-v3` | HF repo for the transformers fallback |
| `SURT_INDEX_DIR` | `<repo>/index` | Where `sggs_tuk.faiss` + metas live |
| `SURT_TUKS_JSON` | `<repo>/data/processed/tuks.json` | Source tuks corpus |
| `SURT_MURIL_MODEL` | `google/muril-base-cased` | MuRIL encoder |

## Troubleshooting

### "CT2 model directory not found"
The default path resolves to `~/models/surt-small-v3-int8`. Either run the
converter (above) or switch **Backend** to `transformers` in the UI to
download from HF on the fly.

### Live mic isn't producing transcripts
- Make sure you clicked **Load / reload model** (status panel goes green).
- First-segment latency is ~15 s (MuRIL + FAISS load lazily). Subsequent
  segments should be ~1–2 s on a modern CPU.
- Try **VAD = off** with **segmenter mode = fixed** and **max segment = 15 s**
  before touching Silero — Silero-VAD is speech-trained and routinely
  rejects sung kirtan.
- If the browser blocks the mic: open DevTools → Console and check for
  `NotAllowedError` / `NotReadableError`.
- Gradio's streaming audio callback serializes behind the model. If
  decoding is slower than realtime (RTF > 1), the mic buffer fills up and
  looks "frozen." Watch the **RTF** meter — it should stay under 0.5× on
  CPU with INT8 and beam=1.

### `'list' object has no attribute 'keys'` during conversion
See the converter workaround section above — patch
`extra_special_tokens` in the snapshot.

### Whisper hallucinates `ਵਾਹਿਗੁਰੂ ਵਾਹਿਗੁਰੂ` or repetitions on silence
Expected on a Gurbani fine-tune. Two defences:
1. Turn VAD back on (silero or energy) to stop feeding silence to the model.
2. Raise `no_speech_threshold` (currently 0.6) — in `asr.py`.

### FAISS segfault at load on Apple Silicon
Your `faiss-cpu` wheel was built for a different arch. Reinstall:
```bash
pip uninstall -y faiss-cpu && pip install --force-reinstall faiss-cpu
```
