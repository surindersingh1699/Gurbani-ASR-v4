# Surt — Complete Training Plan

> ## ⚠️ HISTORICAL — DO NOT FOLLOW
>
> This is the **v1** training plan (sehaj path only). It is kept for reference and to understand past decisions. **It does not reflect the current direction.**
>
> **Current plan:** [`PLAN.md`](PLAN.md) — v3: 500h kirtan dataset built from YouTube + Gemini 2.5 Flash Lite transcription.
>
> Artifacts from this plan (e.g. `surt-small-v1`, the original sehaj-only dataset) may still be referenced, but any new work should align with `PLAN.md`.

### Gurbani Kirtan Real-Time Shabad Identification

---

## Overview

| | |
|---|---|
| **Goal** | CPU real-time shabad identification from live kirtan audio |
| **Final model** | Whisper Tiny INT8 ONNX (~42MB) + 3-layer SGGS FAISS index |
| **Total GPU cost** | ~$85 (Phase 1 free on Colab Pro + RunPod for phases 2–4) |
| **Calendar time** | ~8 weeks part-time evenings |
| **Data** | 100h sehaj path FLAC on HuggingFace (line-level) + 700h kirtan (shabad-level) |

### Compute platform per phase

| Phase | Platform | Why |
|---|---|---|
| 0 — Data prep + SGGS index | Local or Colab CPU | No GPU needed |
| 1 — Fine-tune Small on 100h | **Colab Pro (free GPU)** | FLAC on HuggingFace streams directly, no disk upload needed |
| 2 — Forced alignment 700h | RunPod A40 | 12–15h uninterrupted; Colab session limit too risky |
| 3 — Curriculum fine-tune 800h | RunPod A40 | 36–42h, needs a stable long run |
| 4 — Distil Small → Tiny | RunPod A40 | 8–10h |
| 5 — Quantize + pipeline | Local CPU | No GPU needed |

### Model lineage

```
openai/whisper-small
  → Phase 1 fine-tune (100h sehaj path)
  → surt_small_v1                        ← used for forced alignment in Phase 2
      → Phase 3 continue training (800h)
      → surt_small_final                 ← teacher for distillation
          → Phase 4 distil to Tiny
          → surt_tiny_student
              → Phase 5 quantize
              → surt_tiny_int8.onnx      ← ships to users
```

---

## Phase 0 — Data Prep + SGGS Index
**GPU cost:** $0 · **Time:** 3–5 days CPU · **Prerequisite for everything**

### Goal
Format the 100h sehaj path into HuggingFace Dataset format. Build all three FAISS index layers from the full SGGS. Both are prerequisites for every subsequent phase and require no GPU.

### Deliverables
- `data/sehaj_path_dataset/` — HuggingFace Dataset, one row per line: `(audio, gurmukhi_text, duration)`
- `index/sggs_tuk.faiss` — 15,744 MuRIL tuk embeddings
- `index/sggs_shabad.faiss` — ~3,500 mean-pooled shabad embeddings
- `index/sggs_ngram.faiss` — ~480k 3–5 word n-gram embeddings
- `index/tuk_meta.pkl` — `tuk_id → {shabad_id, ang, raag, text}` lookup

### The 3-layer SGGS index explained

| Layer | Unit | Vectors | Index size | When to search |
|---|---|---|---|---|
| 1 — n-gram | 3–5 word chunks | ~480k | ~1.4GB | Fallback — confidence < 0.55 |
| 2 — tuk | One pankiti/line | 15,744 | ~46MB | Always first — primary workhorse |
| 3 — shabad | Full shabad (mean-pooled) | ~3,500 | ~10MB | Confirmation — cross-check tuk match |

**Inference decision tree** (every 3s audio window):
1. Whisper transcribes → embed with MuRIL → query vector
2. Search Layer 2 (tuk). Top score > 0.82 and top-3 from same shabad? → step 4
3. Otherwise search Layer 1 (n-gram fallback). Score still < 0.55? → no prediction this window
4. Confirm with Layer 3 (shabad). Does shabad-level search agree?
5. 3-window agreement: same shabad 3× in a row with confidence > 0.70 → display to user

```python
from datasets import Dataset, Audio
from sentence_transformers import SentenceTransformer
import faiss, numpy as np, pickle, librosa

# Format sehaj path dataset
rows = []
for clip in sehaj_path_clips:
    audio, sr = librosa.load(clip.wav_path, sr=16000, mono=True)
    rows.append({
        "audio": {"array": audio, "sampling_rate": 16000},
        "text":  clip.gurmukhi_text,
        "duration": len(audio) / sr,
    })
ds = Dataset.from_list(rows).cast_column("audio", Audio(sampling_rate=16000))
ds.save_to_disk("data/sehaj_path_dataset")

# Build tuk index (run once, ~20 min CPU)
muril = SentenceTransformer("google/muril-base-cased")
tuks, meta = load_all_sggs_tuks()  # your SGGS parser
vecs = muril.encode(tuks, batch_size=64,
                    normalize_embeddings=True,
                    show_progress_bar=True)
idx = faiss.IndexFlatIP(768)
idx.add(vecs)
faiss.write_index(idx, "index/sggs_tuk.faiss")
pickle.dump(meta, open("index/tuk_meta.pkl", "wb"))
```

### Exit criterion
- Dataset loads cleanly in HuggingFace
- FAISS search on `"ਸਤਿ ਨਾਮੁ"` returns correct SGGS tuks in top-3
- All 3 index files exist and load without error

---

## Phase 1 — Fine-tune Whisper Small on 100h Sehaj Path
**GPU cost:** $0 (Colab Pro free GPU) · **Platform:** Google Colab Pro · **Model produced:** `surt_small_v1`

### Why Colab Pro works perfectly here
Your sehaj path data is already on HuggingFace as FLAC files (~9GB total). HuggingFace streaming loads audio directly into Colab without touching Colab's disk at all — no upload, no disk space issues.

**Expected training time by GPU (Colab Pro):**

| GPU | Time | Frequency on Pro |
|---|---|---|
| A100 (40GB) | ~5–6h | Sometimes, off-peak |
| V100 (16GB) | ~12–14h | Common |
| T4 (16GB) | ~20–24h | Most common |

### Key design decisions
- **Full fine-tuning** — train encoder + decoder together. Whisper saw very little Punjabi (~0.06% of training data). The encoder needs to adapt to Gurbani acoustics.
- **Discriminative learning rates** — decoder gets 2× higher LR than encoder. The encoder already knows general audio; the decoder is learning Gurmukhi almost from scratch.
- **Mool Mantar as `initial_prompt`** — free accuracy improvement, zero training cost.
- **Checkpoints pushed to HuggingFace every 300 steps** — if the Colab session dies, restart and resume. No work lost.
- **Google Drive as local checkpoint buffer** — saves to Drive first, then pushes to HuggingFace.

### Before running — 3 setup steps
1. Create a private model repo on HuggingFace: `your-username/surt-small-v1`
2. Store your HuggingFace token in Colab Secrets (key icon in left sidebar → add `HF_TOKEN`)
3. Run the column inspection cell first to confirm your dataset column names

### Complete Colab Pro notebook

```python
# ── Cell 1: Mount Drive + install ────────────────────────────────
from google.colab import drive
drive.mount("/content/drive")

!pip install transformers datasets accelerate evaluate \
             audiomentations huggingface_hub -q -U
```

```python
# ── Cell 2: Config — edit these ──────────────────────────────────
HF_DATASET    = "your-username/sehaj-path-gurbani"
HF_MODEL_REPO = "your-username/surt-small-v1"
CKPT_DIR      = "/content/drive/MyDrive/surt/phase1"
AUDIO_COL     = "audio"      # update if your column is named differently
TEXT_COL      = "text"       # might be "sentence" or "transcription"
RESUME        = True         # set False for a completely fresh run

from google.colab import userdata
HF_TOKEN = userdata.get("HF_TOKEN")   # stored in Colab Secrets — never hardcode
```

```python
# ── Cell 3: Inspect dataset columns — run this first ─────────────
from datasets import load_dataset
from huggingface_hub import login

login(token=HF_TOKEN)
sample = load_dataset(HF_DATASET, split="train[:3]")
print("Columns:", sample.column_names)
print("First row types:", {k: type(v) for k, v in sample[0].items()})
# Confirm AUDIO_COL and TEXT_COL match before proceeding
```

```python
# ── Cell 4: Load + preprocess (streaming — no disk usage) ────────
from datasets import load_dataset, Audio
from transformers import WhisperProcessor
import audiomentations as A
import numpy as np

processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small", language="pa", task="transcribe"
)

augment = A.Compose([
    A.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.4),
    A.RoomSimulator(p=0.3),
    A.TimeStretch(min_rate=0.9, max_rate=1.1, p=0.2),
    A.PitchShift(min_semitones=-1, max_semitones=1, p=0.2),
])

def prepare_example(batch, augment_audio=False):
    audio = batch[AUDIO_COL]
    arr   = np.array(audio["array"], dtype=np.float32)
    if augment_audio:
        arr = augment(samples=arr, sample_rate=16000)
    batch["input_features"] = processor(
        arr, sampling_rate=16000, return_tensors="np"
    ).input_features[0]
    label_ids = processor.tokenizer(batch[TEXT_COL]).input_ids
    batch["labels"] = [
        -100 if t == processor.tokenizer.pad_token_id else t
        for t in label_ids
    ]
    return batch

# Stream from HuggingFace — never touches Colab disk
ds_train = load_dataset(HF_DATASET, split="train",
                        streaming=True, token=HF_TOKEN)
ds_train = ds_train.cast_column(AUDIO_COL, Audio(sampling_rate=16000))
ds_train = ds_train.shuffle(seed=42, buffer_size=1000)
ds_train = ds_train.map(lambda b: prepare_example(b, augment_audio=True))

# Validation: fixed 300-example slice loaded into memory
ds_val = load_dataset(HF_DATASET, split="train[:300]", token=HF_TOKEN)
ds_val = ds_val.cast_column(AUDIO_COL, Audio(sampling_rate=16000))
ds_val = ds_val.map(lambda b: prepare_example(b, augment_audio=False))
```

```python
# ── Cell 5: Model — auto-detect GPU and set batch size ───────────
from transformers import WhisperForConditionalGeneration
from torch.optim import AdamW
import torch

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model.generation_config.language = "pa"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None

gpu = torch.cuda.get_device_name(0)
print(f"GPU: {gpu}")

BATCH = 8 if "A100" in gpu else 4
ACCUM = 4 if "A100" in gpu else 8
print(f"Batch: {BATCH}  Accumulation: {ACCUM}  Effective: {BATCH * ACCUM}")

optimizer = AdamW([
    {"params": model.model.encoder.parameters(), "lr": 5e-5},
    {"params": model.model.decoder.parameters(), "lr": 1e-4},
    {"params": model.proj_out.parameters(),      "lr": 1e-4},
])
```

```python
# ── Cell 6: Training with Drive + HuggingFace checkpoint safety ──
from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments,
                           DataCollatorForSeq2Seq, TrainerCallback)
from huggingface_hub import HfApi
import evaluate, os

wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    labels    = pred.label_ids
    preds     = pred.predictions
    labels    = np.where(labels != -100, labels,
                         processor.tokenizer.pad_token_id)
    pred_str  = processor.batch_decode(preds,  skip_special_tokens=True)
    label_str = processor.batch_decode(labels, skip_special_tokens=True)
    return {"wer": round(wer_metric.compute(
        predictions=pred_str, references=label_str), 4)}

data_collator = DataCollatorForSeq2Seq(
    model=model, tokenizer=processor.feature_extractor, padding=True
)
os.makedirs(CKPT_DIR, exist_ok=True)

args = Seq2SeqTrainingArguments(
    output_dir                  = CKPT_DIR,
    num_train_epochs            = 3,
    per_device_train_batch_size = BATCH,
    gradient_accumulation_steps = ACCUM,
    gradient_checkpointing      = True,   # needed on T4
    warmup_steps                = 400,
    fp16                        = True,
    evaluation_strategy         = "steps",
    eval_steps                  = 300,
    save_steps                  = 300,    # save to Drive every 300 steps
    save_total_limit            = 3,
    predict_with_generate       = True,
    generation_max_length       = 225,
    logging_steps               = 50,
    load_best_model_at_end      = True,
    metric_for_best_model       = "wer",
    greater_is_better           = False,
    dataloader_num_workers      = 2,
    report_to                   = "none",
)

# Push best checkpoint to HuggingFace after every eval
# Colab session death loses at most 300 steps — everything else is on HuggingFace
class PushBestToHub(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if state.best_model_checkpoint:
            HfApi().upload_folder(
                folder_path    = state.best_model_checkpoint,
                repo_id        = HF_MODEL_REPO,
                repo_type      = "model",
                commit_message = f"step {state.global_step} "
                                 f"wer={metrics.get('eval_wer', '?')}",
                token          = HF_TOKEN,
            )
            print(f"  Pushed to HuggingFace: step {state.global_step}")

# Resume from Drive checkpoint if session was interrupted
checkpoint = None
if RESUME and os.path.isdir(CKPT_DIR):
    ckpts = sorted([d for d in os.listdir(CKPT_DIR)
                    if d.startswith("checkpoint")])
    if ckpts:
        checkpoint = os.path.join(CKPT_DIR, ckpts[-1])
        print(f"Resuming from: {checkpoint}")

trainer = Seq2SeqTrainer(
    model           = model,
    args            = args,
    train_dataset   = ds_train,
    eval_dataset    = ds_val,
    data_collator   = data_collator,
    compute_metrics = compute_metrics,
    callbacks       = [PushBestToHub()],
    optimizers      = (optimizer, None),
)
trainer.train(resume_from_checkpoint=checkpoint)
```

```python
# ── Cell 7: Spot-check + push final model ────────────────────────
MOOL_MANTAR = "ੴ ਸਤਿ ਨਾਮੁ ਕਰਤਾ ਪੁਰਖੁ ਨਿਰਭਉ ਨਿਰਵੈਰੁ ਅਕਾਲ ਮੂਰਤਿ ਅਜੂਨੀ ਸੈਭੰ ਗੁਰ ਪ੍ਰਸਾਦਿ"

for i in range(5):
    ex    = ds_val[i]
    arr   = np.array(ex[AUDIO_COL]["array"], dtype=np.float32)
    feats = processor(arr, sampling_rate=16000,
                      return_tensors="pt").input_features.to("cuda")
    ids   = model.generate(feats, language="<|pa|>",
                            initial_prompt=MOOL_MANTAR)
    pred  = processor.batch_decode(ids, skip_special_tokens=True)[0]
    ref   = ex[TEXT_COL]
    print(f"REF:  {ref}")
    print(f"PRED: {pred}\n")

trainer.save_model(f"{CKPT_DIR}/final")
HfApi().upload_folder(
    folder_path    = f"{CKPT_DIR}/final",
    repo_id        = HF_MODEL_REPO,
    repo_type      = "model",
    commit_message = "Phase 1 final — surt_small_v1",
    token          = HF_TOKEN,
)
print(f"Done: https://huggingface.co/{HF_MODEL_REPO}")
```

> **Mool Mantar prompt for inference:**
> ```python
> MOOL_MANTAR = "ੴ ਸਤਿ ਨਾਮੁ ਕਰਤਾ ਪੁਰਖੁ ਨਿਰਭਉ ਨਿਰਵੈਰੁ ਅਕਾਲ ਮੂਰਤਿ ਅਜੂਨੀ ਸੈਭੰ ਗੁਰ ਪ੍ਰਸਾਦਿ"
> result = model.transcribe(audio, language="pa", initial_prompt=MOOL_MANTAR)
> ```
> Keep under ~80 tokens — Mool Mantar is ~40 tokens, ideal.

### Exit criterion — do not proceed to Phase 2 until all pass
- WER on held-out sehaj path **< 15%**
- WER on 10 held-out kirtan test clips **< 35%**
- SGGS retrieval hit@3 on kirtan clips **> 75%**
- No English hallucinations on non-speech frames
- Final model visible at `huggingface.co/your-username/surt-small-v1`

---

## Phase 2 — Forced Alignment: Auto-label 700h Kirtan
**GPU cost:** ~$18 (RunPod A40, ~12–15h) · **Reuses:** `surt_small_v1` (read-only)

### Why RunPod, not Colab, from here
Phase 2 needs 12–15 uninterrupted GPU hours to process 700h of audio. Colab Pro sessions are not guaranteed to last that long. A single interruption mid-alignment means reprocessing hours of audio. RunPod at ~$1.50/hr on an A40 gives a stable SSH session you can leave running overnight — Claude Code can manage this fully unattended.

### Why forced alignment, not free transcription
You already know which shabad is in each clip. Forced alignment uses that known text as a constraint — the model finds **where** each word occurs in the audio rather than guessing **what** was said. This produces timestamps, not transcriptions. The output is `(start_time, end_time, text, confidence)` for each line.

```python
import whisper_timestamped as wt
import json

# Load Phase 1 checkpoint — NO weight updates in this phase
model = wt.load_model("./checkpoints/surt_small_v1", device="cuda")

CONFIDENCE_THRESHOLD = 0.70
results = []

for clip in kirtan_700h_clips:
    known_text = sggs_lookup[clip.shabad_id]  # known shabad text

    result = wt.transcribe(
        model,
        clip.audio_path,
        language         = "pa",
        initial_prompt   = known_text[:150],   # anchor to known shabad
        word_timestamps  = True,
        vad              = True,               # skip silence/tabla solos
    )

    for seg in result["segments"]:
        if not seg["words"]:
            continue
        avg_conf = sum(w["confidence"] for w in seg["words"]) / len(seg["words"])

        if avg_conf < CONFIDENCE_THRESHOLD:
            continue   # discard low-confidence — better to skip than corrupt

        results.append({
            "clip_id":    clip.id,
            "shabad_id":  clip.shabad_id,
            "start":      seg["start"],
            "end":        seg["end"],
            "text":       seg["text"].strip(),
            "confidence": avg_conf,
            "label_type": "silver",            # vs "gold" for sehaj path
        })

with open("data/kirtan_aligned.jsonl", "w") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Produced {len(results)} aligned segments")
# Target: at least 60% of 700h yields high-confidence segments
```

### Confidence tiers for Phase 3 training

| Tier | Confidence | Label | Used in |
|---|---|---|---|
| Gold | — | Sehaj path (human-labeled) | Stage A + B |
| Silver high | > 0.80 | Aligned kirtan | Stage A |
| Silver medium | 0.60–0.80 | Aligned kirtan | Stage B only |
| Discard | < 0.60 | Not used | Never |

### Exit criterion
- At least **60% of 700h** yields segments above 0.70 confidence
- Manual spot-check of 50 random segments: text matches audible content
- If < 50% passes: return to Phase 1 and improve WER before proceeding

---

## Phase 3 — Fine-tune on Combined 800h Corpus (Curriculum)
**GPU cost:** ~$55 (RunPod A40, ~36–42h) · **Starts from:** `surt_small_v1` · **Produces:** `surt_small_final`

### Goal
Continue training the Phase 1 model on the combined corpus: 100h gold sehaj path + aligned kirtan. Curriculum training introduces noisy data gradually to avoid corrupting clean-data knowledge.

> **Critical:** always load from `./checkpoints/surt_small_v1`, not from `openai/whisper-small`. Starting from the wrong checkpoint wastes all Phase 1 work.

### Stage A — Gold + high-confidence silver (2 epochs)

```python
from datasets import concatenate_datasets, load_from_disk
from transformers import WhisperForConditionalGeneration
from torch.optim import AdamW

# Load Phase 1 checkpoint as starting point
model = WhisperForConditionalGeneration.from_pretrained(
    "./checkpoints/surt_small_v1"   # ← Phase 1, NOT openai/whisper-small
)

sehaj       = load_from_disk("data/sehaj_path_dataset")
kirtan_high = load_kirtan_aligned("data/kirtan_aligned.jsonl",
                                   min_confidence=0.80)
stage_a_data = concatenate_datasets([sehaj, kirtan_high])

# Lower LR than Phase 1 — continuing, not starting over
optimizer_a = AdamW([
    {"params": model.model.encoder.parameters(), "lr": 2e-5},  # was 5e-5
    {"params": model.model.decoder.parameters(), "lr": 5e-5},  # was 1e-4
    {"params": model.proj_out.parameters(),      "lr": 5e-5},
])

# Train 2 epochs → save to ./checkpoints/surt_small_stage_a
```

### Stage B — Add medium-confidence silver (1 epoch)

```python
# Load Stage A checkpoint
model = WhisperForConditionalGeneration.from_pretrained(
    "./checkpoints/surt_small_stage_a"
)

kirtan_med   = load_kirtan_aligned("data/kirtan_aligned.jsonl",
                                    min_confidence=0.60)
stage_b_data = concatenate_datasets([sehaj, kirtan_high, kirtan_med])

# Drop LR again — noisy data, be conservative
optimizer_b = AdamW([
    {"params": model.model.encoder.parameters(), "lr": 1e-5},
    {"params": model.model.decoder.parameters(), "lr": 3e-5},
    {"params": model.proj_out.parameters(),      "lr": 3e-5},
])

# Train 1 epoch → save to ./checkpoints/surt_small_final
```

### Exit criterion
- WER on kirtan test set **< 22%** (must beat Phase 1's ~26–30%)
- Retrieval hit@3 on noisy kirtan (laptop mic, 3m distance) **> 85%**
- WER on sehaj path must not regress beyond Phase 1 + 2 points

---

## Phase 4 — Distil Whisper Small → Tiny
**GPU cost:** ~$12 (RunPod A40, ~8–10h) · **Teacher:** `surt_small_final` · **Produces:** `surt_tiny_student`

### Goal
Compress the Phase 3 Small model into a Tiny-sized student. The student learns to mimic the teacher's encoder representations (MSE loss) and produce correct Gurmukhi tokens (cross-entropy loss). CPU latency drops from ~1s to ~220ms per 3s chunk.

### Why distil rather than fine-tune Tiny directly
The Small teacher was trained on 800h of Gurbani and learned rich phoneme representations. Distillation transfers that depth into Tiny's smaller body. A Tiny fine-tuned directly on the same 800h produces a weaker model — it can't absorb the same complexity without the teacher signal.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Seq2SeqTrainer, WhisperForConditionalGeneration

class DistillationTrainer(Seq2SeqTrainer):
    def __init__(self, teacher_model, proj_layer, alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.teacher = teacher_model.eval()
        self.proj    = proj_layer     # teacher dim (768) → student dim (384)
        self.alpha   = alpha          # 0.5 = equal weight task + distil loss

    def compute_loss(self, model, inputs, return_outputs=False):
        student_out = model(**inputs)
        task_loss   = student_out.loss   # CE on Gurmukhi tokens

        with torch.no_grad():
            teacher_out = self.teacher(**inputs)

        # Align teacher encoder dim (768) to student (384)
        teacher_enc = self.proj(
            teacher_out.encoder_last_hidden_state
        )
        student_enc = student_out.encoder_last_hidden_state
        distil_loss = F.mse_loss(student_enc, teacher_enc.detach())

        loss = (1 - self.alpha) * task_loss + self.alpha * distil_loss
        return (loss, student_out) if return_outputs else loss


teacher = WhisperForConditionalGeneration.from_pretrained(
    "./checkpoints/surt_small_final"
).eval().to("cuda")

student = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-tiny"   # fresh Tiny — not fine-tuned
).to("cuda")

# Projection: Small d_model=768 → Tiny d_model=384
proj = nn.Linear(768, 384).to("cuda")

distil_trainer = DistillationTrainer(
    teacher_model = teacher,
    proj_layer    = proj,
    alpha         = 0.5,
    model         = student,
    train_dataset = stage_a_data,   # gold + high-conf silver
    optimizers    = (AdamW([
        {"params": student.model.encoder.parameters(), "lr": 3e-5},
        {"params": student.model.decoder.parameters(), "lr": 8e-5},
        {"params": proj.parameters(),                  "lr": 8e-5},
    ]), None),
    # ... args, data_collator, compute_metrics
)
distil_trainer.train()
# Saves to: ./checkpoints/surt_tiny_student/
```

### Exit criterion
- Tiny student WER within **5 points** of Small teacher on kirtan test set
- CPU latency **< 300ms** per 3s chunk on 8-core laptop
- No regression on sehaj path WER beyond 3 points vs Small

---

## Phase 5 — Quantize to INT8 ONNX + Wire Full Inference Pipeline
**GPU cost:** $0 · **Time:** 2–3 days · **Produces:** `surt_tiny_int8.onnx` (ships to users)

### Goal
Export the distilled Tiny model to ONNX and quantize to INT8. Wire up the complete Surt real-time pipeline: VAD → Whisper Tiny INT8 → MuRIL embed → 3-layer FAISS → 3-window agreement → display shabad.

### Export and quantize

```python
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.onnxruntime import ORTQuantizer

# Step 1: export to ONNX
ort_model = ORTModelForSpeechSeq2Seq.from_pretrained(
    "./checkpoints/surt_tiny_student",
    export=True
)
ort_model.save_pretrained("./models/surt_tiny_onnx")

# Step 2: quantize to INT8
qconfig   = AutoQuantizationConfig.avx512_vnni(
    is_static=False, per_channel=False
)
quantizer = ORTQuantizer.from_pretrained("./models/surt_tiny_onnx")
quantizer.quantize(
    save_dir             = "./models/surt_tiny_int8",
    quantization_config  = qconfig
)
# Result: ~42MB model, ~220ms per 3s chunk on 8-core CPU
```

### Complete real-time inference pipeline

```python
import sounddevice as sd
import numpy as np
import faiss, pickle
from collections import deque
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from sentence_transformers import SentenceTransformer
from transformers import WhisperProcessor

SR         = 16000
WINDOW_S   = 3.0
HOP_S      = 1.0
CONFIDENCE = 0.70
AGREE_N    = 3

MOOL_MANTAR = "ੴ ਸਤਿ ਨਾਮੁ ਕਰਤਾ ਪੁਰਖੁ ਨਿਰਭਉ ਨਿਰਵੈਰੁ ਅਕਾਲ ਮੂਰਤਿ ਅਜੂਨੀ ਸੈਭੰ ਗੁਰ ਪ੍ਰਸਾਦਿ"

# Load once at startup
processor  = WhisperProcessor.from_pretrained("./models/surt_tiny_int8")
asr        = ORTModelForSpeechSeq2Seq.from_pretrained("./models/surt_tiny_int8")
muril      = SentenceTransformer("google/muril-base-cased")
idx_tuk    = faiss.read_index("index/sggs_tuk.faiss")
idx_shabad = faiss.read_index("index/sggs_shabad.faiss")
tuk_meta   = pickle.load(open("index/tuk_meta.pkl", "rb"))
window     = deque(maxlen=AGREE_N)


def identify_chunk(audio: np.ndarray):
    # 1. ASR with Gurbani prompt
    features = processor(audio, sampling_rate=SR,
                         return_tensors="pt").input_features
    ids      = asr.generate(features,
                             language="<|pa|>",
                             initial_prompt=MOOL_MANTAR)
    text     = processor.batch_decode(ids, skip_special_tokens=True)[0]
    if len(text.strip()) < 5:
        return None

    # 2. Embed with MuRIL
    q = muril.encode([text], normalize_embeddings=True)

    # 3. Layer 2 — tuk search (always first)
    D, I     = idx_tuk.search(q, k=5)
    top_score = float(D[0][0])
    top_tuks  = [tuk_meta[i] for i in I[0]]
    top_shabad = top_tuks[0]["shabad_id"]
    all_same   = all(t["shabad_id"] == top_shabad for t in top_tuks[:3])

    if top_score < 0.65:
        return None   # too noisy — emit no prediction

    # 4. Layer 3 — shabad confirmation
    Ds, Is    = idx_shabad.search(q, k=1)
    confirmed = (tuk_meta[Is[0][0]]["shabad_id"] == top_shabad)
    confidence = top_score * (1.1 if confirmed else 0.9)

    return {
        "shabad_id":  top_shabad,
        "tuk":        top_tuks[0],
        "confidence": confidence,
        "text":       text,
    }


def run():
    buf      = np.zeros(int(SR * WINDOW_S), dtype=np.float32)
    hop_size = int(SR * HOP_S)

    def callback(indata, frames, time, status):
        nonlocal buf
        buf = np.roll(buf, -frames)
        buf[-frames:] = indata[:, 0]

        result = identify_chunk(buf)
        window.append(result)

        if (len(window) == AGREE_N
                and all(r and r["shabad_id"] == window[0]["shabad_id"]
                        for r in window)
                and all(r["confidence"] > CONFIDENCE for r in window)):
            tuk = window[0]["tuk"]
            print(f"\n  Shabad identified: {window[0]['shabad_id']}")
            print(f"  Line: {tuk['text']}")
            print(f"  Ang: {tuk['ang']}  |  Confidence: {window[0]['confidence']:.0%}")

    with sd.InputStream(samplerate=SR, channels=1,
                        blocksize=hop_size, callback=callback):
        print("Listening for kirtan... (Ctrl+C to stop)")
        while True:
            sd.sleep(1000)


if __name__ == "__main__":
    run()
```

### Final artifact sizes

| File | Size | Purpose |
|---|---|---|
| `surt_tiny_int8.onnx` | ~42 MB | ASR model |
| `sggs_tuk.faiss` | ~46 MB | Primary retrieval index |
| `sggs_shabad.faiss` | ~10 MB | Confirmation index |
| `sggs_ngram.faiss` | ~1.4 GB | Fallback index |
| `tuk_meta.pkl` | ~8 MB | Metadata lookup |
| MuRIL (INT8) | ~100 MB | Query embedder |

> The n-gram index is large. If bundle size matters, ship without it and only use Layers 2 and 3 — coverage drops slightly but the app stays under 250MB total.

### Exit criterion — Surt is shippable when
- Identifies correct shabad in **< 3s** from onset of a clearly sung line
- Retrieval hit@3 on real gurdwara test recordings **> 85%**
- CPU latency per window **< 350ms** on a 4-core laptop
- No English hallucinations on tabla solos or transitions

---

## Cost + Time Summary

| Phase | Platform | GPU cost | GPU hours | Wall time |
|---|---|---|---|---|
| 0 | Local / Colab CPU | $0 | 0 | 3–5 days |
| 1 | **Colab Pro (free GPU)** | **$0** | 5–24h | 1 weekend |
| 2 | RunPod A40 | ~$18 | 12–15h | 2 days |
| 3 | RunPod A40 | ~$55 | 36–42h | 4–5 days |
| 4 | RunPod A40 | ~$12 | 8–10h | 1 day |
| 5 | Local CPU | $0 | 0 | 2–3 days |
| **Total** | | **~$85** | **~70h GPU** | **~8 weeks** |

Phase 1 is free thanks to Colab Pro — saves ~$27 vs the original RunPod estimate.

> **Optional upgrade:** Fine-tune Whisper Medium on the 800h corpus and use it as the distillation teacher instead of Small. Improves final Tiny WER by ~5 points and retrieval hit@3 by ~5%. Additional cost: ~$73. Total becomes ~$158.

---

## Claude Code + Colab Pro — Can it run unattended?

**Short answer: No for Colab, yes for everything else.**

Claude Code runs on your local machine via terminal. Google Colab runs in a browser tab. Claude Code has no way to open a browser, log into Google, click "connect runtime," and run cells. Colab Pro has no SSH or API that Claude Code could hook into.

### What Claude Code CAN run fully unattended for Surt

| Phase | Task | Unattended? |
|---|---|---|
| 0 | Build SGGS FAISS index | ✅ Yes — pure Python, runs locally |
| 1 | **Generate** the Colab notebook file | ✅ Yes — writes the .ipynb |
| 1 | **Run** the notebook in Colab | ❌ No — you click Run All once |
| 2 | Forced alignment on RunPod via SSH | ✅ Yes |
| 3 | Curriculum fine-tune on RunPod via SSH | ✅ Yes |
| 4 | Distillation on RunPod via SSH | ✅ Yes |
| 5 | Quantize + wire inference pipeline | ✅ Yes — runs locally |

### The Phase 1 workflow with Claude Code

Claude Code writes the notebook. You upload it and click one button. After that, Colab runs completely unattended — Drive saves every 300 steps, HuggingFace gets every best checkpoint. If the session dies (rare on Pro), you restart and resume automatically.

```bash
# In your terminal with Claude Code
claude "generate the Phase 1 Colab notebook for Surt as phase1_colab.ipynb
        using the spec in surt_training_plan.md"

# Claude Code produces: phase1_colab.ipynb
# You: go to colab.research.google.com, upload the file, click Runtime > Run All
# Walk away — Drive + HuggingFace handle the rest
```

### RunPod phases — fully unattended with Claude Code

For Phases 2, 3, and 4 on RunPod, Claude Code handles everything:

```bash
claude "SSH into RunPod at [your-ip], run Phase 2 forced alignment
        using surt_small_v1 from HuggingFace, save output to
        kirtan_aligned.jsonl, report back when done"
```

Claude Code will SSH in, install dependencies, run the script, monitor progress, and let you know when it's complete or if something fails.

### Summary

- **Phase 1**: Claude Code writes the notebook → you click Run All once → walks away
- **Phases 2–4 (RunPod)**: Claude Code fully unattended via SSH
- **Phases 0 and 5 (local)**: Claude Code fully unattended

---

## Install requirements

```bash
pip install transformers datasets evaluate accelerate
pip install optimum[onnxruntime]
pip install whisper-timestamped
pip install sentence-transformers faiss-cpu
pip install audiomentations sounddevice librosa
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

*Surt — ਸੁਰਤਿ — consciousness attuned to the Shabad.*