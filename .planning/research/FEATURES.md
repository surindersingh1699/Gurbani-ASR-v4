# Feature Research

**Domain:** Whisper fine-tuning pipeline for low-resource Indic language (Gurmukhi/Punjabi) ASR
**Researched:** 2026-04-04
**Confidence:** HIGH (verified against HuggingFace Transformers v4.56 docs via Context7, audiomentations docs, and training plan spec)

## Feature Landscape

### Table Stakes (Training Fails or Wastes Money Without These)

Features that every serious Whisper fine-tuning pipeline must have. Missing any of these means training either crashes, produces garbage, or loses work on ephemeral cloud instances.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **HuggingFace dataset streaming** | 100h FLAC dataset is ~9GB. RunPod disk is expensive and ephemeral. Streaming avoids disk copy entirely. `load_dataset(..., streaming=True)` is the standard pattern. | LOW | Confirmed via HF Transformers docs. Cast audio column to `Audio(sampling_rate=16000)` for resampling on-the-fly. |
| **Seq2SeqTrainer with `predict_with_generate`** | Whisper is an encoder-decoder model. Standard `Trainer` cannot do autoregressive generation during eval. Without `predict_with_generate=True`, WER computation is impossible during training. | LOW | This is the standard HF pattern for ASR fine-tuning. `Seq2SeqTrainer` + `Seq2SeqTrainingArguments` required, not plain `Trainer`. |
| **WER metric computation during eval** | WER is the only meaningful metric for ASR. Without it, you have no signal on whether training is improving or diverging. Must decode predictions back to text and compare against references. | LOW | Uses `evaluate.load("wer")`. Must handle `-100` label masking by replacing with `pad_token_id` before decode. |
| **Gradient checkpointing** | A40 has 48GB VRAM, but full Whisper Small activations + optimizer states can exceed this at reasonable batch sizes. Gradient checkpointing trades ~30% slower training for ~60% VRAM reduction. | LOW | Single flag: `gradient_checkpointing=True` in `Seq2SeqTrainingArguments`. Confirmed in Transformers docs. |
| **Mixed precision (fp16)** | Without fp16, training is ~2x slower and uses ~2x more VRAM. On A40 (Ampere architecture), fp16 is well-supported. This is non-negotiable for cost efficiency. | LOW | `fp16=True` in training args. A40 also supports `bf16=True` which avoids fp16 overflow issues, but fp16 is fine for Whisper. |
| **Checkpoint saving + resume** | RunPod sessions can die (preemption, network issues, maintenance). Without incremental checkpoints and resume capability, you lose all training progress. At $1.50/hr, losing 6 hours = $9 wasted. | MEDIUM | `save_steps=300`, `save_total_limit=3`, `resume_from_checkpoint=path`. The Trainer handles optimizer state restoration automatically. |
| **HuggingFace Hub push on checkpoint** | Local checkpoints on RunPod are lost when the session ends. Pushing best checkpoint to HF Hub after every eval is the safety net. Without this, even saved checkpoints are at risk. | MEDIUM | Custom `TrainerCallback.on_evaluate` using `HfApi().upload_folder()`. This is the critical difference between "lost all work" and "lost at most 300 steps." |
| **Whisper language/task config** | Whisper uses special tokens to select language and task. Without setting `language="pa"` and `task="transcribe"`, the model defaults to English and generates English text from Punjabi audio (hallucination). | LOW | Set on processor (`WhisperProcessor.from_pretrained(..., language="pa", task="transcribe")`) AND on `model.generation_config`. Both are required. |
| **Label masking for pad tokens** | Whisper tokenizer produces pad tokens. If these are not masked to `-100` in labels, the model trains on predicting padding, which wastes capacity and degrades output quality. | LOW | Standard preprocessing: `labels = [-100 if t == pad_token_id else t for t in label_ids]`. |
| **Data collator with padding** | Audio sequences have variable lengths. Without a proper data collator that pads input features and labels to batch max length, training crashes on shape mismatches. | LOW | `DataCollatorForSeq2Seq(model=model, tokenizer=processor.feature_extractor, padding=True)`. |
| **Warmup steps** | Whisper fine-tuning is notoriously unstable in early steps. Without warmup, the model can diverge immediately and never recover, wasting the entire training run. | LOW | `warmup_steps=400` in training args. 400 steps is the community consensus for Whisper Small on ~100h data. |
| **Auto GPU detection + batch size adjustment** | Different GPUs (T4, V100, A40, A100) have vastly different VRAM. A fixed batch size either wastes A100 capacity or OOMs on T4. Must detect and adapt. | LOW | `torch.cuda.get_device_name(0)` + conditional batch/accumulation logic. Alternative: `auto_find_batch_size=True` in training args (binary search, slightly wasteful). |

### Differentiators (Competitive Advantage for Gurmukhi-Specific Quality)

Features that separate a production-quality Gurmukhi ASR model from a naive fine-tune. These address the specific challenges of low-resource Indic language training where Whisper has minimal prior exposure (~0.06% Punjabi in pretraining data).

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Discriminative learning rates (encoder vs decoder)** | Whisper's encoder already knows general audio representations. The decoder knows almost zero Gurmukhi. A uniform LR either under-trains the decoder or over-trains the encoder (catastrophic forgetting of acoustic features). Encoder at 5e-5, decoder at 1e-4 gives each component what it needs. | MEDIUM | Requires custom `AdamW` optimizer with parameter groups instead of the Trainer's default single LR. Pass via `optimizers=(optimizer, None)` to `Seq2SeqTrainer`. This overrides the Trainer's built-in optimizer, so you also lose the built-in LR scheduler (must implement separately or accept constant LR after warmup). |
| **Audio augmentation (noise, reverb, stretch, pitch)** | The 100h sehaj path is clean studio-quality audio. But the target use case is live kirtan with room acoustics, background noise, distance microphones, and varying singing speeds. Without augmentation, the model overfits to clean audio and fails on real-world kirtan. | MEDIUM | Uses `audiomentations.Compose` with `AddGaussianNoise`, `RoomSimulator`, `TimeStretch`, `PitchShift`. Confirmed via audiomentations docs: `RoomSimulator` uses pyroomacoustics for realistic reverb; `TimeStretch` uses signalsmith-stretch. Applied in preprocessing with probability gating (p=0.2-0.4). |
| **Mool Mantar as `initial_prompt`** | Whisper supports a text prompt that biases the decoder toward expected vocabulary. The Mool Mantar contains foundational Gurmukhi vocabulary that anchors the model to the correct script and domain. This is free accuracy at zero training cost. | LOW | Set during `model.generate(initial_prompt=MOOL_MANTAR)`. Must stay under ~80 tokens. Mool Mantar is ~40 tokens, ideal. Only used during inference/eval, not training. |
| **Full fine-tuning (not LoRA/adapter)** | Whisper saw ~0.06% Punjabi during pretraining. The encoder needs deep adaptation to Gurbani phonetics (nasal consonants, tonal distinctions, unique vowel sounds). The decoder needs to learn Gurmukhi tokenization almost from scratch. LoRA/adapters cannot achieve the depth of adaptation needed for a language this far from the pretraining distribution. | LOW (fewer moving parts than LoRA setup) | Full fine-tuning is actually simpler to implement than LoRA (no PEFT dependency, no rank tuning, no adapter merging). The trade-off is higher VRAM usage, which gradient checkpointing handles. |
| **Validation on fixed subset (not streamed)** | When training data is streamed, eval must be a fixed in-memory subset for reproducible WER tracking. Without this, eval metrics bounce randomly and you cannot tell if training is converging. | LOW | `load_dataset(HF_DATASET, split="train[:300]")` loads 300 examples into memory. Small enough to not matter for VRAM, large enough for stable WER estimation. |
| **Gurmukhi-specific text normalization** | Gurmukhi has multiple visually similar characters, optional diacritics (sihari, bihari, aunkar), and compound characters. Without normalization before WER computation, the metric penalizes stylistic differences that are not actual errors. | MEDIUM | Requires custom normalizer that handles Gurmukhi Unicode normalization (NFC), optional diacritics, and Gurbani-specific spelling conventions. Not provided by HuggingFace or Whisper out of the box. This is Phase 1 scope but often missed. |
| **Structured logging with step-level metrics** | Training runs cost real money ($1.50/hr). Without step-level loss and WER logging, you cannot diagnose problems until after the run completes. Early divergence detection saves hours of wasted compute. | LOW | `logging_steps=50` in training args. Can also use `report_to=["tensorboard"]` but `"none"` + custom logging is simpler for headless RunPod SSH sessions. |
| **Curriculum-aware data ordering** | For Phase 1, shuffling with a buffer (`shuffle(seed=42, buffer_size=1000)`) is sufficient. But the infrastructure should support ordered data presentation for Phase 3 where gold data comes before silver. Building this awareness into the pipeline now avoids a rewrite later. | LOW | Streaming shuffle with `buffer_size=1000` for Phase 1. Phase 3 uses `concatenate_datasets` with explicit ordering. The design pattern carries forward. |

### Anti-Features (Deliberately NOT Build for Phase 1)

Features that seem valuable but actively harm Phase 1 outcomes through added complexity, compute waste, or premature optimization.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| **LoRA / QLoRA / Adapter fine-tuning** | "Saves VRAM, trains faster, cheaper" | Whisper has 0.06% Punjabi exposure. LoRA rank 8-64 adapts maybe 1-5% of parameters. The encoder fundamentally does not understand Gurmukhi phonetics and needs deep weight changes that low-rank updates cannot achieve. You will get a model that hallucinates English on Punjabi audio. | Full fine-tuning with gradient checkpointing. VRAM fits on A40 (48GB). |
| **Multi-GPU / DeepSpeed / FSDP** | "Training will be faster" | Single A40 with gradient checkpointing and fp16 trains 100h in ~12-18 hours. Multi-GPU adds distributed training complexity (communication overhead, debugging difficulty, RunPod multi-GPU pricing). The cost savings do not justify the complexity for a 100h dataset. | Single A40, gradient accumulation to simulate larger batch sizes. |
| **Weights & Biases / MLflow experiment tracking** | "Need to track experiments properly" | Phase 1 has exactly ONE experiment: fine-tune Whisper Small on 100h sehaj path. There are no hyperparameter sweeps, no architecture comparisons, no ablations. W&B adds a dependency, requires account setup on RunPod, and provides no value for a single run. | Console logging + `logging_steps=50`. Review logs via SSH. WER at each eval step is sufficient. |
| **Evaluation scripts with automatic exit criteria** | "Automate the go/no-go decision" | The exit criteria for Phase 1 are nuanced: WER < 15% on sehaj path, WER < 35% on kirtan, retrieval hit@3 > 75%, no English hallucinations. Automating this requires building a full evaluation harness with kirtan test data, FAISS retrieval pipeline, and hallucination detection. This is more work than Phase 1 training itself. | Manual evaluation: spot-check 5-10 examples, compute WER on held-out set, listen to a few predictions. Quick and sufficient for a single model. |
| **Dynamic batch size / auto_find_batch_size** | "Automatically finds optimal batch size" | Binary search wastes the first 5-10 minutes of a paid GPU session testing batch sizes that OOM. The GPU set is known (A40), the model is known (Whisper Small), the batch size is known (8 + 4 accumulation). No need for runtime discovery. | Hardcode batch size per GPU in a simple conditional. |
| **Custom tokenizer / extended vocabulary** | "Gurmukhi needs special tokens" | Whisper's tokenizer already includes Gurmukhi Unicode block coverage. Adding tokens requires resizing embeddings, which destabilizes fine-tuning and invalidates pretrained decoder weights. The tokenizer is not the bottleneck; the encoder's acoustic model is. | Use Whisper's built-in tokenizer with `language="pa"`. It handles Gurmukhi. |
| **Data filtering / quality scoring pipeline** | "Remove bad examples before training" | The sehaj path dataset is human-labeled, studio-recorded, line-level aligned. It is already high quality. Building a quality filter adds complexity and risks removing valid training examples. Data quality problems are a Phase 2/3 concern (when working with auto-aligned kirtan). | Trust the gold dataset. Quality filtering is for Phase 3 silver data. |
| **Spectrogram augmentation (SpecAugment / freq masking)** | "Standard in ASR training" | Whisper's preprocessor already converts audio to log-mel spectrograms with a fixed pipeline. Applying SpecAugment requires intercepting the feature extraction pipeline and modifying the spectrogram tensor directly. This is fragile, poorly documented for Whisper specifically, and less effective than waveform-level augmentation for domain adaptation. | Waveform-level augmentation via audiomentations (noise, reverb, stretch, pitch) applied before Whisper's feature extractor. More natural, better studied, easier to implement. |
| **Learning rate scheduler tuning** | "Need cosine annealing / one-cycle policy" | When using discriminative learning rates with a custom optimizer, the Trainer's built-in scheduler is bypassed. Implementing a custom scheduler adds complexity. Linear warmup then constant LR is sufficient for a 3-epoch fine-tune on 100h. The model converges well without fancy scheduling. | Linear warmup (400 steps) via the custom optimizer. Constant LR after warmup. |
| **Colab notebook as training platform** | "Free GPU, why not" | Colab Pro sessions can be interrupted unpredictably (90 min - 12h). Phase 1 needs 12-18h of continuous training. A single interruption mid-training risks corruption if the checkpoint callback hasn't fired. Colab has no SSH (Claude Code cannot manage it unattended). The "free" cost is illusory when you factor in babysitting time and restart risk. | RunPod A40 at $1.50/hr. ~$18-27 total. Stable SSH, Claude Code manages unattended, checkpoints safe on HF Hub. |

## Feature Dependencies

```
[Dataset streaming from HF]
    └──requires──> [Whisper processor + audio resampling]
                       └──requires──> [Audio augmentation pipeline]
                       └──requires──> [Label tokenization + pad masking]
                                          └──requires──> [Data collator]

[Seq2SeqTrainer]
    └──requires──> [Seq2SeqTrainingArguments]
    └──requires──> [Data collator]
    └──requires──> [WER compute_metrics]
    └──requires──> [Model with language/task config]

[Discriminative learning rates]
    └──requires──> [Custom AdamW optimizer with param groups]
    └──conflicts──> [Trainer's built-in optimizer/scheduler]

[HuggingFace Hub push on checkpoint]
    └──requires──> [Checkpoint saving (save_steps)]
    └──requires──> [HF Hub authentication (HF_TOKEN)]
    └──enhances──> [Training resume from checkpoint]

[Training resume from checkpoint]
    └──requires──> [Checkpoint saving]
    └──requires──> [Stable checkpoint directory (not ephemeral)]

[Audio augmentation]
    └──enhances──> [Model generalization to kirtan audio]
    └──requires──> [audiomentations library]

[Mool Mantar initial_prompt]
    └──enhances──> [WER metric computation (eval only)]
    └──independent of──> [Training loop]

[Gradient checkpointing]
    └──enables──> [Larger batch sizes on limited VRAM]
    └──conflicts──> [Training speed (30% slower)]

[GPU auto-detection]
    └──determines──> [Batch size + accumulation steps]
    └──enhances──> [Portability across GPU types]
```

### Dependency Notes

- **Discriminative LR conflicts with Trainer's built-in optimizer:** When you pass a custom optimizer via `optimizers=(optimizer, None)`, the Trainer ignores `learning_rate`, `lr_scheduler_type`, and `warmup_steps` from training args. Warmup must be handled separately (either via a custom scheduler or by accepting constant LR).
- **HF Hub push requires checkpoint saving:** The `PushBestToHub` callback fires on `on_evaluate`, which only works if `eval_steps` aligns with `save_steps` and `state.best_model_checkpoint` is populated.
- **Audio augmentation requires audiomentations:** Not a HuggingFace dependency. Must be installed separately. The `RoomSimulator` transform additionally requires `pyroomacoustics`.
- **Streaming conflicts with random access:** Streamed datasets do not support indexing. Validation must be loaded as a regular (non-streamed) dataset to allow `compute_metrics` to work with `predict_with_generate`.

## MVP Definition

### Launch With (v1 — Phase 1 Training Pipeline)

Minimum viable pipeline that produces `surt_small_v1` on HuggingFace Hub.

- [x] **HF dataset streaming + audio resampling** -- foundation of data pipeline, no disk dependency
- [x] **Whisper processor with `language="pa"`, `task="transcribe"`** -- prevents English hallucinations
- [x] **Seq2SeqTrainer with `predict_with_generate`** -- standard Whisper training loop
- [x] **WER compute_metrics** -- the only metric that matters
- [x] **Gradient checkpointing + fp16** -- fits training in A40 VRAM, 2x speed
- [x] **Checkpoint save every 300 steps + HF Hub push** -- $0 lost work guarantee
- [x] **Training resume from checkpoint** -- handles RunPod interruptions
- [x] **Discriminative learning rates** -- critical for Gurmukhi adaptation quality
- [x] **Audio augmentation (noise, reverb, stretch, pitch)** -- bridges clean sehaj path to noisy kirtan target domain
- [x] **Mool Mantar as initial_prompt during eval** -- free accuracy gain
- [x] **GPU auto-detection with batch size adjustment** -- works on whatever RunPod assigns
- [x] **Manual spot-check script** -- 5 examples decoded and printed for human review

### Add After Validation (v1.x — After Phase 1 Produces Acceptable WER)

Features to add once `surt_small_v1` meets exit criteria (WER < 15% sehaj path).

- [ ] **Gurmukhi text normalization for WER** -- trigger: WER looks worse than actual quality due to diacritic mismatches
- [ ] **TensorBoard logging** -- trigger: need to diagnose loss curves across multiple training attempts
- [ ] **Formal evaluation harness** -- trigger: entering Phase 3 curriculum training where go/no-go decisions are automated
- [ ] **Learning rate scheduler (cosine/linear decay)** -- trigger: training shows late-stage divergence or oscillation

### Future Consideration (v2+ — Phases 2-5)

Features to defer until Phase 1 model is validated and downstream phases begin.

- [ ] **Curriculum data loading (gold before silver)** -- Phase 3 specific
- [ ] **Confidence-based data filtering** -- Phase 2/3 for auto-aligned kirtan data
- [ ] **Distillation trainer (teacher-student)** -- Phase 4 specific
- [ ] **ONNX export + INT8 quantization** -- Phase 5 specific
- [ ] **Multi-GPU / FSDP training** -- Phase 3 if 800h dataset requires it (unlikely on A40)

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| HF dataset streaming | HIGH | LOW | P1 |
| Seq2SeqTrainer + predict_with_generate | HIGH | LOW | P1 |
| WER compute_metrics | HIGH | LOW | P1 |
| Gradient checkpointing + fp16 | HIGH | LOW | P1 |
| Checkpoint save + HF Hub push | HIGH | MEDIUM | P1 |
| Training resume from checkpoint | HIGH | MEDIUM | P1 |
| Whisper language/task config | HIGH | LOW | P1 |
| Label masking + data collator | HIGH | LOW | P1 |
| Warmup steps | HIGH | LOW | P1 |
| Discriminative learning rates | HIGH | MEDIUM | P1 |
| Audio augmentation | HIGH | MEDIUM | P1 |
| Mool Mantar initial_prompt | MEDIUM | LOW | P1 |
| GPU auto-detection | MEDIUM | LOW | P1 |
| Gurmukhi text normalization | MEDIUM | MEDIUM | P2 |
| Structured logging | MEDIUM | LOW | P2 |
| TensorBoard integration | LOW | LOW | P3 |
| Evaluation harness automation | MEDIUM | HIGH | P3 |
| Custom LR scheduler | LOW | MEDIUM | P3 |

**Priority key:**
- P1: Must have for Phase 1 training run (training fails or wastes money without it)
- P2: Should have, add if time permits before first training run
- P3: Nice to have, defer to later phases or subsequent training iterations

## Comparable Pipeline Analysis

| Feature | HF Blog "Fine-Tune Whisper" | Distil-Whisper (HF) | IndicWhisper (AI4Bharat) | Our Approach (Surt Phase 1) |
|---------|----------------------------|---------------------|--------------------------|----------------------------|
| Base fine-tuning loop | Seq2SeqTrainer, single LR | Custom distillation trainer | Seq2SeqTrainer, single LR | Seq2SeqTrainer, **discriminative LR** |
| Data augmentation | None (clean data assumed) | None (distillation, not fine-tuning) | SpecAugment on spectrograms | **Waveform augmentation** (noise, reverb, stretch, pitch) |
| Checkpoint safety | push_to_hub=True | Local + Hub | Local only | **Custom callback: Hub push on every best eval** |
| Language handling | Single language token | Multi-language | Multi-language with language ID | **Single language (pa) + Mool Mantar prompt** |
| Compute target | Colab / single GPU | Multi-GPU (A100 cluster) | Multi-GPU | **Single A40 with gradient checkpointing** |
| Dataset handling | In-memory | Streaming | In-memory | **Streaming from HF Hub** |
| Eval strategy | WER only | WER + downstream tasks | WER + CER | **WER + manual spot-check** |

## Sources

- HuggingFace Transformers v4.56.2 documentation (via Context7, HIGH confidence): Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration, gradient checkpointing, training arguments API
- audiomentations documentation (via Context7, HIGH confidence): Compose, AddGaussianNoise, RoomSimulator, TimeStretch, PitchShift parameters and usage
- Project training plan `surt_training_plan.md` (project-specific, HIGH confidence): Phase structure, model lineage, exit criteria, compute constraints
- PROJECT.md (project-specific, HIGH confidence): Requirements, constraints, key decisions
- HuggingFace blog "Fine-Tune Whisper For Multilingual ASR" (MEDIUM confidence, training data knowledge): Standard pipeline pattern for Whisper fine-tuning
- AI4Bharat IndicWhisper (MEDIUM confidence, training data knowledge): Indic language ASR fine-tuning patterns, SpecAugment usage
- Distil-Whisper paper and codebase (MEDIUM confidence, training data knowledge): Knowledge distillation patterns for Whisper

---
*Feature research for: Whisper fine-tuning pipeline for Gurmukhi ASR*
*Researched: 2026-04-04*
