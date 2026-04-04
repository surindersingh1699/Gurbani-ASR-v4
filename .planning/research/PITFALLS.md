# Pitfalls Research

**Domain:** Whisper fine-tuning for Gurbani ASR (low-resource Indic, Gurmukhi script)
**Researched:** 2026-04-04
**Confidence:** MEDIUM-HIGH (verified via HuggingFace official blog, Context7 Transformers docs, HuggingFace datasets streaming docs, OpenAI Whisper GitHub discussions; RunPod-specific items rely on training data + community patterns)

## Critical Pitfalls

### Pitfall 1: Whisper Hallucinations on Silence and Non-Speech Audio

**What goes wrong:**
Whisper generates fluent but entirely fabricated text when fed silence, tabla solos, harmonium interludes, or crowd noise. This is not random garbage -- Whisper produces plausible-looking Gurmukhi (or worse, switches to English/Hindi) that passes casual inspection but is completely wrong. These hallucinated tokens then corrupt downstream FAISS retrieval, returning confident but incorrect shabad matches.

**Why it happens:**
Whisper was trained to always produce text output. It has no robust "this is not speech" signal for low-resource languages. The decoder is autoregressive and will happily generate tokens conditioned on its own prior outputs, creating self-reinforcing hallucination loops. With only ~0.06% Punjabi in pre-training data, the model's language model prior for Gurmukhi is weak, making it more prone to mode collapse into English or Hindi during non-speech segments.

**How to avoid:**
1. Implement VAD (Voice Activity Detection) as a pre-filter before sending audio to Whisper. Use `silero-vad` or `webrtcvad` to gate audio chunks -- only transcribe segments with detected speech energy above threshold.
2. Set `no_speech_threshold` in generation config (default 0.6, consider raising to 0.7-0.8 for this domain).
3. Post-filter: discard any transcription shorter than 5 Gurmukhi characters or containing Latin script characters.
4. During training, ensure the dataset does NOT contain silence-only segments with empty labels -- this teaches the model that silence = empty output, but Whisper's architecture fights this.

**Warning signs:**
- Evaluation outputs show English text mixed with Gurmukhi
- Repeated phrases or tokens in output (e.g., the same tuk repeated 5 times)
- Very high confidence scores on non-speech audio segments
- FAISS retrieval returns wildly different shabads for consecutive windows of the same audio

**Phase to address:**
Phase 1 (training data filtering + generation config) and Phase 5 (VAD pre-filter in inference pipeline)

---

### Pitfall 2: Sampling Rate Mismatch Silently Destroys Training

**What goes wrong:**
Audio is fed to Whisper at the wrong sampling rate (e.g., 44.1kHz or 48kHz instead of 16kHz). Whisper does not error -- it silently processes the audio, but the mel spectrogram is computed assuming 16kHz input. At 48kHz, the audio sounds like it is playing at 1/3 speed to the model. Training "converges" to a garbage model that produces nonsensical output.

**Why it happens:**
HuggingFace `Audio` feature requires explicit `sampling_rate=16000` casting. If the dataset is uploaded at its native sample rate (FLAC from recording equipment is often 44.1kHz or 48kHz) and this cast is forgotten, the WhisperProcessor happily computes features on the wrong-rate audio without raising any error.

**How to avoid:**
1. Always call `ds.cast_column("audio", Audio(sampling_rate=16000))` immediately after loading the dataset -- before any other processing.
2. Add an assertion in the preprocessing function: `assert audio["sampling_rate"] == 16000, f"Expected 16kHz, got {audio['sampling_rate']}Hz"`.
3. Spot-check by playing back a few samples after preprocessing to verify they sound correct at 16kHz.

**Warning signs:**
- Training loss decreases normally but WER on evaluation is terrible (>80%)
- Model outputs look like random token soup rather than Gurmukhi words
- Mel spectrograms look stretched or compressed compared to reference

**Phase to address:**
Phase 1 (data loading and preprocessing step -- must be correct from the very first training run)

---

### Pitfall 3: `forced_decoder_ids` Conflict with `generation_config`

**What goes wrong:**
The model generates in the wrong language (English or Hindi instead of Punjabi) or in the wrong task mode (translating instead of transcribing). Fine-tuning appears to work but the model's generation behavior is inconsistent. Older tutorials set `forced_decoder_ids` on the model config; newer Transformers versions use `generation_config`. Setting both or neither causes silent misbehavior.

**Why it happens:**
Whisper uses special decoder prefix tokens: `<|startoftranscript|> <|pa|> <|transcribe|> <|notimestamps|>`. These must be forced during generation. The HuggingFace Transformers API changed how this works between versions. Setting `model.config.forced_decoder_ids` is deprecated; the correct approach is `model.generation_config.language = "pa"` and `model.generation_config.task = "transcribe"` with `model.generation_config.forced_decoder_ids = None`. If you follow an outdated tutorial, you get silent misconfiguration.

**How to avoid:**
1. Use the current API pattern (verified via HuggingFace blog, HIGH confidence):
   ```python
   model.generation_config.language = "pa"
   model.generation_config.task = "transcribe"
   model.generation_config.forced_decoder_ids = None
   ```
2. After loading the model, print `model.generation_config` and verify `language`, `task`, and `forced_decoder_ids` are set correctly.
3. Test generation on a single sample before starting training to confirm output is Gurmukhi, not English.

**Warning signs:**
- Model outputs are in English or Hindi despite training on Gurmukhi data
- Outputs start with timestamp tokens when you did not want them
- Different behavior between `trainer.predict()` and manual `model.generate()` calls
- Deprecation warnings about `forced_decoder_ids` in training logs

**Phase to address:**
Phase 1 (model initialization -- must be correct before training begins)

---

### Pitfall 4: Streaming Dataset + Epoch-Based Training = Silent Failure

**What goes wrong:**
Training with `num_train_epochs=3` on a streaming `IterableDataset` either: (a) runs forever because the dataset has no known length, (b) silently trains on only a fraction of the data, or (c) crashes with an opaque error about dataset length. The Trainer cannot compute steps-per-epoch without knowing dataset size.

**Why it happens:**
HuggingFace `IterableDataset` (streaming=True) does not have a `__len__` method. The `Seq2SeqTrainer` needs the dataset length to compute total training steps from `num_train_epochs`. With streaming, you must use `max_steps` instead. The training plan currently specifies `num_train_epochs=3` -- this is incompatible with streaming mode.

**How to avoid:**
1. Use `max_steps` instead of `num_train_epochs` when using streaming datasets. Calculate: `max_steps = (num_examples * num_epochs) / (batch_size * gradient_accumulation_steps)`. For 100h at ~30s average = ~12,000 examples, 3 epochs with effective batch 32 = ~1,125 steps.
2. Set `dataloader_num_workers=0` for streaming datasets (Context7 verified: "set `dataloader_num_workers=0` for streaming datasets as they require special handling").
3. If the dataset is small enough (~9GB FLAC), consider downloading it fully rather than streaming. This allows epoch-based training and proper shuffling. On an A40 with 48GB VRAM, disk space is usually not the bottleneck.

**Warning signs:**
- Error: `TypeError: object of type 'IterableDataset' has no len()`
- Training shows 0 total steps in progress bar
- Training appears to complete instantly
- No proper epoch boundaries in logs

**Phase to address:**
Phase 1 (training configuration -- decide streaming vs. download before first run)

---

### Pitfall 5: Gurmukhi Tokenizer Vocabulary Starvation

**What goes wrong:**
Whisper's tokenizer has very few Gurmukhi-specific tokens. Most Gurmukhi text gets decomposed into byte-level fallback tokens (individual UTF-8 bytes). This means: (a) label sequences are 3-5x longer than equivalent English, (b) the model needs far more decoder capacity per word, (c) `generation_max_length=225` may truncate longer gurbani passages mid-word, producing broken Unicode. WER metrics become misleading because byte-level tokenization inflates the denominator.

**Why it happens:**
Whisper's multilingual tokenizer was trained on a web corpus dominated by English, Chinese, and major European languages. Punjabi constituted ~0.06% of training data, so the BPE tokenizer learned very few Gurmukhi subword units. A single Gurmukhi word like "ਗੁਰਪ੍ਰਸਾਦਿ" might require 8-12 tokens versus 1-2 for an English word.

**How to avoid:**
1. Set `generation_max_length` high enough -- at least 448 (Whisper's max). For gurbani lines, 225 is likely sufficient for single tuks but may truncate if the model runs away.
2. Monitor average label length (in tokens) during preprocessing. If mean exceeds 100 tokens, consider whether some examples are too long and should be split.
3. Do NOT attempt to retrain or extend the tokenizer -- this breaks the pre-trained encoder-decoder alignment and requires retraining from scratch.
4. Use character-level WER or CER (Character Error Rate) as a supplementary metric. Token-level WER is misleading for byte-fallback tokenization.

**Warning signs:**
- Average label sequence length is >100 tokens for short (5-10 second) audio clips
- Truncation warnings during tokenization
- Model outputs end mid-character with broken Unicode (e.g., partial UTF-8 byte sequences)
- WER appears high but actual transcriptions look reasonable on inspection

**Phase to address:**
Phase 1 (preprocessing and evaluation metric setup)

---

### Pitfall 6: Checkpoint Resume Restores Model but Not Optimizer/Scheduler State

**What goes wrong:**
After a RunPod session interruption, training resumes from a checkpoint but the learning rate scheduler resets to its initial value (or warmup restarts). The model experiences a learning rate spike mid-training, causing loss to spike and potentially destroying hours of fine-tuning progress. Discriminative learning rates (encoder 5e-5, decoder 1e-4) make this worse because the custom optimizer is not part of the standard Trainer checkpoint.

**Why it happens:**
The training plan uses a custom `AdamW` optimizer with discriminative LRs passed via `optimizers=(optimizer, None)`. When `Seq2SeqTrainer` saves a checkpoint, it saves `optimizer.pt` and `scheduler.pt` alongside the model. But if you reinitialize the custom optimizer in your script and pass it to the Trainer before resuming, the Trainer may use the NEW optimizer instead of loading the saved one. Additionally, the `PushBestToHub` callback only pushes model weights to HuggingFace -- not optimizer/scheduler state.

**How to avoid:**
1. Let the Trainer manage the optimizer: instead of passing `optimizers=(optimizer, None)`, configure discriminative LRs through the Trainer's parameter groups by subclassing `Seq2SeqTrainer.create_optimizer()`.
2. If using a custom optimizer, do NOT recreate it before `trainer.train(resume_from_checkpoint=...)`. Let the Trainer load it from the checkpoint directory.
3. Push full checkpoint directories (including `optimizer.pt`, `scheduler.pt`, `trainer_state.json`) to HuggingFace -- not just model weights. Or keep full checkpoints on persistent storage (Google Drive for Colab, network volume for RunPod).
4. After resuming, check `trainer.state.global_step` and verify the learning rate matches expectations for that step.

**Warning signs:**
- After resume, loss spikes sharply then slowly recovers (LR reset)
- Learning rate in logs shows warmup pattern restarting mid-training
- `trainer.state.global_step` shows 0 after resume instead of the checkpoint step
- Different WER before and after resume on the same eval set (model degraded)

**Phase to address:**
Phase 1 (checkpoint strategy design -- must be correct before the first session interruption)

---

### Pitfall 7: Label Padding Not Masked with -100 Causes Pad-Token Learning

**What goes wrong:**
The model learns to predict padding tokens as part of the transcription. During inference, this manifests as trailing garbage tokens, repeated `<pad>` tokens in output, or the model learning to "fill space" instead of stopping when the transcription is complete. Loss appears low but the model is partially learning a trivial padding-prediction task.

**Why it happens:**
When batching variable-length label sequences, shorter sequences are padded to match the longest in the batch. If these padding positions are not replaced with `-100` (PyTorch's ignore_index for cross-entropy loss), the model receives gradient signal to predict pad tokens. The standard HuggingFace Whisper data collator handles this, but custom data collators or preprocessing pipelines often miss it.

**How to avoid:**
1. In the data collator, replace padding token IDs with -100:
   ```python
   labels = labels_batch["input_ids"].masked_fill(
       labels_batch.attention_mask.ne(1), -100
   )
   ```
2. If preprocessing labels individually (as in the training plan), mask pad tokens:
   ```python
   batch["labels"] = [
       -100 if t == processor.tokenizer.pad_token_id else t
       for t in label_ids
   ]
   ```
3. Verify by inspecting a batch from the DataLoader: `labels` tensor should contain -100 wherever padding exists.

**Warning signs:**
- Training loss drops very quickly to near-zero (model is "cheating" by predicting pads)
- Inference outputs have trailing repeated tokens or silence tokens
- Model outputs are correct for the first few words then degrade

**Phase to address:**
Phase 1 (data collator implementation)

---

### Pitfall 8: BOS Token Doubled in Label Sequence

**What goes wrong:**
The `<|startoftranscript|>` (BOS) token appears twice at the start of every label sequence. The model learns to always produce a double-BOS at the start of generation, which either causes the first real token to be skipped or shifts the entire output by one position, degrading WER by a constant offset on every single example.

**Why it happens:**
The `WhisperProcessor.tokenizer()` may prepend BOS automatically. The `DataCollatorForSeq2Seq` or `Seq2SeqTrainer` then prepends it again as `decoder_start_token_id`. This double-prepend is a known gotcha documented in the HuggingFace fine-tuning blog. The fix is to check and strip the first BOS if present.

**How to avoid:**
1. In the data collator, check if BOS is already present and strip it:
   ```python
   if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
       labels = labels[:, 1:]
   ```
2. After tokenizing a sample label, print the token IDs and verify BOS appears exactly once.
3. Use the standard `DataCollatorForSeq2Seq` which handles this correctly when properly configured.

**Warning signs:**
- Every output starts with an unexpected token or space
- WER has a consistent ~2-5% offset that does not improve with training
- Decoded labels start with a blank or duplicate special token

**Phase to address:**
Phase 1 (data collator implementation)

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Skipping proper train/val/test split (using first 300 examples as val from training set) | Faster setup, no separate validation data needed | Validation WER is optimistically biased; model may overfit to validation examples that are also in training | Phase 1 MVP only -- create a proper held-out test set before making go/no-go decisions on Phase 2 |
| Using `save_total_limit=3` on RunPod without persistent storage | Saves disk space on ephemeral RunPod storage | If all 3 checkpoints are from a degraded training region (after a bad LR spike), there is no way to roll back to an earlier good state | Acceptable only if HuggingFace Hub push is verified working -- Hub is the real checkpoint store |
| Not implementing CER alongside WER | Simpler eval code | WER is misleading for byte-level tokenized Gurmukhi; you cannot meaningfully compare WER numbers to published benchmarks on other languages | Never -- always implement CER from the start |
| Hardcoding batch size based on GPU name string matching | Quick auto-config for known GPUs | Fails silently on GPUs with different VRAM than expected (e.g., A40 48GB vs A40 24GB variant), or on GPUs not in the if/else chain | Phase 1 only -- switch to `torch.cuda.mem_get_info()` based auto-tuning for later phases |
| Evaluating only on sehaj path, not kirtan | Faster iteration, matched domain | Sehaj path is clean studio-quality audio; kirtan has reverb, harmonium, tabla, congregation noise. Model may ace sehaj path but fail on the actual target domain | Never -- always maintain a small kirtan test set even in Phase 1 |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| HuggingFace Hub checkpoint push | Pushing only model weights (via `upload_folder` on best checkpoint) -- loses optimizer state, scheduler state, and trainer state needed for resume | Push full checkpoint directory OR maintain full checkpoints on persistent storage (Drive/network volume) separately from the Hub model push |
| HuggingFace streaming dataset | Using `num_workers > 0` in DataLoader with streaming IterableDataset | Set `dataloader_num_workers=0` for streaming datasets. Multi-worker loading with streaming causes data duplication or hangs (Context7 verified) |
| HuggingFace streaming shuffle | Assuming `.shuffle(buffer_size=1000)` provides true randomization | Shuffle buffer only randomizes within a sliding window of 1000 examples. If dataset is ordered (e.g., by recording session or ang number), the model sees correlated batches. Use `buffer_size >= 5000` or download the dataset for proper shuffling |
| RunPod SSH session | Assuming SSH connection staying alive means the pod stays alive | RunPod pods can be preempted (spot instances) or hit idle timeouts. Run training inside `tmux` or `screen` so training survives SSH disconnection. Also set a keep-alive on the SSH connection |
| RunPod disk storage | Saving checkpoints to pod's local filesystem assuming persistence | RunPod pod storage is ephemeral -- destroyed when pod stops. Use network volumes for persistent storage, or push to HuggingFace Hub immediately after each checkpoint save |
| audiomentations library | Applying augmentation to the raw audio array without ensuring float32 dtype | audiomentations expects `np.float32` arrays. Integer arrays or float64 will either error or produce clipped/distorted augmented audio. Always cast: `arr = np.array(audio["array"], dtype=np.float32)` |
| `evaluate` WER metric | Computing WER on raw tokenizer output (with special tokens) | Must use `skip_special_tokens=True` in `batch_decode` for both predictions and references, otherwise special tokens inflate WER |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Streaming dataset re-decodes audio every epoch | Training speed degrades over time; CPU utilization stays at 100% while GPU is idle | For a 9GB dataset that fits in memory, download fully and use a regular `Dataset` instead of streaming. Streaming makes sense for >50GB datasets | Immediately on the first epoch -- audio decoding becomes the bottleneck, not GPU compute |
| `predict_with_generate=True` with large eval set | Evaluation takes 10-30x longer than a training step because generate() is autoregressive; eval every 300 steps means training is mostly evaluating | Keep eval set small (200-500 examples) and increase `eval_steps` to 500-1000. Or disable `predict_with_generate` during training and only run generation-based eval at end | When eval takes >5 minutes and happens every 300 steps |
| `gradient_checkpointing=True` without necessity | Training speed drops ~20-30% | Only enable on GPUs with insufficient VRAM (T4 16GB, V100 16GB). On A40 48GB with batch=8, gradient checkpointing is likely unnecessary and just slows training | When using A40/A100 with moderate batch sizes |
| Mel spectrogram computed on CPU during preprocessing in streaming mode | GPU sits idle waiting for CPU preprocessing; effective GPU utilization <50% | Pre-compute features if possible. Or use `num_proc` with non-streaming dataset. With streaming, the feature extraction runs in the DataLoader worker -- with `num_workers=0` (required for streaming), this serializes on the main thread | From the very start of training -- this is the bottleneck for streaming audio datasets |

## "Looks Done But Isn't" Checklist

- [ ] **Model generation config:** Often missing explicit `language="pa"` and `task="transcribe"` -- verify by printing `model.generation_config` after initialization
- [ ] **Audio resampling:** Often missing `cast_column("audio", Audio(sampling_rate=16000))` -- verify by checking `sample["audio"]["sampling_rate"]` on a loaded example
- [ ] **Label padding masking:** Often missing `-100` replacement for pad tokens -- verify by printing a batch of labels and confirming `-100` appears where padding should be
- [ ] **BOS token deduplication:** Often missing the check for double-BOS -- verify by decoding a label sequence and checking for duplicate start tokens
- [ ] **Checkpoint resume completeness:** Often missing optimizer/scheduler state in pushed checkpoints -- verify by checking that `optimizer.pt` and `scheduler.pt` exist in checkpoint directory
- [ ] **Eval set independence:** Often using a slice of training data as eval -- verify that eval examples are not in the training set (especially with streaming + `.take()`)
- [ ] **WER computation correctness:** Often computing WER with special tokens included -- verify by manually decoding a prediction and checking for `<|...>` tokens in the string
- [ ] **Augmentation applied only to train:** Often accidentally augmenting eval data too -- verify that eval preprocessing path has `augment_audio=False`
- [ ] **RunPod tmux/screen:** Often starting training in bare SSH session -- verify training process persists after SSH disconnect test

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Hallucination on silence | MEDIUM | Add VAD pre-filter to inference pipeline; retrain with silence segments removed from dataset if severe; add `no_speech_threshold` to generation config |
| Sampling rate mismatch | HIGH | Must retrain from scratch -- the entire model learned wrong acoustic representations. No shortcut. Verify rate and restart Phase 1 |
| Wrong language/task tokens | HIGH | Must retrain from scratch -- model learned to decode into wrong language. Checkpoint is not salvageable |
| Streaming + epochs failure | LOW | Switch to `max_steps` or download the dataset. No retraining needed if caught before training starts |
| Tokenizer starvation (truncation) | LOW | Increase `generation_max_length` to 448. If labels were truncated during training, reprocess dataset with higher max length and retrain affected epochs |
| Checkpoint resume without optimizer | MEDIUM | If caught early, restart from the last good full checkpoint (Drive/Hub). If the model has trained many steps with wrong LR, those steps are partially wasted -- resume from the pre-spike checkpoint |
| Pad token learning | MEDIUM | Fix data collator and retrain. The corrupted checkpoint may be partially recoverable by continuing training with correct collator, but a fresh restart is cleaner |
| Double BOS token | LOW | Fix in data collator and retrain. Existing model can likely be corrected by continuing training with fixed labels for 200-500 steps |

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Hallucinations on silence/non-speech | Phase 1 (data filtering) + Phase 5 (VAD) | Generate on 5 silence-only clips; output should be empty or near-empty |
| Sampling rate mismatch | Phase 1 (data loading) | Assert `sr == 16000` on first batch; play back a sample |
| `forced_decoder_ids` / generation_config | Phase 1 (model init) | Generate one sample before training; verify output is Gurmukhi |
| Streaming + epochs incompatibility | Phase 1 (training config) | Verify `max_steps` is set (not `num_train_epochs`) when using streaming, OR download dataset and use regular Dataset |
| Gurmukhi tokenizer starvation | Phase 1 (preprocessing) | Print mean/max label token length; verify `generation_max_length` >= max observed |
| Checkpoint resume without optimizer | Phase 1 (checkpoint strategy) | After first checkpoint + resume cycle, verify `trainer.state.global_step > 0` and LR matches expected schedule |
| Pad token not masked with -100 | Phase 1 (data collator) | Inspect one training batch; labels tensor should contain -100 at pad positions |
| Double BOS token | Phase 1 (data collator) | Decode label[0] of a batch; verify single BOS |
| RunPod session death losing local data | Phase 1 (infra setup) | Verify HuggingFace Hub push succeeds; test resume from Hub checkpoint |
| Augmentation on eval set | Phase 1 (preprocessing) | Verify eval pipeline does not call augmentation functions |
| Shuffle buffer too small for streaming | Phase 1 (data loading) | If dataset is ordered (by ang/session), verify batches contain diverse examples |
| Sehaj-path-only eval masking kirtan weakness | Phase 1 (eval setup) | Maintain 10+ kirtan test clips; evaluate on both clean and noisy audio |
| Forced alignment confidence too low (Phase 2) | Phase 2 (alignment) | If <50% of 700h passes threshold, return to Phase 1 and improve base model WER first |
| Curriculum training from wrong checkpoint | Phase 3 (model loading) | Assert loaded model is `surt_small_v1`, NOT `openai/whisper-small`; print checkpoint path |
| Distillation projection dimension mismatch | Phase 4 (model setup) | Verify projection layer maps 768->384 exactly; test forward pass on one batch before training loop |

## Sources

- HuggingFace Blog: "Fine-Tune Whisper" (https://huggingface.co/blog/fine-tune-whisper) -- PRIMARY source for data collator gotchas, language token configuration, sampling rate issues, BOS token deduplication. HIGH confidence.
- HuggingFace Transformers Context7 docs -- Verified `Seq2SeqTrainingArguments`, streaming dataset requirements (`dataloader_num_workers=0`, `max_steps` vs epochs), gradient checkpointing guidance. HIGH confidence.
- HuggingFace Datasets Streaming docs (https://huggingface.co/docs/datasets/stream) -- Verified shuffle buffer behavior, `IterableDataset` limitations (no `__len__`), checkpoint/resume via `state_dict()`. HIGH confidence.
- OpenAI Whisper GitHub Discussions -- Community reports of hallucination issues, low-resource language struggles, domain-specific fine-tuning challenges. MEDIUM confidence (community reports, not official guidance).
- Project training plan (`surt_training_plan.md`) -- Domain-specific context for Gurbani/Gurmukhi, discriminative LR design decisions, checkpoint strategy. HIGH confidence (project-specific).
- RunPod session behavior, SSH stability, ephemeral storage patterns -- Based on training data + community patterns. MEDIUM confidence (not verified against current RunPod docs due to tool restrictions).

---
*Pitfalls research for: Whisper fine-tuning on Gurbani (Gurmukhi script, low-resource Indic ASR)*
*Researched: 2026-04-04*
