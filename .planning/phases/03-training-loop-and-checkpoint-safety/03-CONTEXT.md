# Phase 3: Training Loop and Checkpoint Safety - Context

**Gathered:** 2026-04-04
**Status:** Ready for planning

<domain>
## Phase Boundary

Wire Seq2SeqTrainer with discriminative learning rates, gradient checkpointing, bf16 precision, WER evaluation, and a checkpoint safety system that preserves progress to HuggingFace Hub. Training must resume seamlessly after spot instance preemption. Creates `surt/train.py` as the main training entry point.

</domain>

<decisions>
## Implementation Decisions

### max_steps target
- Hardcode `MAX_STEPS = 5000` in config.py (not calculated from epochs)
- ~5 effective epochs over 64k rows with effective batch 64
- No early stopping logic in this phase — run all 5,000 steps, pick best checkpoint afterward

### Hub push strategy
- Push on WER improvement ("best") PLUS periodic safety push (e.g., every 3rd eval regardless of WER)
- Push full model folder: model weights + processor + tokenizer + generation_config — Hub repo is instantly loadable with `from_pretrained()`
- Push to a **separate training repo** (e.g., `surindersinghssj/surt-small-v1-training`), NOT the final `surindersingh/surt-small-v1` repo — keep the main repo clean until training completes

### Resume behavior
- Auto-detect: if `OUTPUT_DIR` contains checkpoints, automatically resume from the latest — no flag needed
- Trust the checkpoint — no compatibility verification before resume
- Track best WER in a file (`best_wer.json` alongside checkpoints) so the Hub push callback knows the previous best WER after resume and doesn't re-push a regression as "best"

### LR scheduler
- Cosine decay to 0 after 400-step warmup (`lr_scheduler_type="cosine"`)
- Discriminative LRs: encoder 5e-5, decoder 1e-4, proj_out 1e-4

### Evaluation time budget
- Evaluation must consume less than 5% of total training wall time
- Optimize eval for speed: eval every 300 steps but keep eval set small (300 examples — already configured in config.py)
- If eval is too slow, reduce `generation_max_length` during eval or limit eval batches

### Claude's Discretion
- Whether to use early stopping (patience-based) or run all max_steps — lean toward simplest approach for v1
- Epoch estimation logging at training start (informational)
- Hub commit message format (include step number, WER, and push reason)
- LR schedule shape per parameter group (same cosine for all groups vs encoder-slower warmup)
- LR config style (explicit constants vs base+ratio)
- Weight decay value (standard 0.01 default is fine)
- Resume status logging format

</decisions>

<specifics>
## Specific Ideas

- "Make sure only 5% of training time is spent on eval, rest on actual training" — eval must be fast
- Training repo separate from final model repo — training pushes to `surt-small-v1-training`, final model gets a clean push to `surt-small-v1` in Phase 4
- Auto-resume is critical for spot instances — just re-run the same script, no flags needed
- best_wer.json tracks state across resume cycles so the Hub push callback preserves monotonic improvement

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `config.py`: BATCH_SIZE, GRAD_ACCUM, EVAL_STEPS, SAVE_STEPS, SAVE_TOTAL_LIMIT, WARMUP_STEPS, GENERATION_MAX_LENGTH, OUTPUT_DIR, LOG_DIR, HF_MODEL_REPO, BASE_MODEL already defined
- `model.py`: `load_model_and_processor()` returns (model, processor) fully configured for Punjabi
- `model.py`: `get_mool_mantar_prompt_ids(processor)` returns prompt_ids for eval generation anchoring
- `data.py`: `get_train_dataset()`, `get_val_dataset()`, `DataCollatorSpeechSeq2SeqWithPadding` — full data pipeline ready

### Established Patterns
- GPU auto-detection in config.py sets BATCH_SIZE and GRAD_ACCUM per GPU type
- Print-based logging: `[module] message` format (e.g., `[config]`, `[model]`, `[data]`)
- Processor injected as parameter (not imported) in data.py — same pattern for train.py

### Integration Points
- `train.py` will import from `surt.config`, `surt.model`, `surt.data`
- New constants needed in config.py: MAX_STEPS, LR values, training Hub repo name
- Custom callback class for Hub push logic (lives in train.py or a new callbacks.py)
- `__main__` entry point: `python -m surt.train` or `python surt/train.py`

</code_context>

<deferred>
## Deferred Ideas

- ETRAIN-01 (custom LR scheduler beyond cosine) — v2 requirement, cosine covers v1
- ETRAIN-02 (TensorBoard logging) — v2, console logging sufficient for single run
- ETRAIN-03 (optimizer state to Hub) — v2, local optimizer state + checkpoint resume is sufficient
- EVAL-01 (Gurmukhi text normalization for WER) — v2, raw jiwer WER for v1
- Final model push to `surt-small-v1` — Phase 4 (CKPT-04)

</deferred>

---

*Phase: 03-training-loop-and-checkpoint-safety*
*Context gathered: 2026-04-04*
