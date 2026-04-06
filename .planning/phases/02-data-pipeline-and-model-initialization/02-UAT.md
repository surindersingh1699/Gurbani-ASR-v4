---
status: testing
phase: 02-data-pipeline-and-model-initialization
source: 02-01-SUMMARY.md, 02-02-SUMMARY.md
started: 2026-04-04T14:00:00Z
updated: 2026-04-04T14:00:00Z
---

## Current Test
<!-- OVERWRITE each test - shows where we are -->

number: 1
name: Model module imports cleanly
expected: |
  Running `python -c "from surt.model import load_model_and_processor, get_mool_mantar_prompt_ids"` completes without error.
awaiting: user response

## Tests

### 1. Model module imports cleanly
expected: Running `python -c "from surt.model import load_model_and_processor, get_mool_mantar_prompt_ids"` completes without error.
result: [pending]

### 2. Whisper loads with Punjabi generation config
expected: After calling load_model_and_processor(), model.generation_config shows language="punjabi", task="transcribe", forced_decoder_ids=None, and max_length=448.
result: [pending]

### 3. Mool Mantar prompt IDs are non-empty
expected: Calling get_mool_mantar_prompt_ids(processor) returns a non-empty list of token IDs (137 tokens expected).
result: [pending]

### 4. Data module imports cleanly
expected: Running `python -c "from surt.data import get_train_dataset, get_val_dataset, DataCollatorSpeechSeq2SeqWithPadding"` completes without error.
result: [pending]

### 5. Dataset constants in config.py
expected: surt/config.py defines DATASET_NAME ("surindersinghssj/gurbani-asr"), TEXT_COLUMN ("transcription"), VAL_SIZE (300), and SHUFFLE_BUFFER.
result: [pending]

### 6. Data collator masks padding with -100
expected: DataCollatorSpeechSeq2SeqWithPadding replaces processor.tokenizer.pad_token_id with -100 in label tensors and strips double-BOS tokens.
result: [pending]

## Summary

total: 6
passed: 0
issues: 0
pending: 6
skipped: 0

## Gaps

[none yet]
