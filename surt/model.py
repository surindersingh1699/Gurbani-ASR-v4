"""
Whisper model and processor initialization for Gurmukhi/Punjabi transcription.

This module configures the Whisper model with the correct language, task, and
generation settings for Gurbani transcription. It addresses several critical
pitfalls identified during research:

1. Language misconfiguration: language="punjabi" and task="transcribe" must be
   set on BOTH the processor AND model.generation_config. Setting them only on
   the processor causes model.generate() to output English during evaluation.

2. forced_decoder_ids: Must be explicitly set to None on model.generation_config.
   The default value forces specific token sequences that conflict with the
   language/task settings. Using model.config.forced_decoder_ids is WRONG --
   it must be model.generation_config.forced_decoder_ids.

3. Mool Mantar prompt_ids: The Mool Mantar (opening verse of Guru Granth Sahib)
   is tokenized and available as prompt_ids for model.generate(). This anchors
   the model's vocabulary in Gurmukhi script during evaluation, improving
   transcription quality.
"""

import os

from transformers import WhisperForConditionalGeneration, WhisperProcessor

from surt.config import BASE_MODEL, GENERATION_MAX_LENGTH, MOOL_MANTAR


def _pick_attn_implementation() -> str | None:
    """Pick the best attention kernel available.

    Priority: flash_attention_2 (if flash-attn installed) > sdpa (built-in fast path).
    SDPA is always available in torch 2.x and gives ~10–15% over eager on Ampere.
    Override via SURT_ATTN_IMPL env var ("sdpa" | "flash_attention_2" | "eager").
    """
    forced = os.environ.get("SURT_ATTN_IMPL", "").strip()
    if forced:
        return forced
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ModuleNotFoundError:
        return "sdpa"


def load_model_and_processor() -> tuple[WhisperForConditionalGeneration, WhisperProcessor]:
    """Load Whisper model and processor configured for Punjabi/Gurmukhi transcription.

    Returns:
        Tuple of (model, processor) with language="punjabi", task="transcribe",
        forced_decoder_ids=None, max_length set on model.generation_config,
        and Mool Mantar prompt_ids wired onto generation_config for eval/inference.
    """
    model_id = os.environ.get("SURT_BASE_MODEL", BASE_MODEL)
    # Keep processor/tokenizer on canonical base model by default. Loading
    # processor assets from training checkpoints can fail across lib versions.
    processor_id = os.environ.get("SURT_PROCESSOR_MODEL", BASE_MODEL)
    attn_impl = _pick_attn_implementation()

    try:
        processor = WhisperProcessor.from_pretrained(
            processor_id,
            language="punjabi",
            task="transcribe",
            use_fast=False,
        )
    except Exception as e:
        if processor_id != BASE_MODEL:
            print(
                f"[model] Processor load failed from {processor_id}: {e}. "
                f"Falling back to {BASE_MODEL}."
            )
            processor_id = BASE_MODEL
            processor = WhisperProcessor.from_pretrained(
                processor_id,
                language="punjabi",
                task="transcribe",
                use_fast=False,
            )
        else:
            raise

    # Full fine-tune (no adapters). Attention kernel picked via env/auto.
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id,
        attn_implementation=attn_impl,
    )

    # CRITICAL — language/task must live on generation_config, not just processor.
    model.generation_config.language = "punjabi"
    model.generation_config.task = "transcribe"
    # CRITICAL — forced_decoder_ids must go on generation_config (not model.config).
    model.generation_config.forced_decoder_ids = None
    model.generation_config.max_length = GENERATION_MAX_LENGTH

    # Mool Mantar anchors Gurmukhi output at eval/inference — only wired when
    # SURT_USE_MOOL_PROMPT=1 (off by default during training since Whisper trains
    # teacher-forced; a training-time prompt biases decode patterns).
    if os.environ.get("SURT_USE_MOOL_PROMPT", "0") == "1":
        try:
            prompt_ids = get_mool_mantar_prompt_ids(processor)
            model.generation_config.prompt_ids = prompt_ids
            print(f"[model] Mool Mantar prompt_ids wired for generation ({len(prompt_ids)} tokens)")
        except Exception as e:
            print(f"[model] Mool Mantar prompt wiring failed (non-fatal): {e}")

    print(
        f"[model] Loaded model={model_id} processor={processor_id} | attn={attn_impl} | "
        f"lang=punjabi task=transcribe max_length={GENERATION_MAX_LENGTH}"
    )

    return model, processor


def get_mool_mantar_prompt_ids(processor: WhisperProcessor) -> list[int]:
    """Tokenize the Mool Mantar for use as prompt_ids during generation.

    The Mool Mantar (opening verse of Guru Granth Sahib) anchors the model's
    vocabulary in Gurmukhi script when passed as prompt_ids to model.generate().
    This helps prevent the model from drifting to English or other scripts
    during inference.

    Args:
        processor: WhisperProcessor with Punjabi tokenizer.

    Returns:
        List of token IDs for the Mool Mantar text.
    """
    token_ids = processor.tokenizer(
        MOOL_MANTAR, add_special_tokens=False
    ).input_ids

    return token_ids
