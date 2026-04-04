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

from transformers import WhisperForConditionalGeneration, WhisperProcessor

from surt.config import BASE_MODEL, GENERATION_MAX_LENGTH, MOOL_MANTAR


def load_model_and_processor() -> tuple[WhisperForConditionalGeneration, WhisperProcessor]:
    """Load Whisper model and processor configured for Punjabi/Gurmukhi transcription.

    Returns:
        Tuple of (model, processor) with language="punjabi", task="transcribe",
        forced_decoder_ids=None, and max_length=448 set on model.generation_config.
    """
    # Load processor with language and task
    processor = WhisperProcessor.from_pretrained(
        BASE_MODEL,
        language="punjabi",
        task="transcribe",
    )

    # Load model (full fine-tuning, no adapters)
    model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)

    # CRITICAL: Set language/task on model.generation_config, not just processor.
    # Without this, model.generate() ignores the processor settings and outputs English.
    model.generation_config.language = "punjabi"
    model.generation_config.task = "transcribe"

    # CRITICAL: Explicitly set forced_decoder_ids to None.
    # The default forces token sequences that conflict with language/task settings.
    # Must use generation_config (NOT model.config.forced_decoder_ids).
    model.generation_config.forced_decoder_ids = None

    # Set max generation length for Gurmukhi (3-5x longer tokenization than English)
    model.generation_config.max_length = GENERATION_MAX_LENGTH

    print(
        f"[model] Loaded {BASE_MODEL} with language=punjabi, "
        f"task=transcribe, max_length={GENERATION_MAX_LENGTH}"
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
