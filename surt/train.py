"""
Training module for Gurbani ASR fine-tuning.

Provides the core training machinery for Whisper fine-tuning:
1. SurtTrainer -- Seq2SeqTrainer subclass with discriminative learning rates
   (encoder at 5e-5, decoder at 1e-4, proj_out at 1e-4)
2. make_compute_metrics -- Factory for WER computation via jiwer
3. build_training_args -- Seq2SeqTrainingArguments builder with all hyperparameters

The Hub push callback and auto-resume entry point are added in Plan 02.
"""

import jiwer
import numpy as np
from torch.optim import AdamW
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from surt.config import (
    BATCH_SIZE,
    DECODER_LR,
    ENCODER_LR,
    EVAL_STEPS,
    GENERATION_MAX_LENGTH,
    GRAD_ACCUM,
    LEARNING_RATE,
    MAX_STEPS,
    OUTPUT_DIR,
    SAVE_STEPS,
    SAVE_TOTAL_LIMIT,
    WARMUP_STEPS,
    WEIGHT_DECAY,
)


class SurtTrainer(Seq2SeqTrainer):
    """Seq2SeqTrainer subclass with discriminative learning rates.

    Overrides create_optimizer() to build three AdamW parameter groups:
    - Encoder parameters at ENCODER_LR (5e-5) -- slower for transfer learning
    - Decoder parameters at DECODER_LR (1e-4) -- base rate
    - proj_out parameters at DECODER_LR (1e-4) -- base rate

    Handles the tied weight between proj_out.weight and
    model.decoder.embed_tokens.weight via id(param) deduplication.
    """

    def create_optimizer(self):
        """Create AdamW optimizer with three discriminative LR parameter groups."""
        if self.optimizer is not None:
            return self.optimizer

        model = self.model

        # Deduplicate by id(param) to handle tied weights.
        # proj_out.weight is tied with model.decoder.embed_tokens.weight --
        # both names appear in named_parameters() but point to the same tensor.
        # Without deduplication, the tied weight gets double gradient updates.
        seen_ids = set()
        encoder_params = []
        decoder_params = []
        proj_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad or id(param) in seen_ids:
                continue
            seen_ids.add(id(param))

            if "model.encoder" in name:
                encoder_params.append(param)
            elif "proj_out" in name:
                # Check proj_out BEFORE decoder because proj_out name
                # does not contain "decoder"
                proj_params.append(param)
            elif "model.decoder" in name:
                decoder_params.append(param)

        optimizer_grouped_parameters = [
            {"params": encoder_params, "lr": ENCODER_LR, "weight_decay": WEIGHT_DECAY},
            {"params": decoder_params, "lr": DECODER_LR, "weight_decay": WEIGHT_DECAY},
            {"params": proj_params, "lr": DECODER_LR, "weight_decay": WEIGHT_DECAY},
        ]

        self.optimizer = AdamW(optimizer_grouped_parameters)

        # Log parameter group sizes for verification
        total = sum(len(g["params"]) for g in optimizer_grouped_parameters)
        print(
            f"[train] Parameter groups: "
            f"encoder={len(encoder_params)}, "
            f"decoder={len(decoder_params)}, "
            f"proj_out={len(proj_params)}, "
            f"total={total}"
        )

        return self.optimizer


def make_compute_metrics(processor):
    """Factory function returning a compute_metrics callable for WER evaluation.

    Uses jiwer directly (not evaluate.load("wer")) for simplicity and
    compatibility. Handles the -100 padding mask by replacing with
    pad_token_id before decoding.

    Args:
        processor: WhisperProcessor with tokenizer for decoding.

    Returns:
        Callable that takes an EvalPrediction and returns {"wer": float}.
    """

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 padding with pad_token_id for decoding
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # Decode predictions and labels to text
        pred_str = processor.tokenizer.batch_decode(
            pred_ids, skip_special_tokens=True
        )
        label_str = processor.tokenizer.batch_decode(
            label_ids, skip_special_tokens=True
        )

        # Compute WER as percentage
        wer = 100 * jiwer.wer(label_str, pred_str)
        return {"wer": wer}

    return compute_metrics


def build_training_args() -> Seq2SeqTrainingArguments:
    """Build Seq2SeqTrainingArguments with all training hyperparameters.

    All values come from surt.config constants. Key design choices:
    - bf16=True (not fp16) for Ampere GPU numerical stability
    - gradient_checkpointing with use_reentrant=False for PyTorch 2.x
    - max_steps instead of num_train_epochs (streaming IterableDataset has no length)
    - eval_strategy (not evaluation_strategy) for transformers v5
    - dataloader_num_workers=0 for streaming IterableDataset safety
    - push_to_hub=False (custom HubPushCallback handles Hub pushes)
    - remove_unused_columns=False (streaming datasets may lack column_names)

    Returns:
        Configured Seq2SeqTrainingArguments.
    """
    args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        max_steps=MAX_STEPS,
        warmup_steps=WARMUP_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        weight_decay=WEIGHT_DECAY,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        predict_with_generate=True,
        generation_max_length=GENERATION_MAX_LENGTH,
        logging_steps=25,
        dataloader_num_workers=0,
        load_best_model_at_end=False,
        push_to_hub=False,
        report_to="none",
        remove_unused_columns=False,
    )

    print(
        f"[train] Training args: max_steps={args.max_steps}, "
        f"bf16={args.bf16}, lr={args.learning_rate}, "
        f"scheduler={args.lr_scheduler_type}"
    )

    return args
