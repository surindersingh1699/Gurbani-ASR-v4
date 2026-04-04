"""
Training module for Gurbani ASR fine-tuning.

Provides the complete training pipeline for Whisper fine-tuning:
1. SurtTrainer -- Seq2SeqTrainer subclass with discriminative learning rates
   (encoder at 5e-5, decoder at 1e-4, proj_out at 1e-4)
2. make_compute_metrics -- Factory for WER computation via jiwer
3. build_training_args -- Seq2SeqTrainingArguments builder with all hyperparameters
4. HubPushCallback -- WER-gated + periodic Hub push with best_wer.json persistence
5. main() -- Entry point with auto-resume from latest checkpoint
"""

import json
import os
from pathlib import Path

import jiwer
import numpy as np
from huggingface_hub import HfApi
from torch.optim import AdamW
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint

from surt.config import (
    BATCH_SIZE,
    DECODER_LR,
    EFFECTIVE_BATCH,
    ENCODER_LR,
    EVAL_STEPS,
    GENERATION_MAX_LENGTH,
    GRAD_ACCUM,
    LEARNING_RATE,
    MAX_STEPS,
    OUTPUT_DIR,
    SAVE_STEPS,
    SAVE_TOTAL_LIMIT,
    TRAINING_HUB_REPO,
    WARMUP_STEPS,
    WEIGHT_DECAY,
)
# surt.data and surt.model are imported inside main() to avoid
# import-time dependency on audiomentations/torch model loading.
# This keeps `import surt.train` lightweight for testing.


class HubPushCallback(TrainerCallback):
    """Callback for WER-gated and periodic model pushes to HuggingFace Hub.

    Implements a two-pronged push strategy for checkpoint safety:
    1. Best push: Upload when WER improves (WER-gated)
    2. Periodic safety push: Upload every Nth eval regardless of WER

    Persists best WER to best_wer.json in OUTPUT_DIR so the metric survives
    across resume cycles on spot instances.

    Pushes the full model folder (weights + processor + tokenizer +
    generation_config) to TRAINING_HUB_REPO, making it instantly loadable
    with from_pretrained().
    """

    def __init__(
        self,
        hub_repo: str,
        processor,
        output_dir: str,
        push_every_n_evals: int = 3,
    ):
        super().__init__()
        self.hub_repo = hub_repo
        self.processor = processor
        self.output_dir = Path(output_dir)
        self.push_every_n_evals = push_every_n_evals
        self.eval_count = 0
        self.api = HfApi()
        self.best_wer_path = self.output_dir / "best_wer.json"
        self.best_wer = self._load_best_wer()

    def _load_best_wer(self) -> float:
        """Load persisted best WER from disk, or return inf if not found."""
        if self.best_wer_path.exists():
            with open(self.best_wer_path) as f:
                data = json.load(f)
            wer = data.get("best_wer", float("inf"))
            print(f"[train] Loaded best WER: {wer} from {self.best_wer_path}")
            return wer
        print("[train] No previous best WER found, starting fresh")
        return float("inf")

    def _save_best_wer(self, wer: float, step: int):
        """Persist best WER and step to disk for resume survival."""
        with open(self.best_wer_path, "w") as f:
            json.dump({"best_wer": wer, "step": step}, f)

    def _push_to_hub(self, model, step: int, wer: float, reason: str):
        """Upload model + processor to Hub as a complete loadable folder."""
        staging_dir = self.output_dir / "hub_staging"
        model.save_pretrained(staging_dir)
        self.processor.save_pretrained(staging_dir)

        commit_msg = f"step {step} | WER {wer:.2f} | {reason}"
        self.api.upload_folder(
            repo_id=self.hub_repo,
            folder_path=str(staging_dir),
            commit_message=commit_msg,
        )
        print(f"[train] Hub push ({reason}): {commit_msg}")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """After each eval step, conditionally push to Hub."""
        self.eval_count += 1
        current_wer = metrics.get("eval_wer", float("inf"))
        step = state.global_step

        is_best = current_wer < self.best_wer
        is_periodic = (self.eval_count % self.push_every_n_evals) == 0

        if is_best:
            self.best_wer = current_wer
            self._save_best_wer(current_wer, step)

        if is_best or is_periodic:
            model = kwargs.get("model")
            reason = "best" if is_best else "periodic"
            self._push_to_hub(model, step, current_wer, reason)

        print(
            f"[train] Eval step {step}: WER={current_wer:.2f} "
            f"(best={self.best_wer:.2f}, evals={self.eval_count})"
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


def main():
    """Entry point for the Surt training pipeline.

    Wires together all components: model, data, trainer, Hub callback, and
    training args. Automatically resumes from the latest checkpoint if
    OUTPUT_DIR contains any -- no flag needed, just re-run the script.

    Usage:
        python -m surt.train
        python surt/train.py
    """
    # Deferred imports -- these pull in heavy dependencies (audiomentations,
    # model weights) that are only needed at training time.
    from surt.config import DATASET_NAME
    from surt.data import (
        DataCollatorSpeechSeq2SeqWithPadding,
        get_train_dataset,
        get_val_dataset,
    )
    from surt.model import load_model_and_processor

    # Step 1: Startup banner
    print("[train] === Surt Training Pipeline ===")
    print(f"[train] Output: {OUTPUT_DIR}")
    print(f"[train] Hub repo: {TRAINING_HUB_REPO}")

    # Step 2: Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 3: Auto-resume detection (CKPT-03)
    # If OUTPUT_DIR contains checkpoints, resume from the latest automatically.
    # No flag needed -- trust the checkpoint, no compatibility verification.
    last_ckpt = get_last_checkpoint(OUTPUT_DIR)
    if last_ckpt:
        print(f"[train] Resuming from checkpoint: {last_ckpt}")
    else:
        print("[train] Starting fresh training run")

    # Step 4: Load model and processor
    model, processor = load_model_and_processor()

    # Step 5: Load datasets
    train_dataset = get_train_dataset(DATASET_NAME, processor)
    val_dataset = get_val_dataset(DATASET_NAME, processor)

    # Step 6: Create data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # Step 7: Build training args
    training_args = build_training_args()

    # Step 8: Create Hub push callback
    hub_callback = HubPushCallback(
        hub_repo=TRAINING_HUB_REPO,
        processor=processor,
        output_dir=OUTPUT_DIR,
        push_every_n_evals=3,
    )

    # Step 9: Create trainer
    trainer = SurtTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(processor),
        processing_class=processor.feature_extractor,  # v5: processing_class, not tokenizer
        callbacks=[hub_callback],
    )

    # Step 10: Train with auto-resume
    # Epoch estimation: ~{MAX_STEPS * EFFECTIVE_BATCH / 64000} effective epochs
    print(
        f"[train] Starting training for {MAX_STEPS} steps "
        f"(~{MAX_STEPS * EFFECTIVE_BATCH / 64000:.1f} effective epochs "
        f"over ~64k examples)"
    )
    trainer.train(resume_from_checkpoint=last_ckpt)
    print("[train] Training complete!")


if __name__ == "__main__":
    main()
