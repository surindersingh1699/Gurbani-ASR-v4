"""
Training module for Gurbani ASR fine-tuning.

Provides:
1. SurtTrainer with discriminative learning rates.
2. WER/CER compute_metrics via jiwer.
3. Training arguments builder with streaming-safe defaults.
4. HubPushCallback for periodic and best-checkpoint pushes.
5. Phase 4 orchestration: pre-flight, smoke run, full run, final model push.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import tempfile
from pathlib import Path

import jiwer
import numpy as np
import torch
from huggingface_hub import HfApi
from torch.optim import AdamW

# Disable cuDNN — the bundled cuDNN 9.19 fails to initialize on some RunPod
# images (CUDNN_STATUS_NOT_INITIALIZED). CUDA conv fallback is ~5% slower but
# rock-solid. Safe to remove once the pod image ships a compatible cuDNN.
torch.backends.cudnn.enabled = False
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint

from surt.config import (
    AUX_TRAIN_DATASET_NAME,
    AUX_TRAIN_PROBABILITY,
    BATCH_SIZE,
    DECODER_LR,
    EFFECTIVE_BATCH,
    ENCODER_LR,
    EVAL_STEPS,
    GENERATION_MAX_LENGTH,
    GRAD_ACCUM,
    HF_MODEL_REPO,
    LEARNING_RATE,
    MAX_STEPS,
    OUTPUT_DIR,
    SAVE_STEPS,
    SAVE_TOTAL_LIMIT,
    TRAINING_HUB_REPO,
    WANDB_ENTITY,
    WANDB_PROJECT,
    WARMUP_STEPS,
    WEIGHT_DECAY,
)
# surt.data/surt.model/surt.smoke_test are imported inside runtime functions to avoid
# import-time dependency on audiomentations and model loading.


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

    def _push_to_hub(self, model, step: int, wer: float, cer: float, reason: str):
        """Upload model + processor to Hub as a complete loadable folder."""
        try:
            staging_dir = self.output_dir / "hub_staging"
            model.save_pretrained(staging_dir)
            self.processor.save_pretrained(staging_dir)

            commit_msg = f"step {step} | WER {wer:.2f} | CER {cer:.2f} | {reason}"
            self.api.upload_folder(
                repo_id=self.hub_repo,
                folder_path=str(staging_dir),
                commit_message=commit_msg,
            )
            print(f"[train] Hub push ({reason}): {commit_msg}")
        except Exception as e:
            print(f"[train] Hub push FAILED (non-fatal): {e}")
            print("[train] Training continues — will retry on next eval")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """After each eval step, conditionally push to Hub.

        Only runs Hub pushes and file writes on rank 0 to prevent race
        conditions in multi-GPU (DDP) training.
        """
        if not state.is_world_process_zero:
            return

        self.eval_count += 1
        current_wer = metrics.get("eval_wer", float("inf"))
        current_cer = metrics.get("eval_cer", float("inf"))
        step = state.global_step

        is_best = current_wer < self.best_wer
        is_periodic = (self.eval_count % self.push_every_n_evals) == 0

        if is_best:
            self.best_wer = current_wer
            self._save_best_wer(current_wer, step)

        if is_best or is_periodic:
            model = kwargs.get("model")
            reason = "best" if is_best else "periodic"
            self._push_to_hub(model, step, current_wer, current_cer, reason)

        print(
            f"[train] Eval step {step}: WER={current_wer:.2f} CER={current_cer:.2f} "
            f"(best_wer={self.best_wer:.2f}, evals={self.eval_count})"
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
        #
        # Each LR group is split into decay/no-decay subgroups.
        # Bias and LayerNorm params should NOT get weight decay (destabilizes training).
        NO_DECAY_KEYWORDS = {"bias", "layer_norm", "layernorm"}

        def _is_no_decay(name: str) -> bool:
            name_lower = name.lower()
            return any(kw in name_lower for kw in NO_DECAY_KEYWORDS)

        seen_ids = set()
        encoder_decay, encoder_no_decay = [], []
        decoder_decay, decoder_no_decay = [], []
        proj_decay, proj_no_decay = [], []

        for name, param in model.named_parameters():
            if not param.requires_grad or id(param) in seen_ids:
                continue
            seen_ids.add(id(param))

            no_decay = _is_no_decay(name)

            if "model.encoder" in name:
                (encoder_no_decay if no_decay else encoder_decay).append(param)
            elif "proj_out" in name:
                # Check proj_out BEFORE decoder because proj_out name
                # does not contain "decoder"
                (proj_no_decay if no_decay else proj_decay).append(param)
            elif "model.decoder" in name:
                (decoder_no_decay if no_decay else decoder_decay).append(param)

        optimizer_grouped_parameters = [
            {"params": encoder_decay, "lr": ENCODER_LR, "weight_decay": WEIGHT_DECAY},
            {"params": encoder_no_decay, "lr": ENCODER_LR, "weight_decay": 0.0},
            {"params": decoder_decay, "lr": DECODER_LR, "weight_decay": WEIGHT_DECAY},
            {"params": decoder_no_decay, "lr": DECODER_LR, "weight_decay": 0.0},
            {"params": proj_decay, "lr": DECODER_LR, "weight_decay": WEIGHT_DECAY},
            {"params": proj_no_decay, "lr": DECODER_LR, "weight_decay": 0.0},
        ]

        # Remove empty groups (proj_no_decay is typically empty)
        optimizer_grouped_parameters = [
            g for g in optimizer_grouped_parameters if g["params"]
        ]

        self.optimizer = AdamW(optimizer_grouped_parameters)

        # Log parameter group sizes for verification
        n_decay = sum(len(g["params"]) for g in optimizer_grouped_parameters if g["weight_decay"] > 0)
        n_no_decay = sum(len(g["params"]) for g in optimizer_grouped_parameters if g["weight_decay"] == 0)
        total = n_decay + n_no_decay
        print(
            f"[train] Parameter groups: "
            f"decay={n_decay}, no_decay={n_no_decay}, total={total}"
        )

        return self.optimizer


def make_compute_metrics(processor):
    """Factory function returning a compute_metrics callable for WER/CER evaluation.

    Uses jiwer directly (not evaluate.load("wer")) for simplicity and
    compatibility. Handles the -100 padding mask by replacing with
    pad_token_id before decoding.

    Args:
        processor: WhisperProcessor with tokenizer for decoding.

    Returns:
        Callable that takes an EvalPrediction and returns {"wer": float, "cer": float}.
    """

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = np.array(pred.label_ids, copy=True)

        # Replace -100 padding with pad_token_id for decoding
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # Decode predictions and labels to text
        pred_str = processor.tokenizer.batch_decode(
            pred_ids, skip_special_tokens=True
        )
        label_str = processor.tokenizer.batch_decode(
            label_ids, skip_special_tokens=True
        )

        # Compute WER/CER as percentage
        wer = 100 * jiwer.wer(label_str, pred_str)
        cer = 100 * jiwer.cer(label_str, pred_str)
        return {"wer": wer, "cer": cer}

    return compute_metrics


def build_training_args(
    *,
    output_dir: str = OUTPUT_DIR,
    max_steps: int = MAX_STEPS,
    eval_steps: int = EVAL_STEPS,
    save_steps: int = SAVE_STEPS,
    logging_steps: int = 25,
    report_to: str | list[str] = "none",
    run_name: str | None = None,
    dataloader_num_workers: int = 0,
) -> Seq2SeqTrainingArguments:
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
        output_dir=output_dir,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        max_steps=max_steps,
        warmup_steps=WARMUP_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        weight_decay=WEIGHT_DECAY,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=SAVE_TOTAL_LIMIT,
        predict_with_generate=True,
        generation_max_length=GENERATION_MAX_LENGTH,
        logging_steps=logging_steps,
        dataloader_num_workers=dataloader_num_workers,
        load_best_model_at_end=False,
        push_to_hub=False,
        report_to=report_to,
        run_name=run_name,
        remove_unused_columns=False,
    )

    print(
        f"[train] Training args: output_dir={args.output_dir}, "
        f"max_steps={args.max_steps}, "
        f"bf16={args.bf16}, lr={args.learning_rate}, "
        f"scheduler={args.lr_scheduler_type}"
    )

    return args


def run_training_job(
    *,
    dataset_name: str,
    output_dir: str,
    max_steps: int,
    eval_steps: int,
    save_steps: int,
    logging_steps: int,
    enable_hub_callback: bool,
    resume_from_last_checkpoint: bool,
    aux_dataset_name: str | None = None,
    aux_probability: float = AUX_TRAIN_PROBABILITY,
    enable_wandb: bool = False,
    run_name: str | None = None,
    streaming: bool = False,
):
    """Run one training job and return (trainer, processor)."""
    from surt.data import (
        DataCollatorSpeechSeq2SeqWithPadding,
        get_train_dataset,
        get_val_dataset,
    )
    from surt.model import load_model_and_processor

    print("[train] === Surt Training Job ===")
    print(f"[train] Output: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    last_ckpt = None
    if resume_from_last_checkpoint:
        last_ckpt = get_last_checkpoint(output_dir)
        if last_ckpt:
            print(f"[train] Resuming from checkpoint: {last_ckpt}")
        else:
            print("[train] Starting fresh training run")
    else:
        print("[train] Starting fresh training run (resume disabled)")

    model, processor = load_model_and_processor()
    train_dataset = get_train_dataset(
        dataset_name,
        processor,
        aux_dataset_name=aux_dataset_name,
        aux_probability=aux_probability,
        streaming=streaming,
    )
    val_dataset = get_val_dataset(dataset_name, processor)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    training_args = build_training_args(
        output_dir=output_dir,
        max_steps=max_steps,
        eval_steps=eval_steps,
        save_steps=save_steps,
        logging_steps=logging_steps,
        report_to=["wandb"] if enable_wandb else "none",
        run_name=run_name,
        dataloader_num_workers=4 if not streaming else 0,
    )

    callbacks = []
    if enable_hub_callback:
        callbacks = [
            HubPushCallback(
                hub_repo=TRAINING_HUB_REPO,
                processor=processor,
                output_dir=output_dir,
                push_every_n_evals=3,
            )
        ]

    trainer = SurtTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(processor),
        processing_class=processor.feature_extractor,
        callbacks=callbacks,
    )

    print(
        f"[train] Starting training for {max_steps} steps "
        f"(~{max_steps * EFFECTIVE_BATCH / 64000:.1f} effective epochs over ~64k examples)"
    )
    trainer.train(resume_from_checkpoint=last_ckpt)
    print("[train] Training complete")
    return trainer, processor


def validate_smoke_training(trainer, smoke_steps: int) -> None:
    """TEST-03: ensure smoke run shows usable loss/LR behavior."""
    history = trainer.state.log_history
    losses = [entry["loss"] for entry in history if "loss" in entry]
    lrs = [entry["learning_rate"] for entry in history if "learning_rate" in entry]

    if len(losses) < 2:
        raise AssertionError(
            f"TEST-03 failed: expected >=2 loss logs in {smoke_steps} steps, got {len(losses)}"
        )
    if min(losses[1:]) >= losses[0]:
        raise AssertionError(
            "TEST-03 failed: loss did not improve from initial logged value"
        )

    if len(lrs) < 2:
        raise AssertionError(
            f"TEST-03 failed: expected >=2 LR logs in {smoke_steps} steps, got {len(lrs)}"
        )
    if lrs[-1] <= lrs[0]:
        raise AssertionError(
            "TEST-03 failed: LR did not increase during warmup in smoke run"
        )

    print("[smoke] TEST-03 PASS")
    print(
        f"[smoke] Loss first={losses[0]:.4f}, best={min(losses):.4f}, "
        f"last={losses[-1]:.4f}"
    )
    print(f"[smoke] LR first={lrs[0]:.8f}, last={lrs[-1]:.8f}")


def push_final_model_to_hub(model, processor, *, repo_id: str, final_step: int) -> None:
    """CKPT-04: push final trained model and processor to HF_MODEL_REPO."""
    api = HfApi()
    with tempfile.TemporaryDirectory(prefix="surt-final-") as staging_dir:
        model.save_pretrained(staging_dir)
        processor.save_pretrained(staging_dir)
        commit_message = f"Phase 4 final - surt_small_v1 (step {final_step})"
        api.upload_folder(
            repo_id=repo_id,
            folder_path=staging_dir,
            commit_message=commit_message,
        )
    print(f"[train] Final model pushed to {repo_id} at step {final_step}")


def parse_args():
    parser = argparse.ArgumentParser(description="Surt training entry point")
    parser.add_argument(
        "--mode",
        choices=["full", "smoke", "phase4", "pilot"],
        default="full",
        help=(
            "full: run pre-flight + full training + final push; "
            "smoke: run pre-flight + 10-step smoke run; "
            "phase4: run pre-flight + smoke + full + final push; "
            "pilot: 1000-step eval-enabled pilot run (no final push)"
        ),
    )
    parser.add_argument(
        "--preset",
        choices=["none", "pilot", "full"],
        default="none",
        help=(
            "Optional schedule preset. "
            "pilot: steps=1000 eval/save=100 logging=10. "
            "full: keep full steps but eval/save cadence=200."
        ),
    )
    parser.add_argument(
        "--smoke-steps",
        type=int,
        default=10,
        help="Number of steps for smoke mode (default: 10)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional override for full training max steps",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=None,
        help="Optional override for evaluation cadence (steps)",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=None,
        help="Optional override for checkpoint save cadence (steps)",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=None,
        help="Optional override for logging cadence (steps)",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip TEST-01/TEST-02 pre-flight checks",
    )
    parser.add_argument(
        "--skip-final-push",
        action="store_true",
        help="Skip CKPT-04 final model push to HF_MODEL_REPO",
    )
    return parser.parse_args()


def main():
    """CLI entry point for full/smoke/phase4 training workflows."""
    args = parse_args()

    # Deferred imports for heavy training dependencies.
    from surt.config import DATASET_NAME
    from surt.model import load_model_and_processor
    from surt.smoke_test import run_preflight_checks

    print("[train] === Surt Training Pipeline ===")
    print(f"[train] Mode: {args.mode}")
    print(f"[train] Output: {OUTPUT_DIR}")
    print(f"[train] Training hub repo: {TRAINING_HUB_REPO}")
    print(f"[train] Final hub repo: {HF_MODEL_REPO}")
    print(
        f"[train] Aux train dataset: {AUX_TRAIN_DATASET_NAME} "
        f"(prob={AUX_TRAIN_PROBABILITY:.2f})"
    )

    run_smoke = args.mode in {"smoke", "phase4"}
    run_full = args.mode in {"full", "phase4"}
    run_pilot = args.mode == "pilot"
    has_wandb_key = bool(os.environ.get("WANDB_API_KEY"))
    enable_wandb = has_wandb_key
    if enable_wandb:
        os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)
        os.environ.setdefault("WANDB_ENTITY", WANDB_ENTITY)
        os.environ.setdefault("WANDB_WATCH", "false")
        os.environ.setdefault("WANDB_LOG_MODEL", "false")
        print(
            f"[train] Weights & Biases enabled "
            f"(project={os.environ.get('WANDB_PROJECT')}, entity={os.environ.get('WANDB_ENTITY')})"
        )
    else:
        print("[train] Weights & Biases disabled (WANDB_API_KEY not set)")

    if not args.skip_preflight:
        model, processor = load_model_and_processor()
        run_preflight_checks(model, processor, dataset_name=DATASET_NAME)
        del model
        del processor
    else:
        print("[smoke] Pre-flight checks skipped by flag")

    if run_smoke:
        smoke_steps = max(2, args.smoke_steps)
        smoke_dir = os.path.join(OUTPUT_DIR, "smoke_test")
        smoke_interval = min(5, smoke_steps)
        smoke_trainer, _ = run_training_job(
            dataset_name=DATASET_NAME,
            output_dir=smoke_dir,
            max_steps=smoke_steps,
            eval_steps=smoke_interval,
            save_steps=smoke_interval,
            logging_steps=1,
            enable_hub_callback=False,
            resume_from_last_checkpoint=False,
            aux_dataset_name=AUX_TRAIN_DATASET_NAME,
            aux_probability=AUX_TRAIN_PROBABILITY,
            enable_wandb=enable_wandb,
            run_name=f"surt-smoke-{datetime.datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
            streaming=True,  # streaming for smoke (no need to load 64k for 10 steps)
        )
        validate_smoke_training(smoke_trainer, smoke_steps=smoke_steps)
        if args.mode == "smoke":
            print("[smoke] Smoke-only mode complete")
            return

    if run_pilot:
        pilot_steps = args.max_steps if args.max_steps is not None else 1000
        pilot_eval_steps = 100
        pilot_save_steps = 100
        pilot_logging_steps = 10
        if args.preset == "full":
            print("[train] Note: --preset full ignored in --mode pilot")

        if args.eval_steps is not None:
            pilot_eval_steps = args.eval_steps
        if args.save_steps is not None:
            pilot_save_steps = args.save_steps
        if args.logging_steps is not None:
            pilot_logging_steps = args.logging_steps

        pilot_dir = os.path.join(OUTPUT_DIR, "pilot")
        pilot_trainer, _ = run_training_job(
            dataset_name=DATASET_NAME,
            output_dir=pilot_dir,
            max_steps=pilot_steps,
            eval_steps=pilot_eval_steps,
            save_steps=pilot_save_steps,
            logging_steps=pilot_logging_steps,
            enable_hub_callback=True,
            resume_from_last_checkpoint=True,
            aux_dataset_name=AUX_TRAIN_DATASET_NAME,
            aux_probability=AUX_TRAIN_PROBABILITY,
            enable_wandb=enable_wandb,
            run_name=f"surt-pilot-{datetime.datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
            streaming=True,
        )
        print(
            f"[train] Pilot complete at step {pilot_trainer.state.global_step} "
            f"(WER/CER logged in eval metrics)"
        )
        return

    if run_full:
        full_steps = args.max_steps if args.max_steps is not None else MAX_STEPS
        full_eval_steps = EVAL_STEPS
        full_save_steps = SAVE_STEPS
        full_logging_steps = 25
        if args.preset == "pilot":
            full_steps = args.max_steps if args.max_steps is not None else 400
            full_eval_steps = 100
            full_save_steps = 100
            full_logging_steps = 10
            print(
                "[train] Using preset=pilot schedule for full-mode runner "
                "(no mode switch)."
            )
        elif args.preset == "full":
            full_eval_steps = 200
            full_save_steps = 200
            full_logging_steps = 25
            print("[train] Using preset=full cadence: eval/save every 200 steps.")

        if args.eval_steps is not None:
            full_eval_steps = args.eval_steps
        if args.save_steps is not None:
            full_save_steps = args.save_steps
        if args.logging_steps is not None:
            full_logging_steps = args.logging_steps

        full_trainer, full_processor = run_training_job(
            dataset_name=DATASET_NAME,
            output_dir=OUTPUT_DIR,
            max_steps=full_steps,
            eval_steps=full_eval_steps,
            save_steps=full_save_steps,
            logging_steps=full_logging_steps,
            enable_hub_callback=True,
            resume_from_last_checkpoint=True,
            aux_dataset_name=AUX_TRAIN_DATASET_NAME,
            aux_probability=AUX_TRAIN_PROBABILITY,
            enable_wandb=enable_wandb,
            run_name=f"surt-full-{datetime.datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
            streaming=False,
        )

        if args.skip_final_push:
            print("[train] Final push skipped by flag")
        else:
            try:
                push_final_model_to_hub(
                    full_trainer.model,
                    full_processor,
                    repo_id=HF_MODEL_REPO,
                    final_step=full_trainer.state.global_step,
                )
            except Exception as e:
                print(f"[train] Final push FAILED (non-fatal): {e}")
                print(
                    f"[train] Model is safe in {TRAINING_HUB_REPO} from callback pushes. "
                    "You can push manually later."
                )

        print("[train] Full training workflow complete")


if __name__ == "__main__":
    main()
