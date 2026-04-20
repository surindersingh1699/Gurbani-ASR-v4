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
import threading
from pathlib import Path

import jiwer
import numpy as np
import torch
from huggingface_hub import HfApi
from torch.optim import AdamW

# cuDNN gate. Historically disabled because cuDNN 9.19 on some RunPod images
# failed with CUDNN_STATUS_NOT_INITIALIZED. On pods where cuDNN initializes
# cleanly, keeping it on is a ~20–30% speedup (conv + matmul kernels).
# Behaviour:
#   - Try a 1-shot cudnn.init by running a tiny conv on GPU.
#   - If it errors, fall back to disabled (legacy safe path).
#   - Override with SURT_DISABLE_CUDNN=1 for manual escape hatch.
if os.environ.get("SURT_DISABLE_CUDNN", "0") == "1":
    torch.backends.cudnn.enabled = False
    print("[train] cuDNN force-disabled via SURT_DISABLE_CUDNN=1")
elif torch.cuda.is_available():
    try:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True  # autotune conv algos for fixed shapes
        _probe = torch.randn(1, 1, 8, 8, device="cuda")
        _ = torch.nn.functional.conv2d(_probe, torch.randn(1, 1, 3, 3, device="cuda"))
        del _probe
        torch.cuda.synchronize()
        print("[train] cuDNN enabled (probe OK, benchmark=True)")
    except Exception as e:
        torch.backends.cudnn.enabled = False
        print(f"[train] cuDNN probe failed, disabled: {e}")
else:
    torch.backends.cudnn.enabled = False

# TF32 matmuls on Ampere+ (free speedup in bf16/fp32 paths, numerically harmless here).
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint

from surt.config import (
    AUX_TRAIN_DATASET_NAME,
    AUX_TRAIN_PROBABILITY,
    BATCH_SIZE,
    DECODER_LR,
    EARLY_STOP_METRIC,
    EARLY_STOP_PATIENCE,
    EFFECTIVE_BATCH,
    ENCODER_LR,
    EVAL_STEPS,
    EXTRA_SEHAJ_DATASET_NAME,
    EXTRA_SEHAJ_TEXT_COLUMN,
    GENERATION_MAX_LENGTH,
    GRAD_ACCUM,
    HF_MODEL_REPO,
    KIRTAN_EVAL_DATASET_NAME,
    KIRTAN_EVAL_SPLIT,
    LEARNING_RATE,
    MAX_STEPS,
    OUTPUT_DIR,
    SAVE_STEPS,
    SAVE_TOTAL_LIMIT,
    SEHAJ_EVAL_DATASET_NAME,
    SEHAJ_EVAL_SPLIT,
    TRAINING_HUB_REPO,
    WANDB_ENTITY,
    WANDB_PROJECT,
    WARMUP_STEPS,
    WEIGHT_DECAY,
)
# surt.data/surt.model/surt.smoke_test are imported inside runtime functions to avoid
# import-time dependency on audiomentations and model loading.


class PlateauEarlyStopCallback(TrainerCallback):
    """Stop training when neither WER nor CER improves for `patience` consecutive evals.

    Watches a single eval split (e.g. "kirtan" or "sehaj_path") — multi-dataset
    eval fires on_evaluate once per split, and we only act on the matching one.
    Tracks per-metric best values independently; "improvement" on this eval means
    either WER or CER beat its own best. If BOTH stayed flat-or-worse for
    `patience` evals in a row, signal the trainer to stop.

    Env overrides:
      - SURT_EARLY_STOP_METRIC ("kirtan" | "sehaj_path" | "none") — split to watch
      - SURT_EARLY_STOP_PATIENCE (int) — how many stale evals before stopping
    """

    def __init__(self, split_name: str, patience: int):
        super().__init__()
        self.split_name = split_name
        self.patience = patience
        self.best_wer = float("inf")
        self.best_cer = float("inf")
        self.stale_count = 0
        self._wer_key = f"eval_{split_name}_wer"
        self._cer_key = f"eval_{split_name}_cer"

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not state.is_world_process_zero or not metrics:
            return
        if self._wer_key not in metrics or self._cer_key not in metrics:
            # Not our split — the other eval dataset fired this call
            return

        wer = metrics[self._wer_key]
        cer = metrics[self._cer_key]

        wer_improved = wer < self.best_wer
        cer_improved = cer < self.best_cer
        if wer_improved:
            self.best_wer = wer
        if cer_improved:
            self.best_cer = cer

        if wer_improved or cer_improved:
            self.stale_count = 0
            print(
                f"[early-stop] {self.split_name}: improved "
                f"(WER {wer:.2f} best {self.best_wer:.2f}, "
                f"CER {cer:.2f} best {self.best_cer:.2f}) — counter reset"
            )
            return

        self.stale_count += 1
        print(
            f"[early-stop] {self.split_name}: no improvement "
            f"({self.stale_count}/{self.patience}) — "
            f"WER {wer:.2f} (best {self.best_wer:.2f}), "
            f"CER {cer:.2f} (best {self.best_cer:.2f})"
        )
        if self.stale_count >= self.patience:
            print(
                f"[early-stop] STOP: neither WER nor CER improved on "
                f"{self.split_name} for {self.patience} consecutive evals"
            )
            control.should_training_stop = True


class HubPushCallback(TrainerCallback):
    """WER+CER-gated and periodic model pushes to HuggingFace Hub.

    Tracks four independent "best" values — WER and CER for both the sehaj
    and kirtan eval splits. A Hub push fires if ANY of the four metrics
    improves, plus a periodic safety push every N evals regardless.

    State persists in best_metrics.json under OUTPUT_DIR so all four bests
    survive spot-instance restarts.

    Pushes the full model folder (weights + processor + tokenizer +
    generation_config) to TRAINING_HUB_REPO, making it instantly loadable
    with from_pretrained().
    """

    _SPLITS = ("sehaj_path", "kirtan")
    _METRICS = ("wer", "cer")

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

        self.best_path = self.output_dir / "best_metrics.json"
        # legacy single-WER file — migrate on first run
        self._legacy_wer_path = self.output_dir / "best_wer.json"
        self.bests = self._load_bests()  # {split: {wer: float, cer: float, step: int}}

    def _empty_bests(self) -> dict:
        return {s: {m: float("inf") for m in self._METRICS} | {"step": -1} for s in self._SPLITS}

    def _load_bests(self) -> dict:
        """Load persisted best metrics, migrating legacy best_wer.json if found."""
        if self.best_path.exists():
            with open(self.best_path) as f:
                data = json.load(f)
            for split in self._SPLITS:
                data.setdefault(split, {m: float("inf") for m in self._METRICS} | {"step": -1})
                for m in self._METRICS:
                    data[split].setdefault(m, float("inf"))
                data[split].setdefault("step", -1)
            print(f"[train] Loaded best metrics from {self.best_path}: {data}")
            return data

        if self._legacy_wer_path.exists():
            with open(self._legacy_wer_path) as f:
                legacy = json.load(f)
            bests = self._empty_bests()
            bests["sehaj_path"]["wer"] = legacy.get("best_wer", float("inf"))
            bests["sehaj_path"]["step"] = legacy.get("step", -1)
            print(f"[train] Migrated legacy best_wer.json → best_metrics.json: {bests}")
            return bests

        print("[train] No previous best metrics found, starting fresh")
        return self._empty_bests()

    def _save_bests(self):
        """Persist all best metrics to disk for resume survival."""
        with open(self.best_path, "w") as f:
            json.dump(self.bests, f, indent=2)

    def _push_to_hub(self, model, step: int, reason: str, metrics_blurb: str):
        """Upload model + processor to Hub in a background thread.

        Saves weights synchronously (fast, local I/O) then uploads
        asynchronously so training is never blocked by slow Hub uploads.
        """
        try:
            staging_dir = self.output_dir / f"hub_staging_{step}"
            model.save_pretrained(staging_dir)
            self.processor.save_pretrained(staging_dir)

            commit_msg = f"step {step} | {metrics_blurb} | {reason}"

            def _upload():
                try:
                    self.api.upload_folder(
                        repo_id=self.hub_repo,
                        folder_path=str(staging_dir),
                        commit_message=commit_msg,
                    )
                    print(f"[train] Hub push ({reason}): {commit_msg}")
                except Exception as e:
                    print(f"[train] Hub push FAILED (non-fatal): {e}")
                finally:
                    import shutil
                    shutil.rmtree(staging_dir, ignore_errors=True)

            t = threading.Thread(target=_upload, daemon=True)
            t.start()
            print(f"[train] Hub upload queued in background: step {step}")
        except Exception as e:
            print(f"[train] Hub save FAILED (non-fatal): {e}")
            print("[train] Training continues — will retry on next eval")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """After each eval step, conditionally push to Hub.

        Multi-dataset eval fires on_evaluate once per split. We count the
        PAIR of evals (sehaj + kirtan) as one "eval" for periodic-push
        accounting, driven off sehaj_path calls (which fire first).
        """
        if not state.is_world_process_zero or not metrics:
            return

        step = state.global_step

        # Identify which split this eval is for (multi-eval fires once per split)
        is_sehaj = "eval_sehaj_path_wer" in metrics
        is_kirtan = "eval_kirtan_wer" in metrics
        if not (is_sehaj or is_kirtan):
            # Single-split legacy path (no aux) — treat as sehaj_path
            is_sehaj = "eval_wer" in metrics

        split = "sehaj_path" if is_sehaj else "kirtan"
        wer_key = f"eval_{split}_wer" if f"eval_{split}_wer" in metrics else "eval_wer"
        cer_key = f"eval_{split}_cer" if f"eval_{split}_cer" in metrics else "eval_cer"

        current_wer = metrics.get(wer_key, float("inf"))
        current_cer = metrics.get(cer_key, float("inf"))

        prev_wer = self.bests[split]["wer"]
        prev_cer = self.bests[split]["cer"]
        improved: list[str] = []
        if current_wer < prev_wer:
            self.bests[split]["wer"] = current_wer
            improved.append(f"{split}_WER")
        if current_cer < prev_cer:
            self.bests[split]["cer"] = current_cer
            improved.append(f"{split}_CER")
        if improved:
            self.bests[split]["step"] = step
            self._save_bests()

        # Periodic-push accounting happens once per pair — sehaj call is the
        # anchor since Trainer fires it first in the eval_dataset dict.
        is_periodic = False
        if is_sehaj:
            self.eval_count += 1
            is_periodic = (self.eval_count % self.push_every_n_evals) == 0

        is_best = bool(improved)
        print(
            f"[train] Eval step {step}: {split} "
            f"WER={current_wer:.2f} (best {self.bests[split]['wer']:.2f}) "
            f"CER={current_cer:.2f} (best {self.bests[split]['cer']:.2f})"
            + (f" — improved: {','.join(improved)}" if improved else "")
        )

        if is_best or is_periodic:
            model = kwargs.get("model")
            reason = "best-" + "+".join(improved) if is_best else "periodic"
            metrics_blurb = (
                f"{split} WER {current_wer:.2f} CER {current_cer:.2f}"
            )
            self._push_to_hub(model, step, reason, metrics_blurb)


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


def _should_disable_grad_ckpt(gpu_name: str) -> bool:
    """Disable gradient_checkpointing on 48GB+ Ampere/Hopper GPUs.

    gradient_checkpointing trades ~35% throughput for VRAM. On A40/A6000/A100/H100
    with Whisper-small + batch 64 + bf16, VRAM peaks at ~30GB — plenty of headroom
    without checkpointing. Keep it ON for 24GB cards (4090/3090/A5000) where VRAM
    is tight.

    Override with SURT_GRAD_CKPT=1 or SURT_GRAD_CKPT=0 to force either way.
    """
    override = os.environ.get("SURT_GRAD_CKPT", "").strip()
    if override == "1":
        return False  # do NOT disable — user forced grad_ckpt ON
    if override == "0":
        return True   # DO disable — user forced grad_ckpt OFF
    roomy = any(tag in gpu_name for tag in ("A40", "A6000", "A100", "H100", "H200", "L40"))
    return roomy


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

    Key design choices:
    - bf16=True for Ampere numerical stability
    - gradient_checkpointing auto-decided by GPU VRAM (off on 48GB+ cards)
    - max_steps (not num_train_epochs) so streaming IterableDataset works
    - dataloader num_workers + persistent_workers + prefetch tuned for throughput
    - push_to_hub=False (custom HubPushCallback handles pushes)
    """
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    use_grad_ckpt = not _should_disable_grad_ckpt(gpu_name)

    # Dataloader knobs only apply when num_workers > 0 (non-streaming path).
    dl_kwargs = {}
    if dataloader_num_workers > 0:
        dl_kwargs = {
            "dataloader_persistent_workers": True,
            "dataloader_prefetch_factor": 4,
            "dataloader_pin_memory": True,
        }

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
        gradient_checkpointing=use_grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False} if use_grad_ckpt else None,
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
        **dl_kwargs,
    )
    print(
        f"[train] gpu={gpu_name}  grad_ckpt={use_grad_ckpt}  "
        f"dl_workers={dataloader_num_workers}  persistent={dl_kwargs.get('dataloader_persistent_workers', False)}"
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
        get_kirtan_val_dataset,
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

    # Extra sehaj override from env — useful for turning it off without
    # editing config (e.g. SURT_EXTRA_SEHAJ_DATASET="" disables).
    extra_sehaj = os.environ.get(
        "SURT_EXTRA_SEHAJ_DATASET", EXTRA_SEHAJ_DATASET_NAME
    ).strip() or None
    extra_sehaj_col = os.environ.get(
        "SURT_EXTRA_SEHAJ_TEXT_COLUMN", EXTRA_SEHAJ_TEXT_COLUMN
    ).strip() or None

    train_dataset = get_train_dataset(
        dataset_name,
        processor,
        aux_dataset_name=aux_dataset_name,
        aux_probability=aux_probability,
        streaming=streaming,
        extra_sehaj_dataset_name=extra_sehaj,
        extra_sehaj_text_column=extra_sehaj_col,
    )
    # v3: eval datasets live in separate Hub repos (SEHAJ_EVAL_DATASET_NAME,
    # KIRTAN_EVAL_DATASET_NAME). The training datasets (DATASET_NAME /
    # AUX_TRAIN_DATASET_NAME) are NOT used for eval — that would be a data leak.
    # Kirtan eval repo has `eval` + `test` splits — training ONLY reads `eval`
    # (KIRTAN_EVAL_SPLIT="eval"). The `test` split is reserved for post-training
    # reporting and must never be touched during training.
    val_dataset = get_val_dataset(
        SEHAJ_EVAL_DATASET_NAME, processor, split=SEHAJ_EVAL_SPLIT
    )

    # If aux training is enabled, we evaluate on both splits (sehaj + kirtan).
    # Otherwise just sehaj.
    eval_dataset = val_dataset
    if aux_dataset_name:
        eval_dataset = {"sehaj_path": val_dataset}
        try:
            kirtan_val = get_kirtan_val_dataset(
                KIRTAN_EVAL_DATASET_NAME, processor, split=KIRTAN_EVAL_SPLIT
            )
            eval_dataset["kirtan"] = kirtan_val
        except Exception as e:
            print(f"[train] WARNING: kirtan val set unavailable: {e}. Evaluating sehaj_path only.")

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
        dataloader_num_workers=0 if streaming else int(os.environ.get("SURT_DL_WORKERS", "8")),
    )

    callbacks = []
    if enable_hub_callback:
        callbacks.append(
            HubPushCallback(
                hub_repo=TRAINING_HUB_REPO,
                processor=processor,
                output_dir=output_dir,
                push_every_n_evals=3,
            )
        )

    # Early stopping: configurable split + patience via env.
    early_stop_split = os.environ.get("SURT_EARLY_STOP_METRIC", EARLY_STOP_METRIC).strip()
    early_stop_patience = int(
        os.environ.get("SURT_EARLY_STOP_PATIENCE", str(EARLY_STOP_PATIENCE))
    )
    if early_stop_split != "none" and early_stop_patience > 0:
        # Only attach when that split will actually evaluate — watching kirtan
        # but running sehaj-only eval would never trigger.
        split_available = (
            (early_stop_split == "kirtan" and aux_dataset_name)
            or (early_stop_split == "sehaj_path")
        )
        if split_available:
            callbacks.append(
                PlateauEarlyStopCallback(
                    split_name=early_stop_split,
                    patience=early_stop_patience,
                )
            )
            print(
                f"[train] Early stop armed: patience={early_stop_patience} "
                f"on eval_{early_stop_split}_{{wer,cer}}"
            )
        else:
            print(
                f"[train] Early stop disabled: split '{early_stop_split}' "
                f"not in current eval config (aux_dataset_name={aux_dataset_name})"
            )

    trainer = SurtTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
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


def push_model_to_hub(
    model,
    processor,
    *,
    repo_id: str,
    commit_message: str,
) -> None:
    """Push model and processor artifacts to a Hub repo."""
    api = HfApi()
    with tempfile.TemporaryDirectory(prefix="surt-final-") as staging_dir:
        model.save_pretrained(staging_dir)
        processor.save_pretrained(staging_dir)
        api.upload_folder(
            repo_id=repo_id,
            folder_path=staging_dir,
            commit_message=commit_message,
        )
    print(f"[train] Model pushed to {repo_id}: {commit_message}")


def push_final_model_to_hub(model, processor, *, repo_id: str, final_step: int) -> None:
    """CKPT-04: push final trained model and processor to HF_MODEL_REPO."""
    push_model_to_hub(
        model,
        processor,
        repo_id=repo_id,
        commit_message=f"surt_small_v3 final (step {final_step})",
    )


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
    parser.add_argument(
        "--pilot-push-repo",
        type=str,
        default=None,
        help="Optional Hub repo to upload model+processor after pilot completes",
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
        pilot_trainer, pilot_processor = run_training_job(
            dataset_name=DATASET_NAME,
            output_dir=pilot_dir,
            max_steps=pilot_steps,
            eval_steps=pilot_eval_steps,
            save_steps=pilot_save_steps,
            logging_steps=pilot_logging_steps,
            enable_hub_callback=True,
            resume_from_last_checkpoint=False,
            aux_dataset_name=AUX_TRAIN_DATASET_NAME,
            aux_probability=AUX_TRAIN_PROBABILITY,
            enable_wandb=enable_wandb,
            run_name=f"surt-pilot-{datetime.datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
            streaming=False,  # non-streaming: pre-cached data + num_workers=4 keeps GPU fed
        )
        print(
            f"[train] Pilot complete at step {pilot_trainer.state.global_step} "
            f"(WER/CER logged in eval metrics)"
        )
        if args.pilot_push_repo:
            print(f"[train] Uploading pilot model to {args.pilot_push_repo} ...")
            push_model_to_hub(
                pilot_trainer.model,
                pilot_processor,
                repo_id=args.pilot_push_repo,
                commit_message=(
                    "kirtan_first_pilot500 "
                    f"(step {pilot_trainer.state.global_step})"
                ),
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
            streaming=False,  # non-streaming: pre-cached data + num_workers=8 keeps GPU fed
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
