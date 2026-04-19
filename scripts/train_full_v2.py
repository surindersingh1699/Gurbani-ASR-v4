#!/usr/bin/env python3
"""
Full v2 training: sehaj + kirtan combined with on-the-fly audio processing.

Key fix: uses on-the-fly featurization in data collator instead of pre-computing
arrow cache (which bloats to 60+GB and crashes spot instances).

Dataset:  surindersinghssj/gurbani-sehaj-kirtan-full-v2 (69k rows, 93h)
Training: 3500 steps, eval every 200, warmup 150, save best+final
Base:     surindersinghssj/surt-small-v2-training
"""

from __future__ import annotations

import datetime
import json
import os
import tempfile
import threading
from pathlib import Path

import torch
torch.backends.cudnn.enabled = False

import jiwer
import numpy as np
from datasets import concatenate_datasets, load_dataset
from huggingface_hub import HfApi, login
from torch.optim import AdamW
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

# ── Config ──────────────────────────────────────────────────────────────
BASE_MODEL = "surindersinghssj/surt-small-v2-training"
COMBINED_DATASET = "surindersinghssj/gurbani-sehaj-kirtan-full-v2"
KIRTAN_DATASET = "surindersinghssj/gurbani-kirtan-dataset-v2"

OUTPUT_DIR = "/workspace/surt/checkpoints"
HUB_TRAINING_REPO = "surindersinghssj/surt-small-v2-training"
HUB_FINAL_REPO = "surindersinghssj/surt-small-v2"

MAX_STEPS = 3500
EVAL_STEPS = 200
SAVE_STEPS = 200
LOGGING_STEPS = 25
WARMUP_STEPS = 150
BATCH_SIZE = 64
GRAD_ACCUM = 1
ENCODER_LR = 5e-5
DECODER_LR = 1e-5
WEIGHT_DECAY = 0.01
GENERATION_MAX_LENGTH = 448
SAVE_TOTAL_LIMIT = 2
KIRTAN_MAX_DURATION = 30.0
KIRTAN_PROBABILITY = 0.60  # 60% kirtan, 40% sehaj during training
NUM_WORKERS = 4

WANDB_PROJECT = "surt"
WANDB_ENTITY = "sabysurinder-surinder"


# ── On-the-fly Data Collator ───────────────────────────────────────────
class OnTheFlyCollator:
    """Process audio to mel spectrograms on-the-fly, no arrow cache needed."""

    def __init__(self, processor, decoder_start_token_id, text_column="transcription"):
        self.processor = processor
        self.decoder_start_token_id = decoder_start_token_id
        self.text_column = text_column

    def __call__(self, features):
        # Extract audio and compute mel spectrograms on-the-fly
        input_features = []
        labels_list = []
        for f in features:
            audio = f["audio"]
            inputs = self.processor(
                audio["array"],
                sampling_rate=audio["sampling_rate"],
                return_tensors="np",
            )
            input_features.append({"input_features": inputs.input_features[0]})
            labels_list.append(
                {"input_ids": self.processor.tokenizer(f[self.text_column]).input_ids}
            )

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(labels_list, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


# ── Hub Push Callback ───────────────────────────────────────────────────
class HubPushCallback(TrainerCallback):
    def __init__(self, hub_repo, processor, output_dir, push_every_n_evals=3):
        self.hub_repo = hub_repo
        self.processor = processor
        self.output_dir = Path(output_dir)
        self.push_every_n_evals = push_every_n_evals
        self.eval_count = 0
        self.api = HfApi()
        self.best_wer = self._load_best_wer()

    def _load_best_wer(self):
        p = self.output_dir / "best_wer.json"
        if p.exists():
            with open(p) as f:
                wer = json.load(f).get("best_wer", float("inf"))
            print(f"[hub] Loaded best WER: {wer}")
            return wer
        return float("inf")

    def _save_best_wer(self, wer, step):
        with open(self.output_dir / "best_wer.json", "w") as f:
            json.dump({"best_wer": wer, "step": step}, f)

    def _push(self, model, step, wer, cer, reason):
        staging = self.output_dir / f"hub_staging_{step}"
        model.save_pretrained(staging)
        self.processor.save_pretrained(staging)
        msg = f"step {step} | WER {wer:.2f} | CER {cer:.2f} | {reason}"

        def _upload():
            try:
                self.api.upload_folder(
                    repo_id=self.hub_repo, folder_path=str(staging), commit_message=msg
                )
                print(f"[hub] Pushed ({reason}): {msg}")
            except Exception as e:
                print(f"[hub] Push FAILED (non-fatal): {e}")
            finally:
                import shutil
                shutil.rmtree(staging, ignore_errors=True)

        threading.Thread(target=_upload, daemon=True).start()

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not state.is_world_process_zero:
            return
        step = state.global_step

        # Kirtan eval — log only
        kirtan_wer = metrics.get("eval_kirtan_wer")
        if kirtan_wer is not None:
            print(
                f"[eval] Step {step}: kirtan WER={kirtan_wer:.2f} "
                f"CER={metrics.get('eval_kirtan_cer', 0):.2f}"
            )
            return

        # Sehaj eval — hub push logic
        wer = metrics.get("eval_sehaj_path_wer", metrics.get("eval_wer", float("inf")))
        cer = metrics.get("eval_sehaj_path_cer", metrics.get("eval_cer", float("inf")))
        self.eval_count += 1

        is_best = wer < self.best_wer
        is_periodic = (self.eval_count % self.push_every_n_evals) == 0

        if is_best:
            self.best_wer = wer
            self._save_best_wer(wer, step)

        if is_best or is_periodic:
            self._push(kwargs.get("model"), step, wer, cer, "best" if is_best else "periodic")

        print(
            f"[eval] Step {step}: sehaj WER={wer:.2f} CER={cer:.2f} "
            f"(best={self.best_wer:.2f})"
        )


# ── Discriminative LR Trainer ──────────────────────────────────────────
class SurtTrainer(Seq2SeqTrainer):
    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        NO_DECAY = {"bias", "layer_norm", "layernorm"}
        seen = set()
        enc_d, enc_nd, dec_d, dec_nd, proj_d, proj_nd = [], [], [], [], [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad or id(param) in seen:
                continue
            seen.add(id(param))
            nd = any(k in name.lower() for k in NO_DECAY)
            if "model.encoder" in name:
                (enc_nd if nd else enc_d).append(param)
            elif "proj_out" in name:
                (proj_nd if nd else proj_d).append(param)
            elif "model.decoder" in name:
                (dec_nd if nd else dec_d).append(param)
        groups = [g for g in [
            {"params": enc_d, "lr": ENCODER_LR, "weight_decay": WEIGHT_DECAY},
            {"params": enc_nd, "lr": ENCODER_LR, "weight_decay": 0.0},
            {"params": dec_d, "lr": DECODER_LR, "weight_decay": WEIGHT_DECAY},
            {"params": dec_nd, "lr": DECODER_LR, "weight_decay": 0.0},
            {"params": proj_d, "lr": DECODER_LR, "weight_decay": WEIGHT_DECAY},
            {"params": proj_nd, "lr": DECODER_LR, "weight_decay": 0.0},
        ] if g["params"]]
        self.optimizer = AdamW(groups)
        print(f"[train] Optimizer: {sum(len(g['params']) for g in groups)} params "
              f"(enc={ENCODER_LR}, dec={DECODER_LR})")
        return self.optimizer


def make_compute_metrics(processor):
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = np.array(pred.label_ids, copy=True)
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {
            "wer": 100 * jiwer.wer(label_str, pred_str),
            "cer": 100 * jiwer.cer(label_str, pred_str),
        }
    return compute_metrics


# ── Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print(f"[main] SURT v2 Full Training — {MAX_STEPS} steps")
    print(f"[main] Eval: {EVAL_STEPS}, Warmup: {WARMUP_STEPS}, Save: {SAVE_STEPS}")
    print(f"[main] Batch: {BATCH_SIZE}, Enc LR: {ENCODER_LR}, Dec LR: {DECODER_LR}")
    print(f"[main] Base: {BASE_MODEL}")
    print(f"[main] Dataset: {COMBINED_DATASET}")
    print(f"[main] Kirtan exposure: {KIRTAN_PROBABILITY*100:.0f}%")
    print("=" * 60)

    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        login(token=hf_token)
        print("[main] HF authenticated")

    has_wandb = bool(os.environ.get("WANDB_API_KEY"))
    if has_wandb:
        os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)
        os.environ.setdefault("WANDB_ENTITY", WANDB_ENTITY)
        os.environ.setdefault("WANDB_WATCH", "false")
        os.environ.setdefault("WANDB_LOG_MODEL", "false")
        print("[main] W&B enabled")

    # Load model
    print(f"[main] Loading model...")
    processor = WhisperProcessor.from_pretrained(BASE_MODEL)
    model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.language = "punjabi"
    model.generation_config.task = "transcribe"
    print(f"[main] Model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    # Load datasets, oversample kirtan to achieve 60% exposure
    print("[data] Loading sehaj + kirtan for oversampled training...")
    combined_ds = load_dataset(COMBINED_DATASET, split="train")
    val_ds = load_dataset(COMBINED_DATASET, split="validation")

    sehaj_train = combined_ds.filter(lambda x: x["source_dataset"] == "sehajpath")
    kirtan_train = combined_ds.filter(lambda x: x["source_dataset"] == "kirtan")
    print(f"[data] Sehaj: {len(sehaj_train)}, Kirtan: {len(kirtan_train)}")

    # Oversample kirtan to get ~60% proportion
    # target: kirtan_total / (sehaj + kirtan_total) = 0.60
    # kirtan_total = 0.60 * sehaj / 0.40 = 1.5 * sehaj
    target_kirtan = int(KIRTAN_PROBABILITY / (1 - KIRTAN_PROBABILITY) * len(sehaj_train))
    repeats = target_kirtan // len(kirtan_train)
    remainder = target_kirtan % len(kirtan_train)

    kirtan_parts = [kirtan_train] * repeats
    if remainder > 0:
        kirtan_parts.append(kirtan_train.select(range(remainder)))
    kirtan_oversampled = concatenate_datasets(kirtan_parts)

    train_ds = concatenate_datasets([sehaj_train, kirtan_oversampled]).shuffle(seed=42)
    actual_pct = len(kirtan_oversampled) / len(train_ds) * 100
    print(f"[data] Oversampled: {len(kirtan_oversampled)} kirtan + {len(sehaj_train)} sehaj = {len(train_ds)} total")
    print(f"[data] Kirtan: {actual_pct:.1f}%, Val: {len(val_ds)}")

    # Kirtan validation for separate eval
    kirtan_val = None
    try:
        kv = load_dataset(KIRTAN_DATASET, split="validation")
        kirtan_val = kv.filter(
            lambda x: float(x.get("duration", 0)) <= KIRTAN_MAX_DURATION
        )
        print(f"[data] Kirtan val: {len(kirtan_val)} examples")
    except Exception as e:
        print(f"[data] Kirtan val unavailable: {e}")

    eval_dataset = val_ds
    if kirtan_val is not None:
        eval_dataset = {"sehaj_path": val_ds, "kirtan": kirtan_val}

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Collators: sehaj uses "transcription", kirtan uses "gurmukhi_text"
    sehaj_collator = OnTheFlyCollator(processor, model.config.decoder_start_token_id, "transcription")
    kirtan_collator = OnTheFlyCollator(processor, model.config.decoder_start_token_id, "gurmukhi_text")

    # For multi-eval, trainer needs a single collator. We use a smart collator
    # that detects the text column based on what's available.
    class SmartCollator:
        def __init__(self, processor, decoder_start_token_id):
            self.processor = processor
            self.decoder_start_token_id = decoder_start_token_id

        def __call__(self, features):
            input_features = []
            labels_list = []
            for f in features:
                audio = f["audio"]
                inputs = self.processor(
                    audio["array"],
                    sampling_rate=audio["sampling_rate"],
                    return_tensors="np",
                )
                input_features.append({"input_features": inputs.input_features[0]})
                # Detect text column
                text = f.get("transcription") or f.get("gurmukhi_text", "")
                labels_list.append(
                    {"input_ids": self.processor.tokenizer(text).input_ids}
                )

            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
            labels_batch = self.processor.tokenizer.pad(labels_list, return_tensors="pt")
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100
            )
            if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
                labels = labels[:, 1:]
            batch["labels"] = labels
            return batch

    collator = SmartCollator(processor, model.config.decoder_start_token_id)

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        max_steps=MAX_STEPS,
        warmup_steps=WARMUP_STEPS,
        learning_rate=DECODER_LR,
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
        logging_steps=LOGGING_STEPS,
        dataloader_num_workers=NUM_WORKERS,
        load_best_model_at_end=False,
        push_to_hub=False,
        report_to=["wandb"] if has_wandb else "none",
        run_name=f"surt-v2-full-{datetime.datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
        remove_unused_columns=False,
    )

    callbacks = [HubPushCallback(HUB_TRAINING_REPO, processor, OUTPUT_DIR)]

    trainer = SurtTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=make_compute_metrics(processor),
        processing_class=processor.feature_extractor,
        callbacks=callbacks,
    )

    # Check for existing checkpoint to resume from
    from transformers.trainer_utils import get_last_checkpoint
    last_ckpt = get_last_checkpoint(OUTPUT_DIR)
    if last_ckpt:
        print(f"[main] Resuming from checkpoint: {last_ckpt}")
    else:
        print(f"[main] Starting fresh training")

    print(f"[main] Training for {MAX_STEPS} steps")
    trainer.train(resume_from_checkpoint=last_ckpt)
    print("[main] Training complete!")

    # Final push
    print(f"[main] Pushing final model to {HUB_FINAL_REPO}...")
    try:
        api = HfApi()
        with tempfile.TemporaryDirectory(prefix="surt-final-") as staging:
            model.save_pretrained(staging)
            processor.save_pretrained(staging)
            api.upload_folder(
                repo_id=HUB_FINAL_REPO,
                folder_path=staging,
                commit_message=f"surt_v2 final (step {trainer.state.global_step})",
            )
        print(f"[main] Final model pushed to {HUB_FINAL_REPO}")
    except Exception as e:
        print(f"[main] Final push FAILED: {e}")
        print(f"[main] Model safe in {HUB_TRAINING_REPO}")

    print("[main] === DONE ===")


if __name__ == "__main__":
    main()
