#!/usr/bin/env python3
"""
Kirtan-only 500-step pilot training.

Loads kirtan dataset directly, normalizes columns inline,
filters to <=30s, trains Whisper Small from v1 checkpoint.
"""

from __future__ import annotations

import datetime
import os
import sys
import threading
import tempfile

# Disable cuDNN (same workaround as main training)
import torch
torch.backends.cudnn.enabled = False

import jiwer
import numpy as np
from datasets import load_dataset
from huggingface_hub import HfApi
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
KIRTAN_DATASET = "surindersinghssj/gurbani-kirtan-dataset-v2"
OUTPUT_DIR = "/workspace/surt/checkpoints/kirtan_only"
HUB_REPO = "surindersinghssj/surt-small-v2-kirtan-only-training"
FINAL_REPO = "surindersinghssj/surt-small-v2-kirtan-only"

MAX_STEPS = 500
EVAL_STEPS = 100
SAVE_STEPS = 250
LOGGING_STEPS = 10
BATCH_SIZE = 64
GRAD_ACCUM = 1
WARMUP_STEPS = 150
ENCODER_LR = 5e-5
DECODER_LR = 1e-5
WEIGHT_DECAY = 0.01
MAX_DURATION = 30.0
GENERATION_MAX_LENGTH = 448
NUM_WORKERS = 4

WANDB_PROJECT = "surt"
WANDB_ENTITY = "sabysurinder-surinder"


# ── Data helpers ────────────────────────────────────────────────────────
def _to_seconds(v):
    return float(v) if v is not None else 0.0


def load_kirtan_data(processor):
    """Load, filter, normalize, and featurize kirtan data."""
    print("[data] Loading kirtan dataset...")
    train_raw = load_dataset(KIRTAN_DATASET, split="train")
    val_raw = load_dataset(KIRTAN_DATASET, split="validation")
    print(f"[data] Raw: train={len(train_raw)}, val={len(val_raw)}")

    # Filter <=30s
    train_f = train_raw.filter(lambda x: _to_seconds(x.get("duration")) <= MAX_DURATION)
    val_f = val_raw.filter(lambda x: _to_seconds(x.get("duration")) <= MAX_DURATION)
    print(f"[data] Filtered <=30s: train={len(train_f)}, val={len(val_f)}")

    train_hours = sum(_to_seconds(x) for x in train_f["duration"]) / 3600
    val_hours = sum(_to_seconds(x) for x in val_f["duration"]) / 3600
    print(f"[data] Train: {train_hours:.1f}h, Val: {val_hours:.1f}h")

    # Featurize
    def prepare(batch):
        audio = batch["audio"]
        inputs = processor(
            audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="np"
        )
        batch["input_features"] = inputs.input_features[0]
        batch["labels"] = processor.tokenizer(batch["gurmukhi_text"]).input_ids
        return batch

    train_ds = train_f.map(prepare, remove_columns=train_f.column_names)
    val_ds = val_f.map(prepare, remove_columns=val_f.column_names)
    print(f"[data] Featurized: train={len(train_ds)}, val={len(val_ds)}")
    return train_ds, val_ds


# ── Data collator ───────────────────────────────────────────────────────
class DataCollator:
    def __init__(self, processor, decoder_start_token_id):
        self.processor = processor
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


# ── Hub push callback ───────────────────────────────────────────────────
class HubPushCallback(TrainerCallback):
    def __init__(self, hub_repo, processor, output_dir):
        self.hub_repo = hub_repo
        self.processor = processor
        self.output_dir = output_dir
        self.api = HfApi()
        self.best_wer = float("inf")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not state.is_world_process_zero:
            return
        step = state.global_step
        wer = metrics.get("eval_wer", float("inf"))
        cer = metrics.get("eval_cer", float("inf"))
        print(f"[train] Eval step {step}: WER={wer:.2f} CER={cer:.2f} (best={self.best_wer:.2f})")
        if wer < self.best_wer:
            self.best_wer = wer
            model = kwargs.get("model")
            staging = os.path.join(self.output_dir, f"hub_staging_{step}")
            model.save_pretrained(staging)
            self.processor.save_pretrained(staging)
            msg = f"kirtan_only step {step} | WER {wer:.2f} | CER {cer:.2f} | best"
            def _upload():
                try:
                    self.api.upload_folder(repo_id=self.hub_repo, folder_path=staging, commit_message=msg)
                    print(f"[train] Hub push: {msg}")
                except Exception as e:
                    print(f"[train] Hub push FAILED (non-fatal): {e}")
                finally:
                    import shutil
                    shutil.rmtree(staging, ignore_errors=True)
            t = threading.Thread(target=_upload, daemon=True)
            t.start()


# ── Discriminative LR trainer ──────────────────────────────────────────
class SurtTrainer(Seq2SeqTrainer):
    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        model = self.model
        NO_DECAY = {"bias", "layer_norm", "layernorm"}
        seen = set()
        enc_d, enc_nd, dec_d, dec_nd, proj_d, proj_nd = [], [], [], [], [], []
        for name, param in model.named_parameters():
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
        groups = [
            {"params": enc_d, "lr": ENCODER_LR, "weight_decay": WEIGHT_DECAY},
            {"params": enc_nd, "lr": ENCODER_LR, "weight_decay": 0.0},
            {"params": dec_d, "lr": DECODER_LR, "weight_decay": WEIGHT_DECAY},
            {"params": dec_nd, "lr": DECODER_LR, "weight_decay": 0.0},
            {"params": proj_d, "lr": DECODER_LR, "weight_decay": WEIGHT_DECAY},
            {"params": proj_nd, "lr": DECODER_LR, "weight_decay": 0.0},
        ]
        groups = [g for g in groups if g["params"]]
        self.optimizer = AdamW(groups)
        total = sum(len(g["params"]) for g in groups)
        print(f"[train] Optimizer: {total} param groups")
        return self.optimizer


# ── Metrics ─────────────────────────────────────────────────────────────
def make_compute_metrics(processor):
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = np.array(pred.label_ids, copy=True)
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = 100 * jiwer.wer(label_str, pred_str)
        cer = 100 * jiwer.cer(label_str, pred_str)
        return {"wer": wer, "cer": cer}
    return compute_metrics


# ── Main ────────────────────────────────────────────────────────────────
def main():
    print("[kirtan] === Kirtan-Only Pilot Training ===")

    # HF token login
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)
        print("[kirtan] HF Hub authenticated")

    # W&B setup
    has_wandb = bool(os.environ.get("WANDB_API_KEY"))
    if has_wandb:
        os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)
        os.environ.setdefault("WANDB_ENTITY", WANDB_ENTITY)
        os.environ.setdefault("WANDB_WATCH", "false")
        os.environ.setdefault("WANDB_LOG_MODEL", "false")
        print("[kirtan] W&B enabled")

    # Load model
    print(f"[kirtan] Loading model: {BASE_MODEL}")
    processor = WhisperProcessor.from_pretrained(BASE_MODEL)
    model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="punjabi", task="transcribe"
    )
    model.config.suppress_tokens = []
    model.generation_config.language = "punjabi"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = model.config.forced_decoder_ids
    print(f"[kirtan] Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    # Load data
    train_ds, val_ds = load_kirtan_data(processor)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Training args
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
        save_total_limit=2,
        predict_with_generate=True,
        generation_max_length=GENERATION_MAX_LENGTH,
        logging_steps=LOGGING_STEPS,
        dataloader_num_workers=NUM_WORKERS,
        load_best_model_at_end=False,
        push_to_hub=False,
        report_to=["wandb"] if has_wandb else "none",
        run_name=f"surt-kirtan-only-{datetime.datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
        remove_unused_columns=False,
    )

    collator = DataCollator(processor, model.config.decoder_start_token_id)
    callbacks = [HubPushCallback(HUB_REPO, processor, OUTPUT_DIR)]

    trainer = SurtTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=make_compute_metrics(processor),
        processing_class=processor.feature_extractor,
        callbacks=callbacks,
    )

    print(f"[kirtan] Starting training: {MAX_STEPS} steps, eval every {EVAL_STEPS}")
    trainer.train()
    print("[kirtan] Training complete!")

    # Final eval
    metrics = trainer.evaluate()
    print(f"[kirtan] Final: WER={metrics.get('eval_wer', '?'):.2f} CER={metrics.get('eval_cer', '?'):.2f}")

    # Push final model
    try:
        api = HfApi()
        with tempfile.TemporaryDirectory(prefix="surt-kirtan-final-") as staging:
            model.save_pretrained(staging)
            processor.save_pretrained(staging)
            api.upload_folder(
                repo_id=FINAL_REPO,
                folder_path=staging,
                commit_message=f"kirtan_only_pilot step {trainer.state.global_step}",
            )
        print(f"[kirtan] Final model pushed to {FINAL_REPO}")
    except Exception as e:
        print(f"[kirtan] Final push failed (non-fatal): {e}")
        print(f"[kirtan] Model safe in {HUB_REPO}")

    print("[kirtan] Done!")


if __name__ == "__main__":
    main()
