"""
Turbo distillation: surt-small-v3 -> surt-small-turbo-baseline-v0.

Recipe (whisper-large-v3-turbo style):
- Encoder: 12 x 768 unchanged, copied verbatim from teacher.
- Decoder: 12 -> 4 layers, init from teacher decoder layers [0, 4, 8, 11].
- Loss: alpha * KL(student, teacher_logits, T=2.0) + (1-alpha) * CE(student, labels).
- Teacher frozen in bf16 (no_grad forward, no optimizer state).

This is a BASELINE artifact (`surt-small-turbo-baseline-v0`). The teacher's
kirtan WER is 54.80%; the student inherits that ceiling -- only the latency /
size tradeoff is informative, not the absolute kirtan number.

A40-saturating schedule:
- Per-device batch 96, grad_accum 1 (effective 96)
- bf16 + SDPA / FA2, no grad-checkpointing
- max_steps 1500 (~1 epoch through ~144k mixed clips at batch 96)
- Encoder LR 1e-5 (encoder is essentially frozen)
- Decoder LR 5e-5 (the part actually being trained)
"""

from __future__ import annotations

import argparse
import datetime
import os

import torch
import torch.nn.functional as F
from transformers import (
    Seq2SeqTrainingArguments,
    WhisperConfig,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from transformers.trainer_utils import get_last_checkpoint

from surt.config import (
    AUX_TRAIN_DATASET_NAME,
    AUX_TRAIN_PROBABILITY,
    DATASET_NAME,
    EXTRA_SEHAJ_DATASET_NAME,
    EXTRA_SEHAJ_TEXT_COLUMN,
    GENERATION_MAX_LENGTH,
    KIRTAN_EVAL_DATASET_NAME,
    KIRTAN_EVAL_SPLIT,
    SAVE_TOTAL_LIMIT,
    SEHAJ_EVAL_DATASET_NAME,
    SEHAJ_EVAL_SPLIT,
    WANDB_ENTITY,
    WANDB_PROJECT,
    WEIGHT_DECAY,
)
from surt.train import (
    HubPushCallback,
    SurtTrainer,
    make_compute_metrics,
    push_model_to_hub,
)


# -- Distillation config --------------------------------------------------
TEACHER_REPO = os.environ.get("SURT_TEACHER_REPO", "surindersinghssj/surt-small-v3")
STUDENT_FINAL_REPO = os.environ.get(
    "SURT_STUDENT_REPO", "surindersinghssj/surt-small-turbo-baseline-v0"
)
STUDENT_TRAINING_REPO = os.environ.get(
    "SURT_STUDENT_TRAINING_REPO",
    "surindersinghssj/surt-small-turbo-baseline-v0-training",
)

STUDENT_DECODER_LAYERS = int(os.environ.get("SURT_STUDENT_DECODER_LAYERS", "4"))
STUDENT_INIT_LAYER_INDICES = [0, 4, 8, 11]

KD_TEMPERATURE = float(os.environ.get("SURT_KD_TEMP", "2.0"))
KD_ALPHA = float(os.environ.get("SURT_KD_ALPHA", "0.5"))

DISTILL_BATCH_SIZE = int(os.environ.get("SURT_DISTILL_BATCH", "96"))
DISTILL_GRAD_ACCUM = int(os.environ.get("SURT_DISTILL_GRAD_ACCUM", "1"))
DISTILL_MAX_STEPS = int(os.environ.get("SURT_DISTILL_MAX_STEPS", "1500"))
DISTILL_WARMUP_STEPS = int(os.environ.get("SURT_DISTILL_WARMUP", "150"))
DISTILL_EVAL_STEPS = int(os.environ.get("SURT_DISTILL_EVAL_STEPS", "250"))
DISTILL_SAVE_STEPS = int(os.environ.get("SURT_DISTILL_SAVE_STEPS", "250"))
DISTILL_ENCODER_LR = float(os.environ.get("SURT_DISTILL_ENCODER_LR", "1e-5"))
DISTILL_DECODER_LR = float(os.environ.get("SURT_DISTILL_DECODER_LR", "5e-5"))
DISTILL_OUTPUT_DIR = os.environ.get(
    "SURT_DISTILL_OUTPUT_DIR", "/workspace/surt/distill_turbo"
)


# -- Model surgery --------------------------------------------------------
def build_student_from_teacher(
    teacher: WhisperForConditionalGeneration,
) -> WhisperForConditionalGeneration:
    """Build student: same encoder + 4-layer decoder, weights init from teacher."""
    student_config = WhisperConfig(**teacher.config.to_dict())
    student_config.decoder_layers = STUDENT_DECODER_LAYERS
    student = WhisperForConditionalGeneration(student_config)

    student.generation_config = teacher.generation_config
    student.generation_config.max_length = GENERATION_MAX_LENGTH

    student.model.encoder.load_state_dict(teacher.model.encoder.state_dict())
    student.model.decoder.embed_tokens.load_state_dict(
        teacher.model.decoder.embed_tokens.state_dict()
    )
    student.model.decoder.embed_positions.load_state_dict(
        teacher.model.decoder.embed_positions.state_dict()
    )
    student.model.decoder.layer_norm.load_state_dict(
        teacher.model.decoder.layer_norm.state_dict()
    )
    student.proj_out.load_state_dict(teacher.proj_out.state_dict())

    if len(STUDENT_INIT_LAYER_INDICES) != STUDENT_DECODER_LAYERS:
        raise ValueError(
            f"STUDENT_INIT_LAYER_INDICES has {len(STUDENT_INIT_LAYER_INDICES)} entries "
            f"but STUDENT_DECODER_LAYERS={STUDENT_DECODER_LAYERS}"
        )

    for s_idx, t_idx in enumerate(STUDENT_INIT_LAYER_INDICES):
        student.model.decoder.layers[s_idx].load_state_dict(
            teacher.model.decoder.layers[t_idx].state_dict()
        )
        print(f"[distill] copied decoder layer teacher[{t_idx}] -> student[{s_idx}]")

    n_s = sum(p.numel() for p in student.parameters())
    n_t = sum(p.numel() for p in teacher.parameters())
    print(
        f"[distill] params: teacher={n_t/1e6:.1f}M  student={n_s/1e6:.1f}M  "
        f"({n_s/n_t:.0%} of teacher)"
    )
    return student


def load_teacher_and_processor(
    device: str = "cuda", dtype: torch.dtype = torch.bfloat16,
) -> tuple[WhisperForConditionalGeneration, WhisperProcessor]:
    print(f"[distill] loading teacher: {TEACHER_REPO}")
    teacher = WhisperForConditionalGeneration.from_pretrained(
        TEACHER_REPO,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    teacher.generation_config.language = "punjabi"
    teacher.generation_config.task = "transcribe"
    teacher.generation_config.forced_decoder_ids = None
    teacher.generation_config.max_length = GENERATION_MAX_LENGTH

    processor = WhisperProcessor.from_pretrained(
        TEACHER_REPO, language="punjabi", task="transcribe", use_fast=False,
    )

    teacher.to(device=device)
    teacher.train(False)  # inference mode (equivalent to .eval())
    for p in teacher.parameters():
        p.requires_grad = False
    print(
        f"[distill] teacher on {device} dtype={dtype}, frozen "
        f"({sum(p.numel() for p in teacher.parameters())/1e6:.1f}M params)"
    )
    return teacher, processor


# -- Trainer with combined CE + KL loss -----------------------------------
class DistillTrainer(SurtTrainer):
    """SurtTrainer with frozen-teacher KL term added to the loss."""

    def __init__(
        self,
        *args,
        teacher_model: WhisperForConditionalGeneration,
        kd_alpha: float = KD_ALPHA,
        kd_temperature: float = KD_TEMPERATURE,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.kd_alpha = kd_alpha
        self.kd_temperature = kd_temperature
        print(
            f"[distill] DistillTrainer ready: alpha={kd_alpha} temp={kd_temperature} "
            f"encoder_lr={DISTILL_ENCODER_LR} decoder_lr={DISTILL_DECODER_LR}"
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        student_outputs = model(**inputs)
        ce_loss = student_outputs.loss

        # Teacher is loaded in bf16; data collator emits fp32 features.
        # Match dtypes explicitly since Trainer's autocast doesn't wrap our
        # manual teacher forward.
        teacher_dtype = next(self.teacher.parameters()).dtype
        with torch.no_grad():
            teacher_outputs = self.teacher(
                input_features=inputs["input_features"].to(dtype=teacher_dtype),
                labels=labels,
            )

        T = self.kd_temperature
        s_logits = student_outputs.logits
        t_logits = teacher_outputs.logits.to(s_logits.dtype)

        s_log_probs = F.log_softmax(s_logits / T, dim=-1)
        t_probs = F.softmax(t_logits / T, dim=-1)
        kl_per_token = F.kl_div(s_log_probs, t_probs, reduction="none").sum(-1)

        mask = (labels != -100).float()
        kl_loss = (kl_per_token * mask).sum() / mask.sum().clamp(min=1.0)
        kl_loss = kl_loss * (T * T)

        loss = (1.0 - self.kd_alpha) * ce_loss + self.kd_alpha * kl_loss

        if self.state.global_step > 0 and self.state.global_step % 25 == 0:
            print(
                f"[distill] step {self.state.global_step}: "
                f"ce={ce_loss.item():.4f}  kl={kl_loss.item():.4f}  total={loss.item():.4f}"
            )

        return (loss, student_outputs) if return_outputs else loss

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        from torch.optim import AdamW

        NO_DECAY = {"bias", "layer_norm", "layernorm"}

        def _is_no_decay(name: str) -> bool:
            n = name.lower()
            return any(kw in n for kw in NO_DECAY)

        seen = set()
        enc_d, enc_n, dec_d, dec_n, proj_d, proj_n = [], [], [], [], [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            no_decay = _is_no_decay(name)
            if "model.encoder" in name:
                (enc_n if no_decay else enc_d).append(p)
            elif "proj_out" in name:
                (proj_n if no_decay else proj_d).append(p)
            elif "model.decoder" in name:
                (dec_n if no_decay else dec_d).append(p)

        groups = [
            {"params": enc_d,  "lr": DISTILL_ENCODER_LR, "weight_decay": WEIGHT_DECAY},
            {"params": enc_n,  "lr": DISTILL_ENCODER_LR, "weight_decay": 0.0},
            {"params": dec_d,  "lr": DISTILL_DECODER_LR, "weight_decay": WEIGHT_DECAY},
            {"params": dec_n,  "lr": DISTILL_DECODER_LR, "weight_decay": 0.0},
            {"params": proj_d, "lr": DISTILL_DECODER_LR, "weight_decay": WEIGHT_DECAY},
            {"params": proj_n, "lr": DISTILL_DECODER_LR, "weight_decay": 0.0},
        ]
        groups = [g for g in groups if g["params"]]
        self.optimizer = AdamW(groups)
        n = sum(sum(p.numel() for p in g["params"]) for g in groups)
        print(f"[distill] optimizer: {len(groups)} groups, {n/1e6:.1f}M trainable params")
        return self.optimizer


# -- Training args (A40-saturating, distillation-specific) ----------------
def build_distill_training_args(
    *,
    output_dir: str,
    max_steps: int,
    warmup_steps: int,
    eval_steps: int,
    save_steps: int,
    logging_steps: int,
    report_to="none",
    run_name: str | None = None,
    dataloader_num_workers: int = 8,
) -> Seq2SeqTrainingArguments:
    dl_kwargs = {}
    if dataloader_num_workers > 0:
        dl_kwargs = {
            "dataloader_persistent_workers": True,
            "dataloader_prefetch_factor": 4,
            "dataloader_pin_memory": True,
        }

    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=DISTILL_BATCH_SIZE,
        per_device_eval_batch_size=DISTILL_BATCH_SIZE,
        gradient_accumulation_steps=DISTILL_GRAD_ACCUM,
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        learning_rate=DISTILL_DECODER_LR,
        lr_scheduler_type="cosine",
        weight_decay=WEIGHT_DECAY,
        bf16=True,
        gradient_checkpointing=False,
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
        f"[distill] training_args: per_device_batch={DISTILL_BATCH_SIZE} "
        f"grad_accum={DISTILL_GRAD_ACCUM} effective={DISTILL_BATCH_SIZE * DISTILL_GRAD_ACCUM} "
        f"max_steps={max_steps} warmup={warmup_steps}"
    )
    return args


# -- Orchestration --------------------------------------------------------
def run_distillation(
    *,
    output_dir: str,
    max_steps: int,
    warmup_steps: int,
    eval_steps: int,
    save_steps: int,
    logging_steps: int,
    streaming: bool = False,
    enable_hub_callback: bool = True,
    resume_from_last_checkpoint: bool = True,
    enable_wandb: bool = False,
    run_name: str | None = None,
):
    from surt.data import (
        DataCollatorSpeechSeq2SeqWithPadding,
        get_kirtan_val_dataset,
        get_train_dataset,
        get_val_dataset,
    )

    print("[distill] === Surt Turbo Distillation ===")
    print(f"[distill] teacher: {TEACHER_REPO}")
    print(f"[distill] student final repo: {STUDENT_FINAL_REPO}")
    print(f"[distill] student training repo: {STUDENT_TRAINING_REPO}")
    print(f"[distill] output_dir: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    last_ckpt = None
    if resume_from_last_checkpoint:
        last_ckpt = get_last_checkpoint(output_dir)
        if last_ckpt:
            print(f"[distill] resuming from checkpoint: {last_ckpt}")
        else:
            print("[distill] starting fresh (no prior checkpoint)")

    teacher, processor = load_teacher_and_processor(device=device, dtype=dtype)

    if last_ckpt:
        print(f"[distill] loading student from checkpoint {last_ckpt}")
        student = WhisperForConditionalGeneration.from_pretrained(
            last_ckpt, attn_implementation="sdpa"
        )
        student.generation_config = teacher.generation_config
        student.generation_config.max_length = GENERATION_MAX_LENGTH
    else:
        student = build_student_from_teacher(teacher)
    student.to(device)

    extra_sehaj = os.environ.get("SURT_EXTRA_SEHAJ_DATASET", EXTRA_SEHAJ_DATASET_NAME).strip() or None
    extra_sehaj_col = os.environ.get("SURT_EXTRA_SEHAJ_TEXT_COLUMN", EXTRA_SEHAJ_TEXT_COLUMN).strip() or None

    train_dataset = get_train_dataset(
        DATASET_NAME,
        processor,
        aux_dataset_name=AUX_TRAIN_DATASET_NAME,
        aux_probability=AUX_TRAIN_PROBABILITY,
        streaming=streaming,
        extra_sehaj_dataset_name=extra_sehaj,
        extra_sehaj_text_column=extra_sehaj_col,
    )
    sehaj_val = get_val_dataset(SEHAJ_EVAL_DATASET_NAME, processor, split=SEHAJ_EVAL_SPLIT)
    eval_dataset = {"sehaj_path": sehaj_val}
    try:
        eval_dataset["kirtan"] = get_kirtan_val_dataset(
            KIRTAN_EVAL_DATASET_NAME, processor, split=KIRTAN_EVAL_SPLIT
        )
    except Exception as e:
        print(f"[distill] WARNING: kirtan val unavailable ({e}); evaluating sehaj only")

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=student.config.decoder_start_token_id,
    )

    args = build_distill_training_args(
        output_dir=output_dir,
        max_steps=max_steps,
        warmup_steps=warmup_steps,
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
                hub_repo=STUDENT_TRAINING_REPO,
                processor=processor,
                output_dir=output_dir,
                push_every_n_evals=2,
            )
        )

    trainer = DistillTrainer(
        model=student,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(processor),
        processing_class=processor.feature_extractor,
        callbacks=callbacks,
        teacher_model=teacher,
        kd_alpha=KD_ALPHA,
        kd_temperature=KD_TEMPERATURE,
    )

    print(f"[distill] starting training: max_steps={max_steps}")
    trainer.train(resume_from_checkpoint=last_ckpt)
    print("[distill] training complete")
    return trainer, processor


# -- CLI ------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Surt turbo distillation")
    p.add_argument(
        "--mode",
        choices=["full", "smoke", "surgery-only"],
        default="full",
    )
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--eval-steps", type=int, default=None)
    p.add_argument("--save-steps", type=int, default=None)
    p.add_argument("--logging-steps", type=int, default=None)
    p.add_argument("--warmup-steps", type=int, default=None)
    p.add_argument("--skip-final-push", action="store_true")
    p.add_argument("--surgery-output-dir", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    if args.mode == "surgery-only":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        teacher, processor = load_teacher_and_processor(device=device, dtype=dtype)
        student = build_student_from_teacher(teacher)
        out = args.surgery_output_dir or os.path.join(DISTILL_OUTPUT_DIR, "student_init")
        os.makedirs(out, exist_ok=True)
        student.save_pretrained(out)
        processor.save_pretrained(out)
        print(f"[distill] surgery-only complete; student initial weights at {out}")
        return

    has_wandb_key = bool(os.environ.get("WANDB_API_KEY"))
    enable_wandb = has_wandb_key
    if enable_wandb:
        os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)
        os.environ.setdefault("WANDB_ENTITY", WANDB_ENTITY)
        os.environ.setdefault("WANDB_WATCH", "false")
        os.environ.setdefault("WANDB_LOG_MODEL", "false")
        print(f"[distill] W&B enabled (project={os.environ.get('WANDB_PROJECT')})")

    if args.mode == "smoke":
        smoke_steps = args.max_steps or 5
        smoke_dir = os.path.join(DISTILL_OUTPUT_DIR, "smoke_test")
        run_distillation(
            output_dir=smoke_dir,
            max_steps=smoke_steps,
            warmup_steps=min(2, smoke_steps),
            eval_steps=min(5, smoke_steps),
            save_steps=min(5, smoke_steps),
            logging_steps=1,
            streaming=True,
            enable_hub_callback=False,
            resume_from_last_checkpoint=False,
            enable_wandb=False,
            run_name=f"surt-distill-smoke-{datetime.datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
        )
        print("[distill] smoke complete")
        return

    full_steps = args.max_steps or DISTILL_MAX_STEPS
    full_warmup = args.warmup_steps or DISTILL_WARMUP_STEPS
    full_eval = args.eval_steps or DISTILL_EVAL_STEPS
    full_save = args.save_steps or DISTILL_SAVE_STEPS
    full_logging = args.logging_steps or 25

    trainer, processor = run_distillation(
        output_dir=DISTILL_OUTPUT_DIR,
        max_steps=full_steps,
        warmup_steps=full_warmup,
        eval_steps=full_eval,
        save_steps=full_save,
        logging_steps=full_logging,
        streaming=False,
        enable_hub_callback=True,
        resume_from_last_checkpoint=True,
        enable_wandb=enable_wandb,
        run_name=f"surt-distill-{datetime.datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
    )

    if args.skip_final_push:
        print("[distill] final push skipped (flag)")
        return

    try:
        push_model_to_hub(
            trainer.model,
            processor,
            repo_id=STUDENT_FINAL_REPO,
            commit_message=(
                f"surt-small-turbo-baseline-v0 final (step {trainer.state.global_step})"
            ),
        )
    except Exception as e:
        print(f"[distill] final push FAILED (non-fatal): {e}")
        print(f"[distill] best checkpoints are safe in {STUDENT_TRAINING_REPO}")


if __name__ == "__main__":
    main()
