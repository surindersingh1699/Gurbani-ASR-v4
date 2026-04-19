"""Fine-tune Surt on kirtan data (291 whisper-aligned samples).

Continues from the best Sehaj Path checkpoint and adapts to kirtan style.
Pushes result to surindersinghssj/surt-small-v1-kirtan.

Usage:
    python scripts/finetune_kirtan.py [--max-steps 500]

Run on GPU (RunPod A40 recommended). ~30 min for 500 steps.
"""

import argparse
import torch
import evaluate
import numpy as np
from dataclasses import dataclass
from datasets import load_dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from huggingface_hub import HfApi

BASE_MODEL = "surindersinghssj/surt-small-v1-training"
DATASET = "surindersinghssj/gurbani-asr-whisper-aligned"
OUTPUT_DIR = "/workspace/surt-kirtan/checkpoints"
HUB_REPO = "surindersinghssj/surt-small-v1-kirtan"
TEXT_COLUMN = "sentence"


@dataclass
class DataCollatorSpeechSeq2Seq:
    processor: object
    decoder_start_token_id: int

    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Strip double BOS if present
        if (labels[:, 0] == self.decoder_start_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--hub-repo", default=HUB_REPO)
    parser.add_argument("--skip-push", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load processor from base whisper (avoids tokenizer compat issues)
    print("Loading processor from openai/whisper-small...")
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    processor.tokenizer.set_prefix_tokens(language="punjabi", task="transcribe")

    # Load model from our best Sehaj Path checkpoint
    print(f"Loading model from {BASE_MODEL}...")
    model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)
    model.generation_config.language = "punjabi"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")

    # Load dataset
    print(f"Loading dataset {DATASET}...")
    ds = load_dataset(DATASET, split="train")
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    print(f"  Total samples: {len(ds)}")

    # Split: 260 train / 31 eval (~90/10)
    split = ds.train_test_split(test_size=31, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]
    print(f"  Train: {len(train_ds)}, Eval: {len(eval_ds)}")

    # Preprocess
    def prepare(example):
        audio = example["audio"]
        input_features = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        labels = processor.tokenizer(example[TEXT_COLUMN]).input_ids
        return {"input_features": input_features, "labels": labels}

    print("Preprocessing train set...")
    train_ds = train_ds.map(prepare, remove_columns=train_ds.column_names, num_proc=4)
    print("Preprocessing eval set...")
    eval_ds = eval_ds.map(prepare, remove_columns=eval_ds.column_names, num_proc=4)

    # Data collator
    collator = DataCollatorSpeechSeq2Seq(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # Metrics
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        w = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
        c = 100 * cer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": w, "cer": c}

    # With 260 train samples and batch_size=8:
    # 1 epoch = ~33 steps
    # 500 steps = ~15 epochs (reasonable)
    eval_steps = 50
    save_steps = 50

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=2,  # effective batch = 16
        max_steps=args.max_steps,
        warmup_steps=50,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        bf16=torch.cuda.is_available(),
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="no",
        predict_with_generate=True,
        generation_max_length=448,
        logging_steps=10,
        push_to_hub=False,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )

    print(f"\nTraining for {args.max_steps} steps...")
    print(f"  Batch: {args.batch_size} x accum {2} = effective {args.batch_size * 2}")
    print(f"  LR: {args.lr}, Scheduler: cosine, Warmup: 50")
    print(f"  Eval every {eval_steps} steps")
    print(f"  ~{args.max_steps // 33} epochs over {len(train_ds)} samples\n")

    trainer.train()

    # Final eval
    print("\nFinal evaluation...")
    metrics = trainer.evaluate()
    print(f"  WER: {metrics['eval_wer']:.2f}%")
    print(f"  CER: {metrics['eval_cer']:.2f}%")

    # Push to hub
    if not args.skip_push:
        print(f"\nPushing to {args.hub_repo}...")
        api = HfApi()
        try:
            api.create_repo(args.hub_repo, exist_ok=True)
        except Exception:
            pass

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            processor.save_pretrained(tmpdir)
            api.upload_folder(
                repo_id=args.hub_repo,
                folder_path=tmpdir,
                commit_message=f"Kirtan fine-tune: {args.max_steps} steps | WER {metrics['eval_wer']:.2f} | CER {metrics['eval_cer']:.2f}",
            )
        print(f"Done! Model pushed to {args.hub_repo}")
    else:
        print("Push skipped. Model saved locally.")


if __name__ == "__main__":
    main()
