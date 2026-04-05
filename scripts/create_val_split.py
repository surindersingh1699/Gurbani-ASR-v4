"""
Create a proper train/validation split on HuggingFace.

Takes 300 rows from offset 50000 as validation, removes them from train.
Pushes both splits back to the same HF dataset repo.

Usage:
    python scripts/create_val_split.py

Requires HF_TOKEN in environment (set in ~/.bashrc on RunPod).
"""

from datasets import DatasetDict, load_dataset

DATASET_NAME = "surindersinghssj/gurbani-asr"
VAL_SIZE = 300
VAL_OFFSET = 50000  # Deep in dataset to avoid overlap with early training


def main():
    print(f"[split] Loading full dataset: {DATASET_NAME}")
    ds = load_dataset(DATASET_NAME, split="train")
    total = len(ds)
    print(f"[split] Total rows: {total}")

    if VAL_OFFSET + VAL_SIZE > total:
        raise ValueError(
            f"VAL_OFFSET ({VAL_OFFSET}) + VAL_SIZE ({VAL_SIZE}) > total ({total})"
        )

    # Select val indices
    val_indices = list(range(VAL_OFFSET, VAL_OFFSET + VAL_SIZE))
    train_indices = list(range(0, VAL_OFFSET)) + list(range(VAL_OFFSET + VAL_SIZE, total))

    val_ds = ds.select(val_indices)
    train_ds = ds.select(train_indices)

    print(f"[split] Train: {len(train_ds)} rows")
    print(f"[split] Validation: {len(val_ds)} rows (from offset {VAL_OFFSET})")

    # Sanity check: no overlap
    assert len(train_ds) + len(val_ds) == total
    assert len(val_ds) == VAL_SIZE

    # Push to HF as a DatasetDict with both splits
    dataset_dict = DatasetDict({"train": train_ds, "validation": val_ds})

    print(f"[split] Pushing to {DATASET_NAME} with train + validation splits...")
    dataset_dict.push_to_hub(DATASET_NAME)
    print("[split] Done! Dataset now has 'train' and 'validation' splits.")


if __name__ == "__main__":
    main()
