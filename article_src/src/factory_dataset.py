import os

from datasets import load_dataset, DatasetDict


def dataset_factory(data_augmentation: bool):
    root_dir = os.path.join("../..", "datastore", "training")
    if data_augmentation:
        train = load_dataset(
            root_dir,
            data_files=["train_FrJUDGE_with_data_augmentation.jsonl"],
        )
        dev = load_dataset(
            root_dir,
            data_files=["dev_FrJUDGE_with_data_augmentation.jsonl"],
        )
        test = load_dataset(
            root_dir,
            data_files=["test_FrJUDGE_with_data_augmentation.jsonl"],
        )
    else:
        train = load_dataset(
            root_dir,
            data_files=["train_FrJUDGE.jsonl"],
        )
        dev = load_dataset(
            root_dir,
            data_files=["dev_FrJUDGE.jsonl"],
        )
        test = load_dataset(
            root_dir,
            data_files=["test_FrJUDGE.jsonl"],
        )

    legal_precision_annotation_dataset = DatasetDict(
        {"train": train["train"], "dev": dev["train"], "test": test["train"]}
    )
    holdout_identical_dataset = load_dataset(
        root_dir, data_files=["holdout_identical_FrJUDGE.jsonl"]
    )
    holdout_unrelated_dataset = load_dataset(
        root_dir, data_files=["holdout_unrelated_FrJUDGE.jsonl"]
    )

    return (
        legal_precision_annotation_dataset,
        holdout_identical_dataset,
        holdout_unrelated_dataset,
    )
