import os
import pickle

from datasets import load_dataset, DatasetDict

root_dir = os.path.join("...", "datastore", "training")

metric_names = [
    "BERTScore F1",
    "QuestEval",
    "LENS",
    "Coverage",
    "MeaningBERT",
    "SBERT Embeddings",
    "SBERT Multi Embeddings",
] + [f"JUDGEBERT_{x}" for x in range(42, 52)]


for data_augmentation in [False, True]:
    for metric_name in metric_names:
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

        file_path = f"{metric_name.replace(' ', '_')}"
        if data_augmentation:
            file_path += "_WithDA"
        file_path += "_predictions.pickle"

        with open(os.path.join("predictions", file_path), "rb") as f:
            predictions = pickle.load(f)

        if "JUDGEBERT" in metric_name:
            predictions = [round(pred, 1) for pred in predictions.label_ids.tolist()]

        legal_precision_annotation_dataset = DatasetDict(
            {"train": train["train"], "dev": dev["train"], "test": test["train"]}
        )

        labels = legal_precision_annotation_dataset["test"]["label"]

        counts_higher = 0
        for prediction, label in zip(predictions, labels):
            if prediction > label:
                counts_higher += 1

        ratio = round(100 * counts_higher / len(predictions), 2)
        print(f"{metric_name} % of scores > than the labels: {ratio}")
