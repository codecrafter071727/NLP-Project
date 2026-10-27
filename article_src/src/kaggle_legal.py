"""Legal clause pairs from the Kaggle dataset of Karthik R V.

Kaggle: karthikrv1107/adaption-legal-clause-simplification (MIT)
Public file: Karthikrv/adaption-legal-clause-simplification on Hugging Face

The Kaggle release has 8,063 records. A record is kept when it has both a
source clause (Input) and a restatement (Output). Records with no source
clause are instructions to draft a document, so they are left out of the table.

The table has no human rating from 0 to 10. RQ2 and RQ3 stay on
datastore/frjudge.csv.
"""

import csv
import json
import os
from pathlib import Path

import numpy as np

KAGGLE_SLUG = "karthikrv1107/adaption-legal-clause-simplification"
HF_REPO = "Karthikrv/adaption-legal-clause-simplification"
HF_FILE = "data/train_0.jsonl"
DATASET_PATH = Path(__file__).resolve().parents[2] / "datastore" / "kaggle_legal_simplification.csv"
SPLIT_SEED = 0
TEST_FRACTION = 0.2
COLUMNS = ("split", "original", "simplification")


def assign_splits(n, test_fraction=TEST_FRACTION, seed=SPLIT_SEED):
    """Hold out a fixed share of rows. The dataset publishes no split of its own."""
    if n < 2:
        raise ValueError("At least two rows are required to make a train and test split")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    n_test = int(n * test_fraction)
    n_test = min(max(n_test, 1), n - 1)
    test_index = set(np.random.default_rng(seed).permutation(n)[:n_test].tolist())
    return ["test" if index in test_index else "train" for index in range(n)]


def pairs_from_jsonl(path):
    """Return (pairs, skipped) from the upstream JSONL."""
    pairs = []
    skipped = 0
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            original = row.get("Input")
            simplification = row.get("Output")
            if not isinstance(original, str) or not original.strip():
                skipped += 1
                continue
            if not isinstance(simplification, str) or not simplification.strip():
                skipped += 1
                continue
            pairs.append((original.strip(), simplification.strip()))
    if not pairs:
        raise ValueError(f"No clause pairs in {path}")
    return pairs, skipped


def write_table(pairs, path, test_fraction=TEST_FRACTION, seed=SPLIT_SEED):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    splits = assign_splits(len(pairs), test_fraction=test_fraction, seed=seed)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for split, (original, simplification) in zip(splits, pairs):
            writer.writerow(
                {
                    "split": split,
                    "original": original,
                    "simplification": simplification,
                }
            )
    return path


def download_source():
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(HF_REPO, HF_FILE, repo_type="dataset"))


def build_dataset(path=DATASET_PATH):
    """Write the CSV from the public copy of the Kaggle dataset."""
    source = download_source()
    pairs, skipped = pairs_from_jsonl(source)
    write_table(pairs, path)
    return {"kept": len(pairs), "skipped": skipped, "path": Path(path)}


def load_split(name, path=DATASET_PATH):
    if name not in ("train", "test"):
        raise ValueError(f"Unknown split: {name}")
    path = Path(path)
    if not path.is_file():
        build_dataset(path)
    rows = []
    with open(path, encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle):
            if record["split"] == name:
                rows.append(
                    {
                        "original": record["original"],
                        "simplification": record["simplification"],
                    }
                )
    if not rows:
        raise ValueError(f"No {name} rows in {path}")
    return rows
