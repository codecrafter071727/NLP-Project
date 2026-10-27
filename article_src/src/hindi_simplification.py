"""Hindi simplification pairs from a public Hugging Face dataset.

Source: raja20221020/hindi-text-simplification-for-finetuning
12,000 rows. Each row has an instruction, a Hindi source sentence, and a
simpler Hindi sentence. The table keeps the source and the simplification.

The dataset card lists no license and no human rating from 0 to 10.
RQ2 and RQ3 stay on datastore/frjudge.csv.
"""

import os
from pathlib import Path

from kaggle_legal import write_table

HF_REPO = "raja20221020/hindi-text-simplification-for-finetuning"
HF_FILE = "data/train-00000-of-00001.parquet"
DATASET_PATH = Path(__file__).resolve().parents[2] / "datastore" / "hindi_simplification.csv"


def pairs_from_records(records):
    """Return (pairs, skipped) from rows with input and output fields."""
    pairs = []
    skipped = 0
    for record in records:
        original = record.get("input")
        simplification = record.get("output")
        if not isinstance(original, str) or not original.strip():
            skipped += 1
            continue
        if not isinstance(simplification, str) or not simplification.strip():
            skipped += 1
            continue
        pairs.append((original.strip(), simplification.strip()))
    if not pairs:
        raise ValueError("No Hindi simplification pairs")
    return pairs, skipped


def download_frame():
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(HF_REPO, HF_FILE, repo_type="dataset")
    return pd.read_parquet(path)


def build_dataset(path=DATASET_PATH):
    """Write the CSV from the public Hindi simplification dataset."""
    frame = download_frame()
    records = frame.to_dict(orient="records")
    pairs, skipped = pairs_from_records(records)
    write_table(pairs, path)
    return {"kept": len(pairs), "skipped": skipped, "path": Path(path)}


def load_split(name, path=DATASET_PATH):
    if name not in ("train", "test"):
        raise ValueError(f"Unknown split: {name}")
    path = Path(path)
    if not path.is_file():
        build_dataset(path)
    import csv

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
