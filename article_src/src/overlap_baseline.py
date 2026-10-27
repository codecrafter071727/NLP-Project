"""Predict legal-meaning scores from word overlap, without a neural model.

The labels in datastore/training are on a 0-10 scale. This script compares two
baselines on the official splits:

- always predict the training-set average
- a ridge regression on overlap features between the original sentence and the simplification

Run from anywhere:

    python article_src/src/overlap_baseline.py
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

DATA_DIR = Path(__file__).resolve().parents[2] / "datastore" / "training"
WORD = re.compile(r"[0-9A-Za-zÀ-ÿ\u0900-\u097F]+", re.UNICODE)


def load_split(name):
    rows = []
    with open(DATA_DIR / name, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def tokens(text):
    return WORD.findall(text.lower())


def features(row):
    original = tokens(row["original"])
    simple = tokens(row["simplification"])
    original_set = set(original)
    simple_set = set(simple)
    shared = original_set & simple_set
    union = original_set | simple_set
    jaccard = len(shared) / len(union) if union else 0.0
    coverage = len(shared) / len(original_set) if original_set else 0.0
    precision = len(shared) / len(simple_set) if simple_set else 0.0
    orig_len = max(len(original), 1)
    simp_len = max(len(simple), 1)
    length_ratio = min(orig_len, simp_len) / max(orig_len, simp_len)
    return np.array(
        [jaccard, coverage, precision, length_ratio, simp_len / orig_len],
        dtype=float,
    )


def matrix(rows):
    return np.vstack([features(row) for row in rows]), np.array(
        [float(row["label"]) for row in rows], dtype=float
    )


def pearson(y_true, y_pred):
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def report(name, y_true, y_pred):
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    print(
        f"{name:22} n={len(y_true):4d}  RMSE={rmse:5.2f}  MAE={mae:5.2f}  "
        f"Pearson={pearson(y_true, y_pred):6.3f}  mean_pred={float(y_pred.mean()):5.2f}"
    )
    return rmse, mae


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    train = load_split("train_FrJUDGE.jsonl")
    dev = load_split("dev_FrJUDGE.jsonl")
    test = load_split("test_FrJUDGE.jsonl")
    identical = load_split("holdout_identical_FrJUDGE.jsonl")
    unrelated = load_split("holdout_unrelated_FrJUDGE.jsonl")

    x_train, y_train = matrix(train)
    x_dev, y_dev = matrix(dev)
    x_test, y_test = matrix(test)
    x_identical, y_identical = matrix(identical)
    x_unrelated, y_unrelated = matrix(unrelated)

    mean_score = float(y_train.mean())
    print("Legal meaning score is 0-10. Higher means the simplification kept the meaning.")
    print(f"Train average label: {mean_score:.2f} on {len(train)} examples\n")

    print("Baseline: always predict the training average")
    report("dev", y_dev, np.full_like(y_dev, mean_score))
    report("test", y_test, np.full_like(y_test, mean_score))
    report("identical holdout", y_identical, np.full_like(y_identical, mean_score))
    report("unrelated holdout", y_unrelated, np.full_like(y_unrelated, mean_score))

    model = Ridge(alpha=1.0)
    model.fit(x_train, y_train)
    print("\nBaseline: ridge regression on word-overlap features")
    report("dev", y_dev, np.clip(model.predict(x_dev), 0, 10))
    test_pred = np.clip(model.predict(x_test), 0, 10)
    report("test", y_test, test_pred)
    report("identical holdout", y_identical, np.clip(model.predict(x_identical), 0, 10))
    report("unrelated holdout", y_unrelated, np.clip(model.predict(x_unrelated), 0, 10))

    names = ["jaccard", "coverage", "precision", "length_ratio", "simp/orig length"]
    print("\nFeature weights:")
    for name, weight in zip(names, model.coef_):
        print(f"  {name:20} {weight:7.3f}")

    errors = np.abs(y_test - test_pred)
    worst = int(np.argmax(errors))
    row = test[worst]
    print("\nLargest test error")
    print(f"  label={y_test[worst]:.1f}  predicted={test_pred[worst]:.1f}")
    print(f"  original: {row['original'][:240]}")
    print(f"  simplification: {row['simplification'][:240]}")


if __name__ == "__main__":
    main()
