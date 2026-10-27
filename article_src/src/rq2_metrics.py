"""Train and score the RQ2 legal-meaning experiment.

Rows come from datastore/frjudge.csv. The model is fit only on the train
split. Dev, test, and the identical and unrelated holdouts are evaluation.
"""

import csv
import json
import pickle
import re
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge

from overlap_baseline import matrix

DATA_DIR = Path(__file__).resolve().parents[2] / "datastore" / "training"
DATASET_PATH = Path(__file__).resolve().parents[2] / "datastore" / "frjudge.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "predictions"
FEATURE_NAMES = ["jaccard", "coverage", "precision", "length_ratio", "simp/orig length"]
SCORE = re.compile(r"\d+")
DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
SPLITS = ("train", "dev", "test", "holdout_identical", "holdout_unrelated")


def build_dataset():
    """Write datastore/frjudge.csv from the official JUDGEBERT splits."""
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DATASET_PATH, "w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=["split", "original", "simplification", "label"],
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        for name in SPLITS:
            source = DATA_DIR / f"{name}_FrJUDGE.jsonl"
            with open(source, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    writer.writerow(
                        {
                            "split": name,
                            "original": row["original"],
                            "simplification": row["simplification"],
                            "label": row["label"],
                        }
                    )


def load_split(name):
    if name not in SPLITS:
        raise ValueError(f"Unknown split: {name}")
    if not DATASET_PATH.is_file():
        build_dataset()
    rows = []
    with open(DATASET_PATH, encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle):
            if record["split"] == name:
                rows.append(
                    {
                        "original": record["original"],
                        "simplification": record["simplification"],
                        "label": float(record["label"]),
                    }
                )
    if not rows:
        raise ValueError(f"No {name} rows in {DATASET_PATH}")
    return rows


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            cleaned = {}
            for field in fieldnames:
                value = row.get(field, "")
                if isinstance(value, float) and value != value:
                    value = ""
                cleaned[field] = value
            writer.writerow(cleaned)


def fit_ridge(train_rows, alpha=1.0):
    features, labels = matrix(train_rows)
    model = Ridge(alpha=alpha)
    model.fit(features, labels)
    return model


def predict_ridge(model, rows):
    features, _labels = matrix(rows)
    return np.clip(model.predict(features), 0, 10)


def parse_score(text):
    """Read the first integer in a model reply. Reject values outside 0-10."""
    if text is None:
        return None
    match = SCORE.search(str(text).translate(DEVANAGARI_DIGITS))
    if match is None:
        return None
    value = int(match.group())
    if value < 0 or value > 10:
        return None
    return value


def _correlation(function, y_true, y_pred):
    if len(y_true) < 3 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan"), float("nan")
    result = function(y_true, y_pred)
    statistic = getattr(result, "statistic", result[0])
    pvalue = getattr(result, "pvalue", result[1])
    return float(statistic), float(pvalue)


def regression_metrics(y_true, y_pred, seed=0, n_boot=1000):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    n = int(len(y_true))
    empty = {
        "n": 0,
        "mae": float("nan"),
        "rmse": float("nan"),
        "pearson": float("nan"),
        "pearson_p": float("nan"),
        "spearman": float("nan"),
        "spearman_p": float("nan"),
        "mae_low": float("nan"),
        "mae_high": float("nan"),
        "mean_pred": float("nan"),
        "mean_label": float("nan"),
    }
    if n == 0:
        return empty
    error = y_pred - y_true
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    pearson, pearson_p = _correlation(pearsonr, y_true, y_pred)
    spearman, spearman_p = _correlation(spearmanr, y_true, y_pred)
    if n == 1:
        mae_low = mae_high = mae
    else:
        rng = np.random.default_rng(seed)
        draw = rng.integers(0, n, size=(n_boot, n))
        boot = np.mean(np.abs(y_true[draw] - y_pred[draw]), axis=1)
        mae_low, mae_high = (float(value) for value in np.quantile(boot, [0.025, 0.975]))
    return {
        "n": n,
        "mae": mae,
        "rmse": rmse,
        "pearson": pearson,
        "pearson_p": pearson_p,
        "spearman": spearman,
        "spearman_p": spearman_p,
        "mae_low": mae_low,
        "mae_high": mae_high,
        "mean_pred": float(np.mean(y_pred)),
        "mean_label": float(np.mean(y_true)),
    }


def save_ridge(model, train_rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "alpha": 1.0,
        "feature_names": FEATURE_NAMES,
        "coefficients": model.coef_.tolist(),
        "intercept": float(model.intercept_),
        "train_rows": len(train_rows),
        "train_file": "datastore/frjudge.csv",
    }
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)


def format_p(value):
    if value != value:
        return "n/a"
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def format_num(value, digits=2):
    if value != value:
        return "n/a"
    return f"{value:.{digits}f}"


def metric_row(name, metrics, missing=0):
    return (
        f"{name:22} {metrics['n']:4d} {missing:5d}  "
        f"{format_num(metrics['mae']):>5}  "
        f"[{format_num(metrics['mae_low'])}, {format_num(metrics['mae_high'])}]  "
        f"{format_num(metrics['rmse']):>5}  "
        f"{format_num(metrics['pearson'], 3):>6}  "
        f"{format_p(metrics['pearson_p']):>6}"
    )


def intervals_overlap(left, right):
    if any(value != value for value in (*left, *right)):
        return True
    return not (left[1] < right[0] or right[1] < left[0])
