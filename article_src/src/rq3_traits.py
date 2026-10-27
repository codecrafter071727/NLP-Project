"""RQ3: can interpretable traits of a simplification be read reliably from the text?

Each trait is a fixed measurement of the original sentence and its simplification.
A trait is reliable when the identical holdout sits on the expected side of the
unrelated holdout, and its correlation with the human rating keeps the same sign
from the training split to the official test split with p < 0.05 on the test split.

Run from the repo root:

    python article_src/src/rq3_traits.py
"""

import re
import sys

import numpy as np

from overlap_baseline import features, tokens
from rq2_metrics import (
    OUTPUT_DIR,
    format_num,
    format_p,
    load_split,
    regression_metrics,
    write_csv,
)

SENTENCE_END = re.compile(r"[.!?।]+")

TRAITS = (
    ("overlap", "higher", "Share of words that appear in both texts."),
    ("words_kept", "higher", "Share of original words that remain in the simplification."),
    ("words_from_original", "higher", "Share of simplification words that come from the original."),
    ("length_closeness", "higher", "1 when the two texts have the same number of words, lower when one is much longer."),
    ("length_change", "closer_to_one", "Simplification length divided by original length. 1 means no change."),
    ("sentence_split", "closer_to_one", "Simplification sentences divided by original sentences. Above 1 means the sentence was split."),
    ("shorter_words", "closer_to_zero", "Average original word length minus average simplification word length."),
)


def _mean_length(words):
    if not words:
        return 0.0
    return float(np.mean([len(word) for word in words]))


def _sentences(text):
    return max(len(SENTENCE_END.findall(text)), 1)


def trait_values(row):
    original = tokens(row["original"])
    simple = tokens(row["simplification"])
    overlap, words_kept, words_from_original, length_closeness, length_change = features(row)
    return {
        "overlap": float(overlap),
        "words_kept": float(words_kept),
        "words_from_original": float(words_from_original),
        "length_closeness": float(length_closeness),
        "length_change": float(length_change),
        "sentence_split": _sentences(row["simplification"]) / _sentences(row["original"]),
        "shorter_words": _mean_length(original) - _mean_length(simple),
    }


def trait_matrix(rows):
    records = [trait_values(row) for row in rows]
    labels = np.array([float(row["label"]) for row in rows], dtype=float)
    columns = {
        name: np.array([record[name] for record in records], dtype=float)
        for name, _check, _meaning in TRAITS
    }
    return columns, labels


def holdout_ok(check, identical_mean, unrelated_mean):
    if check == "higher":
        return identical_mean > unrelated_mean
    if check == "lower":
        return identical_mean < unrelated_mean
    if check == "closer_to_one":
        return abs(identical_mean - 1.0) < abs(unrelated_mean - 1.0)
    if check == "closer_to_zero":
        return abs(identical_mean) < abs(unrelated_mean)
    raise ValueError(check)


def is_reliable(train_r, test_r, test_p, identical_mean, unrelated_mean, check):
    if any(value != value for value in (train_r, test_r, test_p)):
        return False
    same_sign = train_r * test_r > 0
    significant = test_p < 0.05
    return same_sign and significant and holdout_ok(check, identical_mean, unrelated_mean)


def evaluate(train_rows, test_rows, dev_rows, identical_rows, unrelated_rows):
    train_columns, train_labels = trait_matrix(train_rows)
    test_columns, test_labels = trait_matrix(test_rows)
    dev_columns, dev_labels = trait_matrix(dev_rows)
    identical_columns, _identical_labels = trait_matrix(identical_rows)
    unrelated_columns, _unrelated_labels = trait_matrix(unrelated_rows)
    results = []
    for name, check, meaning in TRAITS:
        train_metrics = regression_metrics(train_labels, train_columns[name])
        test_metrics = regression_metrics(test_labels, test_columns[name])
        dev_metrics = regression_metrics(dev_labels, dev_columns[name])
        identical_mean = float(np.mean(identical_columns[name]))
        unrelated_mean = float(np.mean(unrelated_columns[name]))
        reliable = is_reliable(
            train_metrics["pearson"],
            test_metrics["pearson"],
            test_metrics["pearson_p"],
            identical_mean,
            unrelated_mean,
            check,
        )
        results.append(
            {
                "name": name,
                "check": check,
                "meaning": meaning,
                "train_r": train_metrics["pearson"],
                "train_p": train_metrics["pearson_p"],
                "dev_r": dev_metrics["pearson"],
                "test_r": test_metrics["pearson"],
                "test_p": test_metrics["pearson_p"],
                "identical_mean": identical_mean,
                "unrelated_mean": unrelated_mean,
                "reliable": reliable,
            }
        )
    return results


def render(results, n_train, n_test):
    lines = [
        "RQ3 official evaluation",
        "",
        "Protocol",
        "Rows are read from datastore/frjudge.csv.",
        "Each trait is measured from the original sentence and its simplification.",
        "The same pair always produces the same trait value.",
        f"Correlation with the human rating is estimated on the training split (n={n_train}) and checked on the test split (n={n_test}).",
        "A trait is reliable when three things hold: the identical holdout sits on the expected side of the unrelated holdout,",
        "the training and test correlations have the same sign, and the test correlation has p < 0.05.",
        "This dataset has no user search logs. The traits describe how a legal sentence was simplified.",
        "",
        f"{'Trait':22} {'train r':>8} {'test r':>8} {'test p':>8} {'identical':>10} {'unrelated':>10} {'reliable':>10}",
    ]
    for row in results:
        lines.append(
            f"{row['name']:22} {format_num(row['train_r'], 3):>8} {format_num(row['test_r'], 3):>8} "
            f"{format_p(row['test_p']):>8} {row['identical_mean']:10.2f} {row['unrelated_mean']:10.2f} "
            f"{'yes' if row['reliable'] else 'no':>10}"
        )
    reliable = [row["name"] for row in results if row["reliable"]]
    lines.append("")
    lines.append(f"Reliable traits: {len(reliable)} of {len(results)}.")
    if reliable:
        lines.append("Reliable: " + ", ".join(reliable) + ".")
    else:
        lines.append("No trait met all three reliability checks.")
    lines.append("")
    lines.append("What each trait measures")
    for row in results:
        lines.append(f"  {row['name']}: {row['meaning']}")
    return "\n".join(lines) + "\n"


def save_table(test_rows, path):
    names = [name for name, _check, _meaning in TRAITS]
    records = []
    for row in test_rows:
        values = trait_values(row)
        records.append({"label": row["label"], **{name: f"{values[name]:.6f}" for name in names}})
    write_csv(path, records, ["label", *names])


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    train = load_split("train")
    dev = load_split("dev")
    test = load_split("test")
    identical = load_split("holdout_identical")
    unrelated = load_split("holdout_unrelated")
    results = evaluate(train, test, dev, identical, unrelated)
    report = render(results, len(train), len(test))
    print(report, end="")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / "rq3_summary.txt"
    report_path = OUTPUT_DIR / "rq3_results.csv"
    table_path = OUTPUT_DIR / "rq3_test_traits.csv"
    summary_path.write_text(report, encoding="utf-8")
    write_csv(
        report_path,
        results,
        [
            "name",
            "train_r",
            "train_p",
            "dev_r",
            "test_r",
            "test_p",
            "identical_mean",
            "unrelated_mean",
            "reliable",
        ],
    )
    save_table(test, table_path)
    print(f"Saved summary to {summary_path}")
    print(f"Saved metrics to {report_path}")
    print(f"Saved test traits to {table_path}")


if __name__ == "__main__":
    main()
