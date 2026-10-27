"""RQ1: which properties of the original sentence shape the simplification?

The dataset has no age, gender, region, education, or LLM-experience fields.
The characteristics below are measured from the original sentence only.
The formulation measures describe the simplification.
A link is reliable when the training and test correlations have the same sign
and the test p-value survives Bonferroni correction.

Run from the repo root:

    python article_src/src/rq1_characteristics.py
"""

import csv
import sys

import numpy as np

from overlap_baseline import tokens
from rq2_metrics import OUTPUT_DIR, format_num, format_p, load_split, regression_metrics, write_csv
from rq3_traits import SENTENCE_END, trait_values

LONG_WORD = 8
CHARACTERISTICS = (
    ("source_words", "Number of words in the original sentence."),
    ("source_sentences", "Number of sentences in the original text."),
    ("source_word_length", "Average number of characters per original word."),
    ("source_long_word_share", "Share of original words with at least 8 characters."),
    ("source_comma_rate", "Commas in the original text divided by its number of sentences."),
)
OUTCOMES = (
    ("simplification_words", "Number of words in the simplification."),
    ("simplification_sentences", "Number of sentences in the simplification."),
    ("words_kept", "Share of original words that remain in the simplification."),
    ("shorter_words", "How much shorter the simplification words are, in characters."),
)


def source_characteristics(text):
    words = tokens(text)
    count = max(len(words), 1)
    sentences = max(len(SENTENCE_END.findall(text)), 1)
    long_words = sum(1 for word in words if len(word) >= LONG_WORD)
    return {
        "source_words": float(len(words)),
        "source_sentences": float(sentences),
        "source_word_length": float(np.mean([len(word) for word in words])) if words else 0.0,
        "source_long_word_share": long_words / count,
        "source_comma_rate": text.count(",") / sentences,
    }


def formulation(row):
    values = trait_values(row)
    return {
        "simplification_words": float(len(tokens(row["simplification"]))),
        "simplification_sentences": float(max(len(SENTENCE_END.findall(row["simplification"])), 1)),
        "words_kept": values["words_kept"],
        "shorter_words": values["shorter_words"],
    }


def corrected_alpha(n_tests, alpha=0.05):
    if n_tests < 1:
        raise ValueError("n_tests must be positive")
    return alpha / n_tests


def is_reliable(train_r, test_r, test_p, alpha):
    if any(value != value for value in (train_r, test_r, test_p)):
        return False
    return train_r * test_r > 0 and test_p < alpha


def evaluate(train_rows, test_rows):
    alpha = corrected_alpha(len(CHARACTERISTICS) * len(OUTCOMES))
    results = []
    for characteristic, characteristic_meaning in CHARACTERISTICS:
        for outcome, outcome_meaning in OUTCOMES:
            train_x = [source_characteristics(row["original"])[characteristic] for row in train_rows]
            train_y = [formulation(row)[outcome] for row in train_rows]
            test_x = [source_characteristics(row["original"])[characteristic] for row in test_rows]
            test_y = [formulation(row)[outcome] for row in test_rows]
            train_metrics = regression_metrics(train_y, train_x)
            test_metrics = regression_metrics(test_y, test_x)
            reliable = is_reliable(
                train_metrics["pearson"],
                test_metrics["pearson"],
                test_metrics["pearson_p"],
                alpha,
            )
            results.append(
                {
                    "characteristic": characteristic,
                    "characteristic_meaning": characteristic_meaning,
                    "outcome": outcome,
                    "outcome_meaning": outcome_meaning,
                    "train_r": train_metrics["pearson"],
                    "train_p": train_metrics["pearson_p"],
                    "test_r": test_metrics["pearson"],
                    "test_p": test_metrics["pearson_p"],
                    "reliable": reliable,
                }
            )
    return results, alpha


def render(results, alpha, n_train, n_test):
    lines = [
        "RQ1 official evaluation",
        "",
        "Protocol",
        "Rows are read from datastore/frjudge.csv.",
        "Each row stores an original sentence, a simplification, and a human rating.",
        "It does not store age, gender, birth region, education, English proficiency, or LLM experience.",
        "Each characteristic is measured from the original sentence only.",
        "Each formulation measure describes the simplification.",
        f"Correlations are estimated on the training split (n={n_train}) and checked on the test split (n={n_test}).",
        f"A link is reliable when the two correlations have the same sign and the test p-value is below {alpha:.4f}.",
        "That threshold is 0.05 divided by the number of characteristic-formulation pairs, the Bonferroni correction.",
        "The identical and unrelated holdouts are not used: their simplifications were constructed, not written from the source sentence.",
        "",
        f"{'Characteristic':24} {'Formulation':26} {'train r':>8} {'test r':>8} {'test p':>8} {'reliable':>10}",
    ]
    for row in results:
        lines.append(
            f"{row['characteristic']:24} {row['outcome']:26} "
            f"{format_num(row['train_r'], 3):>8} {format_num(row['test_r'], 3):>8} "
            f"{format_p(row['test_p']):>8} {'yes' if row['reliable'] else 'no':>10}"
        )
    reliable = [row for row in results if row["reliable"]]
    lines.append("")
    lines.append(f"Reliable links: {len(reliable)} of {len(results)}.")
    if reliable:
        for row in reliable:
            direction = "higher" if row["test_r"] > 0 else "lower"
            lines.append(
                f"  A higher {row['characteristic']} goes with a {direction} {row['outcome']} "
                f"(test r {format_num(row['test_r'], 3)}, p {format_p(row['test_p'])})."
            )
    else:
        lines.append("No characteristic-formulation link met the corrected test threshold.")
    return "\n".join(lines) + "\n"


def save_tsv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    characteristic_names = [name for name, _meaning in CHARACTERISTICS]
    outcome_names = [name for name, _meaning in OUTCOMES]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", *characteristic_names, *outcome_names])
        for row in rows:
            source = source_characteristics(row["original"])
            written = formulation(row)
            writer.writerow(
                [
                    row["label"],
                    *[f"{source[name]:.6f}" for name in characteristic_names],
                    *[f"{written[name]:.6f}" for name in outcome_names],
                ]
            )


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    train = load_split("train")
    test = load_split("test")
    results, alpha = evaluate(train, test)
    report = render(results, alpha, len(train), len(test))
    print(report, end="")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / "rq1_summary.txt"
    report_path = OUTPUT_DIR / "rq1_results.csv"
    table_path = OUTPUT_DIR / "rq1_test_rows.csv"
    summary_path.write_text(report, encoding="utf-8")
    write_csv(
        report_path,
        results,
        [
            "characteristic",
            "outcome",
            "train_r",
            "train_p",
            "test_r",
            "test_p",
            "reliable",
        ],
    )
    save_tsv(test, table_path)
    print(f"Saved summary to {summary_path}")
    print(f"Saved metrics to {report_path}")
    print(f"Saved test rows to {table_path}")


if __name__ == "__main__":
    main()
