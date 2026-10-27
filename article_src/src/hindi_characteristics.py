"""Source-characteristic check on the Hindi simplification table.

Uses the same characteristics, formulation measures, and Bonferroni rule as RQ1.
The Hindi table has no human rating, so this check does not train the ridge
model and does not call the judge prompts.

Run from the repo root:

    python article_src/src/hindi_characteristics.py
"""

import sys

from hindi_simplification import DATASET_PATH, HF_REPO, build_dataset, load_split
from kaggle_characteristics import save_rows
from rq1_characteristics import evaluate
from rq2_metrics import OUTPUT_DIR, format_num, format_p, write_csv


def render(results, alpha, n_train, n_test, kept, skipped):
    lines = [
        "Hindi simplification check",
        "",
        "Protocol",
        "Rows are read from datastore/hindi_simplification.csv.",
        f"Source dataset: Hugging Face {HF_REPO}.",
        "The dataset card lists 12,000 Hindi source sentences and simpler Hindi sentences.",
        "The dataset card lists no license and no human rating from 0 to 10.",
        "Each kept row stores the Hindi source as original and the simpler sentence as simplification.",
        f"Upstream records with both fields: {kept}. Records left out: {skipped}.",
        "The ridge model and the judge prompts stay on datastore/frjudge.csv.",
        "The dataset publishes no train or test split. This check holds out 20 percent of the kept rows with a fixed shuffle, seed 0.",
        f"Correlations are estimated on the training rows (n={n_train}) and checked on the held-out rows (n={n_test}).",
        f"A link is reliable when the two correlations have the same sign and the test p-value is below {alpha:.4f}.",
        "That threshold is 0.05 divided by the number of characteristic-formulation pairs, the Bonferroni correction.",
        "Word counts include Devanagari characters, and sentence counts include the Hindi danda.",
        "The characteristics and formulation measures are the ones defined for RQ1.",
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


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    info = build_dataset() if not DATASET_PATH.is_file() else None
    train = load_split("train")
    test = load_split("test")
    if info is None:
        from hindi_simplification import download_frame, pairs_from_records

        _pairs, skipped = pairs_from_records(download_frame().to_dict(orient="records"))
        kept = len(train) + len(test)
    else:
        kept = info["kept"]
        skipped = info["skipped"]

    results, alpha = evaluate(train, test)
    report = render(results, alpha, len(train), len(test), kept, skipped)
    print(report, end="")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / "hindi_summary.txt"
    report_path = OUTPUT_DIR / "hindi_results.csv"
    table_path = OUTPUT_DIR / "hindi_test_rows.csv"
    summary_path.write_text(report, encoding="utf-8")
    write_csv(
        report_path,
        results,
        ["characteristic", "outcome", "train_r", "train_p", "test_r", "test_p", "reliable"],
    )
    save_rows(test, table_path)
    print(f"Saved the table to {DATASET_PATH}")
    print(f"Saved summary to {summary_path}")
    print(f"Saved metrics to {report_path}")
    print(f"Saved test rows to {table_path}")


if __name__ == "__main__":
    main()
