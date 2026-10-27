"""Score Hindi prompts on Hindi simplifications and report the scores.

The Hindi table has no human rating from 0 to 10, so these numbers are the
judge's own scores and the overlap statistics. They are not an error against
a human label.

Run from the repo root:

    python article_src/src/hindi_judge.py
"""

import pickle
import sys
from collections import Counter

import numpy as np
from scipy.stats import pearsonr

from hindi_simplification import DATASET_PATH, load_split
from overlap_baseline import features
from rq1_characteristics import source_characteristics
from rq2_metrics import OUTPUT_DIR, _correlation, format_num, format_p, parse_score, write_csv
from rq2_prompt_ratings import (
    DEFAULT_MODEL,
    SHORT_HI,
    display_model,
    fill_prompt,
    generate_reply,
    load_generator,
    long_hindi_prompt,
    resolve_model,
)
from rq3_traits import trait_values

SAMPLE_SIZE = 90
RIDGE_PATH = OUTPUT_DIR / "rq2_ridge.pkl"


def hindi_prompts():
    return [
        ("short_hi", "Short Hindi", SHORT_HI),
        ("long_hi", "Long Hindi", long_hindi_prompt()),
    ]


def overlap_scores(rows):
    """Apply the French-trained overlap model. The result is not a human rating."""
    with open(RIDGE_PATH, "rb") as handle:
        model = pickle.load(handle)["model"]
    matrix = np.vstack([features(row) for row in rows])
    return np.clip(model.predict(matrix), 0, 10)


def mean(values):
    return float(np.mean(values)) if len(values) else float("nan")


def score_rows(rows, model_name):
    prompts = hindi_prompts()
    tokenizer, model, device = load_generator(model_name)
    print(f"Scoring {len(rows)} Hindi rows on {device} with {model_name}.", flush=True)
    total = len(rows) * len(prompts)
    done = 0
    for row in rows:
        for key, title, template in prompts:
            reply = generate_reply(tokenizer, model, device, fill_prompt(template, row))
            row[key] = parse_score(reply)
            row[f"{key}_reply"] = reply
            done += 1
            if done % 15 == 0 or done == total:
                print(f"[{done}/{total}] {title}: {row[key]}", flush=True)
    return rows


def render(test_rows, scored_rows, model_name):
    traits = [trait_values(row) for row in test_rows]
    source_words = [source_characteristics(row["original"])["source_words"] for row in test_rows]
    simple_words = [source_characteristics(row["simplification"])["source_words"] for row in test_rows]
    lines = [
        "Hindi judge on the Hindi dataset",
        "",
        "Protocol",
        "Rows are read from datastore/hindi_simplification.csv.",
        "The Hindi prompts are applied to Hindi source sentences and Hindi simplifications.",
        "The table has no human rating from 0 to 10, so this section reports the judge scores and the overlap statistics.",
        f"Overlap statistics use the full test split (n={len(test_rows)}).",
        f"Judge scores use the first {len(scored_rows)} test rows. The split itself is the fixed seed-0 holdout.",
        f"Judge: {display_model(model_name)}",
        "The overlap score is the ridge model trained on the French training rows. It is not a human rating.",
        "",
        "Test-split overlap statistics",
        f"  Mean words in the source: {format_num(mean(source_words))}",
        f"  Mean words in the simplification: {format_num(mean(simple_words))}",
        f"  Mean share of source words kept: {format_num(mean([item['words_kept'] for item in traits]), 3)}",
        f"  Mean word overlap: {format_num(mean([item['overlap'] for item in traits]), 3)}",
        "",
        f"{'Prompt':16} {'n':>4} {'miss':>5} {'mean':>6} {'zero':>6} {'distinct':>8} {'r(kept)':>8} {'p':>8}",
    ]
    records = []
    for key, title, _template in hindi_prompts():
        scores = [row[key] for row in scored_rows]
        present = [(row, score) for row, score in zip(scored_rows, scores) if score is not None]
        missing = len(scores) - len(present)
        values = [score for _row, score in present]
        kept = [trait_values(row)["words_kept"] for row, _score in present]
        correlation, p_value = _correlation(pearsonr, kept, values)
        counts = Counter(values)
        lines.append(
            f"{title:16} {len(values):4d} {missing:5d} {format_num(mean(values)):>6} "
            f"{counts.get(0, 0):6d} {len(counts):8d} {format_num(correlation, 3):>8} {format_p(p_value):>8}"
        )
        lines.append(
            "  scores: " + ", ".join(f"{score}:{counts[score]}" for score in sorted(counts))
        )
        records.append(
            {
                "prompt": key,
                "n": len(values),
                "missing": missing,
                "mean": mean(values),
                "zeros": counts.get(0, 0),
                "distinct": len(counts),
                "pearson_words_kept": correlation,
                "pearson_p": p_value,
            }
        )
    overlap = [float(row["overlap_score"]) for row in scored_rows]
    lines.extend(
        [
            "",
            f"Mean overlap-model score on these {len(scored_rows)} rows: {format_num(mean(overlap))}.",
            "",
            "Examples",
        ]
    )
    ranked = sorted(scored_rows, key=lambda row: trait_values(row)["words_kept"])
    for heading, row in (("Lowest share of words kept", ranked[0]), ("Highest share of words kept", ranked[-1])):
        kept = trait_values(row)["words_kept"]
        lines.extend(
            [
                "",
                heading,
                f"  words_kept={kept:.2f}",
                f"  overlap_score={row['overlap_score']:.2f}",
                f"  original: {row['original']}",
                f"  simplification: {row['simplification']}",
            ]
        )
        for key, title, _template in hindi_prompts():
            score = row.get(key)
            shown = "missing" if score is None else str(score)
            lines.append(f"  {title}: {shown}")
    return "\n".join(lines) + "\n", records


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    test_rows = load_split("test")
    scored_rows = [dict(row) for row in test_rows[:SAMPLE_SIZE]]
    predictions = overlap_scores(scored_rows)
    for row, prediction in zip(scored_rows, predictions):
        row["overlap_score"] = float(prediction)
    model_name = resolve_model(DEFAULT_MODEL)
    score_rows(scored_rows, model_name)
    report, records = render(test_rows, scored_rows, model_name)
    print(report, end="")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / "hindi_judge_summary.txt"
    scores_path = OUTPUT_DIR / "hindi_judge_scores.csv"
    metrics_path = OUTPUT_DIR / "hindi_judge_results.csv"
    summary_path.write_text(report, encoding="utf-8")
    write_csv(
        scores_path,
        scored_rows,
        ["original", "simplification", "overlap_score", "short_hi", "long_hi"],
    )
    write_csv(
        metrics_path,
        records,
        ["prompt", "n", "missing", "mean", "zeros", "distinct", "pearson_words_kept", "pearson_p"],
    )
    print(f"Saved summary to {summary_path}")
    print(f"Saved scores to {scores_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Hindi table: {DATASET_PATH}")


if __name__ == "__main__":
    main()
