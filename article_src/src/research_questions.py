"""Continuation of JUDGEBERT on the official splits.

RQ1 measures how the original sentence shapes the simplification.
RQ2 trains the overlap model and compares the French and English judge prompts with the human ratings.
Hindi prompts are scored on the Hindi simplification table.
RQ3 checks which simplification traits still match the human rating on the test split.

The long French judge prompt is the one already defined in evaluate_tools.py.
The overlap traits are the features already defined in overlap_baseline.py.
The Kaggle legal-clause table and the Hindi simplification table are checked with the same source characteristics as RQ1.

Run from the repo root:

    python article_src/src/research_questions.py
"""

import sys

from rq2_metrics import OUTPUT_DIR


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    from hindi_characteristics import main as run_hindi
    from hindi_judge import main as run_hindi_judge
    from kaggle_characteristics import main as run_kaggle
    from rq1_characteristics import main as run_rq1
    from rq2_prompt_ratings import main as run_rq2
    from rq3_traits import main as run_rq3

    run_rq1()
    run_rq3()
    run_rq2()
    run_kaggle()
    run_hindi()
    run_hindi_judge()
    print(f"Saved the continuation report to {write_combined()}")


def write_combined():
    parts = []
    for name in (
        "rq1_summary.txt",
        "rq2_summary.txt",
        "rq3_summary.txt",
        "kaggle_summary.txt",
        "hindi_summary.txt",
        "hindi_judge_summary.txt",
    ):
        path = OUTPUT_DIR / name
        if path.is_file():
            parts.append(path.read_text(encoding="utf-8").rstrip())
    combined = OUTPUT_DIR / "research_questions.txt"
    combined.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    return combined


if __name__ == "__main__":
    main()
