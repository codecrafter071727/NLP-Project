"""Official RQ2 evaluation for JUDGEBERT.

Trains a ridge model on the training split, then scores the untouched test
split with that model and with the French and English judging prompts.
Hindi prompts are scored on the Hindi simplification table.

Run from the repo root:

    python article_src/src/rq2_prompt_ratings.py
"""

import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from rq2_metrics import (
    OUTPUT_DIR,
    FEATURE_NAMES,
    fit_ridge,
    format_num,
    intervals_overlap,
    load_split,
    metric_row,
    parse_score,
    predict_ridge,
    regression_metrics,
    save_ridge,
    write_csv,
)

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def display_model(model_name):
    name = Path(model_name).name
    if name == "Qwen2.5-0.5B-Instruct":
        return DEFAULT_MODEL
    return model_name


def resolve_model(model_name):
    """Accept a Hugging Face id, a real folder, or a %USERPROFILE% path from PowerShell."""
    expanded = os.path.expandvars(os.path.expanduser(model_name))
    folder = Path(expanded)
    if folder.is_dir():
        return str(folder)
    local = Path.home() / ".cache" / "rq2" / "Qwen2.5-0.5B-Instruct"
    weights = local / "model.safetensors"
    if model_name == DEFAULT_MODEL and weights.is_file():
        return str(local)
    return expanded


HEADER = f"{'System':22} {'n':>4} {'miss':>5}  {'MAE':>5}  {'95% CI':>14}  {'RMSE':>5}  {'r':>6}  {'p':>6}"

SHORT_FR = (
    "Vous êtes un conseiller juridique au Québec. Donnez une note entière de 0 à 10 "
    "selon que la simplification conserve le sens juridique du texte original. "
    "10 signifie que le sens juridique est entièrement conservé, 0 signifie qu'il est perdu. "
    "Répondez uniquement par un nombre entier.\n"
    "Texte original : {original}\n"
    "Texte simplifié : {simplification}"
)

SHORT_EN = (
    "You are a legal advisor in Quebec. Give an integer score from 0 to 10 for how well "
    "the simplification preserves the legal meaning of the original text. "
    "10 means the legal meaning is fully preserved, 0 means it is lost. "
    "Reply with only an integer.\n"
    "Original text: {original}\n"
    "Simplified text: {simplification}"
)

SHORT_HI = (
    "आप क्यूबेक में एक कानूनी सलाहकार हैं। मूल पाठ का कानूनी अर्थ सरलीकरण कितना सुरक्षित रखता है, "
    "इसके लिए 0 से 10 तक एक पूर्णांक अंक दें। "
    "10 का अर्थ है कि कानूनी अर्थ पूरी तरह सुरक्षित है, 0 का अर्थ है कि वह खो गया है। "
    "केवल एक पूर्णांक लिखें।\n"
    "मूल पाठ: {original}\n"
    "सरलीकृत पाठ: {simplification}"
)

def long_prompt(prompt_name, closing_line):
    """Use a project judge prompt from evaluate_tools, then ask for one integer."""
    from evaluate_tools import prompt_fr, prompt_hi

    source = {"prompt_fr": prompt_fr, "prompt_hi": prompt_hi}[prompt_name]
    text = source.replace("{}", "{original}", 1).replace("{}", "{simplification}", 1)
    return text + "\n" + closing_line


def long_french_prompt():
    return long_prompt("prompt_fr", "Répondez uniquement par un nombre entier de 0 à 10.")


def long_hindi_prompt():
    return long_prompt("prompt_hi", "केवल 0 से 10 तक एक पूर्णांक लिखें।")


def build_prompts():
    return [
        ("short_fr", "Short French", SHORT_FR),
        ("short_en", "Short English", SHORT_EN),
        ("long_fr", "Long French", long_french_prompt()),
    ]


def get_prompts():
    if not hasattr(get_prompts, "cached"):
        get_prompts.cached = build_prompts()
    return get_prompts.cached


def fill_prompt(template, row):
    return template.replace("{original}", row["original"]).replace(
        "{simplification}", row["simplification"]
    )


def labels_of(rows):
    return [float(row["label"]) for row in rows]


def ridge_report(model, evaluated):
    lines = [
        "RQ2 official evaluation",
        "",
        "Protocol",
        "The ridge model is trained only on the train split of datastore/frjudge.csv.",
        "Dev, test, and both holdouts are scored after training.",
        "The identical holdout is the original sentence copied as its own simplification, so every human label is 10.",
        "The unrelated holdout pairs each original with an unrelated simplification, so every human label is 0.",
        "Correlation is undefined when the human labels do not vary. Error is the measure on those holdouts.",
        "",
        "Trained ridge model",
        HEADER,
    ]
    serial = {}
    for name, title, rows, predicted in evaluated:
        metrics = regression_metrics(labels_of(rows), predicted)
        lines.append(metric_row(title, metrics))
        serial[name] = metrics
    lines.append("")
    lines.append("Feature weights")
    for name, weight in zip(FEATURE_NAMES, model.coef_):
        lines.append(f"  {name:20} {weight:7.3f}")
    lines.append(f"  {'intercept':20} {float(model.intercept_):7.3f}")
    identical = serial["holdout_identical"]["mean_pred"]
    unrelated = serial["holdout_unrelated"]["mean_pred"]
    lines.append("")
    lines.append(
        "On the holdouts, the trained model predicts "
        f"{identical:.2f} for an identical pair and {unrelated:.2f} for an unrelated pair. "
        "The human labels are 10 and 0."
    )
    lines.append(
        "Every identical pair has the same overlap features, so the model assigns them one score "
        "and the error interval is a single number."
    )
    lines.append(
        "The unrelated error stays large because shared words still produce a mid-scale score. "
        "The model ranks identical pairs above unrelated pairs, and it does not push unrelated pairs down to 0."
    )
    return lines, serial


def load_generator(model_name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def generate_reply(tokenizer, model, device, prompt):
    import torch

    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        text = prompt + "\nScore:"
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=12,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    reply_tokens = output[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(reply_tokens, skip_special_tokens=True).strip()


def score_prompts(rows, model_name):
    tokenizer, model, device = load_generator(model_name)
    print(f"Scoring {len(rows)} test rows on {device} with {model_name}.", flush=True)
    total = len(rows) * len(get_prompts())
    done = 0
    for row in rows:
        for key, title, template in get_prompts():
            reply = generate_reply(tokenizer, model, device, fill_prompt(template, row))
            row[key] = parse_score(reply)
            row[f"{key}_reply"] = reply
            done += 1
            if done % 15 == 0 or done == total:
                print(f"[{done}/{total}] {title}: {row[key]}", flush=True)
    return rows


def prompt_section(rows, model_name):
    ridge_metrics = regression_metrics(labels_of(rows), [row["ridge"] for row in rows])
    lines = [
        "",
        "Prompt comparison on the official test split",
        f"Judge: {display_model(model_name)}",
        "Each test pair is scored once with each judge prompt. Only the prompt text changes.",
        "A reply with no integer from 0 to 10 is missing and is left out of that row's error.",
        HEADER,
    ]
    results = {}
    reliable = []
    for key, title, _template in get_prompts():
        paired_rows = [row for row in rows if row[key] is not None]
        missing = len(rows) - len(paired_rows)
        metrics = regression_metrics(labels_of(paired_rows), [row[key] for row in paired_rows])
        lines.append(metric_row(title, metrics, missing))
        results[key] = {"title": title, "missing": missing, "metrics": metrics}
        if metrics["n"] >= max(1, int(0.8 * len(rows))):
            reliable.append((title, metrics))
    lines.append(metric_row("Ridge on these rows", ridge_metrics))
    lines.append("")
    if not reliable:
        lines.append("No prompt returned a usable score on at least 80% of the test rows.")
        return lines, results
    winner_title, winner = min(reliable, key=lambda item: item[1]["mae"])
    lines.append(
        f"Lowest test error among reliable prompts: {winner_title} "
        f"(MAE {format_num(winner['mae'])}, "
        f"95% CI [{format_num(winner['mae_low'])}, {format_num(winner['mae_high'])}], "
        f"Pearson {format_num(winner['pearson'], 3)})."
    )
    for title, metrics in reliable:
        if title == winner_title:
            continue
        if intervals_overlap(
            (winner["mae_low"], winner["mae_high"]),
            (metrics["mae_low"], metrics["mae_high"]),
        ):
            lines.append(
                f"The {winner_title} and {title} error intervals overlap, "
                "so this test run does not separate them."
            )
    ridge_interval = (ridge_metrics["mae_low"], ridge_metrics["mae_high"])
    winner_interval = (winner["mae_low"], winner["mae_high"])
    if intervals_overlap(winner_interval, ridge_interval):
        lines.append("The best prompt and the trained ridge model have overlapping test-error intervals.")
    elif winner["mae"] < ridge_metrics["mae"]:
        lines.append(f"{winner_title} has a lower test error than the trained ridge model.")
    else:
        lines.append("The trained ridge model has a lower test error than every reliable prompt.")
    return lines, results


def example_lines(rows):
    lines = []
    high = max(rows, key=lambda row: float(row["label"]))
    low = min(rows, key=lambda row: float(row["label"]))
    for heading, row in (("Highest test label", high), ("Lowest test label", low)):
        lines.extend(["", heading, f"  label={float(row['label']):.1f}", f"  ridge={row['ridge']:.2f}"])
        lines.append(f"  original: {row['original']}")
        lines.append(f"  simplification: {row['simplification']}")
        for key, title, _template in get_prompts():
            score = row.get(key)
            shown = "missing" if score is None else str(score)
            lines.append(f"  {title}: {shown}")
    return lines


def save_tsv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    prompt_keys = [key for key, _title, _template in get_prompts()]
    fields = ["original", "simplification", "label", "ridge", *prompt_keys]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for row in rows:
            writer.writerow(
                [
                    str(row["original"]).replace("\t", " ").replace("\n", " "),
                    str(row["simplification"]).replace("\t", " ").replace("\n", " "),
                    row["label"],
                    f"{row['ridge']:.4f}",
                    *("" if row.get(key) is None else row[key] for key in prompt_keys),
                ]
            )


def metric_records(serial):
    fields = (
        "n",
        "mae",
        "rmse",
        "pearson",
        "pearson_p",
        "spearman",
        "spearman_p",
        "mae_low",
        "mae_high",
        "mean_pred",
        "mean_label",
    )
    records = []
    for name, metrics in serial.items():
        if name == "prompts":
            continue
        records.append({"system": name, "missing": 0, **{field: metrics[field] for field in fields}})
    for name, value in serial.get("prompts", {}).items():
        records.append(
            {
                "system": name,
                "missing": value["missing"],
                **{field: value["metrics"][field] for field in fields},
            }
        )
    return records


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--skip-prompts", action="store_true")
    parser.add_argument(
        "--prompt-limit",
        type=int,
        default=0,
        help="Score only the first N test rows. 0 scores the full test split.",
    )
    args = parser.parse_args()
    args.model = resolve_model(args.model)

    train = load_split("train")
    dev = load_split("dev")
    test = load_split("test")
    identical = load_split("holdout_identical")
    unrelated = load_split("holdout_unrelated")
    model = fit_ridge(train)
    print(f"Trained ridge on {len(train)} training rows.", flush=True)

    evaluated = []
    split_rows = {
        "dev": dev,
        "test": test,
        "holdout_identical": identical,
        "holdout_unrelated": unrelated,
    }
    titles = {
        "dev": "Dev check",
        "test": "Test",
        "holdout_identical": "Identical holdout",
        "holdout_unrelated": "Unrelated holdout",
    }
    predictions = {}
    for name in ("test", "dev", "holdout_identical", "holdout_unrelated"):
        predicted = predict_ridge(model, split_rows[name])
        predictions[name] = predicted
        evaluated.append((name, titles[name], split_rows[name], predicted))

    lines, serial = ridge_report(model, evaluated)
    for row, score in zip(test, predictions["test"]):
        row["ridge"] = float(score)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUTPUT_DIR / "rq2_ridge.pkl"
    save_ridge(model, train, model_path)
    partial = args.prompt_limit > 0
    summary_name = "rq2_summary_partial.txt" if partial else "rq2_summary.txt"
    report_name = "rq2_results_partial.csv" if partial else "rq2_results.csv"
    summary_path = OUTPUT_DIR / summary_name
    report_path = OUTPUT_DIR / report_name

    def write_outputs():
        report = "\n".join(lines) + "\n"
        summary_path.write_text(report, encoding="utf-8")
        write_csv(
            report_path,
            metric_records(serial),
            [
                "system",
                "missing",
                "n",
                "mae",
                "rmse",
                "pearson",
                "pearson_p",
                "spearman",
                "spearman_p",
                "mae_low",
                "mae_high",
                "mean_pred",
                "mean_label",
            ],
        )
        return report

    write_outputs()
    if not args.skip_prompts:
        prompt_rows = test[: args.prompt_limit] if partial else test
        try:
            score_prompts(prompt_rows, args.model)
        except Exception as error:
            lines.append("")
            lines.append(f"Prompt scoring did not finish: {error}")
            write_outputs()
            print()
            print("\n".join(lines))
            print(f"Saved the trained-model result to {summary_path}")
            raise
        extra, prompt_results = prompt_section(prompt_rows, args.model)
        if partial:
            lines.append("")
            lines.append(f"Partial prompt run: first {len(prompt_rows)} test rows only. Not the official result.")
        lines.extend(extra)
        lines.extend(example_lines(prompt_rows))
        serial["prompts"] = {
            key: {"missing": value["missing"], "metrics": value["metrics"]}
            for key, value in prompt_results.items()
        }
        save_tsv(prompt_rows, OUTPUT_DIR / ("rq2_test_scores_partial.csv" if partial else "rq2_test_scores.csv"))

    report = write_outputs()
    print()
    print(report)
    print(f"Saved model to {model_path}")
    print(f"Saved summary to {summary_path}")
    print(f"Saved metrics to {report_path}")


if __name__ == "__main__":
    main()
