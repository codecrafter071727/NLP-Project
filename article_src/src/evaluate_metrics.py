import argparse
import os
import pickle
import urllib
import warnings
from zipfile import ZipFile

import nltk
import torch
import wandb
from evaluate import load
from lens import LENS
from poutyne import set_seeds
from questeval.questeval_metric import QuestEval
from sentence_transformers import SentenceTransformer
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
)

from metrics.model_salience import CoverageModel
from factory_dataset import dataset_factory
from tools import (
    bool_parse,
)

warnings.filterwarnings("ignore")

nltk.download("stopwords")

# To handle LENS deterministic pattern fixed
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

torch.multiprocessing.set_sharing_strategy("file_system")

r2_metric = load("r_squared")
pearsonr_metric = load("pearsonr")

bertscore = load("bertscore")

questeval = QuestEval(
    task="text2text",
    language="en",
    device="cuda:0",
    do_weighter=False,
    list_scores=["bertscore", "answerability"],
    qg_batch_size=36,
)

lens_model_ckpt = os.path.join("LENS", "checkpoints", "lens_model.ckpt")
if not os.path.exists(lens_model_ckpt):
    url = "https://github.com/GRAAL-Research/MeaningBERT/releases/download/dependencies_model_release/LENS.zip"
    filehandle, _ = urllib.request.urlretrieve(url)
    with ZipFile(filehandle, "r") as zip_file:
        zip_file.extractall(".", members=None, pwd=None)
lens = LENS(lens_model_ckpt, rescale=True)

coverage_model_ckpt = os.path.join("coverage_roberta.bin")
if not os.path.exists(coverage_model_ckpt):
    url = (
        "https://github.com/GRAAL-Research/MeaningBERT/releases/download/dependencies_model_release/"
        "coverage_roberta.bin"
    )
    filehandle, _ = urllib.request.urlretrieve(url, filename=coverage_model_ckpt)

coverage_kis = CoverageModel(model_file=coverage_model_ckpt)

meaning_bert = load("davebulaval/meaningbert")

best_sentence_emb = SentenceTransformer("sentence-t5-xxl", device="cuda:0")
multilingual_sentence_emb = SentenceTransformer(
    "distiluse-base-multilingual-cased-v1", device="cuda:0"
)


def multiply(x):
    return round(x * 10, 1)


def divide(x):
    return round(x / 10, 1)


def compute_other_metrics_performance(
    test_set,
    holdout_identical_set,
    holdout_unrelated_set,
    logger,
    device: str = "cuda:0",
):
    compute_other_metrics_on_test_set(test_set=test_set, logger=logger, device=device)
    compute_other_metrics_on_holdout_identical_set(
        holdout_identical_set=holdout_identical_set, logger=logger, device=device
    )

    compute_other_metrics_on_holdout_unrelated_set(
        holdout_unrelated_set=holdout_unrelated_set, logger=logger, device=device
    )


def compute_other_metrics_on_test_set(test_set, logger, device: str = "cuda:0"):
    bertscore_precision_scores = []
    bertscore_recall_scores = []
    bertscore_f1_scores = []
    questeval_scores = []
    lens_scores = []
    coverage_scores = []
    meaningbert_scores = []
    sentence_emb_scores = []
    sentence_mutli_emb_scores = []
    ratings = []

    # Processing of all scores on the dataset
    for eval_data in tqdm(
        test_set, desc="Doing test evaluation on best selected metrics"
    ):
        original = [eval_data["original"]]
        simplification = [eval_data["simplification"]]
        rating = eval_data["label"]
        ratings.append(rating)

        # Default model is the best
        # We use the precision, recall and F1
        # Between 0 and 1: 1 is the best
        # We multiply by 10 to get a 0-10 scores
        bertscore_score = bertscore.compute(
            predictions=simplification, references=original, lang="en", device=device
        )
        bertscore_precision_scores.append(multiply(bertscore_score["precision"][0]))
        bertscore_recall_scores.append(multiply(bertscore_score["recall"][0]))
        bertscore_f1_scores.append(multiply(bertscore_score["f1"][0]))

        # Between 0 and 1: 1 is the best
        # We multiply by 10 to get a 0-10 scores
        questeval_scores.append(
            multiply(
                questeval.corpus_questeval(sources=original, hypothesis=simplification)[
                    "ex_level_scores"
                ][0]
            )
        )

        # Between 0 and 100: 100 is the best
        # We divide by 10 to get a 0-10 scores
        lens_scores.append(
            divide(
                lens.score(
                    complex=original,
                    simplified=simplification,
                    references=[original],
                    batch_size=1,
                    gpus=1,
                )[0]
            )
        )

        # Between 0 and 1: 1 is the best
        # We multiply by 10 to get a 0-10 scores
        coverage_scores.append(
            multiply(
                coverage_kis.score(bodies=original, decodeds=simplification)["scores"][
                    0
                ]
            )
        )

        # Between 0 and 100: 100 is the best
        # We divide by 10 to get a 0-10 scores
        meaningbert_scores.append(
            divide(
                meaning_bert.compute(references=original, predictions=simplification)[
                    "scores"
                ][0]
            )
        )

        # Between 0 and 1: 1 is the best
        # We multiply by 10 to get a 0-10 scores
        sentence_emb_scores.append(
            multiply(
                best_sentence_emb.similarity(
                    best_sentence_emb.encode(original),
                    best_sentence_emb.encode(simplification),
                ).tolist()[0][0]
            )
        )

        sentence_mutli_emb_scores.append(
            multiply(
                multilingual_sentence_emb.similarity(
                    multilingual_sentence_emb.encode(original),
                    multilingual_sentence_emb.encode(simplification),
                ).tolist()[0][0]
            )
        )

    # Compilation of the scores per metrics
    for metric_name, metric_score in zip(
        [
            "BERTScore precision",
            "BERTScore recall",
            "BERTScore F1",
            "QuestEval",
            "LENS",
            "Coverage",
            "MeaningBERT",
            "SBERT Embeddings",
            "SBERT Multi Embeddings",
        ],
        [
            bertscore_precision_scores,
            bertscore_recall_scores,
            bertscore_f1_scores,
            questeval_scores,
            lens_scores,
            coverage_scores,
            meaningbert_scores,
            sentence_emb_scores,
            sentence_mutli_emb_scores,
        ],
    ):
        pearson_corr = pearsonr_metric.compute(
            predictions=metric_score, references=ratings, return_pvalue=True
        )
        logger.log({f"test/{metric_name}_pearson_corr": pearson_corr["pearsonr"]})
        logger.log({f"test/{metric_name}_pearson_pvalue": pearson_corr["p-value"]})

        r_squared = r2_metric.compute(predictions=metric_score, references=ratings)
        logger.log({f"test/{metric_name}_R2": r_squared})

        rmse_score = mean_squared_error(metric_score, ratings, squared=False)
        logger.log({f"test/{metric_name}_RMSE": rmse_score})

        os.makedirs("predictions", exist_ok=True)

        file_path = f"{metric_name.replace(' ', '_')}"
        if data_augmentation:
            file_path += "_WithDA"
        file_path += "_predictions.pickle"
        with open(os.path.join("predictions", file_path), "wb") as file:
            pickle.dump(metric_score, file)


def compute_other_metrics_on_holdout_identical_set(
    holdout_identical_set, logger, device: str = "cuda:0"
):
    bertscore_precision_scores = []
    bertscore_recall_scores = []
    bertscore_f1_scores = []
    questeval_scores = []
    lens_scores = []
    coverage_scores = []
    meaningbert_scores = []
    sentence_emb_scores = []
    sentence_mutli_emb_scores = []
    ratings = []

    # Processing of all scores on the dataset
    for eval_data in tqdm(
        holdout_identical_set,
        desc="Doing holdout evaluation on identical on best selected metrics",
    ):
        original = [eval_data["original"]]
        simplification = [eval_data["simplification"]]
        rating = eval_data["label"]
        ratings.append(rating)

        # Default model is the best
        # We use the precision, recall and F1
        # Between 0 and 1: 1 is the best
        # We multiply by 10 to get a 0-10 scores
        bertscore_score = bertscore.compute(
            predictions=simplification, references=original, lang="en", device=device
        )
        bertscore_precision_scores.append(multiply(bertscore_score["precision"][0]))
        bertscore_recall_scores.append(multiply(bertscore_score["recall"][0]))
        bertscore_f1_scores.append(multiply(bertscore_score["f1"][0]))

        # Between 0 and 1: 1 is the best
        # We multiply by 10 to get a 0-10 scores
        questeval_scores.append(
            multiply(
                questeval.corpus_questeval(sources=original, hypothesis=simplification)[
                    "ex_level_scores"
                ][0]
            )
        )

        # Between 0 and 100: 100 is the best
        # We divide by 10 to get a 0-10 scores
        lens_scores.append(
            divide(
                lens.score(
                    complex=original,
                    simplified=simplification,
                    references=[original],
                    batch_size=1,
                    gpus=1,
                )[0]
            )
        )

        # Between 0 and 1: 1 is the best
        # We multiply by 10 to get a 0-10 scores
        coverage_scores.append(
            multiply(
                coverage_kis.score(bodies=original, decodeds=simplification)["scores"][
                    0
                ]
            )
        )

        # Between 0 and 100: 100 is the best
        # We divide by 10 to get a 0-10 scores
        meaningbert_scores.append(
            divide(
                meaning_bert.compute(references=original, predictions=simplification)[
                    "scores"
                ][0]
            )
        )

        # Between 0 and 1: 1 is the best
        # We multiply by 10 to get a 0-10 scores
        sentence_emb_scores.append(
            multiply(
                best_sentence_emb.similarity(
                    best_sentence_emb.encode(original),
                    best_sentence_emb.encode(simplification),
                ).tolist()[0][0]
            )
        )

        sentence_mutli_emb_scores.append(
            multiply(
                multilingual_sentence_emb.similarity(
                    multilingual_sentence_emb.encode(original),
                    multilingual_sentence_emb.encode(simplification),
                ).tolist()[0][0]
            )
        )

    # Compilation of the scores per metrics
    for metric_name, metric_score in zip(
        [
            "BERTScore precision",
            "BERTScore recall",
            "BERTScore F1",
            "QuestEval",
            "LENS",
            "Coverage",
            "MeaningBERT",
            "SBERT Embeddings",
            "SBERT Multi Embeddings",
        ],
        [
            bertscore_precision_scores,
            bertscore_recall_scores,
            bertscore_f1_scores,
            questeval_scores,
            lens_scores,
            coverage_scores,
            meaningbert_scores,
            sentence_emb_scores,
            sentence_mutli_emb_scores,
        ],
    ):
        counts_95 = [round(s) > 95 for s in metric_score]
        ratio_95 = sum(counts_95) / len(counts_95) * 100
        counts_99 = [round(s) > 99 for s in metric_score]
        ratio_99 = sum(counts_99) / len(counts_99) * 100
        counts_equals = [round(s) == 100 for s in metric_score]
        ratio_equals = sum(counts_equals) / len(counts_equals) * 100

        logger.log({f"test/{metric_name}_ratio_equals": ratio_equals})
        logger.log({f"test/{metric_name}_ratio_99": ratio_99})
        logger.log({f"test/{metric_name}_ratio_95": ratio_95})


def compute_other_metrics_on_holdout_unrelated_set(
    holdout_unrelated_set, logger, device: str = "cuda:0"
):
    bertscore_precision_scores = []
    bertscore_recall_scores = []
    bertscore_f1_scores = []
    questeval_scores = []
    lens_scores = []
    coverage_scores = []
    meaningbert_scores = []
    sentence_emb_scores = []
    sentence_mutli_emb_scores = []
    ratings = []

    # Processing of all scores on the dataset
    for eval_data in tqdm(
        holdout_unrelated_set,
        desc="Doing holdout evaluation on unrelated best selected metrics",
    ):
        original = [eval_data["original"]]
        simplification = [eval_data["simplification"]]
        rating = eval_data["label"]
        ratings.append(rating)

        # Default model is the best
        # We use the precision, recall and F1
        # Between 0 and 1: 1 is the best
        # We multiply by 10 to get a 0-10 scores
        bertscore_score = bertscore.compute(
            predictions=simplification, references=original, lang="en", device=device
        )
        bertscore_precision_scores.append(multiply(bertscore_score["precision"][0]))
        bertscore_recall_scores.append(multiply(bertscore_score["recall"][0]))
        bertscore_f1_scores.append(multiply(bertscore_score["f1"][0]))

        # Between 0 and 1: 1 is the best
        # We multiply by 10 to get a 0-10 scores
        questeval_scores.append(
            multiply(
                questeval.corpus_questeval(sources=original, hypothesis=simplification)[
                    "ex_level_scores"
                ][0]
            )
        )

        # Between 0 and 100: 100 is the best
        # We divide by 10 to get a 0-10 scores
        lens_scores.append(
            divide(
                lens.score(
                    complex=original,
                    simplified=simplification,
                    references=[original],
                    batch_size=1,
                    gpus=1,
                )[0]
            )
        )

        # Between 0 and 1: 1 is the best
        # We multiply by 10 to get a 0-10 scores
        coverage_scores.append(
            multiply(
                coverage_kis.score(bodies=original, decodeds=simplification)["scores"][
                    0
                ]
            )
        )

        # Between 0 and 100: 100 is the best
        # We divide by 10 to get a 0-10 scores
        meaningbert_scores.append(
            divide(
                meaning_bert.compute(references=original, predictions=simplification)[
                    "scores"
                ][0]
            )
        )

        # Between 0 and 1: 1 is the best
        # We multiply by 10 to get a 0-10 scores
        sentence_emb_scores.append(
            multiply(
                best_sentence_emb.similarity(
                    best_sentence_emb.encode(original),
                    best_sentence_emb.encode(simplification),
                ).tolist()[0][0]
            )
        )

        sentence_mutli_emb_scores.append(
            multiply(
                multilingual_sentence_emb.similarity(
                    multilingual_sentence_emb.encode(original),
                    multilingual_sentence_emb.encode(simplification),
                ).tolist()[0][0]
            )
        )

    # Compilation of the scores per metrics
    for metric_name, metric_score in zip(
        [
            "BERTScore precision",
            "BERTScore recall",
            "BERTScore F1",
            "QuestEval",
            "LENS",
            "Coverage",
            "MeaningBERT",
            "SBERT Embeddings",
            "SBERT Multi Embeddings",
        ],
        [
            bertscore_precision_scores,
            bertscore_recall_scores,
            bertscore_f1_scores,
            questeval_scores,
            lens_scores,
            coverage_scores,
            meaningbert_scores,
            sentence_emb_scores,
            sentence_mutli_emb_scores,
        ],
    ):
        counts_1 = [round(s) < 1 for s in metric_score]
        ratio_1 = sum(counts_1) / len(counts_1) * 100
        counts_5 = [round(s) < 5 for s in metric_score]
        ratio_5 = sum(counts_5) / len(counts_5) * 100
        counts_equals = [round(s) == 0 for s in metric_score]
        ratio_equals = sum(counts_equals) / len(counts_equals) * 100

        logger.log({f"test/{metric_name}_ratio_equals": ratio_equals})
        logger.log({f"test/{metric_name}_ratio_1": ratio_1})
        logger.log({f"test/{metric_name}_ratio_5": ratio_5})


parser = argparse.ArgumentParser()
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="The seed to use for training.",
)

parser.add_argument(
    "--root",
    type=str,
    default=".",
    help="Root directory.",
)

parser.add_argument(
    "--data_augmentation",
    type=bool_parse,
    default=False,
    help="Either or not to do data augmentation.",
)

parser.add_argument(
    "--model_name",
    type=str,
    default="bert-base-uncased",
    help="The model name to use as the starting model checkpoint.",
)

args = parser.parse_args()

seed = args.seed
root = args.root
data_augmentation = args.data_augmentation

wandb.init(
    project="huggingface",
    name=f"evaluate_other_metrics_with_DA-{data_augmentation}_seed-{seed}",
)

set_seeds(seed=seed)

(
    legal_precision_annotation_dataset,
    holdout_identical_dataset,
    holdout_unrelated_dataset,
) = dataset_factory(data_augmentation=data_augmentation)

checkpoint = args.model_name
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(
        example["original"], example["simplification"], truncation=True, padding=True
    )


tokenized_legal_annotation_dataset = legal_precision_annotation_dataset.map(
    tokenize_function, batched=True
)
tokenize_holdout_identical_dataset = holdout_identical_dataset.map(
    tokenize_function, batched=True
)
tokenize_holdout_unrelated_dataset = holdout_unrelated_dataset.map(
    tokenize_function, batched=True
)

compute_other_metrics_performance(
    tokenized_legal_annotation_dataset["test"],
    tokenize_holdout_identical_dataset["train"],
    tokenize_holdout_unrelated_dataset["train"],
    wandb,
)
