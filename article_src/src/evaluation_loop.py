import os
from functools import partial

import numpy as np
import torch
import wandb
from dotenv import dotenv_values

from factory_dataset import dataset_factory
from evaluate_tools import evaluation_llm_prompting
from factory_model_tokenizer import model_tokenizer_factory
from metrics.metrics import (
    compute_metrics,
    eval_compute_metrics_identical,
    eval_compute_metrics_unrelated,
)

# To make Wandb silent
os.environ["WANDB_SILENT"] = "true"


def evaluation_loop(
    model_name: str,
    data_augmentation: bool,
):
    params = {"batched": True, "batch_size": 16}
    output_dir = "predictions"
    os.makedirs(output_dir, exist_ok=True)

    secrets = dotenv_values(".env")

    huggingface_token = secrets["huggingface_token"]

    device = torch.device("cuda")

    clean_model_name = model_name.split("/")[-1]

    wandb.init(
        project=f"kil_llm_eval",
        config={"model_name": model_name, "data_augmentation": data_augmentation},
        name=f"{clean_model_name}_with_DA-{data_augmentation}",
    )

    model, tokenizer = model_tokenizer_factory(
        model_name=model_name,
        token=huggingface_token,
        device=device,
    )

    evaluation_fn = partial(
        evaluation_llm_prompting,
        tokenizer=tokenizer,
        model=model,
    )

    (
        legal_precision_annotation_dataset,
        holdout_identical_dataset,
        holdout_unrelated_dataset,
    ) = dataset_factory(data_augmentation=data_augmentation)

    process_dataset = legal_precision_annotation_dataset["test"].map(
        evaluation_fn, desc=f"----Doing model {model_name} for test set-----", **params
    )

    prediction = np.array(process_dataset["prediction"])
    labels = legal_precision_annotation_dataset["test"]["label"]
    payload = {f"test/{k}": v for k, v in compute_metrics((prediction, labels)).items()}
    process_dataset.to_csv(
        os.path.join(
            output_dir, f"{model_name.replace('/', '_')}_predictions_test.tsv"
        ),
        index=False,
        sep="\t",
    )

    process_dataset = holdout_identical_dataset["train"].map(
        evaluation_fn,
        desc=f"----Doing model {model_name} for identical set-----",
        **params,
    )
    prediction = np.array(process_dataset["prediction"])
    labels = holdout_identical_dataset["train"]["label"]
    payload.update(
        {
            f"test/{k}": v
            for k, v in eval_compute_metrics_identical((prediction, labels)).items()
        }
    )
    process_dataset.to_csv(
        os.path.join(
            output_dir, f"{model_name.replace('/', '_')}_predictions_identical.tsv"
        ),
        index=False,
        sep="\t",
    )

    process_dataset = holdout_unrelated_dataset["train"].map(
        evaluation_fn,
        desc=f"----Doing model {model_name} for unrelated set-----",
        **params,
    )
    prediction = np.array(process_dataset["prediction"])
    labels = holdout_unrelated_dataset["train"]["label"]
    payload.update(
        {
            f"test/{k}": v
            for k, v in eval_compute_metrics_unrelated((prediction, labels)).items()
        }
    )
    process_dataset.to_csv(
        os.path.join(
            output_dir, f"{model_name.replace('/', '_')}_predictions_unrelated.tsv"
        ),
        index=False,
        sep="\t",
    )

    wandb.log({**payload})

    # We close the run since we will start a new one in the for loop for the next model.
    wandb.finish(exit_code=0)
