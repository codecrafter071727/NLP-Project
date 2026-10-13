import argparse

from tools import bool_parse

from evaluation_loop import evaluation_loop

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_name",
        type=str,
        help="Model name to process the evaluation loop on.",
    )

    parser.add_argument(
        "data_augmentation",
        type=bool_parse,
        default=True,
        help="Whether to compute use the data augmentation dataset.",
    )

    args = parser.parse_args()

    model_name = args.model_name
    data_augmentation = args.data_augmentation

    evaluation_loop(
        model_name=model_name,
        data_augmentation=data_augmentation,
    )
