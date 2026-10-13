# We use load_dataset from HG datasets library since it handle well JSONL
import os

from datasets import load_dataset

annotated_data = load_dataset(
    os.path.join("../datastore", "released"),
    data_files=["FrJUDGE.jsonl"],
)

annotated_data["train"].to_json(
    os.path.join("../datastore", "released", "FrJUDGE.jsonl"),
    force_ascii=False,
)
