import subprocess

from transformers import logging

from models import model_names

logging.set_verbosity_warning()

for data_augmentation in [False, True]:
    for model_name in model_names:
        subprocess.run(
            f"python3 evaluate_llm.py {model_name} {data_augmentation}",
            shell=True,
        )
