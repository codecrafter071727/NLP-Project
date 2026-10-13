import random
from pathlib import Path

from prodigy.components.loaders import JSONL
from prodigy.core import Arg, recipe
from prodigy.util import set_hashes

options = [
    {
        "id": "1",
        "text": "Description de l'avenant/de la garantie",
    },
    {
        "id": "2",
        "text": "Condition(s) d’application",
    },
    {
        "id": "3",
        "text": "Exclusion(s) ou restriction(s)",
    },
    {
        "id": "4",
        "text": "Dommages",
    },
    {
        "id": "5",
        "text": "Indemnités",
    },
    {
        "id": "6",
        "text": "Définition",
    },
    {
        "id": "7",
        "text": "Frais",
    },
    {
        "id": "8",
        "text": "Prime",
    },
    {
        "id": "9",
        "text": "Obligation(s) de l’assuré",
    },
    {
        "id": "10",
        "text": "Conséquence(s) du non-respect des obligations",
    },
    {
        "id": "11",
        "text": "Obligation(s) de l’assureur",
    },
    {
        "id": "12",
        "text": "Droit(s) de l’assuré",
    },
    {
        "id": "13",
        "text": "Droit(s) de l’assureur",
    },
    {
        "id": "14",
        "text": "Subrogation",
    },
    {
        "id": "15",
        "text": "Prise d’effet, renouvellement",
    },
    {
        "id": "16",
        "text": "Fin du contrat, résiliation",
    },
    {
        "id": "17",
        "text": "Recours",
    },
    {
        "id": "18",
        "text": "Autres (inscrire dans commentaires si ajouts requis)",
    },
]

root = Path(__file__).parent


# Helper functions for adding user provided labels to annotation tasks.
def add_label_options_to_stream(stream):
    for task in stream:
        task["options"] = options
        task["original"] = task["text"]
        task["simplified"] = task["simplified_text"]
        yield task


@recipe(
    "legal_meaning",
    dataset=Arg(help="Dataset to save answers to."),
    source=("The source data as a JSONL file", "positional", None, str),
)
def legal_meaning(dataset: str, source: str):
    def validate_answer(eg):
        difficulty_evaluation = eg.get("difficulty_evaluation", None)

        if difficulty_evaluation is None:
            raise ValueError("Sélectionner le niveau de difficulté.")

        high_level_annotation = eg.get("accept")
        if len(high_level_annotation) == 0:
            raise ValueError("Sélectionner la qualification.")

        evaluation = eg.get("evaluation", None)
        if evaluation is None:
            raise ValueError("Sélectionner le niveau de précision légale.")

        justifications = eg.get("justifications", None)
        if justifications is None:
            raise ValueError("Justifier votre niveau de précision légale.")

    # Load the stream from a JSONL file and return a generator that yields a
    # dictionary for each example in the data.
    stream = JSONL(source)
    stream = list(
        (set_hashes(eg, input_keys=("text", "simplified_text")) for eg in stream)
    )

    # Randomized the stream
    # We need to do it before add_label_options_to_stream since it return a generator
    random.shuffle(stream)

    # Add labels to each task in stream
    stream = add_label_options_to_stream(stream)

    blocks = [
        {"view_id": "html", "html_template": (root / "recipe.html").read_text()},
        {
            "view_id": "text_input",
            "field_rows": 3,
            "field_placeholder": "Tapez ici...",
            "field_label": "Identifier le niveau de difficulté selon les choix suivants:",
            "field_id": "difficulty_evaluation",
            "field_suggestions": [
                "Plus simple à lire",
                "Aussi simple à lire",
                "Plus difficile à lire",
                "Pas de simplification",
            ],
        },
        {
            "view_id": "html",
            "html_template": "Étape 2 de l'évaluation: Selon vous, quelle est la qualification du texte?",
        },
        {
            "view_id": "choice",
            "text": None,
            "field_id": "meaning_evaluation",
        },
        {
            "view_id": "html",
            "html_template": "Étape 3 de l'évaluation: Selon vous, quel est le niveau de précision légale du texte "
            "généré par le modèle sur une échelle de 1 à 10?",
        },
        {
            "view_id": "text_input",
            "field_rows": 1,
            "field_label": "Identifier le niveau de précision légale selon les choix suivants:",
            "field_placeholder": "Tapez ici...",
            "field_id": "evaluation",
            "field_suggestions": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
        },
        {
            "view_id": "html",
            "html_template": "Étape 4 de l'évaluation: Justifier la présence d'erreur(s) fatale(s) "
            "ou pour chaque imprécision:",
        },
        {
            "view_id": "text_input",
            "field_rows": 3,
            "field_placeholder": "Tapez ici...",
            "field_id": "justifications",
        },
    ]

    return {
        "view_id": "blocks",
        "dataset": dataset,  # save annotations in this dataset
        "stream": stream,
        "config": {  # Additional config settings, mostly for app UI
            "exclude_by": "input",  # Hash value used to filter out already seen examples
            "blocks": blocks,
        },
        "validate_answer": validate_answer,
    }
