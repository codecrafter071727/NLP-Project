# REF: https://huggingface.co/spaces/zoebat20/BLiMP/blob/main/app.py
# Zero shot HF doc: https://huggingface.co/docs/transformers/v4.47.1/en/main_classes/pipelines#transformers.ZeroShotClassificationPipeline

import torch
from transformers import pipeline

prompt_en = (
    "You will act as a legal advisor for laymen in Quebec. Your task is to classify if a tuple of sentences of a "
    "complex legal sentence that has been simplified conveys the same legal meaning. The legal meaning measures "
    "how well the output text coveys the legal details and exceptions and does not misrepresent the law. "
    "A score between 6 and 10 means the simplification seems to entail the legal details and exceptions "
    "properly and does not misrepresent the law, it is considered accurate. A score between 2 and 5 means "
    "the simplified sentence seems to entail the legal details and exceptions improperly and slightly misrepresents "
    "the law. A score of 1 means the simplification is completely off track. You can remove one point for each "
    "hallucination, omission or consistency between the original text and the simplification. We refer to "
    "hallucinations as facts included in the simplification that do not appear in the original text, omissions "
    "as cases where essential facts in the original are not in the simplified text, and consistency as "
    "inconsistency in the vocabulary used."
    "Original text: {}"
    "Simplified text: {}"
)

prompt_fr = (
    "Vous agirez en tant que conseiller juridique au Québec. Votre tâche consiste à classer si un "
    "tuple de phrases d'une phrase juridique complexe qui a été simplifiée transmet le même sens juridique. Le sens "
    "juridique mesure à quel point le texte de sortie couvre les détails et les exceptions juridiques et ne déforme "
    "pas la loi. Un score compris entre 6 et 10 signifie que la simplification semble inclure correctement les détails "
    "et exceptions juridiques et ne pas déformer le droit, elle est donc considérée comme exacte, tandis qu'un score "
    "compris entre 2 et 5 signifie que la phrase simplifiée semble impliquer de manière inappropriée les détails et "
    "exceptions juridiques et déformer légèrement le droit. "
    "La note de 1 signifie que la simplification est complètement erronée. Vous pouvez retirer un point pour chaque "
    "hallucination, omission ou cohérence entre le texte original et la simplification. Nous entendons par "
    "hallucinations les faits inclus dans la simplification qui n'apparaissent pas dans le texte original, "
    "par omissions les cas où des faits essentiels de l'original ne figurent pas dans le texte simplifié, et "
    "par cohérence les incohérences dans le vocabulaire utilisé dans la simplification. "
    'Texte original : "{}" '
    'Texte simplifié : "{}"'
)

prompt_hi = (
    "आप क्यूबेक में आम लोगों के लिए कानूनी सलाहकार की भूमिका निभाएँगे। आपका कार्य यह जाँचना है कि "
    "एक जटिल कानूनी वाक्य का सरलीकृत रूप वही कानूनी अर्थ रखता है या नहीं। कानूनी अर्थ इस बात को मापता है "
    "कि परिणाम पाठ कानूनी विवरण और अपवादों को कितनी अच्छी तरह रखता है और कानून को गलत रूप में नहीं दिखाता। "
    "6 से 10 तक का अंक मतलब है कि सरलीकरण कानूनी विवरण और अपवादों को ठीक से रखता है और कानून को गलत नहीं दिखाता, "
    "इसलिए उसे सही माना जाता है। 2 से 5 तक का अंक मतलब है कि सरलीकृत वाक्य कानूनी विवरण और अपवादों को ठीक से नहीं रखता "
    "और कानून को थोड़ा गलत दिखाता है। 1 का अंक मतलब है कि सरलीकरण पूरी तरह गलत है। "
    "मूल पाठ और सरलीकरण के बीच हर भ्रम, छूटे तथ्य या असंगति के लिए आप एक अंक घटा सकते हैं। "
    "भ्रम वे तथ्य हैं जो सरलीकरण में हैं पर मूल पाठ में नहीं। छूटे तथ्य वे आवश्यक तथ्य हैं जो मूल में हैं पर सरलीकृत पाठ में नहीं। "
    "असंगति का अर्थ सरलीकरण में प्रयुक्त शब्दावली का मेल न बैठना है। "
    'मूल पाठ: "{}" '
    'सरलीकृत पाठ: "{}"'
)


def evaluation_llm_prompting(rows, tokenizer, model):
    with torch.no_grad():
        originals = rows["original"]
        simplifieds = rows["simplification"]

        formated_prompts = [
            prompt_fr.format(original, simplified)
            for original, simplified in zip(originals, simplifieds)
        ]

        pipe = pipeline(
            model=model,
            tokenizer=tokenizer,
            task="zero-shot-classification",
            torch_dtype="float16",
            num_workers=2,
        )
        outputs = pipe(
            formated_prompts,
            candidate_labels=[f"{i}" for i in range(1, 11)],
            batch_size=len(originals),
        )

        # Labels are sorted in likelihood order.
        predictions = [int(output["labels"][0]) for output in outputs]

        return {"prediction": predictions}
