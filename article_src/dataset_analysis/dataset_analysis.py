import os
import re
from functools import partial
from statistics import mean

import spacy
from datasets import load_dataset
from textstat.textstat import textstatistics

n_cores = 9

try:
    nlp = spacy.load("fr_core_news_sm")
except OSError as e:
    import spacy.cli

    spacy.cli.download("fr_core_news_sm")
    nlp = spacy.load("fr_core_news_sm")


def tokenize(model, text):
    return [token for token in model.tokenizer(text)]


text_statistic = textstatistics()
text_statistic.set_lang(lang="fr")

pattern = re.compile(r"([0-9]+\.[0-9]+)")


def compute_stats(row, key):
    text = row[key].lower()

    tokens = [
        token.text
        for token in tokenize(nlp, text)
        if not token.is_punct
        and not "\n" in token.text
        and not token.is_digit
        and not "|" in token.text
        and not "$" in token.text
        and not "<" in token.text
        and not ">" in token.text
        and not " " in token.text
        and pattern.match(token.text) is None
    ]

    lexical_words = [
        token.text
        for token in tokenize(nlp, text)
        if not token.is_punct
        and not "\n" in token.text
        and not token.is_digit
        and not "|" in token.text
        and not "$" in token.text
        and not "<" in token.text
        and not ">" in token.text
        and not " " in token.text
        and not token.is_stop
        and pattern.match(token.text) is None
    ]

    # Vocab size statistics
    vocabulary_cardinality.update(tokens)  # Vocab size
    all_words.append(tokens)  # For lexical richness
    vocabulary_lexical_words_cardinality.update(
        lexical_words
    )  # Vocab lexical words size

    # Size statistics
    average_number_of_tokens.append(len(tokens))
    average_number_of_lexical_tokens.append(len(lexical_words))

    doc = nlp(text)
    sentences = list(doc.sents)

    # Average number of sentence statistic
    average_num_sentences.append(len(sentences))

    doc_sentence_len = []
    for sentence in sentences:
        doc_sentence_len.append(
            len(
                [
                    token.text
                    for token in tokenize(nlp, str(sentence))
                    if not token.is_punct
                    and not "\n" in token.text
                    and not token.is_digit
                    and not "|" in token.text
                    and not "$" in token.text
                    and not "<" in token.text
                    and not ">" in token.text
                    and not " " in token.text
                    and pattern.match(token.text) is None
                ]
            )
        )

    doc_lexical_sentence_len = []
    for sentence in sentences:
        doc_lexical_sentence_len.append(
            len(
                [
                    token.text
                    for token in tokenize(nlp, str(sentence))
                    if not token.is_punct
                    and not "\n" in token.text
                    and not token.is_digit
                    and not "|" in token.text
                    and not "$" in token.text
                    and not "<" in token.text
                    and not ">" in token.text
                    and not " " in token.text
                    and not token.is_stop
                    and pattern.match(token.text) is None
                ]
            )
        )

    # Average sentence length statistics
    average_sentence_length.extend(doc_sentence_len)
    average_sen_len_lexical.extend(doc_lexical_sentence_len)

    # Difficulty levels
    fk_grades.append(text_statistic.flesch_reading_ease(text))


root = "..."

insurance_text_simplification = load_dataset(
    os.path.join(root, "datastore", "annotated"),
    data_files=["insurance_text_simplifications_annotated.jsonl"],
)

vocabulary_cardinality = set()
all_words = []
vocabulary_lexical_words_cardinality = set()

average_number_of_tokens = []
average_number_of_lexical_tokens = []

average_num_sentences = []

average_sentence_length = []
average_sen_len_lexical = []

fk_grades = []

partial_compute_stats = partial(compute_stats, key="text")
insurance_text_simplification.map(partial_compute_stats)

print(" -- Source Statistics -- ")
print(f"The len of the dataset is {len(insurance_text_simplification['train'])}")
print(f"The vocab size is {len(vocabulary_cardinality)}")
print(f"The avg number of tokens is {mean(average_number_of_tokens)}")
print(f"The number of tokens is {sum(average_number_of_tokens)}")
print(f"The avg number of lexical tokens is {mean(average_number_of_lexical_tokens)}")
print(f"The average number of sentence is {mean(average_num_sentences)}")
print(f"The average len sentence is {mean(average_sentence_length)}")
print(f"The average len sentence lexical is {mean(average_sen_len_lexical)}")
print(
    f"The lexical richness is {round(len(vocabulary_cardinality) / len(all_words), 2)}"
)
print(f"The FK is {mean(fk_grades)}")
print(max(fk_grades))
print(
    text_statistic.flesch_reading_ease(
        "La ville et la province de l’adresse écrite à cet article 1 constituent les lieux d’usage principal, de remisage et de stationnement du véhicule désigné. Si ce n’est pas le cas, le preneur ou l’assuré désigné doit le déclarer."
    )
)
print(
    text_statistic.flesch_reading_ease(
        "L’assuré désigné doit assumer la franchise relative à la protection applicable. Par contre, si les dommages sont occasionnés par la foudre ou l’incendie, la franchise ne s’applique pas."
    )
)

# Simplified text
vocabulary_cardinality = set()
all_words = []
vocabulary_lexical_words_cardinality = set()

average_number_of_tokens = []
average_number_of_lexical_tokens = []

average_num_sentences = []

average_sentence_length = []
average_sen_len_lexical = []

fk_grades = []

partial_compute_stats = partial(compute_stats, key="simplified_text")
insurance_text_simplification.map(partial_compute_stats)

print(" -- Simplified Statistics -- ")
print(f"The len of the dataset is {len(insurance_text_simplification['train'])}")
print(f"The vocab size is {len(vocabulary_cardinality)}")
print(f"The avg number of tokens is {mean(average_number_of_tokens)}")
print(f"The avg number of lexical tokens is {mean(average_number_of_lexical_tokens)}")
print(f"The average number of sentence is {mean(average_num_sentences)}")
print(f"The average len sentence is {mean(average_sentence_length)}")
print(f"The average len sentence lexical is {mean(average_sen_len_lexical)}")
print(
    f"The lexical richness is {round(len(vocabulary_cardinality) / len(all_words), 2)}"
)
print(f"The FK is {mean(fk_grades)}")

##### For CLEAR
print("------------------ CLEAR-----------------------------")
clear = load_dataset(
    os.path.join(root, "datastore", "corpus_coling"),
    data_files=["clear-train-src.txt", "clear-valid-src.txt", "clear-test-src.txt"],
)

vocabulary_cardinality = set()
all_words = []
vocabulary_lexical_words_cardinality = set()

average_number_of_tokens = []
average_number_of_lexical_tokens = []

average_num_sentences = []

average_sentence_length = []
average_sen_len_lexical = []

fk_grades = []

partial_compute_stats = partial(compute_stats, key="text")
clear.map(partial_compute_stats)

print(" -- Source Statistics -- ")
print(f"The len of the dataset is {len(clear['train'])}")
print(f"The vocab size is {len(vocabulary_cardinality)}")
print(f"The avg number of tokens is {mean(average_number_of_tokens)}")
print(f"The avg number of lexical tokens is {mean(average_number_of_lexical_tokens)}")
print(f"The average number of sentence is {mean(average_num_sentences)}")
print(f"The average len sentence is {mean(average_sentence_length)}")
print(f"The average len sentence lexical is {mean(average_sen_len_lexical)}")
print(
    f"The lexical richness is {round(len(vocabulary_cardinality) / len(all_words), 2)}"
)
print(f"The FK is {mean(fk_grades)}")

# Simplified text
clear = load_dataset(
    os.path.join(root, "datastore", "corpus_coling"),
    data_files=["clear-train-dst.txt", "clear-valid-dst.txt", "clear-test-dst.txt"],
)

vocabulary_cardinality = set()
all_words = []
vocabulary_lexical_words_cardinality = set()

average_number_of_tokens = []
average_number_of_lexical_tokens = []

average_num_sentences = []

average_sentence_length = []
average_sen_len_lexical = []

fk_grades = []

partial_compute_stats = partial(compute_stats, key="text")
clear.map(partial_compute_stats)

print(" -- Simplified Statistics -- ")
print(f"The len of the dataset is {len(clear['train'])}")
print(f"The vocab size is {len(vocabulary_cardinality)}")
print(f"The avg number of tokens is {mean(average_number_of_tokens)}")
print(f"The avg number of lexical tokens is {mean(average_number_of_lexical_tokens)}")
print(f"The average number of sentence is {mean(average_num_sentences)}")
print(f"The average len sentence is {mean(average_sentence_length)}")
print(f"The average len sentence lexical is {mean(average_sen_len_lexical)}")
print(
    f"The lexical richness is {round(len(vocabulary_cardinality) / len(all_words), 2)}"
)
print(f"The FK is {mean(fk_grades)}")

##### For WIKIFR
print("------------------ WIKIFR-----------------------------")
wikifr = load_dataset(
    os.path.join(root, "datastore", "corpus_coling"),
    data_files=[
        "FR-wiki-full-aner-ori-train-src.txt",
        "FR-wiki-full-aner-ori-valid-src.txt",
        "FR-wiki-full-aner-ori-test-src.txt",
    ],
)

vocabulary_cardinality = set()
all_words = []
vocabulary_lexical_words_cardinality = set()

average_number_of_tokens = []
average_number_of_lexical_tokens = []

average_num_sentences = []

average_sentence_length = []
average_sen_len_lexical = []

fk_grades = []

partial_compute_stats = partial(compute_stats, key="text")
wikifr.map(partial_compute_stats)

print(" -- Source Statistics -- ")
print(f"The len of the dataset is {len(wikifr['train'])}")
print(f"The vocab size is {len(vocabulary_cardinality)}")
print(f"The avg number of tokens is {mean(average_number_of_tokens)}")
print(f"The avg number of lexical tokens is {mean(average_number_of_lexical_tokens)}")
print(f"The average number of sentence is {mean(average_num_sentences)}")
print(f"The average len sentence is {mean(average_sentence_length)}")
print(f"The average len sentence lexical is {mean(average_sen_len_lexical)}")
print(
    f"The lexical richness is {round(len(vocabulary_cardinality) / len(all_words), 2)}"
)
print(f"The FK is {mean(fk_grades)}")

# Simplified text
wikifr = load_dataset(
    os.path.join(root, "datastore", "corpus_coling"),
    data_files=[
        "FR-wiki-full-aner-ori-train-dst.txt",
        "FR-wiki-full-aner-ori-valid-dst.txt",
        "FR-wiki-full-aner-ori-test-dst.txt",
    ],
)

vocabulary_cardinality = set()
all_words = []
vocabulary_lexical_words_cardinality = set()

average_number_of_tokens = []
average_number_of_lexical_tokens = []

average_num_sentences = []

average_sentence_length = []
average_sen_len_lexical = []

fk_grades = []

partial_compute_stats = partial(compute_stats, key="text")
wikifr.map(partial_compute_stats)

print(" -- Simplified Statistics -- ")
print(f"The len of the dataset is {len(wikifr['train'])}")
print(f"The vocab size is {len(vocabulary_cardinality)}")
print(f"The avg number of tokens is {mean(average_number_of_tokens)}")
print(f"The avg number of lexical tokens is {mean(average_number_of_lexical_tokens)}")
print(f"The average number of sentence is {mean(average_num_sentences)}")
print(f"The average len sentence is {mean(average_sentence_length)}")
print(f"The average len sentence lexical is {mean(average_sen_len_lexical)}")
print(
    f"The lexical richness is {round(len(vocabulary_cardinality) / len(all_words), 2)}"
)
print(f"The FK is {mean(fk_grades)}")
