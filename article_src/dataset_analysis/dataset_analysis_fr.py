from collections import Counter
from statistics import mean

import pandas as pd
import seaborn as sns
import spacy
from datasets import load_dataset
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from textstat.textstat import textstatistics

n_cores = 4

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


def compute_stats(data_row):
    text = data_row.lower()

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
        and not "new_page" in token.text
    ]
    tokens_len = len(tokens)

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
        and not "new_page" in token.text
        and not token.is_stop
    ]
    lexical_words_len = len(lexical_words)

    vocabulary_set = set(tokens)
    vocabulary_lexical_words_set = set(lexical_words)

    doc = nlp(text)
    sentences = list(doc.sents)
    num_sentences = len(sentences)

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
                    and not "new_page" in token.text
                ]
            )
        )
    average_sen_len = mean(doc_sentence_len)

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
                ]
            )
        )
    average_sen_len_lexical = mean(doc_lexical_sentence_len)

    fk_grade = text_statistic.flesch_reading_ease(text)

    lexical_words_counter = Counter(lexical_words)

    return (
        tokens_len,
        lexical_words_len,
        vocabulary_set,
        vocabulary_lexical_words_set,
        num_sentences,
        average_sen_len,
        average_sen_len_lexical,
        fk_grade,
        lexical_words_counter,
    )


nlp = spacy.load("fr_core_news_sm")

dataset = load_dataset("davebulaval/RISCBAC", "fr")

merge_data = []
for split in dataset:
    for data in dataset.data[split].table["text"]:
        merge_data.append(str(data))

number_of_tokens_per_document = []
number_of_lexical_words_per_document = []

vocabulary = set()
vocabulary_lexical_words = set()

number_of_sentences_per_document = []

average_sentences_len_per_document = []
average_sentence_len_lexical_words_per_document = []

fk_grades = []
overall_lexical_words_counter = Counter()

process_data = list(
    Parallel(n_jobs=n_cores, verbose=2, backend="multiprocessing")(
        delayed(compute_stats)(data_row) for data_row in merge_data
    )
)

for data in process_data:
    number_of_tokens_per_document.append(data[0])
    number_of_lexical_words_per_document.append(data[1])

    vocabulary.update(data[2])
    vocabulary_lexical_words.update(data[3])

    number_of_sentences_per_document.append(data[4])

    average_sentences_len_per_document.append(data[5])
    average_sentence_len_lexical_words_per_document.append(data[6])

    fk_grades.append(data[7])

    overall_lexical_words_counter.update(data[8])

del process_data

average_number_of_tokens = mean(number_of_tokens_per_document)
average_number_of_lexical_words = mean(number_of_lexical_words_per_document)
vocabulary_size = len(vocabulary)
vocabulary_size_lexical_words = len(vocabulary_lexical_words)
average_number_of_sentences = mean(number_of_sentences_per_document)
average_sentences_len = mean(average_sentences_len_per_document)
average_sentence_len_lexical_words = mean(
    average_sentence_len_lexical_words_per_document
)
lexical_diversity = vocabulary_size_lexical_words / sum(
    number_of_lexical_words_per_document
)
average_fk_grades = mean(fk_grades)

with open("results.txt", "w") as file:
    print(
        f"The average number of tokens in the dataset is : {average_number_of_tokens}"
    )
    print(
        f"The average number of lexical words tokens in the dataset is : {average_number_of_lexical_words}"
    )

    print(f"The number of vocabulary size is : {vocabulary_size}")
    print(
        f"The number of vocabulary size of lexical words is : {vocabulary_size_lexical_words}"
    )

    print(f"The average number of sentences is : {average_number_of_sentences}")
    print(f"The average sentence len is: {average_sentences_len}")
    print(
        f"The average sentence len lexical words is: {average_sentence_len_lexical_words}"
    )
    print(f"The lexical richness is: {lexical_diversity}")
    print(f"The average FK score: {average_fk_grades}")
    print(
        f"The top 50 words frequencies are: {overall_lexical_words_counter.most_common(50)}"
    )

    print(
        f"The average number of tokens in the dataset is : {average_number_of_tokens}",
        file=file,
    )
    print(
        f"The average number of lexical words tokens in the dataset is : {average_number_of_lexical_words}",
        file=file,
    )

    print(f"The number of vocabulary size is : {vocabulary_size}", file=file)
    print(
        f"The number of vocabulary size of lexical words is : {vocabulary_size_lexical_words}",
        file=file,
    )

    print(
        f"The average number of sentences is : {average_number_of_sentences}", file=file
    )
    print(f"The average sentence len is: {average_sentences_len}", file=file)
    print(
        f"The average sentence len lexical words is: {average_sentence_len_lexical_words}",
        file=file,
    )
    print(f"The lexical richness is: {lexical_diversity}", file=file)
    print(f"The average FK score: {average_fk_grades}", file=file)
    print(
        f"The top 50 words frequencies are: {overall_lexical_words_counter.most_common(50)}",
        file=file,
    )

data = pd.DataFrame.from_dict(
    {
        "Words": list(zip(*overall_lexical_words_counter.most_common(50)))[0],
        "Values": list(zip(*overall_lexical_words_counter.most_common(50)))[1],
    }
)
plt.figure(figsize=(12, 8))
chart = sns.lineplot(data=data, x="Words", y="Values")
plt.xticks(rotation=45, ha="right")
plt.xlabel("")
plt.ylabel("")
plt.savefig("top_n_values.png", bbox_inches="tight")
plt.show()
