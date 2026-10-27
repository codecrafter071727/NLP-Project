# JUDGEBERT: Assessing Legal Meaning Preservation Between Sentences

This repository contains the official JUDGEBERT: Assessing Legal Meaning Preservation Between Sentences dataset (see
datastore directory) and the source code used to generate the tables and training of the model illustrated
in [JUDGEBERT: Assessing Legal Meaning Preservation Between Sentences]().

## About the Dataset

### Example of Sentences in the Dataset

![img.png](figs/fig1.png)


### Statistics About the Dataset Compared to Other Similar Datasets
![img_1.png](figs/fig2.png)

### Dataset Structure

#### Data Fields

- `label`: the binary label of the sentence, where `0` means ungrammatical and `1` means grammatical.
- `sentence`: the sentence.
- `source`: the URL source of the sentence.
- `category`: the aggregated BDL category of the sentence linguistic phenomena.

### Download the Dataset

You can manually download our dataset splits available in `dataset`, or you can use the HuggingFace dataset class as
follows:

```python
from datasets import load_dataset

dataset = load_dataset("json", data_dir="", data_files=["FrJUDGE.jsonl"])
```

### License

This dataset is under [CC-BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en).

### To Cite

```
@inproceedings{beauchemin2025judgebert,
  title={JUDGEBERT: Assessing Legal Meaning Preservation Between Sentences},
  author={Beauchemin, David and Albert-Rochette, Michelle and Khoury, Richard and D{\'e}ziel, Pierre-Luc},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  pages={92--118},
  year={2025}
}
```

## About the Source Code

In the directory `article_src`, you can find the source code used to clean the dataset, compute the statistics and train the models.

## Continuation: RQ1, RQ2, and RQ3

These questions read `datastore/frjudge.csv`. That table has one row per sentence pair, with the columns `split`, `original`, `simplification`, and `label`. The overlap features come from `article_src/src/overlap_baseline.py`. The long French and long Hindi judge prompts come from `article_src/src/evaluate_tools.py`, beside the English prompt already defined there. Run all three from the repo root:

```text
python article_src/src/research_questions.py
```

The combined result is written to `article_src/src/predictions/research_questions.txt`.

The same command also reads `datastore/kaggle_legal_simplification.csv`. That table is the Kaggle dataset `karthikrv1107/adaption-legal-clause-simplification` (MIT): English legal and policy clauses with plain-language restatements. It is built from the public Hugging Face copy of that dataset. Records with no source clause are left out. The table has no human rating from 0 to 10, so the ridge model and the judge prompts stay on `datastore/frjudge.csv`. The Kaggle rows get the same source-characteristic check as RQ1, on a fixed 80/20 split.

`datastore/hindi_simplification.csv` is the public Hindi simplification dataset `raja20221020/hindi-text-simplification-for-finetuning`: 12,000 Hindi source sentences and simpler Hindi sentences. The dataset card lists no human rating from 0 to 10, so the ridge model and the judge prompts stay on `datastore/frjudge.csv`. The Hindi rows get the same source-characteristic check, on a fixed 80/20 split. The Hindi judge prompts are scored on the first 90 Hindi test rows, not on the French sentences. Word counts include Devanagari, and sentence counts include the Hindi danda.
