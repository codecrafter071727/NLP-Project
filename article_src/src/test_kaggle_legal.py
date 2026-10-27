"""Checks for the Kaggle legal-clause table."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from kaggle_legal import COLUMNS, assign_splits, pairs_from_jsonl, write_table


class SplitAssignmentTest(unittest.TestCase):
    def test_the_holdout_is_fixed(self):
        first = assign_splits(100, test_fraction=0.2, seed=0)
        second = assign_splits(100, test_fraction=0.2, seed=0)
        self.assertEqual(first, second)
        self.assertEqual(first.count("test"), 20)
        self.assertEqual(first.count("train"), 80)

    def test_every_row_is_train_or_test(self):
        splits = assign_splits(10, test_fraction=0.2, seed=1)
        self.assertEqual(set(splits), {"train", "test"})
        self.assertEqual(len(splits), 10)


class TableBuildTest(unittest.TestCase):
    def test_rows_without_a_source_clause_are_left_out(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "data.jsonl"
            source.write_text(
                "\n".join(
                    [
                        json.dumps({"Input": "The insurer may cancel.", "Output": "The insurer can cancel."}),
                        json.dumps({"Input": None, "Output": "A drafted note with no source clause."}),
                        json.dumps(
                            {
                                "Input": 'Payment is due "within 30 days", unless agreed.',
                                "Output": "Payment is due within 30 days.",
                            }
                        ),
                        json.dumps({"Input": "   ", "Output": "Blank source."}),
                    ]
                ),
                encoding="utf-8",
            )
            pairs, skipped = pairs_from_jsonl(source)
            self.assertEqual(skipped, 2)
            self.assertEqual(len(pairs), 2)
            table = write_table(pairs, Path(folder) / "kaggle.csv", test_fraction=0.5, seed=0)
            with open(table, encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(list(rows[0].keys()), list(COLUMNS))
            self.assertEqual(len(rows), 2)
            originals = [row["original"] for row in rows]
            self.assertIn("The insurer may cancel.", originals)
            self.assertTrue(any("within 30 days" in text for text in originals))


if __name__ == "__main__":
    unittest.main()
