"""Checks for the RQ2 training and scoring calculations."""

import unittest

import numpy as np

from rq2_metrics import fit_ridge, load_split, parse_score, predict_ridge, regression_metrics
from rq2_prompt_ratings import SHORT_EN, SHORT_FR, SHORT_HI


class ParseScoreTest(unittest.TestCase):
    def test_reads_the_first_integer_inside_the_scale(self):
        self.assertEqual(parse_score("7"), 7)
        self.assertEqual(parse_score("Score: 10"), 10)
        self.assertEqual(parse_score("The rating is 0."), 0)
        self.assertEqual(parse_score("अंक ० है"), 0)
        self.assertEqual(parse_score("१०"), 10)

    def test_rejects_text_without_a_usable_score(self):
        self.assertIsNone(parse_score("Je suis un conseiller."))
        self.assertIsNone(parse_score("The score is 12."))
        self.assertIsNone(parse_score(None))

    def test_hindi_prompt_asks_for_the_same_score_as_french_and_english(self):
        for prompt in (SHORT_FR, SHORT_EN, SHORT_HI):
            self.assertIn("{original}", prompt)
            self.assertIn("{simplification}", prompt)
            self.assertIn("0", prompt)
            self.assertIn("10", prompt)


class MetricsTest(unittest.TestCase):
    def test_perfect_predictions_have_zero_error(self):
        labels = [1, 4, 7, 10]
        metrics = regression_metrics(labels, labels, n_boot=100)
        self.assertEqual(metrics["n"], 4)
        self.assertAlmostEqual(metrics["mae"], 0.0)
        self.assertAlmostEqual(metrics["rmse"], 0.0)
        self.assertAlmostEqual(metrics["pearson"], 1.0)

    def test_constant_prediction_has_no_correlation(self):
        metrics = regression_metrics([1, 5, 9], [4, 4, 4], n_boot=50)
        self.assertEqual(metrics["n"], 3)
        self.assertAlmostEqual(metrics["mae"], 3.0)
        self.assertTrue(np.isnan(metrics["pearson"]))

    def test_constant_labels_have_no_correlation(self):
        metrics = regression_metrics([10, 10, 10], [9, 8, 10], n_boot=50)
        self.assertTrue(np.isnan(metrics["pearson"]))
        self.assertAlmostEqual(metrics["mae"], 1.0)


class RidgeProtocolTest(unittest.TestCase):
    def test_model_fits_on_train_and_scores_every_official_split(self):
        train = load_split("train")
        self.assertEqual(len(train), 186)
        model = fit_ridge(train)
        self.assertEqual(len(model.coef_), 5)
        for name, expected in (
            ("dev", 21),
            ("test", 90),
            ("holdout_identical", 297),
            ("holdout_unrelated", 297),
        ):
            rows = load_split(name)
            self.assertEqual(len(rows), expected)
            predicted = predict_ridge(model, rows)
            self.assertEqual(len(predicted), expected)
            self.assertTrue(np.isfinite(predicted).all())
            self.assertGreaterEqual(predicted.min(), 0)
            self.assertLessEqual(predicted.max(), 10)


if __name__ == "__main__":
    unittest.main()
