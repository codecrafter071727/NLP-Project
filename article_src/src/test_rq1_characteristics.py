"""Checks for the RQ1 source characteristics and reliability rule."""

import unittest

from rq1_characteristics import corrected_alpha, is_reliable, source_characteristics


class SourceCharacteristicTest(unittest.TestCase):
    def test_characteristics_use_only_the_original_sentence(self):
        original = "Le contrat, tres long, inclut le chapitre extraordinaire."
        first = source_characteristics(original)
        second = source_characteristics(original)
        self.assertEqual(first, second)
        self.assertGreater(first["source_words"], 3)
        self.assertGreater(first["source_comma_rate"], 0)
        self.assertGreater(first["source_long_word_share"], 0)

    def test_a_short_sentence_has_fewer_words_than_a_long_one(self):
        short = source_characteristics("Le contrat finit.")
        long = source_characteristics(
            "Le contrat d'assurance automobile comprend plusieurs chapitres et exceptions."
        )
        self.assertLess(short["source_words"], long["source_words"])


class ReliabilityRuleTest(unittest.TestCase):
    def test_threshold_shrinks_as_the_number_of_tests_grows(self):
        self.assertAlmostEqual(corrected_alpha(20), 0.0025)

    def test_same_sign_and_corrected_significance_is_reliable(self):
        self.assertTrue(is_reliable(0.4, 0.3, 0.001, 0.0025))

    def test_a_result_above_the_corrected_threshold_is_not_reliable(self):
        self.assertFalse(is_reliable(0.4, 0.3, 0.01, 0.0025))

    def test_an_opposite_sign_is_not_reliable(self):
        self.assertFalse(is_reliable(0.4, -0.3, 0.001, 0.0025))


if __name__ == "__main__":
    unittest.main()
