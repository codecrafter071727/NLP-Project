"""Checks for the RQ3 trait measurements and reliability rule."""

import unittest

from rq3_traits import holdout_ok, is_reliable, trait_values


class TraitValueTest(unittest.TestCase):
    def test_identical_text_is_fully_kept(self):
        text = "Le contrat inclut le chapitre A."
        values = trait_values({"original": text, "simplification": text, "label": 10})
        self.assertAlmostEqual(values["overlap"], 1.0)
        self.assertAlmostEqual(values["words_kept"], 1.0)
        self.assertAlmostEqual(values["words_from_original"], 1.0)
        self.assertAlmostEqual(values["length_closeness"], 1.0)
        self.assertAlmostEqual(values["length_change"], 1.0)
        self.assertAlmostEqual(values["sentence_split"], 1.0)
        self.assertAlmostEqual(values["shorter_words"], 0.0)

    def test_new_words_lower_the_faithfulness_traits(self):
        values = trait_values(
            {
                "original": "Le contrat inclut le chapitre A.",
                "simplification": "Bonjour le monde ailleurs.",
                "label": 1,
            }
        )
        self.assertLess(values["overlap"], 0.5)
        self.assertLess(values["words_kept"], 0.5)
        self.assertLess(values["words_from_original"], 0.5)


class ReliabilityRuleTest(unittest.TestCase):
    def test_same_sign_significant_and_correct_holdout_is_reliable(self):
        self.assertTrue(is_reliable(0.4, 0.3, 0.01, 0.9, 0.2, "higher"))

    def test_opposite_sign_is_not_reliable(self):
        self.assertFalse(is_reliable(0.4, -0.3, 0.01, 0.9, 0.2, "higher"))

    def test_non_significant_test_is_not_reliable(self):
        self.assertFalse(is_reliable(0.4, 0.3, 0.2, 0.9, 0.2, "higher"))

    def test_wrong_holdout_direction_is_not_reliable(self):
        self.assertFalse(is_reliable(0.4, 0.3, 0.01, 0.2, 0.9, "higher"))

    def test_closer_to_one_accepts_the_identical_side(self):
        self.assertTrue(holdout_ok("closer_to_one", 1.0, 0.4))
        self.assertFalse(holdout_ok("closer_to_one", 0.2, 0.9))


if __name__ == "__main__":
    unittest.main()
