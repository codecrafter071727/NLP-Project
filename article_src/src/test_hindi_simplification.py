"""Checks for Hindi tokenization and the Hindi simplification table."""

import unittest

from hindi_simplification import pairs_from_records
from overlap_baseline import tokens
from rq3_traits import SENTENCE_END


class HindiTextTest(unittest.TestCase):
    def test_devanagari_words_are_counted(self):
        words = tokens("इंजनों को तेल की आवश्यकता होती है।")
        self.assertGreaterEqual(len(words), 5)
        self.assertTrue(all(word for word in words))

    def test_french_words_are_still_counted(self):
        words = tokens("Le contrat d'assurance finit.")
        self.assertGreaterEqual(len(words), 4)

    def test_a_danda_ends_a_sentence(self):
        self.assertEqual(len(SENTENCE_END.findall("पहला वाक्य। दूसरा वाक्य।")), 2)


class HindiPairTest(unittest.TestCase):
    def test_blank_rows_are_left_out(self):
        pairs, skipped = pairs_from_records(
            [
                {"input": "तेल की आवश्यकता होती है।", "output": "तेल चाहिए।"},
                {"input": "   ", "output": "खाली।"},
                {"input": "मीडिया अध्ययन।", "output": None},
            ]
        )
        self.assertEqual(skipped, 2)
        self.assertEqual(len(pairs), 1)


if __name__ == "__main__":
    unittest.main()
