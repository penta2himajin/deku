"""JA place names pass through without a closed gloss table."""

from __future__ import annotations

import unittest

from deku import normalize as nz


class TestNoClosedPlaceGloss(unittest.TestCase):
    def test_places_table_gone_or_empty(self):
        places = getattr(nz, "_PLACES", None)
        self.assertTrue(places is None or places == {})

    def test_capital_keeps_japanese_place_token(self):
        en, detail = nz.normalize_question("日本の首都はどこですか？")
        self.assertEqual(en, "What is the capital of 日本?")
        self.assertEqual(detail.get("normalized_from"), "ja")


if __name__ == "__main__":
    unittest.main()
