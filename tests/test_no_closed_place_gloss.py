"""English-only front door: no JA→EN templates or place gloss."""

from __future__ import annotations

import unittest

from deku import normalize as nz
from deku import route as rt


class TestNoClosedPlaceGloss(unittest.TestCase):
    def test_places_table_gone_or_empty(self):
        places = getattr(nz, "_PLACES", None)
        self.assertTrue(places is None or places == {})

    def test_no_ja_normalize_bridge(self):
        self.assertFalse(hasattr(nz, "normalize_question"))
        q, detail = nz.prepare_question("日本の首都はどこですか？")
        self.assertEqual(q, "日本の首都はどこですか？")
        self.assertFalse(detail.get("normalized_from"))

    def test_japanese_refused(self):
        got = rt.rule_route("日本の首都はどこですか？")
        self.assertEqual(got.tool, "refuse")
        self.assertEqual(got.detail.get("reason"), "non_english")


if __name__ == "__main__":
    unittest.main()
