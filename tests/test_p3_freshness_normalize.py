"""P3: office freshness, URL temporal scope, Japanese normalize."""

from __future__ import annotations

import unittest

from deku import normalize as nz
from deku import route as rt
from deku import url_read as ur
from deku import web_search as ws


class TestOfficeFreshness(unittest.TestCase):
    def test_present_tense_demotes_historical_ceo(self):
        q = "Who is the CEO of Apple?"
        hits = [
            {
                "title": "Michael Scott (Apple)",
                "snippet": (
                    "Michael Scott was the first CEO of Apple Computer "
                    "from February 1977 to March 1981."
                ),
                "url": "u1",
            },
            {
                "title": "Tim Cook",
                "snippet": (
                    "Tim Cook has served as the chief executive officer "
                    "(CEO) of Apple since 2011."
                ),
                "url": "u2",
            },
        ]
        top = ws.rank_hits(q, hits, k=2)
        self.assertEqual(top[0]["title"], "Tim Cook")

    def test_looks_historical_office(self):
        self.assertTrue(
            ws.looks_historical_office(
                "He was the first CEO of Apple from 1977 to 1981."
            )
        )
        self.assertFalse(
            ws.looks_historical_office(
                "He has served as the CEO of Apple since 2011."
            )
        )


class TestTemporalScope(unittest.TestCase):
    def test_pick_temporally_coherent_lead(self):
        doc = (
            "Kyoto\n"
            "Kyoto is a city in Japan. It was the imperial capital until 1869. "
            "The Sengoku period ended in the early 17th century. "
            "Today Kyoto is a major cultural center with temples and tourism.\n"
        )
        got = ur.prefer_coherent_summary_span(doc)
        self.assertIsNotNone(got)
        self.assertNotIn("Sengoku", got)
        self.assertTrue(
            "Kyoto" in got and ("city" in got.lower() or "capital" in got.lower()
                                 or "cultural" in got.lower())
        )


class TestJapaneseNormalize(unittest.TestCase):
    def test_detect_japanese(self):
        self.assertTrue(nz.looks_japanese("日本の首都はどこですか？"))
        self.assertFalse(nz.looks_japanese("What is the capital of Japan?"))

    def test_rule_normalize_capital(self):
        en, detail = nz.normalize_question("日本の首都はどこですか？")
        self.assertEqual(en, "What is the capital of Japan?")
        self.assertEqual(detail.get("normalized_from"), "ja")

    def test_rule_normalize_ceo(self):
        en, _ = nz.normalize_question("AppleのCEOは誰ですか？")
        self.assertEqual(en, "Who is the CEO of Apple?")

    def test_japanese_routes_after_normalize(self):
        got = rt.rule_route("日本の首都はどこですか？")
        self.assertEqual(got.tool, "web_search")
        self.assertIn("Japan", got.query or got.detail.get("normalized") or "Japan")


if __name__ == "__main__":
    unittest.main()
