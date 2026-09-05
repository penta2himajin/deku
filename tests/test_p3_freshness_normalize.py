"""P3: office freshness, URL temporal scope, English-only (no JA bridge)."""

from __future__ import annotations

import unittest

from deku import normalize as nz
from deku import refuse as rf
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


class TestJapaneseRejected(unittest.TestCase):
    def test_detect_japanese(self):
        self.assertTrue(nz.looks_japanese("日本の首都はどこですか？"))
        self.assertFalse(nz.looks_japanese("What is the capital of Japan?"))

    def test_prepare_does_not_translate(self):
        q, detail = nz.prepare_question("日本の首都はどこですか？")
        self.assertEqual(q, "日本の首都はどこですか？")
        self.assertNotIn("normalized_from", detail)

    def test_japanese_is_hard_refuse(self):
        self.assertTrue(rf.is_hard_refuse("日本の首都はどこですか？"))
        self.assertEqual(rf.classify("日本の首都はどこですか？"), "non_english")
        self.assertEqual(
            rf.message("non_english", audience="agent"), "refused:non_english"
        )

    def test_japanese_routes_to_refuse(self):
        got = rt.rule_route("日本の首都はどこですか？")
        self.assertEqual(got.tool, "refuse")
        self.assertEqual(got.detail.get("reason"), "non_english")
        dispatched = rt.dispatch(
            "AppleのCEOは誰ですか？", router="rule", audience="agent"
        )
        self.assertEqual(dispatched.status, "refused")
        self.assertEqual(dispatched.detail.get("reason"), "non_english")
        self.assertEqual(dispatched.answer, "refused:non_english")


if __name__ == "__main__":
    unittest.main()
