"""Holes found in P7 live composition probes."""

from __future__ import annotations

import unittest

from deku import multi_hop as mh
from deku import orchestrate as orch
from deku import route as rt
from deku import web_search as ws


class TestOpinionJoinRefuse(unittest.TestCase):
    def test_should_buy_stock_refuses(self):
        q = "Who is the CEO of Apple and should I buy the stock?"
        self.assertGreaterEqual(len(mh.decompose(q)), 2)
        self.assertIsNone(orch.select_and_build(q))
        self.assertEqual(rt.rule_route(q).tool, "refuse")


class TestCeoSurnameGrounding(unittest.TestCase):
    def test_ceo_compose_from_bio_without_ceo_word(self):
        doc = (
            "Sundar Pichai\n"
            "Pichai joined Google in 2004, where he led product management.\n"
        )
        reply = ws.compose_reply(
            "Sundar Pichai",
            "The CEO of Google is Sundar Pichai.",
            doc,
            question="Who is the CEO of Google?",
        )
        self.assertIsNotNone(reply)
        self.assertIn("Sundar Pichai", reply)


class TestFounderSingular(unittest.TestCase):
    def test_wiki_founders_dig_gone(self):
        self.assertFalse(hasattr(ws, "wiki_founders"))


class TestCapitalCoreBleed(unittest.TestCase):
    def test_capital_core_does_not_bleed_next_sentence(self):
        doc = (
            "Capital of Japan\n"
            "The capital of Japan is Tokyo. Throughout history, the "
            "national capital of Japan has been in locations other than Tokyo.\n"
        )
        self.assertEqual(
            ws.fact_core_from_doc("What is the capital of Japan?", doc),
            "Tokyo",
        )
        self.assertTrue(
            ws.is_degenerate_core(
                "Tokyo. Throughout", "What is the capital of Japan?"
            )
        )


class TestCapitalRanking(unittest.TestCase):
    def test_capital_of_japan_prefers_capital_page(self):
        q = "What is the capital of Japan?"
        hits = [
            {
                "title": "Japan",
                "snippet": "Japan is an island country in East Asia.",
                "url": "u1",
                "path": "",
            },
            {
                "title": "Capital of Japan",
                "snippet": "Tokyo is the capital of Japan and the largest city.",
                "url": "u2",
                "path": "",
            },
            {
                "title": "Capital punishment in Japan",
                "snippet": "Capital punishment is a legal penalty in Japan.",
                "url": "u3",
                "path": "",
            },
        ]
        top = ws.rank_hits(q, hits, k=1)[0]
        self.assertEqual(top["title"], "Capital of Japan")


if __name__ == "__main__":
    unittest.main()
