"""Minimal expand + generic lexical rerank (no shape templates)."""

from __future__ import annotations

import unittest

from deku import web_search as ws


class TestMinimalExpand(unittest.TestCase):
    def test_no_shape_or_entity_injection(self):
        qs = ws.expand_search_queries(
            "Who wrote Hamlet?", "Who wrote Hamlet?"
        )
        self.assertIn("Who wrote Hamlet?", qs)
        joined = " | ".join(qs).casefold()
        self.assertNotIn("shakespeare", joined)
        self.assertNotIn("orwell", joined)
        self.assertNotIn("author", joined)

    def test_includes_wiki_friendly_query(self):
        qs = ws.expand_search_queries(
            "Who is the CEO of Apple?", "CEO Apple"
        )
        self.assertTrue(len(qs) >= 2)


class TestGenericLexicalRank(unittest.TestCase):
    def test_topic_match_beats_near_miss(self):
        q = "What is the capital of Peru?"
        hits = [
            {
                "title": "Perugia",
                "snippet": "Perugia is a city in Italy.",
                "url": "u1",
            },
            {
                "title": "Lima",
                "snippet": "Lima is the capital of Peru.",
                "url": "u2",
            },
        ]
        top = ws.rank_hits(q, hits, k=1)[0]
        self.assertEqual(top["title"], "Lima")

    def test_ceo_prefers_officer_attestation_over_bare_company(self):
        q = "Who is the CEO of Apple?"
        hits = [
            {
                "title": "Apple Inc.",
                "snippet": "Apple Inc. is an American multinational.",
                "url": "u1",
            },
            {
                "title": "Example Officer",
                "snippet": (
                    "Example Officer is the chief executive officer of Apple."
                ),
                "url": "u2",
            },
        ]
        top = ws.rank_hits(q, hits, k=1)[0]
        self.assertEqual(top["title"], "Example Officer")

    def test_maker_page_beats_bare_product_title(self):
        q = "What company makes the WidgetPhone?"
        hits = [
            {
                "title": "WidgetPhone",
                "snippet": "WidgetPhone is a smartphone.",
                "url": "u1",
            },
            {
                "title": "Globex",
                "snippet": "The WidgetPhone was developed by Globex.",
                "url": "u2",
            },
        ]
        top = ws.rank_hits(q, hits, k=1)[0]
        self.assertEqual(top["title"], "Globex")


if __name__ == "__main__":
    unittest.main()
