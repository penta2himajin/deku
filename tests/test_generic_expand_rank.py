"""Generic typed expand + lexical rank (alongside legacy entity literals)."""

from __future__ import annotations

import unittest

from deku import web_search as ws


class TestGenericExpandShapes(unittest.TestCase):
    def test_who_wrote_adds_role_queries_not_only_authors(self):
        qs = ws.expand_search_queries("Who wrote Zembla?", "wrote Zembla")
        joined = " | ".join(qs).casefold()
        self.assertIn("zembla", joined)
        self.assertIn("zembla author", joined)
        self.assertIn("zembla writer", joined)
        self.assertTrue(
            any("(novel)" in q.casefold() or "(play)" in q.casefold() for q in qs),
            qs,
        )

    def test_who_founded_adds_inc_and_founder_shapes(self):
        qs = ws.expand_search_queries("Who founded AcmeWidgets?", "AcmeWidgets")
        joined = " | ".join(qs).casefold()
        self.assertIn("acmewidgets", joined)
        self.assertIn("acmewidgets founder", joined)
        self.assertTrue(
            any("inc" in q.casefold() for q in qs),
            qs,
        )

    def test_makes_product_adds_manufacturer_shapes(self):
        qs = ws.expand_search_queries(
            "What company makes the WidgetPhone?",
            "WidgetPhone manufacturer",
        )
        joined = " | ".join(qs).casefold()
        self.assertIn("widgetphone", joined)
        self.assertIn("manufacturer", joined)
        self.assertIn("developed by", joined)


class TestLexicalRelationRank(unittest.TestCase):
    def test_founded_prefers_org_page_with_founder_language(self):
        q = "Who founded AcmeWidgets?"
        hits = [
            {
                "title": "Unrelated Person",
                "snippet": "Bill Gates is mentioned in passing.",
                "url": "u1",
            },
            {
                "title": "AcmeWidgets",
                "snippet": "AcmeWidgets was founded by Jane Example in 2001.",
                "url": "u2",
            },
        ]
        top = ws.rank_hits(q, hits, k=1)[0]
        self.assertEqual(top["title"], "AcmeWidgets")

    def test_wrote_prefers_work_page_with_author_language(self):
        q = "Who wrote Zembla?"
        hits = [
            {
                "title": "Zembla (restaurant)",
                "snippet": "Zembla is a hamburger restaurant.",
                "url": "u1",
            },
            {
                "title": "Zembla",
                "snippet": "Zembla is a novel written by Alice Example.",
                "url": "u2",
            },
        ]
        top = ws.rank_hits(q, hits, k=1)[0]
        self.assertEqual(top["title"], "Zembla")

    def test_makes_prefers_developer_attestation(self):
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
