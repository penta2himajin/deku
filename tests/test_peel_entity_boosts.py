"""Peel closed entity boosts (KNOWN_CAPITALS, named incumbents, CEO names)."""

from __future__ import annotations

import unittest
from unittest import mock

from deku import web_search as ws


class TestNoKnownCapitalsTable(unittest.TestCase):
    def test_known_capitals_empty_or_gone(self):
        # Product spine must not rely on a closed country→city map.
        caps = getattr(ws, "KNOWN_CAPITALS", None)
        self.assertTrue(caps is None or caps == {})

    def test_ranks_capital_for_country_not_in_any_map(self):
        q = "What is the capital of Sweden?"
        hits = [
            {
                "title": "Sweden",
                "snippet": "Sweden is a Nordic country in Northern Europe.",
                "url": "u1",
            },
            {
                "title": "Capital of Sweden",
                "snippet": "Stockholm is the capital of Sweden.",
                "url": "u2",
            },
            {
                "title": "Gothenburg",
                "snippet": "Gothenburg is a city in Sweden.",
                "url": "u3",
            },
        ]
        top = ws.rank_hits(q, hits, k=1)[0]
        self.assertEqual(top["title"], "Capital of Sweden")

    def test_ranks_city_page_that_states_capital(self):
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


class TestNoNamedIncumbentBoosts(unittest.TestCase):
    def test_expand_emperor_uses_office_not_hardcoded_name(self):
        qs = ws.expand_search_queries(
            "Who is the current emperor of Japan?",
            "Who is the current emperor of Japan?",
        )
        joined = " | ".join(qs).casefold()
        self.assertIn("current emperor of japan", joined)
        self.assertFalse(
            any(q.strip().casefold() == "naruhito" for q in qs),
            qs,
        )

    def test_ceo_rank_without_named_person_title_boost(self):
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
            {
                "title": "Tim Cook",
                "snippet": "Cook is an American business executive.",
                "url": "u3",
            },
        ]
        top = ws.rank_hits(q, hits, k=1)[0]
        self.assertEqual(top["title"], "Example Officer")

    def test_president_rank_without_macron_name_boost(self):
        q = "Who is the president of France?"
        hits = [
            {
                "title": "Presidency of Charles de Gaulle",
                "snippet": "Charles de Gaulle was president from 1959 to 1969.",
                "url": "u1",
            },
            {
                "title": "Presidency of Alice Example",
                "snippet": (
                    "Alice Example has served as president of France since 2024."
                ),
                "url": "u2",
            },
            {
                "title": "Emmanuel Macron",
                "snippet": "French politician.",
                "url": "u3",
            },
        ]
        top = ws.rank_hits(q, hits, k=1)[0]
        self.assertEqual(top["title"], "Presidency of Alice Example")


class TestEmperorFactCoreGeneric(unittest.TestCase):
    def test_emperor_core_from_title_line_not_literal_naruhito(self):
        doc = (
            "Example Emperor\n"
            "Example Emperor is Emperor of Japan.\n"
        )
        got = ws.fact_core_from_doc(
            "Who is the current emperor of Japan?", doc
        )
        self.assertEqual(got, "Example Emperor")

    def test_emperor_core_from_throne_accession_bio(self):
        doc = (
            "Example Emperor\n"
            "Example Emperor — He acceded to the Chrysanthemum Throne in 2019.\n"
        )
        got = ws.fact_core_from_doc(
            "Who is the current emperor of Japan?", doc
        )
        self.assertEqual(got, "Example Emperor")


if __name__ == "__main__":
    unittest.main()
