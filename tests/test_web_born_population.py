"""Regression tests for born / population fact shapes (live failure modes)."""

from __future__ import annotations

import unittest
from unittest import mock

from deku import web_search as ws


class TestCoreFitsBornPopulation(unittest.TestCase):
    def test_rejects_person_fragment_for_birthplace(self):
        self.assertFalse(
            ws.core_fits_question("Where was Tim Cook born?", "Tim")
        )
        self.assertFalse(
            ws.core_fits_question("Where was Tim Cook born?", "The")
        )
        self.assertFalse(
            ws.core_fits_question("Where was Tim Cook born?", "May 1975")
        )
        self.assertTrue(
            ws.core_fits_question("Where was Tim Cook born?", "Mobile, Alabama")
        )

    def test_rejects_article_for_population(self):
        self.assertFalse(
            ws.core_fits_question("What is the population of Tokyo?", "The")
        )
        self.assertTrue(
            ws.core_fits_question("What is the population of Tokyo?", "14 million")
        )


class TestFactCoreLiveShapes(unittest.TestCase):
    def test_population_city_proper_over(self):
        doc = (
            "Tokyo\n"
            "Tokyo, officially the Tokyo Metropolis, is the capital and most "
            "populous city of Japan. The population of the city proper was "
            "over 14 million as of 2023.\n"
        )
        self.assertEqual(
            ws.fact_core_from_doc("What is the population of Tokyo?", doc),
            "14 million",
        )

    def test_born_in_place(self):
        doc = (
            "Tim Cook\n"
            "Cook was born in Mobile, Alabama, and grew up in nearby Robertsdale.\n"
        )
        self.assertEqual(
            ws.fact_core_from_doc("Where was Tim Cook born?", doc),
            "Mobile, Alabama",
        )

    def test_born_on_date_in_city_of_place(self):
        # Common Wikipedia bio shape (lead omits place; early-life has this).
        doc = (
            "Tim Cook\n"
            "Timothy Donald Cook was born on November 1, 1960, in the city of "
            "Mobile, Alabama.\n"
        )
        self.assertEqual(
            ws.fact_core_from_doc("Where was Tim Cook born?", doc),
            "Mobile, Alabama",
        )


class TestRankBornDisambiguation(unittest.TestCase):
    def test_penalizes_disambiguation(self):
        hits = [
            {
                "title": "Tim Cook (disambiguation)",
                "snippet": "Tim Cook (born 1960) is an American business executive",
                "url": "https://en.wikipedia.org/wiki/Tim_Cook_(disambiguation)",
            },
            {
                "title": "Tim Cook",
                "snippet": (
                    "Timothy Donald Cook (born November 1, 1960) is CEO of Apple. "
                    "Cook was born in Mobile, Alabama."
                ),
                "url": "https://en.wikipedia.org/wiki/Tim_Cook",
            },
        ]
        ranked = ws.rank_hits("Where was Tim Cook born?", hits, k=2)
        self.assertEqual(ranked[0]["title"], "Tim Cook")


class TestHitsPackPopulation(unittest.TestCase):
    def test_keeps_population_sentence(self):
        hits = [
            {
                "title": "Tokyo",
                "snippet": (
                    "Tokyo, officially the Tokyo Metropolis, is the capital and "
                    "most populous city of Japan. The population of the city "
                    "proper was over 14 million as of 2023. The Greater Tokyo "
                    "Area is large."
                ),
                "url": "https://en.wikipedia.org/wiki/Tokyo",
            }
        ]
        doc = ws.hits_to_document(hits, question="What is the population of Tokyo?")
        self.assertIn("14 million", doc)


class TestWikiBirthPlaceParse(unittest.TestCase):
    def test_wiki_birth_place_dig_gone(self):
        self.assertFalse(hasattr(ws, "wiki_birth_place"))

    def test_enrich_uses_full_extract_for_born_where(self):
        # Lead-only extract lacks place; full page extract has early-life sentence.
        hits = [
            {
                "title": "Tim Cook",
                "snippet": (
                    "Timothy Donald Cook (born November 1, 1960) is CEO of Apple."
                ),
                "url": "https://en.wikipedia.org/wiki/Tim_Cook",
            }
        ]
        full = (
            "Timothy Donald Cook (born November 1, 1960) is CEO of Apple.\n"
            "== Early life ==\n"
            "Timothy Donald Cook was born on November 1, 1960, in the city of "
            "Mobile, Alabama.\n"
        )
        with mock.patch.object(ws, "wiki_page_extract", return_value=full):
            out = ws.enrich_hits_for_answer("Where was Tim Cook born?", hits)
        self.assertIn("Mobile", out[0].get("snippet") or "")
        self.assertEqual(out[0].get("enriched"), "wiki_summary")


if __name__ == "__main__":
    unittest.main()
