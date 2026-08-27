"""Generic when/founded/released/birthday rank + core (no entity tables)."""

from __future__ import annotations

import unittest
from unittest import mock

from deku import web_search as ws


class TestWhenFoundedRank(unittest.TestCase):
    def test_exact_org_title_beats_foundation_compound(self):
        q = "When was AcmeCorp founded?"
        hits = [
            {
                "title": "AcmeCorp Foundation Class Library",
                "snippet": "AcmeCorp Foundation Class Library is a toolkit.",
                "url": "u1",
            },
            {
                "title": "AcmeCorp",
                "snippet": "AcmeCorp is an American multinational technology company.",
                "url": "u2",
            },
            {
                "title": "AcmeCorp Foundry",
                "snippet": "AcmeCorp Foundry is a cloud product.",
                "url": "u3",
            },
        ]
        top = ws.rank_hits(q, hits, k=1)[0]
        self.assertEqual(top["title"], "AcmeCorp")


class TestWhenReleasedTopicAndCore(unittest.TestCase):
    def test_question_topic_for_released(self):
        self.assertEqual(
            ws.question_topic("When was the WidgetPhone released?"),
            "WidgetPhone",
        )

    def test_history_page_with_unveil_year_ranks_above_noise(self):
        q = "When was the WidgetPhone released?"
        hits = [
            {
                "title": "WidgetPhone 16",
                "snippet": "The WidgetPhone 16 is a recent model.",
                "url": "u1",
            },
            {
                "title": "History of the WidgetPhone",
                "snippet": (
                    "The first WidgetPhone was unveiled at Expo 2007 "
                    "and released later that year."
                ),
                "url": "u2",
            },
        ]
        top = ws.rank_hits(q, hits, k=1)[0]
        self.assertEqual(top["title"], "History of the WidgetPhone")

    def test_fact_core_unveiled_year(self):
        doc = (
            "History of the WidgetPhone\n"
            "The first WidgetPhone was unveiled at Expo 2007 and "
            "released later that year.\n"
        )
        self.assertEqual(
            ws.fact_core_from_doc("When was the WidgetPhone released?", doc),
            "2007",
        )


class TestBirthdayOfficeIncumbentPick(unittest.TestCase):
    def test_given_name_page_not_preferred_over_exact_person(self):
        q = "What is the birthday of the current Emperor of Japan?"
        hits = [
            {
                "title": "ExampleName (given name)",
                "snippet": "ExampleName is a masculine given name.",
                "url": "u1",
            },
            {
                "title": "ExampleName",
                "snippet": "ExampleName (born 23 February 1960) is Emperor of Japan.",
                "url": "u2",
            },
            {
                "title": "Emperor of Japan",
                "snippet": "The Emperor of Japan is the hereditary monarch.",
                "url": "u3",
            },
        ]
        top = ws.rank_hits(q, hits, k=1)[0]
        self.assertEqual(top["title"], "ExampleName")

    def test_birthday_from_person_hit_without_office_dig(self):
        # Office digs removed: birthday must come from ranked person page + extract.
        q = "What is the birthday of the current Emperor of Japan?"
        hits = [
            {
                "title": "ExampleName (given name)",
                "snippet": "ExampleName is a masculine given name.",
                "url": "https://en.wikipedia.org/wiki/ExampleName_(given_name)",
            },
            {
                "title": "ExampleName",
                "snippet": (
                    "ExampleName (born 23 February 1960) is the Emperor of Japan. "
                    "He acceded to the Chrysanthemum Throne in 2019."
                ),
                "url": "https://en.wikipedia.org/wiki/ExampleName",
            },
        ]
        with mock.patch.object(
            ws, "search", return_value=hits
        ), mock.patch.object(
            ws, "enrich_hits_for_answer", side_effect=lambda q, hs, **k: hs
        ), mock.patch.object(
            ws, "minicpm_extract", return_value=("23 February 1960", "ok")
        ), mock.patch.object(
            ws, "minicpm_summarize", return_value=""
        ):
            out = ws.run(q)
        self.assertFalse(out.detail.get("incumbent_fetched"))
        self.assertIn("1960", out.document or "")
        self.assertEqual(out.status, "ok")
        self.assertRegex(out.answer or "", r"1960|February")


class TestEmperorOfficeDigsRemoved(unittest.TestCase):
    def test_office_page_title_gone(self):
        self.assertFalse(hasattr(ws, "office_page_title"))

    def test_current_emperor_birthday_prefers_incumbent_bio(self):
        q = "What is the birthday of the current Emperor of Japan?"
        hits = [
            {
                "title": "Empress Michiko",
                "snippet": (
                    "Michiko Shōda was born on 20 October 1934. "
                    "She is the Empress Emerita of Japan."
                ),
                "url": "u1",
            },
            {
                "title": "Naruhito",
                "snippet": (
                    "Naruhito (born 23 February 1960) is Emperor of Japan. "
                    "He acceded to the Chrysanthemum Throne in 2019."
                ),
                "url": "u2",
            },
        ]
        top = ws.rank_hits(q, hits, k=1)[0]
        self.assertEqual(top["title"], "Naruhito")


if __name__ == "__main__":
    unittest.main()
