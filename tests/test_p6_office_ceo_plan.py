"""P6: office articles, Wikidata CEO, cross-entity web pairs."""

from __future__ import annotations

import unittest
from unittest import mock

from deku import orchestrate as orch
from deku import route as rt
from deku import web_search as ws


class TestOfficeArticles(unittest.TestCase):
    def test_uk_keeps_the(self):
        self.assertEqual(
            ws.office_page_title("Who is the prime minister of the United Kingdom?"),
            "Prime Minister of the United Kingdom",
        )

    def test_us_keeps_the(self):
        self.assertEqual(
            ws.office_page_title("Who is the president of the United States?"),
            "President of the United States",
        )

    def test_japan_unchanged(self):
        self.assertEqual(
            ws.office_page_title("Who is the prime minister of Japan?"),
            "Prime Minister of Japan",
        )

    def test_topic_echo_matches_polity_without_article(self):
        # Echo gate: core "United Kingdom" vs question "... the United Kingdom"
        self.assertTrue(
            ws.core_echoes_topic(
                "Who is the prime minister of the United Kingdom?",
                "United Kingdom",
            )
        )
        self.assertTrue(
            ws.core_echoes_topic(
                "Who is the president of the United States?",
                "United States",
            )
        )


class TestWikidataCeo(unittest.TestCase):
    def test_parse_ceo_claim(self):
        # Minimal Wikidata entity JSON shape for P169 preferred rank.
        entity = {
            "claims": {
                "P169": [
                    {
                        "rank": "preferred",
                        "mainsnak": {
                            "datavalue": {
                                "value": {"id": "Q42"},
                            }
                        },
                        "qualifiers": {},
                    },
                    {
                        "rank": "normal",
                        "mainsnak": {
                            "datavalue": {
                                "value": {"id": "Q1"},
                            }
                        },
                        "qualifiers": {
                            "P582": [{"datavalue": {"value": {"time": "+2011-08-24T00:00:00Z"}}}]
                        },
                    },
                ]
            }
        }
        self.assertEqual(ws.wikidata_ceo_id_from_entity(entity), "Q42")

    def test_ceo_core_from_wikidata(self):
        with mock.patch.object(ws, "wikidata_search_entity", return_value="Q123"):
            with mock.patch.object(ws, "wikidata_entity", return_value={
                "claims": {
                    "P169": [{
                        "rank": "preferred",
                        "mainsnak": {
                            "datavalue": {"value": {"id": "Q999"}},
                        },
                    }]
                }
            }):
                with mock.patch.object(
                    ws, "wikidata_label", return_value="Kenta Kon"
                ):
                    got = ws.wikidata_ceo_name("Toyota")
        self.assertEqual(got, "Kenta Kon")

    def test_has_person_name_rejects_corp_suffix(self):
        self.assertFalse(
            ws.has_person_name("Sony Interactive Entertainment")
        )
        self.assertFalse(ws.has_person_name("Toyota Motor Corporation"))
        self.assertTrue(ws.has_person_name("Tim Cook"))
        self.assertTrue(ws.has_person_name("Satya Nadella"))

    def test_mononym_fits_who_ceo(self):
        self.assertTrue(
            ws.core_fits_question("Who is the CEO of Google?", "Pichai")
        )


class TestCrossEntityWeb(unittest.TestCase):
    def test_different_entities_still_plan(self):
        plan = orch.select_and_build(
            "Who is the CEO of Sony and where is Apple headquartered?"
        )
        self.assertIsNotNone(plan)
        self.assertEqual(plan.plan_id, "web_independent")
        self.assertEqual(len(plan.steps), 2)

    def test_routes_multi_hop_not_refuse(self):
        got = rt.rule_route(
            "Who is the CEO of Sony and where is Apple headquartered?"
        )
        self.assertEqual(got.tool, "multi_hop")


if __name__ == "__main__":
    unittest.main()
