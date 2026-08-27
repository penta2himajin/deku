"""P6: person-name gates and cross-entity web pairs (office/Wikidata digs removed)."""

from __future__ import annotations

import unittest

from deku import orchestrate as orch
from deku import route as rt
from deku import web_search as ws


class TestTopicEchoPolity(unittest.TestCase):
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


class TestPersonNameGates(unittest.TestCase):
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
