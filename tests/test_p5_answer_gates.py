"""P5: succession reject, incumbency, answer-type, intent-conditional titles."""

from __future__ import annotations

import unittest

from deku import web_search as ws


class TestPredecessorReject(unittest.TestCase):
    def test_succeeding_clause_is_predecessor(self):
        doc = (
            "Premiership of Shigeru Ishiba\n"
            "Shigeru Ishiba's tenure as prime minister of Japan began on "
            "1 October 2024, succeeding Fumio Kishida."
        )
        self.assertTrue(
            ws.is_predecessor_core("Fumio Kishida", doc, "Who is the prime minister of Japan?")
        )
        self.assertFalse(
            ws.is_predecessor_core("Shigeru Ishiba", doc, "Who is the prime minister of Japan?")
        )

    def test_compose_prefers_summary_over_predecessor_core(self):
        doc = (
            "Premiership of Shigeru Ishiba\n"
            "Shigeru Ishiba's tenure as prime minister of Japan began on "
            "1 October 2024, succeeding Fumio Kishida. "
            "Ishiba has served as the prime minister of Japan since 2024."
        )
        summary = "The prime minister of Japan is Shigeru Ishiba."
        reply = ws.compose_reply(
            "Fumio Kishida",
            summary,
            doc,
            question="Who is the prime minister of Japan?",
        )
        self.assertIsNotNone(reply)
        self.assertIn("Ishiba", reply)
        self.assertNotIn("Kishida", reply)


class TestAnswerTypeGate(unittest.TestCase):
    def test_core_echoes_topic_rejected(self):
        self.assertTrue(
            ws.core_echoes_topic("Who is the CEO of Toyota?", "Toyota")
        )
        self.assertFalse(
            ws.core_echoes_topic("Who is the CEO of Toyota?", "Kenta Kon")
        )

    def test_core_fits_who_requires_person(self):
        self.assertFalse(
            ws.core_fits_question("Who is the CEO of Toyota?", "Toyota")
        )
        self.assertTrue(
            ws.core_fits_question("Who is the CEO of Toyota?", "Kenta Kon")
        )


class TestRoleObjectIntent(unittest.TestCase):
    def test_demote_car_when_asking_who(self):
        self.assertTrue(
            ws.looks_role_object_title(
                "Prime Minister's Official Car (Japan)",
                question="Who is the prime minister of Japan?",
            )
        )

    def test_keep_car_when_asking_about_car(self):
        self.assertFalse(
            ws.looks_role_object_title(
                "Prime Minister's Official Car (Japan)",
                question="What is the official car of the prime minister of Japan?",
            )
        )

    def test_rank_keeps_car_page_for_car_question(self):
        q = "What is the official car of the prime minister of Japan?"
        hits = [
            {
                "title": "Prime Minister's Official Car (Japan)",
                "snippet": "The official state car used by the Prime Minister of Japan.",
                "url": "u1",
            },
            {
                "title": "Premiership of Fumio Kishida",
                "snippet": "Fumio Kishida's tenure as prime minister.",
                "url": "u2",
            },
        ]
        top = ws.rank_hits(q, hits, k=2)
        self.assertEqual(top[0]["title"], "Prime Minister's Official Car (Japan)")


class TestIncumbentDigsRemoved(unittest.TestCase):
    def test_office_and_incumbent_helpers_gone(self):
        self.assertFalse(hasattr(ws, "office_page_title"))
        self.assertFalse(hasattr(ws, "preferred_incumbent_core"))
        self.assertFalse(hasattr(ws, "wiki_incumbent_from_page"))


if __name__ == "__main__":
    unittest.main()
