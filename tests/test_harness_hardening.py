"""P0–P2 harness hardening: reply ladder, entity gate, predicates, multi-hop."""

from __future__ import annotations

import unittest

from deku import multi_hop as mh
from deku import orchestrate as orch
from deku import route as rt
from deku import web_search as ws


class TestDegenerateCore(unittest.TestCase):
    def test_rejects_question_echo_and_glue(self):
        q = "What is the capital of Australia?"
        self.assertTrue(ws.is_degenerate_core("the capital", q))
        self.assertTrue(ws.is_degenerate_core("located in", q))
        self.assertTrue(ws.is_degenerate_core("W", q))
        self.assertTrue(ws.is_degenerate_core("capital", q))
        self.assertFalse(ws.is_degenerate_core("Canberra", q))
        self.assertFalse(ws.is_degenerate_core("Redmond, Washington", q))


class TestComposeReplyLadder(unittest.TestCase):
    def test_grounded_summary_wins_over_garbage_core(self):
        q = "What is the capital of Peru?"
        doc = (
            "Lima\n"
            "Lima is the capital of Peru, located in the valleys of the "
            "Chillón, Rímac and Lurín rivers."
        )
        summary = "The capital of Peru is Lima, a city in central Peru."
        got = ws.compose_reply("Perugia", summary, doc, question=q)
        self.assertIsNotNone(got)
        self.assertIn("Lima", got)
        self.assertNotIn("Perugia", got)

    def test_garbage_core_alone_does_not_become_answer(self):
        q = "What is the capital of Australia?"
        doc = (
            "Canberra\n"
            "Canberra is the capital city of Australia."
        )
        got = ws.compose_reply("the capital", None, doc, question=q)
        # Prefer a source sentence with a real place, not the echo core.
        self.assertTrue(got is None or "Canberra" in (got or ""))
        self.assertNotEqual((got or "").strip().lower(), "the capital")


class TestPredicateGrounding(unittest.TestCase):
    def test_founded_requires_founder_language(self):
        q = "Who founded Stripe?"
        doc = (
            "Greg Brockman\n"
            "He began his career at Stripe in 2010 and became CTO in 2013."
        )
        self.assertFalse(
            ws.predicate_supported(q, "Greg Brockman", doc)
        )
        doc2 = (
            "Patrick Collison\n"
            "In 2010, Collison co-founded Stripe with his brother John."
        )
        self.assertTrue(
            ws.predicate_supported(q, "Patrick Collison", doc2)
        )


class TestEntityIdentity(unittest.TestCase):
    def test_peru_prefers_lima_over_perugia(self):
        q = "What is the capital of Peru?"
        hits = [
            {
                "title": "Perugia",
                "snippet": "Perugia is a city in Italy.",
                "url": "https://en.wikipedia.org/wiki/Perugia",
            },
            {
                "title": "Lima",
                "snippet": "Lima is the capital of Peru.",
                "url": "https://en.wikipedia.org/wiki/Lima",
            },
            {
                "title": "Peru",
                "snippet": "The capital of Peru is Lima.",
                "url": "https://en.wikipedia.org/wiki/Peru",
            },
        ]
        top = ws.rank_hits(q, hits, k=2)
        self.assertIn(top[0]["title"], ("Lima", "Peru"))
        self.assertNotEqual(top[0]["title"], "Perugia")
        self.assertFalse(ws.hit_title_matches_topic("Perugia", "Peru"))

    def test_hamlet_prefers_play_over_novel(self):
        q = "Who wrote Hamlet?"
        hits = [
            {
                "title": "The Hamlet (novel)",
                "snippet": "The Hamlet is a novel by William Faulkner.",
                "url": "u1",
            },
            {
                "title": "Hamlet",
                "snippet": "Hamlet is a tragedy written by William Shakespeare.",
                "url": "u2",
            },
        ]
        top = ws.rank_hits(q, hits, k=2)
        self.assertEqual(top[0]["title"], "Hamlet")

    def test_who_founded_prefers_org_title(self):
        q = "Who founded Stripe?"
        hits = [
            {
                "title": "Greg Brockman",
                "snippet": "He began his career at Stripe in 2010.",
                "url": "u1",
            },
            {
                "title": "Stripe, Inc.",
                "snippet": "Stripe was co-founded by Patrick and John Collison.",
                "url": "u2",
            },
        ]
        top = ws.rank_hits(q, hits, k=2)
        self.assertEqual(top[0]["title"], "Stripe, Inc.")


class TestMinHitScore(unittest.TestCase):
    def test_floor_raised(self):
        self.assertGreaterEqual(ws.MIN_HIT_SCORE, 4)


class TestRelatedMultiHop(unittest.TestCase):
    def test_unrelated_dual_web_refused(self):
        q = "Who is the CEO of Apple and what is the boiling point of water?"
        plan = orch.select_and_build(q)
        self.assertIsNone(plan)
        got = rt.rule_route(q)
        self.assertEqual(got.tool, "refuse")

    def test_related_dependent_still_plans(self):
        q = "Who is the CEO of Microsoft and where was he born?"
        plan = orch.select_and_build(q)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.plan_id, "web_dependent")

    def test_capital_population_binds_answer_city(self):
        # After "capital of Kenya" → Nairobi, "its population" should ask Nairobi.
        bind = mh.bind_core(
            "What is the capital of Kenya?",
            "Nairobi",
            "what is its population?",
        )
        self.assertEqual(bind, "Nairobi")
        self.assertEqual(
            mh.rewrite_followup("what is its population?", bind),
            "What is the population of Nairobi?",
        )


class TestCrossToolPlan(unittest.TestCase):
    def test_git_and_web_plan(self):
        q = "What is the last commit message and who is the CEO of Apple?"
        plan = orch.select_and_build(q)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.plan_id, "git_and_web")
        tools = [s.tool for s in plan.steps]
        self.assertEqual(set(tools), {"git_search", "web_search"})


if __name__ == "__main__":
    unittest.main()
