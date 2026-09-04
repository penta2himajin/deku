"""Generic calc tool (structured ops only; free-form math stays refused)."""

from __future__ import annotations

import unittest
from datetime import date

from deku import calc
from deku import orchestrate as orch
from deku import refuse as rf
from deku import route as rt


class TestYearsSince(unittest.TestCase):
    def test_prose_date(self):
        self.assertEqual(
            calc.years_since("1 November 1960", today=date(2026, 8, 21)),
            65,
        )

    def test_iso_date(self):
        self.assertEqual(
            calc.years_since("1960-11-01", today=date(2026, 8, 21)),
            65,
        )

    def test_bad_input(self):
        self.assertIsNone(calc.years_since("not a date"))


class TestCalcRun(unittest.TestCase):
    def test_years_since_query(self):
        got = calc.run("years_since: 1 November 1960", today=date(2026, 8, 21))
        self.assertEqual(got.status, "ok")
        self.assertEqual(got.detail.get("core"), "65")
        self.assertEqual(got.answer, "65")
        self.assertEqual(got.detail.get("op"), "years_since")

    def test_unknown_op_abstains(self):
        got = calc.run("2 + 2")
        self.assertEqual(got.status, "cannot_answer")


class TestNamedAgePlan(unittest.TestCase):
    def test_named_how_old_builds_web_then_calc(self):
        q = "How old is Tim Cook?"
        self.assertFalse(rf.is_hard_refuse(q))
        plan = orch.select_and_build(q)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.plan_id, "age_years")
        self.assertEqual(len(plan.steps), 2)
        self.assertEqual(plan.steps[0].tool, "web_search")
        self.assertIn("Tim Cook", plan.steps[0].query)
        self.assertIn("born", plan.steps[0].query.lower())
        self.assertEqual(plan.steps[1].tool, "calc")
        self.assertTrue(plan.steps[1].bind_prior)

    def test_named_how_old_routes_multi_hop(self):
        q = "How old is Tim Cook?"
        self.assertEqual(rt.rule_route(q).tool, "multi_hop")

    def test_age_years_integrate(self):
        from deku import web_search as ws

        def fake_web(query, seed=0, root=".", **kw):
            return ws.Result(
                intent="search",
                answer="Tim Cook was born on 1 November 1960.",
                status="ok",
                detail={"core": "1 November 1960"},
            )

        got = orch.run(
            "How old is Tim Cook?",
            runners={
                "web_search": fake_web,
                "calc": lambda q, seed=0, root=".", **kw: calc.run(
                    q, today=date(2026, 8, 21)
                ),
            },
        )
        self.assertEqual(got.status, "ok")
        self.assertEqual(got.answer, "Tim Cook is 65 years old.")
        self.assertEqual(got.detail.get("plan_id"), "age_years")


if __name__ == "__main__":
    unittest.main()
