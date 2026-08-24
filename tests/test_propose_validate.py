"""Validated composition: propose steps + validate plan (no builder whitelist)."""

from __future__ import annotations

import unittest

from deku import orchestrate as orch
from deku import route as rt


class TestProposeValidate(unittest.TestCase):
    def test_dir_and_web_proposes(self):
        plan = orch.select_and_build(
            "What does the README say about deku and who is the CEO of Apple?"
        )
        self.assertIsNotNone(plan)
        tools = [s.tool for s in plan.steps]
        self.assertIn("dir_search", tools)
        self.assertIn("web_search", tools)
        self.assertGreaterEqual(len(plan.steps), 2)

    def test_git_and_dir_proposes(self):
        plan = orch.select_and_build(
            "Who authored the last commit and what is the PREFILL string?"
        )
        self.assertIsNotNone(plan)
        tools = [s.tool for s in plan.steps]
        self.assertEqual(set(tools), {"git_search", "dir_search"})

    def test_web_and_dir_proposes(self):
        plan = orch.select_and_build(
            "Who is the CEO of Apple and what is the PREFILL string?"
        )
        self.assertIsNotNone(plan)
        self.assertEqual(
            {s.tool for s in plan.steps},
            {"web_search", "dir_search"},
        )
        self.assertEqual(
            rt.rule_route(
                "Who is the CEO of Apple and what is the PREFILL string?"
            ).tool,
            "multi_hop",
        )

    def test_opinion_still_no_plan(self):
        plan = orch.select_and_build(
            "Who is the CEO of Apple? Also, is that a good thing for the company?"
        )
        self.assertIsNone(plan)
        self.assertEqual(
            rt.rule_route(
                "Who is the CEO of Apple? Also, is that a good thing for the company?"
            ).tool,
            "refuse",
        )

    def test_validate_rejects_empty_query(self):
        self.assertIsNone(
            orch.validate_plan([
                orch.Step(tool="web_search", query="Who is the CEO of Apple?"),
                orch.Step(tool="web_search", query="  "),
            ])
        )

    def test_validate_rejects_unknown_tool(self):
        self.assertIsNone(
            orch.validate_plan([
                orch.Step(tool="web_search", query="Who is the CEO of Apple?"),
                orch.Step(tool="shell", query="ls"),
            ])
        )

    def test_validate_rejects_bind_on_first(self):
        self.assertIsNone(
            orch.validate_plan([
                orch.Step(tool="web_search", query="Who is the CEO of Apple?", bind_prior=True),
                orch.Step(tool="web_search", query="Where was he born?"),
            ])
        )

    def test_validate_rejects_bind_without_anaphora(self):
        self.assertIsNone(
            orch.validate_plan([
                orch.Step(tool="web_search", query="Who is the CEO of Apple?"),
                orch.Step(
                    tool="web_search",
                    query="What is the capital of France?",
                    bind_prior=True,
                ),
            ])
        )

    def test_propose_then_validate_web_dependent(self):
        steps = orch.propose_steps(
            "Who is the CEO of Apple and where was he born?"
        )
        plan = orch.validate_plan(steps)
        self.assertIsNotNone(plan)
        self.assertTrue(plan.dependent)
        self.assertTrue(plan.steps[1].bind_prior)
        self.assertEqual(plan.plan_id, "web_dependent")

    def test_validate_rejects_too_many_steps(self):
        steps = [
            orch.Step(tool="web_search", query=f"Who is person {i}?")
            for i in range(4)
        ]
        self.assertIsNone(orch.validate_plan(steps))

    def test_diff_and_web_proposes(self):
        plan = orch.select_and_build(
            "What is in the unstaged diff and who is the CEO of Apple?"
        )
        self.assertIsNotNone(plan)
        self.assertEqual(
            {s.tool for s in plan.steps},
            {"diff_search", "web_search"},
        )


class TestValidatedRun(unittest.TestCase):
    def test_cross_tool_run_partial_ok(self):
        def web_run(q, seed=0, root=".", **kw):
            r = type("R", (), {})()
            r.hits, r.document, r.detail = [], "", {"core": "Tim Cook"}
            r.status, r.answer = "ok", "The CEO of Apple is Tim Cook."
            return r

        def dir_run(q, seed=0, root=".", **kw):
            r = type("R", (), {})()
            r.hits, r.document, r.detail = [], "", {"core": "ANSWER: "}
            r.status, r.answer = "ok", 'PREFILL is set to "ANSWER: ".'
            return r

        got = orch.run(
            "Who is the CEO of Apple and what is the PREFILL string?",
            runners={"web_search": web_run, "dir_search": dir_run},
        )
        self.assertEqual(got.status, "ok")
        self.assertIn("Tim Cook", got.answer or "")
        self.assertIn("PREFILL", got.answer or "")


if __name__ == "__main__":
    unittest.main()
