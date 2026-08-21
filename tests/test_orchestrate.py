import sys
import unittest
from pathlib import Path

from deku import orchestrate as orch
from deku import route as rt


class TestSelectBuild(unittest.TestCase):
    def test_web_independent(self):
        plan = orch.select_and_build(
            "Who is the CEO of Apple and what is the capital of France?"
        )
        self.assertIsNotNone(plan)
        self.assertEqual(plan.plan_id, "web_independent")
        self.assertEqual([s.tool for s in plan.steps], ["web_search", "web_search"])

    def test_web_dependent(self):
        plan = orch.select_and_build(
            "Who is the CEO of Apple and where was he born?"
        )
        self.assertIsNotNone(plan)
        self.assertEqual(plan.plan_id, "web_dependent")
        self.assertTrue(plan.steps[1].bind_prior)

    def test_dir_pair(self):
        plan = orch.select_and_build(
            "What is the PREFILL string and what is MAX_TOKENS?"
        )
        self.assertIsNotNone(plan)
        self.assertEqual(plan.plan_id, "dir_pair")
        self.assertEqual([s.tool for s in plan.steps], ["dir_search", "dir_search"])

    def test_git_and_diff(self):
        plan = orch.select_and_build(
            "What is the last commit message and what is in the unstaged diff?"
        )
        self.assertIsNotNone(plan)
        self.assertEqual(plan.plan_id, "git_and_diff")
        tools = [s.tool for s in plan.steps]
        self.assertEqual(set(tools), {"git_search", "diff_search"})

    def test_mixed_web_dir_no_plan(self):
        # Ambiguous mix — do not invent a cross-tool plan.
        plan = orch.select_and_build(
            "Who is the CEO of Apple and what is the PREFILL string?"
        )
        self.assertIsNone(plan)
        self.assertTrue(
            orch.mixed_tools_without_plan(
                "Who is the CEO of Apple and what is the PREFILL string?"
            )
        )

    def test_single_question_no_plan(self):
        self.assertIsNone(orch.select_and_build("Who is the CEO of Apple?"))
        self.assertIsNone(orch.select_and_build("What is the PREFILL string?"))


class TestRoutePlans(unittest.TestCase):
    def test_hard_prefers_git_diff_plan(self):
        got = rt.rule_route(
            "What is the last commit message and what is in the unstaged diff?"
        )
        self.assertEqual(got.tool, "multi_hop")
        self.assertEqual(got.detail.get("plan_id"), "git_and_diff")

    def test_hard_prefers_dir_pair(self):
        got = rt.rule_route(
            "What is the PREFILL string and what is MAX_TOKENS?"
        )
        self.assertEqual(got.tool, "multi_hop")
        self.assertEqual(got.detail.get("plan_id"), "dir_pair")


class TestRunCatalog(unittest.TestCase):
    def test_dir_pair_with_runners(self):
        def dir_run(q, seed=0, root=".", **kw):
            r = type("R", (), {})()
            r.hits, r.document, r.detail = [], "", {}
            if "PREFILL" in q:
                r.status, r.answer = "ok", "PREFILL is Answer:"
            else:
                r.status, r.answer = "ok", "MAX_TOKENS is 64"
            return r

        got = orch.run(
            "What is the PREFILL string and what is MAX_TOKENS?",
            runners={"dir_search": dir_run},
        )
        self.assertEqual(got.status, "ok")
        self.assertEqual(got.detail.get("plan_id"), "dir_pair")
        self.assertIn("PREFILL", got.answer or "")
        self.assertIn("MAX_TOKENS", got.answer or "")
        self.assertIn("1.", got.answer or "")

    def test_git_diff_with_runners(self):
        def git_run(q, seed=0, root=".", **kw):
            r = type("R", (), {})()
            r.hits, r.document, r.detail = [], "", {}
            r.status, r.answer = "ok", "Last commit: feat: demo"
            return r

        def diff_run(q, seed=0, root=".", **kw):
            r = type("R", (), {})()
            r.hits, r.document, r.detail = [], "", {}
            r.status, r.answer = "ok", "Unstaged: harness/foo.py"
            return r

        got = orch.run(
            "What is the last commit message and what is in the unstaged diff?",
            runners={"git_search": git_run, "diff_search": diff_run},
        )
        self.assertEqual(got.status, "ok")
        self.assertEqual(got.detail.get("plan_id"), "git_and_diff")
        self.assertIn("feat: demo", got.answer or "")
        self.assertIn("foo.py", got.answer or "")


if __name__ == "__main__":
    unittest.main()
