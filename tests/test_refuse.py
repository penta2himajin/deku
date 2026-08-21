import sys
import unittest
from pathlib import Path

from deku import refuse as rf
from deku import route as rt


class TestClassify(unittest.TestCase):
    def test_math(self):
        self.assertEqual(rf.classify("What is 2+2?"), "math")

    def test_code(self):
        self.assertEqual(rf.classify("Write a sort function"), "code")

    def test_fix_bug_is_code(self):
        self.assertEqual(rf.classify("Fix the bug in route.py"), "code")
        self.assertTrue(rf.is_hard_refuse("Fix the bug in route.py"))

    def test_underspecified_path(self):
        q = "Show me the commit log of the last commit that changed this part."
        self.assertEqual(rf.classify(q), "underspecified")
        self.assertTrue(rf.is_underspecified_path(q))
        self.assertFalse(
            rf.is_underspecified_path(
                "What is the commit message of the last commit that changed harness/route.py?"
            )
        )
        self.assertFalse(
            rf.is_underspecified_path("What is the last commit message?")
        )

    def test_chitchat(self):
        self.assertEqual(rf.classify("hello there"), "chitchat")

    def test_deep_reasoning(self):
        self.assertEqual(
            rf.classify("Compare capitalism and socialism in a long essay"),
            "deep_reasoning",
        )
        self.assertEqual(
            rf.classify("Explain why quantum entanglement works step by step"),
            "deep_reasoning",
        )

    def test_message_nonempty(self):
        for reason in rf.REASONS:
            self.assertTrue(len(rf.message(reason)) > 20)


class TestDispatchRefuse(unittest.TestCase):
    def test_math_has_explicit_answer(self):
        got = rt.dispatch("What is 2+2?", router="rule")
        self.assertEqual(got.tool, "refuse")
        self.assertEqual(got.status, "refused")
        self.assertEqual(got.detail.get("reason"), "math")
        self.assertIn("math", (got.answer or "").lower())

    def test_code_points_at_allowed_tools(self):
        got = rt.dispatch("Write a sort function", router="rule")
        self.assertEqual(got.status, "refused")
        self.assertEqual(got.detail.get("reason"), "code")
        self.assertIn("git", (got.answer or "").lower())

    def test_deep_refuses_before_web(self):
        got = rt.rule_route(
            "Compare capitalism and socialism in a long essay"
        )
        self.assertEqual(got.tool, "refuse")

    def test_this_part_git_clarifies(self):
        got = rt.rule_route(
            "Show me the commit log of the last commit that changed this part."
        )
        self.assertEqual(got.tool, "clarify")
        self.assertEqual(got.detail.get("reason"), "path")
        self.assertFalse(rf.is_hard_refuse(
            "Show me the commit log of the last commit that changed this part."
        ))

    def test_fix_bug_refuses(self):
        got = rt.rule_route("Fix the bug in route.py")
        self.assertEqual(got.tool, "refuse")
        self.assertEqual(got.detail.get("reason"), "code")


if __name__ == "__main__":
    unittest.main()
