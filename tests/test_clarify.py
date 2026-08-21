"""Tests for clarify (ask for missing details instead of hard refuse)."""

from __future__ import annotations

import unittest

from deku import clarify as cl
from deku import refuse as rf
from deku import route as rt


class TestClarifyDetect(unittest.TestCase):
    def test_path_history_needs_clarify(self):
        q = "Show me the commit log of the last commit that changed this part."
        self.assertEqual(cl.detect(q), "path")
        self.assertIn("file path", cl.question_for(q).lower())

    def test_explicit_path_does_not_clarify(self):
        q = "What is the commit message of the last commit that changed deku/route.py?"
        self.assertIsNone(cl.detect(q))

    def test_summarize_without_url_clarifies(self):
        q = "Summarize this document for me."
        self.assertEqual(cl.detect(q), "url")
        self.assertIn("url", cl.question_for(q).lower())

    def test_math_does_not_clarify(self):
        self.assertIsNone(cl.detect("What is 2+2?"))


class TestRouteClarify(unittest.TestCase):
    def test_this_part_clarifies_instead_of_refuse(self):
        got = rt.rule_route(
            "Show me the commit log of the last commit that changed this part."
        )
        self.assertEqual(got.tool, "clarify")
        self.assertEqual(got.detail.get("reason"), "path")

    def test_dispatch_clarify_status(self):
        got = rt.dispatch(
            "Show me the commit log of the last commit that changed this part.",
            router="rule",
            live_answer=False,
        )
        self.assertEqual(got.tool, "clarify")
        self.assertEqual(got.status, "clarify")
        self.assertIn("path", (got.answer or "").lower())
        # Not the old hard refuse line.
        self.assertNotEqual(got.answer, rf.message("underspecified"))


if __name__ == "__main__":
    unittest.main()
