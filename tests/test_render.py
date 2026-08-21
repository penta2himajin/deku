import sys
import unittest
from pathlib import Path

from deku import render as rd
from deku import web_search as ws


class TestTemplateReply(unittest.TestCase):
    def test_ceo_template(self):
        doc = "Tim Cook is the CEO of Apple Inc."
        got = ws.template_reply(
            "Who is the CEO of Apple?", "Tim Cook", doc,
        )
        self.assertEqual(got, "The CEO of Apple is Tim Cook.")

    def test_capital_template(self):
        doc = "Paris is the capital of France."
        got = ws.template_reply(
            "What is the capital of France?", "Paris", doc,
        )
        self.assertEqual(got, "The capital of France is Paris.")

    def test_ungrounded_core_rejected(self):
        doc = "Paris is the capital of France."
        self.assertIsNone(ws.template_reply(
            "Who is the CEO of Apple?", "Tim Cook", doc,
        ))

    def test_compose_prefers_template_over_long_source(self):
        doc = (
            "William Shakespeare\n"
            "The Tragedy of Romeo and Juliet, often shortened to Romeo and Juliet, "
            "is a tragedy written by William Shakespeare about the romance between "
            "two young Italians from feuding families.\n"
            "Source: x"
        )
        got = ws.compose_reply(
            "William Shakespeare",
            "Someone invented this.",
            doc,
            question="Who wrote Romeo and Juliet?",
        )
        self.assertEqual(got, "Romeo and Juliet was written by William Shakespeare.")


class TestSentenceFilter(unittest.TestCase):
    def test_skips_anaphoric_because(self):
        doc = (
            "Boiling\n"
            "Because of this, water boils at 100 °C, rounded from scientific.\n"
            "At standard pressure, water boils at 100 °C.\n"
            "Source: x"
        )
        got = ws.sentence_with_core("100", doc, question="What is the boiling point of water?")
        self.assertIsNotNone(got)
        self.assertNotIn("Because of this", got)
        self.assertIn("100", got)


class TestRender(unittest.TestCase):
    def test_assignment(self):
        self.assertEqual(
            rd.assignment("PREFILL", '"ANSWER: "'),
            'PREFILL is set to "ANSWER: ".',
        )

    def test_git_message(self):
        self.assertEqual(
            rd.git_message("feat: add router"),
            'The last commit message is "feat: add router".',
        )

    def test_git_author(self):
        self.assertIn("Test User", rd.git_author("Test User", "abc1234dead", "fix: x"))
        self.assertIn("abc1234dead", rd.git_author("Test User", "abc1234deadbeef", "fix: x"))

    def test_git_files(self):
        got = rd.git_files("feat: x", ["a.py", "b.py"])
        self.assertIn("a.py", got)
        self.assertIn("feat: x", got)

    def test_diff_assignment_change(self):
        got = rd.diff_line("extract.py", "+PREFILL = 'ANSWER: '")
        self.assertIn("PREFILL", got)
        self.assertIn("ANSWER", got)
        self.assertIn("extract.py", got)


if __name__ == "__main__":
    unittest.main()
