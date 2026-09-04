"""Grounded web reply path (no relation templates)."""

from __future__ import annotations

import unittest

from deku import web_search as ws


class TestGroundedCompose(unittest.TestCase):
    def test_prefers_source_sentence_over_template_phrasing(self):
        doc = (
            "Tim Cook\n"
            "Tim Cook is the chief executive officer of Apple Inc.\n"
            "Source: wiki\n"
        )
        reply = ws.compose_reply(
            "Tim Cook", None, doc, question="Who is the CEO of Apple?",
        )
        self.assertIsNotNone(reply)
        self.assertIn("Tim Cook", reply)
        self.assertIn("chief executive", reply.lower())
        self.assertNotIn("The CEO of Apple is Tim Cook.", reply)

    def test_capital_uses_document_sentence(self):
        doc = (
            "Paris\n"
            "Paris is the capital of France and a major European city.\n"
        )
        reply = ws.compose_reply(
            "Paris", None, doc, question="What is the capital of France?",
        )
        self.assertIsNotNone(reply)
        self.assertIn("Paris", reply)
        self.assertIn("capital", reply.lower())

    def test_best_sentence_class_agnostic(self):
        doc = (
            "Noise about unrelated topics.\n"
            "The Widget Corporation appointed Jane Doe as its chief officer in 2020.\n"
            "More noise.\n"
        )
        sent = ws.best_sentence_for_question(
            "Who is the chief officer of Widget Corporation?", doc
        )
        self.assertIsNotNone(sent)
        self.assertIn("Jane Doe", sent)


class TestLegacyRemoved(unittest.TestCase):
    def test_template_reply_gone(self):
        self.assertFalse(hasattr(ws, "template_reply"))

    def test_classless_flag_gone(self):
        self.assertFalse(hasattr(ws, "classless_web_enabled"))

    def test_predicate_supported_gone(self):
        self.assertFalse(hasattr(ws, "predicate_supported"))

    def test_office_and_span_shortcuts_gone(self):
        self.assertFalse(hasattr(ws, "office_core_from_hit"))
        self.assertFalse(hasattr(ws, "prefer_answer_span"))
        self.assertFalse(hasattr(ws, "age_years_from_birth_date"))


if __name__ == "__main__":
    unittest.main()
