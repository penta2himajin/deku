"""Classless web reply path: grounded sentence/summary, not relation templates."""

from __future__ import annotations

import os
import unittest
from unittest import mock

from deku import web_search as ws


class TestClasslessCompose(unittest.TestCase):
    def setUp(self):
        os.environ["DEKU_CLASSLESS_WEB"] = "1"

    def tearDown(self):
        os.environ.pop("DEKU_CLASSLESS_WEB", None)

    def test_prefers_source_sentence_over_ceo_template(self):
        doc = (
            "Tim Cook\n"
            "Tim Cook is the chief executive officer of Apple Inc.\n"
            "Source: wiki\n"
        )
        # Class path would template "The CEO of Apple is Tim Cook."
        reply = ws.compose_reply(
            "Tim Cook", None, doc, question="Who is the CEO of Apple?",
        )
        self.assertIsNotNone(reply)
        self.assertIn("Tim Cook", reply)
        # Must not be the closed-class template phrasing as the only path signal;
        # sentence from the doc is preferred.
        self.assertIn("chief executive", reply.lower())
        self.assertNotEqual(
            reply,
            ws.template_reply("Who is the CEO of Apple?", "Tim Cook", doc),
        )

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
        templ = ws.template_reply(
            "What is the capital of France?", "Paris", doc
        )
        self.assertNotEqual(reply, templ)

    def test_legacy_templates_when_disabled(self):
        os.environ["DEKU_CLASSLESS_WEB"] = "0"
        doc = "Paris\nParis is the capital of France.\n"
        reply = ws.compose_reply(
            "Paris", None, doc, question="What is the capital of France?",
        )
        self.assertEqual(
            reply,
            ws.template_reply(
                "What is the capital of France?", "Paris", doc
            ),
        )

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


class TestClasslessFlag(unittest.TestCase):
    def test_default_on(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DEKU_CLASSLESS_WEB", None)
            self.assertTrue(ws.classless_web_enabled())

    def test_off(self):
        with mock.patch.dict(os.environ, {"DEKU_CLASSLESS_WEB": "0"}):
            self.assertFalse(ws.classless_web_enabled())


if __name__ == "__main__":
    unittest.main()
