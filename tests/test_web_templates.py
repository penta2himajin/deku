"""Tests for expanded web fact templates and fact_core extractors."""

from __future__ import annotations

import unittest

from deku import web_search as ws


class TestExpandedTemplates(unittest.TestCase):
    def test_born(self):
        doc = "Tim Cook\nTimothy Donald Cook was born in Mobile, Alabama.\n"
        self.assertEqual(
            ws.template_reply("Where was Tim Cook born?", "Mobile, Alabama", doc),
            "Tim Cook was born in Mobile, Alabama.",
        )

    def test_population(self):
        doc = "Tokyo\nTokyo has a population of 14 million.\n"
        self.assertEqual(
            ws.template_reply("What is the population of Tokyo?", "14 million", doc),
            "The population of Tokyo is 14 million.",
        )

    def test_headquarters(self):
        doc = "Apple Inc.\nApple is headquartered in Cupertino, California.\n"
        self.assertEqual(
            ws.template_reply(
                "Where is Apple headquartered?", "Cupertino, California", doc
            ),
            "Apple is headquartered in Cupertino, California.",
        )

    def test_released(self):
        doc = "iPhone\nThe first iPhone was released in 2007.\n"
        self.assertEqual(
            ws.template_reply("When was the iPhone released?", "2007", doc),
            "The iPhone was released in 2007.",
        )


class TestExpandedFactCore(unittest.TestCase):
    def test_born_core(self):
        doc = "Tim Cook\nTimothy Donald Cook was born in Mobile, Alabama.\n"
        self.assertEqual(
            ws.fact_core_from_doc("Where was Tim Cook born?", doc),
            "Mobile, Alabama",
        )

    def test_population_core(self):
        doc = "Tokyo\nAs of 2023, Tokyo has a population of about 14 million.\n"
        self.assertEqual(
            ws.fact_core_from_doc("What is the population of Tokyo?", doc),
            "14 million",
        )

    def test_headquarters_core(self):
        doc = "Apple Inc.\nApple is headquartered in Cupertino, California.\n"
        self.assertEqual(
            ws.fact_core_from_doc("Where is Apple headquartered?", doc),
            "Cupertino, California",
        )

    def test_released_core(self):
        doc = "iPhone\nThe first iPhone was released in 2007.\n"
        self.assertEqual(
            ws.fact_core_from_doc("When was the iPhone released?", doc),
            "2007",
        )


    def test_founded_year(self):
        doc = "Microsoft\nMicrosoft was founded in 1975.\n"
        self.assertEqual(
            ws.template_reply("When was Microsoft founded?", "1975", doc),
            "Microsoft was founded in 1975.",
        )
        self.assertEqual(
            ws.fact_core_from_doc("When was Microsoft founded?", doc),
            "1975",
        )


if __name__ == "__main__":
    unittest.main()
