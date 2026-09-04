"""Peel closed brand / product allowlists into generic rules."""

from __future__ import annotations

import unittest

from deku import multi_hop as mh
from deku import web_search as ws


class TestMakerCoreFitsNoBrandAllowlist(unittest.TestCase):
    def test_rejects_product_embedded_in_core(self):
        self.assertFalse(
            ws.core_fits_question(
                "What company makes the WidgetPhone?", "Globex WidgetPhone"
            )
        )

    def test_accepts_short_maker_without_allowlist(self):
        self.assertTrue(
            ws.core_fits_question("What company makes the WidgetPhone?", "Globex")
        )


class TestConcreteTopicCamelCase(unittest.TestCase):
    def test_iphone_without_brand_list(self):
        self.assertTrue(mh.has_concrete_topic("When was the iPhone released?"))
        self.assertFalse(mh.needs_prior("When was the iPhone released?"))

    def test_playstation_camel_or_cap(self):
        self.assertTrue(mh.has_concrete_topic("What company makes the PlayStation?"))


class TestSpouseNoiseGeneric(unittest.TestCase):
    def test_best_sentence_skips_spouse_not_named_person(self):
        doc = (
            "France\n"
            "Alice Example is the wife of the president. "
            "Bob Example is the president of France.\n"
        )
        got = ws.best_sentence_for_question(
            "Who is the president of France?", doc
        )
        self.assertIsNotNone(got)
        self.assertIn("Bob Example", got)
        self.assertNotIn("wife", got.casefold())


if __name__ == "__main__":
    unittest.main()
