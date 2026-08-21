"""Tests for typed slot fallback (rule-first; Needle optional)."""

from __future__ import annotations

import unittest

from deku import slots


class TestRuleSlot(unittest.TestCase):
    def test_birthday_is_date(self):
        self.assertEqual(
            slots.rule_slot("What is the birthday of the current Emperor of Japan?"),
            "date",
        )

    def test_born_where_is_place(self):
        self.assertEqual(
            slots.rule_slot("Where was Tim Cook born?"),
            "place",
        )

    def test_ceo_is_person(self):
        self.assertEqual(
            slots.rule_slot("Who is the CEO of Apple?"),
            "person",
        )

    def test_population_is_number(self):
        self.assertEqual(
            slots.rule_slot("What is the population of Tokyo?"),
            "number",
        )

    def test_chitchat_is_none(self):
        self.assertEqual(slots.rule_slot("hello there"), "none")


class TestTypedExtract(unittest.TestCase):
    def test_date_from_born_parenthetical(self):
        doc = (
            "Naruhito\n"
            "Naruhito (born 23 February 1960) is Emperor of Japan.\n"
        )
        core = slots.extract_typed(
            "date",
            "What is the birthday of the current Emperor of Japan?",
            doc,
        )
        self.assertEqual(core, "23 February 1960")

    def test_date_rejects_ungrounded(self):
        doc = "The Emperor's Birthday is a public holiday in Japan.\n"
        self.assertIsNone(
            slots.extract_typed(
                "date",
                "What is the birthday of the current Emperor of Japan?",
                doc,
            )
        )


class TestTypedReply(unittest.TestCase):
    def test_birthday_template(self):
        doc = "Naruhito (born 23 February 1960) is Emperor of Japan.\n"
        got = slots.typed_reply(
            "date",
            "What is the birthday of Naruhito?",
            "23 February 1960",
            doc,
        )
        self.assertEqual(got, "The birthday of Naruhito is 23 February 1960.")


class TestNeedleSlotOptional(unittest.TestCase):
    def test_classify_falls_back_without_needle(self):
        slot, src = slots.classify_slot(
            "What is the birthday of Naruhito?",
            use_needle=True,
        )
        self.assertEqual(slot, "date")
        self.assertEqual(src, "rule")

    def test_needle_helper_returns_none_when_unavailable(self):
        # Must not raise if cactus-needle / needle is not installed.
        got = slots.needle_slot("What is the birthday of Naruhito?")
        self.assertTrue(got is None or got in slots.SLOTS)


if __name__ == "__main__":
    unittest.main()
