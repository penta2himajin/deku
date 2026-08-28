"""Lexical core extractors (no closed slot labels)."""

from __future__ import annotations

import unittest

from deku import lexical_core as lex


class TestLexicalCore(unittest.TestCase):
    def test_birthday_date(self):
        doc = (
            "Naruhito\n"
            "Naruhito (born 23 February 1960) is Emperor of Japan.\n"
        )
        got = lex.lexical_core_from_doc(
            "What is the birthday of the current Emperor of Japan?", doc
        )
        self.assertEqual(got, "23 February 1960")

    def test_born_place(self):
        doc = (
            "Tim Cook\n"
            "Cook was born in Mobile, Alabama, and grew up in Robertsdale.\n"
        )
        got = lex.lexical_core_from_doc("Where was Tim Cook born?", doc)
        self.assertEqual(got, "Mobile, Alabama")

    def test_population(self):
        doc = (
            "Tokyo\n"
            "The population of the city proper was over 14 million as of 2023.\n"
        )
        got = lex.lexical_core_from_doc("What is the population of Tokyo?", doc)
        self.assertEqual(got, "14 million")

    def test_founded_year(self):
        doc = "Microsoft\nMicrosoft was founded in 1975.\n"
        got = lex.lexical_core_from_doc("When was Microsoft founded?", doc)
        self.assertEqual(got, "1975")


class TestSlotsRemoved(unittest.TestCase):
    def test_slots_module_gone(self):
        import importlib

        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("deku.slots")


if __name__ == "__main__":
    unittest.main()
