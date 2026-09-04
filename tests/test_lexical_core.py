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


    def test_founded_year_by_names(self):
        doc = (
            "Microsoft\n"
            "Founded in 1975 by Bill Gates and Paul Allen to market BASIC.\n"
        )
        got = lex.lexical_core_from_doc("Who founded Microsoft?", doc)
        self.assertEqual(got, "Bill Gates and Paul Allen")

    def test_extract_date_public(self):
        doc = "Naruhito\nNaruhito (born 23 February 1960) is Emperor of Japan.\n"
        self.assertEqual(
            lex.extract_date(
                "What is the birthday of the current Emperor of Japan?", doc
            ),
            "23 February 1960",
        )

    def test_birth_date_alias_gone(self):
        self.assertFalse(hasattr(lex, "birth_date_from_doc"))


if __name__ == "__main__":
    unittest.main()
