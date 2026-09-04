"""Shared question-shape cue helpers."""

from __future__ import annotations

import unittest

from deku import question_cues as qc


class TestQuestionCues(unittest.TestCase):
    def test_who_office(self):
        self.assertTrue(qc.asks_who_office("Who is the CEO of Apple?"))
        self.assertTrue(qc.asks_who_office("Who is the emperor of Japan?"))
        self.assertFalse(qc.asks_who_office("Where is Apple headquartered?"))

    def test_birthday_vs_how_old(self):
        self.assertTrue(qc.asks_birthday("How old is Tim Cook?"))
        self.assertTrue(qc.asks_birthday_strict("What is Tim Cook's birthday?"))
        self.assertFalse(qc.asks_birthday_strict("How old is Tim Cook?"))

    def test_founded_when(self):
        self.assertTrue(qc.asks_founded_when("When was Microsoft founded?"))
        self.assertFalse(qc.asks_founded_when("Who founded Microsoft?"))

    def test_past_present_tenure(self):
        self.assertTrue(
            qc.looks_past_tenure(
                "He was the first CEO of Apple from 1977 to 1981."
            )
        )
        self.assertTrue(
            qc.looks_present_tenure(
                "She has served as CEO of Globex since 2019."
            )
        )
        self.assertFalse(
            qc.looks_present_tenure(
                "He was the first CEO of Apple from 1977 to 1981."
            )
        )


if __name__ == "__main__":
    unittest.main()
