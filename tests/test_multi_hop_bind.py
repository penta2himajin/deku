"""Regression tests for multi-hop bind / needs_prior."""

from __future__ import annotations

import unittest

from deku import multi_hop as mh
from deku import refuse as rf
from deku import route as rt


class TestNeedsPrior(unittest.TestCase):
    def test_pronoun_needs_prior(self):
        self.assertTrue(mh.needs_prior("where was he born?"))
        self.assertTrue(mh.needs_prior("what is his birthday?"))
        self.assertTrue(mh.needs_prior("what is its population?"))

    def test_independent_topic_does_not_need_prior(self):
        self.assertFalse(mh.needs_prior("When was the iPhone released?"))
        self.assertFalse(mh.needs_prior("What is the population of Tokyo?"))
        self.assertFalse(mh.needs_prior("What is the capital of France?"))

    def test_thin_anaphora_needs_prior(self):
        self.assertTrue(mh.needs_prior("Where was born?"))
        self.assertTrue(mh.needs_prior("When was it founded?"))


class TestBindCore(unittest.TestCase):
    def test_its_population_binds_place_from_prior_query(self):
        bind = mh.bind_core(
            "What is the capital of France?",
            "Paris",
            "what is its population?",
        )
        self.assertEqual(bind, "France")
        self.assertEqual(
            mh.rewrite_followup("what is its population?", bind),
            "What is France's population?",
        )

    def test_his_birthday_binds_person_core(self):
        bind = mh.bind_core(
            "Who is the CEO of Apple?",
            "Tim Cook",
            "what is his birthday?",
        )
        self.assertEqual(bind, "Tim Cook")
        self.assertEqual(
            mh.rewrite_followup("what is his birthday?", bind),
            "What is Tim Cook's birthday?",
        )


class TestHowOldRefuse(unittest.TestCase):
    def test_how_old_is_refused(self):
        q = "Who is the CEO of Apple and how old is he?"
        self.assertEqual(rf.classify(q), "age")
        got = rt.rule_route(q)
        self.assertEqual(got.tool, "refuse")
        self.assertEqual(got.detail.get("reason"), "age")


if __name__ == "__main__":
    unittest.main()
