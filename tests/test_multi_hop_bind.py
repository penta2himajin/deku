"""Regression tests for multi-hop bind / needs_prior."""

from __future__ import annotations

import unittest

from deku import multi_hop as mh
from deku import refuse as rf
from deku import route as rt
from deku import web_search as ws


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

    def test_published_needs_prior(self):
        self.assertTrue(mh.needs_prior("When was it published?"))


class TestBindCore(unittest.TestCase):
    def test_its_population_binds_place_from_prior_query(self):
        bind = mh.bind_core(
            "What is the capital of France?",
            "Paris",
            "what is its population?",
        )
        self.assertEqual(bind, "Paris")
        self.assertEqual(
            mh.rewrite_followup("what is its population?", bind),
            "What is the population of Paris?",
        )

    def test_its_headquarters_rewrites_to_headquartered(self):
        bind = mh.bind_core(
            "Who is the CEO of Microsoft?",
            "Satya Nadella",
            "where is its headquarters?",
        )
        self.assertEqual(bind, "Microsoft")
        self.assertEqual(
            mh.rewrite_followup("where is its headquarters?", bind),
            "Where is Microsoft headquartered?",
        )

    def test_who_founded_binds_org_not_year(self):
        bind = mh.bind_core(
            "When was Tesla founded?",
            "2003",
            "who founded it?",
        )
        self.assertEqual(bind, "Tesla")
        self.assertEqual(
            mh.rewrite_followup("who founded it?", bind),
            "Who founded Tesla?",
        )

    def test_published_binds_work_title(self):
        bind = mh.bind_core(
            "Who wrote 1984?",
            "George Orwell",
            "when was it published?",
        )
        self.assertEqual(bind, "1984")
        self.assertEqual(
            mh.rewrite_followup("when was it published?", bind),
            "When was 1984 published?",
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


class TestCoreFitsHq(unittest.TestCase):
    def test_hq_rejects_org_named_in_question(self):
        self.assertFalse(
            ws.core_fits_question(
                "Where is Microsoft headquartered?",
                "Microsoft",
            )
        )
        self.assertTrue(
            ws.core_fits_question(
                "Where is Microsoft headquartered?",
                "Redmond, Washington",
            )
        )

    def test_published_fact_core(self):
        doc = (
            "Nineteen Eighty-Four\n"
            "Nineteen Eighty-Four is a novel by George Orwell "
            "first published in 1949."
        )
        self.assertEqual(
            ws.fact_core_from_doc("When was 1984 published?", doc),
            "1949",
        )

    def test_founded_year_from_wikitext_shape(self):
        # wiki_founded_year is live; here we only check fact_core after enrich shape.
        doc = "Tesla, Inc.\nTesla, Inc. was founded in 2003.\nHeadquartered in Austin."
        self.assertEqual(
            ws.fact_core_from_doc("When was Tesla founded?", doc),
            "2003",
        )

    def test_population_ignores_decimal_fragment(self):
        doc = (
            "Population of Japan\n"
            "In April 2025, Japan's population was roughly 123.4 million people."
        )
        self.assertEqual(
            ws.fact_core_from_doc("What is the population of Japan?", doc),
            "123.4 million",
        )
        self.assertTrue(ws.population_figure_grounded("123.4 million", doc))
        self.assertFalse(ws.population_figure_grounded("4 million", doc))


class TestHowOldRefuse(unittest.TestCase):
    def test_compound_how_old_is_refused(self):
        q = "Who is the CEO of Apple and how old is he?"
        self.assertEqual(rf.classify(q), "age")
        got = rt.rule_route(q)
        self.assertEqual(got.tool, "refuse")
        self.assertEqual(got.detail.get("reason"), "age")

    def test_named_how_old_routes_to_multi_hop(self):
        q = "How old is Tim Cook?"
        self.assertNotEqual(rf.classify(q), "age")
        self.assertFalse(rf.is_hard_refuse(q))
        self.assertEqual(rt.rule_route(q).tool, "multi_hop")


class TestAgeYears(unittest.TestCase):
    def test_age_from_birth_date(self):
        from datetime import date

        from deku import calc

        years = calc.years_since(
            "1 November 1960", today=date(2026, 8, 21)
        )
        self.assertEqual(years, 65)


if __name__ == "__main__":
    unittest.main()
