"""Tests for multi-hop rewrite / bind helpers."""

from __future__ import annotations

import unittest

from deku import multi_hop as mh


class TestBindRewrite(unittest.TestCase):
    def test_he_born_uses_person_core(self):
        self.assertEqual(
            mh.rewrite_followup("where was he born?", "Tim Cook"),
            "Where was Tim Cook born?",
        )

    def test_it_founded_binds_org_from_prior_query(self):
        bind = mh.bind_core(
            "Who founded Microsoft?",
            "Bill Gates",
            "when was it founded?",
        )
        self.assertEqual(bind, "Microsoft")
        self.assertEqual(
            mh.rewrite_followup("when was it founded?", bind),
            "When was Microsoft founded?",
        )

    def test_needs_prior_for_it_founded(self):
        self.assertTrue(mh.needs_prior("when was it founded?"))


if __name__ == "__main__":
    unittest.main()
