"""P4: canonicalize, confidence gate, unified join + partial hops."""

from __future__ import annotations

import unittest

from deku import canonicalize as can
from deku import multi_hop as mh
from deku import normalize as nz
from deku import orchestrate as orch
from deku import route as rt
from deku import web_search as ws


class TestCanonicalize(unittest.TestCase):
    def test_chief_executive_paraphrase(self):
        en, d = can.canonicalize_question(
            "Who currently runs Apple as chief executive?"
        )
        self.assertEqual(en, "Who is the CEO of Apple?")
        self.assertTrue(d.get("canonicalized"))

    def test_leads_as_ceo(self):
        en, _ = can.canonicalize_question(
            "Who currently leads Apple as chief executive?"
        )
        self.assertEqual(en, "Who is the CEO of Apple?")

    def test_possessive_population(self):
        en, _ = can.canonicalize_question("What is Tokyo's population?")
        self.assertEqual(en, "What is the population of Tokyo?")

    def test_prepare_chains_ja_then_canon(self):
        en, d = nz.prepare_question("日本の首都はどこですか？")
        self.assertEqual(en, "What is the capital of 日本?")
        self.assertEqual(d.get("normalized_from"), "ja")

    def test_route_uses_canonical_ceo(self):
        got = rt.rule_route("Who currently runs Apple as chief executive?")
        self.assertEqual(got.tool, "web_search")
        self.assertIn("CEO", got.query or "")
        self.assertIn("Apple", got.query or "")


class TestConfidenceGate(unittest.TestCase):
    def test_role_object_title(self):
        self.assertTrue(
            ws.looks_role_object_title("Prime Minister's Official Car (Japan)")
        )
        self.assertFalse(ws.looks_role_object_title("Shigeru Ishiba"))

    def test_rank_demotes_official_car(self):
        q = "Who is the prime minister of Japan?"
        hits = [
            {
                "title": "Prime Minister's Official Car (Japan)",
                "snippet": "The official state car of the Prime Minister of Japan.",
                "url": "u1",
            },
            {
                "title": "Shigeru Ishiba",
                "snippet": (
                    "Shigeru Ishiba is a Japanese politician who has served "
                    "as Prime Minister of Japan since 2024."
                ),
                "url": "u2",
            },
        ]
        top = ws.rank_hits(q, hits, k=2)
        self.assertEqual(top[0]["title"], "Shigeru Ishiba")

    def test_office_core_rejects_role_object(self):
        hit = {
            "title": "Prime Minister's Official Car (Japan)",
            "snippet": "state car",
            "url": "u",
        }
        doc = "Prime Minister's Official Car (Japan)\nThe official state car."
        self.assertIsNone(ws.office_core_from_hit(
            "Who is the prime minister of Japan?", hit, doc
        ))


class TestJoinAndPartial(unittest.TestCase):
    def test_decompose_also_clause(self):
        subs = mh.decompose(
            "Who is the CEO of Apple? Also, is that a good thing for the company?"
        )
        self.assertGreaterEqual(len(subs), 2)

    def test_decompose_then(self):
        subs = mh.decompose(
            "What is the capital of France, then what is its population?"
        )
        self.assertGreaterEqual(len(subs), 2)

    def test_opinion_tail_refuses_not_wrong_ok(self):
        got = rt.rule_route(
            "Who is the CEO of Apple? Also, is that a good thing for the company?"
        )
        self.assertEqual(got.tool, "refuse")

    def test_partial_keeps_successful_hop(self):
        plan = orch.Plan(
            plan_id="web_pair",
            steps=[
                orch.Step(tool="web_search", query="Who is the CEO of Apple?"),
                orch.Step(tool="web_search", query="What is the capital of Zembla?"),
            ],
            dependent=False,
        )

        def fake_run(tool, q, **kw):
            r = type("R", (), {})()
            r.hits, r.document, r.detail = [], "", {}
            if "Apple" in q:
                r.status, r.answer = "ok", "The CEO of Apple is Tim Cook."
                r.detail = {"core": "Tim Cook"}
            else:
                r.status, r.answer = "cannot_answer", None
            return r

        got = orch.run(
            "x",
            plan=plan,
            runners={"web_search": lambda q, **kw: fake_run("web_search", q, **kw)},
        )
        self.assertEqual(got.status, "partial")
        self.assertIn("Tim Cook", got.answer or "")
        self.assertIn("could not answer", (got.answer or "").lower())


if __name__ == "__main__":
    unittest.main()
