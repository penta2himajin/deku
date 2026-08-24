"""Contract, audience refuse, url composition, no BUILDERS."""

from __future__ import annotations

import unittest

from deku import orchestrate as orch
from deku import refuse as refuse_mod
from deku import route as rt


class TestNoBuilders(unittest.TestCase):
    def test_builders_removed(self):
        self.assertFalse(hasattr(orch, "BUILDERS"))
        self.assertFalse(hasattr(orch, "build_web_pair"))

    def test_propose_still_covers_legacy_shapes(self):
        plan = orch.select_and_build(
            "Who is the CEO of Apple and where was he born?"
        )
        self.assertEqual(plan.plan_id, "web_dependent")
        plan2 = orch.select_and_build(
            "What is the PREFILL string and what is MAX_TOKENS?"
        )
        self.assertEqual(plan2.plan_id, "dir_pair")


class TestUrlComposition(unittest.TestCase):
    def test_url_and_web_plans(self):
        plan = orch.select_and_build(
            "Summarize https://example.com and who is the CEO of Apple?"
        )
        self.assertIsNotNone(plan)
        self.assertEqual(
            {s.tool for s in plan.steps},
            {"url_read", "web_search"},
        )
        self.assertEqual(plan.plan_id, "url+web")


class TestRefuseAudience(unittest.TestCase):
    def test_human_vs_agent(self):
        human = refuse_mod.message("math", audience="human")
        agent = refuse_mod.message("math", audience="agent")
        self.assertIn("math", human.lower())
        self.assertEqual(agent, "refused:math")
        self.assertNotEqual(human, agent)

    def test_dispatch_agent_refuse(self):
        got = rt.dispatch("What is 2+2?", audience="agent", live_answer=False)
        self.assertEqual(got.status, "refused")
        self.assertEqual(got.answer, "refused:math")
        env = rt.envelope(got)
        self.assertEqual(env["reason"], "math")
        self.assertEqual(env["next_hint"]["action"], "ask_in_scope_fact")


class TestEnvelope(unittest.TestCase):
    def test_multi_hop_envelope_cores(self):
        def web_run(q, seed=0, root=".", **kw):
            r = type("R", (), {})()
            r.hits, r.document, r.detail = [], "", {"core": "Tim Cook"}
            r.status, r.answer = "ok", "The CEO of Apple is Tim Cook."
            return r

        def dir_run(q, seed=0, root=".", **kw):
            r = type("R", (), {})()
            r.hits, r.document, r.detail = [], "", {"core": "ANSWER: "}
            r.status, r.answer = "ok", 'PREFILL is set to "ANSWER: ".'
            return r

        got = orch.run(
            "Who is the CEO of Apple and what is the PREFILL string?",
            runners={"web_search": web_run, "dir_search": dir_run},
        )
        self.assertEqual(got.status, "ok")
        self.assertEqual(got.detail.get("cores")[0], "Tim Cook")
        self.assertTrue(
            (got.detail.get("cores") or [""])[1].startswith("ANSWER:"),
            msg=got.detail.get("cores"),
        )
        self.assertEqual(got.detail.get("next_hint", {}).get("action"), "none")


if __name__ == "__main__":
    unittest.main()
