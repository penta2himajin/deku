"""Human + agent local fact worker: value with path / locations."""

from __future__ import annotations

import unittest

from deku import dir_search as ds
from deku import render as rd
from deku import route as rt


class TestAssignmentWithPath(unittest.TestCase):
    def test_render_includes_path(self):
        self.assertEqual(
            rd.assignment("PREFILL", '"ANSWER: "', path="deku/extract.py"),
            'PREFILL is set to "ANSWER: " in deku/extract.py.',
        )

    def test_render_where_emphasis(self):
        self.assertEqual(
            rd.assignment(
                "PREFILL",
                '"ANSWER: "',
                path="deku/extract.py",
                where=True,
            ),
            'PREFILL is defined in deku/extract.py as "ANSWER: ".',
        )

    def test_find_assignment_path_from_doc(self):
        doc = (
            "deku/extract.py\n"
            'PREFILL = "ANSWER: "\n'
            "Source: file:deku/extract.py\n"
        )
        line, val, path = ds.find_assignment("What is the PREFILL string?", doc)
        self.assertEqual(line, 'PREFILL = "ANSWER: "')
        self.assertIn("ANSWER", val or "")
        self.assertEqual(path, "deku/extract.py")

    def test_repo_prefill_answers_with_path(self):
        got = ds.run("What is the PREFILL string?", root=".", seed=0)
        self.assertEqual(got.status, "ok")
        self.assertIn("PREFILL", got.answer or "")
        self.assertIn("extract.py", got.answer or "")
        self.assertTrue(
            (got.detail.get("locations") or [{}])[0].get("path", "").endswith(
                "extract.py"
            ),
            msg=got.detail.get("locations"),
        )

    def test_where_is_prefill_routes_and_answers(self):
        q = "Where is PREFILL set?"
        self.assertEqual(rt.rule_route(q).tool, "dir_search")
        got = ds.run(q, root=".", seed=0)
        self.assertEqual(got.status, "ok")
        self.assertIn("extract.py", got.answer or "")
        self.assertIn("ANSWER", got.answer or "")

    def test_envelope_exposes_locations(self):
        got = rt.dispatch(
            "What is the PREFILL string?",
            live_answer=False,
            audience="agent",
        )
        env = rt.envelope(got)
        self.assertEqual(env["status"], "ok")
        locs = env.get("locations") or []
        self.assertTrue(locs, msg=env)
        self.assertTrue(
            str(locs[0].get("path", "")).endswith("extract.py"),
            msg=locs,
        )
        self.assertIn("extract.py", env.get("answer") or "")


if __name__ == "__main__":
    unittest.main()
