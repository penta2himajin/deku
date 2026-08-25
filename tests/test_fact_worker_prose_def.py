"""Definition and prose facts include defining path / locations."""

from __future__ import annotations

import unittest

from deku import dir_search as ds
from deku import render as rd
from deku import route as rt


class TestDefinitionLocate(unittest.TestCase):
    def test_render_definition_with_path(self):
        self.assertEqual(
            rd.definition(
                "def find_assignment(question: str, document: str):",
                "If notes contain IDENT = value, return line, value, path",
                path="deku/dir_search.py",
            ),
            "find_assignment in deku/dir_search.py: If notes contain IDENT = value, return line, value, path.",
        )
        self.assertEqual(
            rd.definition(
                "def find_assignment(question: str, document: str):",
                "If notes contain IDENT = value, return line, value, path",
                path="deku/dir_search.py",
                where=True,
            ),
            "find_assignment is defined in deku/dir_search.py: If notes contain IDENT = value, return line, value, path.",
        )

    def test_definition_reply_attaches_path(self):
        doc = (
            "deku/dir_search.py\n"
            "def find_assignment(question: str, document: str):\n"
            '    """If notes contain IDENT = value, return line, value, path."""\n'
            "    pass\n"
            "Source: file:deku/dir_search.py\n"
        )
        defn = ds.find_definition("What does find_assignment do?", doc)
        path = ds.path_near_line(doc, defn or "")
        reply = ds.definition_reply(doc, defn, path=path)
        self.assertIn("find_assignment", reply)
        self.assertIn("dir_search.py", reply)

    def test_where_is_find_assignment_from_repo(self):
        q = "Where is find_assignment defined?"
        self.assertEqual(rt.rule_route(q).tool, "dir_search")
        got = ds.run(q, root=".", seed=0)
        self.assertEqual(got.status, "ok", msg=got.detail)
        self.assertIn("find_assignment", got.answer or "")
        self.assertIn("dir_search.py", got.answer or "")
        locs = got.detail.get("locations") or []
        self.assertTrue(
            locs and "dir_search.py" in str(locs[0].get("path", "")),
            msg=locs,
        )


class TestProseLocate(unittest.TestCase):
    def test_prose_cite(self):
        self.assertEqual(
            rd.prose_cite(
                "A MiniCPM5-1B–oriented local task harness.",
                "README.md",
            ),
            "A MiniCPM5-1B–oriented local task harness (see README.md).",
        )
        self.assertEqual(
            rd.prose_cite(
                "A MiniCPM5-1B–oriented local task harness.",
                "README.md",
                where=True,
            ),
            "README.md says: A MiniCPM5-1B–oriented local task harness.",
        )

    def test_project_about_cites_readme(self):
        got = ds.run("What is this project about?", root=".", seed=0)
        self.assertEqual(got.status, "ok")
        self.assertIn("README.md", got.answer or "")
        locs = got.detail.get("locations") or []
        self.assertTrue(
            any((loc.get("path") or "").endswith("README.md") for loc in locs),
            msg=locs,
        )

    def test_where_readme_describes_project(self):
        q = "Where is this project described?"
        self.assertEqual(rt.rule_route(q).tool, "dir_search")
        got = ds.run(q, root=".", seed=0)
        self.assertEqual(got.status, "ok", msg=got.detail)
        self.assertIn("README.md", got.answer or "")
        self.assertTrue(
            (got.answer or "").startswith("README.md")
            or "says:" in (got.answer or "").lower()
            or "(see README.md)" in (got.answer or ""),
            msg=got.answer,
        )


if __name__ == "__main__":
    unittest.main()
