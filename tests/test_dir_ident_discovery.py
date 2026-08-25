"""Repo-discovered bare ALLCAPS config idents (replaces _DIR_BARE allowlist)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from deku import route as rt
from deku import route_cues as cues


class TestDiscoverBareConfigIdents(unittest.TestCase):
    def tearDown(self):
        cues.clear_ident_cache()

    def test_discovers_prefill_from_this_repo(self):
        found = cues.discover_bare_config_idents(".")
        self.assertIn("PREFILL", found)
        self.assertIn("TEMP", found)
        self.assertIn("STOPS", found)
        # frozenset / catalog assigns must not become dir idents
        self.assertNotIn("MATH", found)
        self.assertNotIn("MESSAGES", found)
        self.assertNotIn("TOOLS", found)
        # Two-letter tokens never match the bare-ident regex (UA was dead in the old allowlist).
        self.assertNotIn("UA", found)

    def test_has_dir_ident_uses_discovery(self):
        self.assertTrue(cues.has_dir_ident("What is the PREFILL string?", root="."))
        self.assertTrue(cues.has_dir_ident("Where is PREFILL set?", root="."))
        self.assertFalse(cues.has_dir_ident("What is NASA?", root="."))
        self.assertFalse(cues.has_dir_ident("Who is the CEO of LVMH?", root="."))
        self.assertFalse(cues.has_dir_ident("What is MATH?", root="."))

    def test_tmp_root_discovers_only_local_literal(self):
        with tempfile.TemporaryDirectory() as d:
            pkg = Path(d) / "pkg"
            pkg.mkdir()
            (pkg / "cfg.py").write_text(
                'MYCONST = "hello"\n'
                "SKIPME = frozenset({1, 2})\n"
                "ALSO = {\"a\": 1}\n",
                encoding="utf-8",
            )
            cues.clear_ident_cache()
            found = cues.discover_bare_config_idents(d)
            self.assertEqual(found, frozenset({"MYCONST"}))
            self.assertTrue(cues.has_dir_ident("What is MYCONST?", root=d))
            self.assertFalse(cues.has_dir_ident("What is PREFILL?", root=d))
            self.assertEqual(
                rt.rule_route("What is MYCONST?", root=d).tool,
                "dir_search",
            )
            self.assertEqual(
                rt.rule_route("What is NASA?", root=d).tool,
                "web_search",
            )

    def test_snake_still_hard_without_discovery(self):
        # MAX_TOKENS is SNAKE_CAPS — regex path, not bare allowlist.
        self.assertTrue(cues.has_dir_ident("What is MAX_TOKENS?"))
        self.assertEqual(
            rt.rule_route("What is MAX_TOKENS?").tool,
            "dir_search",
        )


if __name__ == "__main__":
    unittest.main()
