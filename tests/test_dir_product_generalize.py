"""Repo-local / product-agnostic dir routing and prose discovery."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from deku import dir_search as ds
from deku import route as rt
from deku import route_cues as cues


class TestSoftDirGeneralized(unittest.TestCase):
    def tearDown(self):
        cues.clear_ident_cache()

    def test_dir_words_drop_product_literals(self):
        pat = cues.DIR_WORDS.pattern.casefold()
        for banned in ("deku", "is_looping", "prefill", "max_tokens", "harness"):
            self.assertNotIn(banned, pat, msg=banned)

    def test_overview_and_project_name_route_dir(self):
        self.assertEqual(
            rt.rule_route("What is this project about?", root=".").tool,
            "dir_search",
        )
        # Package name from pyproject / dirname — not a hard-coded cue.
        self.assertIn("deku", {n.casefold() for n in cues.discover_project_names(".")})
        self.assertEqual(
            rt.rule_route("What language models does deku run?", root=".").tool,
            "dir_search",
        )

    def test_foreign_models_question_not_forced_dir(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "pyproject.toml").write_text(
                '[project]\nname = "acme"\n', encoding="utf-8"
            )
            cues.clear_ident_cache()
            got = rt.rule_route(
                "What language models does OpenAI run?", root=d
            )
            self.assertEqual(got.tool, "web_search")

    def test_lowercase_discovered_ident_soft_dir(self):
        self.assertEqual(
            rt.rule_route("what is the prefill string?", root=".").tool,
            "dir_search",
        )

    def test_how_does_client_still_dir(self):
        self.assertEqual(
            rt.rule_route(
                "How does the client guard against repetition?", root="."
            ).tool,
            "dir_search",
        )


class TestProsePathDiscovery(unittest.TestCase):
    def tearDown(self):
        cues.clear_ident_cache()
        ds.clear_prose_cache()

    def test_this_repo_finds_readme(self):
        paths = ds.prose_paths(".")
        self.assertIn("README.md", paths)
        self.assertIn("AGENTS.md", paths)

    def test_tmp_only_existing(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "README.md").write_text("# Hi\n", encoding="utf-8")
            ds.clear_prose_cache()
            self.assertEqual(ds.prose_paths(d), ("README.md",))
            self.assertIn("README", ds.doc_allcaps_names(d))


class TestNoProductBoosts(unittest.TestCase):
    def test_lead_uses_overlap_not_minicpm_name(self):
        doc = (
            "README.md\n"
            "A small, quick local-LLM harness: WidgetModel-1B on localhost.\n"
            "bin/         # serve wraps the local server\n"
        )
        lead = ds.prose_lead_sentence(
            doc, "What language models does this project run?"
        )
        self.assertIsNotNone(lead)
        self.assertIn("WidgetModel", lead)
        self.assertNotIn("bin/", lead)

    def test_overview_avoids_path_map(self):
        doc = (
            "README.md\n"
            "A small harness for local models on localhost.\n"
            "bin/         # serve wrapper\n"
        )
        lead = ds.prose_lead_sentence(doc, "What is this project about?")
        self.assertIn("harness", (lead or "").lower())
        self.assertNotIn("bin/", lead or "")


if __name__ == "__main__":
    unittest.main()
