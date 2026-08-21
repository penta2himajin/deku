"""dir_search: walk a directory, rank chunks, MiniCPM extract + grounded reply.

Same division of labour as web_search, without Needle/LFM:
  intent/query   rule_* (deterministic)
  search/rank    walk files → chunks → extract.term_score
  answer         assignment/def first, else MiniCPM compose + abstain
"""
import sys
import tempfile
import unittest
from pathlib import Path

from deku import dir_search as ds


class TestChunk(unittest.TestCase):
    def test_splits_on_blank_lines(self):
        text = "alpha\n\nbeta line\nmore beta\n\ngamma"
        chunks = ds.chunk_text(text, path="x.py")
        bodies = [c["snippet"] for c in chunks]
        self.assertTrue(any("alpha" in b and "beta" not in b for b in bodies))
        self.assertTrue(any("beta line" in b for b in bodies))

    def test_skips_empty(self):
        self.assertEqual(ds.chunk_text("\n\n  \n", path="x"), [])

    def test_markdown_splits_on_headings(self):
        text = "# Title\n\nIntro para.\n\n## Setup\n\nInstall stuff."
        chunks = ds.chunk_text(text, path="README.md")
        self.assertTrue(any("Setup" in c["snippet"] for c in chunks))


class TestIndex(unittest.TestCase):
    def test_indexes_text_files_under_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.py").write_text("PREFILL = 'ANSWER: '\nTEMP = 0.2\n")
            (root / "skip.bin").write_bytes(b"\x00\x01")
            (root / ".venv").mkdir()
            (root / ".venv" / "noise.py").write_text("secret = 1\n")
            hits = ds.index_dir(root)
            paths = {h["path"] for h in hits}
            self.assertTrue(any(p.endswith("a.py") for p in paths))
            self.assertFalse(any(".venv" in p for p in paths))
            self.assertFalse(any(p.endswith("skip.bin") for p in paths))


class TestRank(unittest.TestCase):
    def test_prefill_chunk_wins(self):
        q = "What is the PREFILL string?"
        hits = [
            {"title": "a.py", "snippet": "TEMP = 0.2\nMAX = 40", "path": "a.py", "url": "a.py"},
            {"title": "b.py", "snippet": "PREFILL = 'ANSWER: '\nused for extract", "path": "b.py", "url": "b.py"},
            {"title": "c.py", "snippet": "unrelated coffee port", "path": "c.py", "url": "c.py"},
        ]
        top = ds.rank_chunks(q, hits, k=2)
        self.assertEqual(top[0]["path"], "b.py")

    def test_assignment_beats_gold_case_string(self):
        q = "What is the PREFILL string?"
        hits = [
            {
                "title": "route_cases.py",
                "snippet": '("What is the PREFILL string?", "dir_search", "hard"),',
                "path": "harness/route_cases.py",
                "url": "r",
            },
            {
                "title": "extract.py",
                "snippet": 'PREFILL = "ANSWER: "',
                "path": "extract.py",
                "url": "e",
            },
        ]
        self.assertEqual(ds.rank_chunks(q, hits, k=1)[0]["path"], "extract.py")

    def test_identifier_bonus_beats_soft_overlap(self):
        q = "What is PREFILL?"
        hits = [
            {"title": "noise.py", "snippet": "what is the string value here", "path": "noise.py", "url": "n"},
            {"title": "extract.py", "snippet": "PREFILL = 'ANSWER: '", "path": "extract.py", "url": "e"},
        ]
        self.assertEqual(ds.rank_chunks(q, hits, k=1)[0]["path"], "extract.py")

    def test_prose_question_prefers_readme(self):
        q = "What is this project about?"
        hits = [
            {"title": "x.py", "snippet": "import os\nx = 1", "path": "util.py", "url": "u"},
            {"title": "README.md", "snippet": "deku is a small local-LLM harness.", "path": "README.md", "url": "r"},
        ]
        self.assertEqual(ds.rank_chunks(q, hits, k=1)[0]["path"], "README.md")

    def test_soft_product_question_is_prose_mode(self):
        self.assertEqual(ds.corpus_mode("What language models does deku run?"), "prose")
        self.assertEqual(ds.corpus_mode("What is PREFILL?"), "code")
        self.assertEqual(ds.corpus_mode("What does is_looping detect?"), "code")

    def test_loop_question_boosts_is_looping_chunk(self):
        q = "How does the client guard against repetition?"
        hits = [
            {"title": "a.md", "snippet": "the client is a CLI", "path": "AGENTS.md", "url": "a"},
            {"title": "s.py", "snippet": "def is_looping(tail):\n    # repetition loops", "path": "client.py", "url": "s"},
        ]
        self.assertEqual(ds.rank_chunks(q, hits, k=1)[0]["path"], "client.py")


class TestIntent(unittest.TestCase):
    def test_config_question_is_search(self):
        self.assertEqual(ds.rule_intent("What is PREFILL in extract.py?"), "search")

    def test_math_refuses(self):
        self.assertEqual(ds.rule_intent("What is 2+2?"), "refuse")

    def test_project_overview_is_search(self):
        self.assertEqual(ds.rule_intent("What is this project about?"), "search")


class TestAssignment(unittest.TestCase):
    def test_finds_prefill_assignment(self):
        doc = 'extract.py\nPREFILL = "ANSWER: "\nTEMP = 0.2\n'
        line, val = ds.find_assignment("What is the PREFILL string?", doc)
        self.assertEqual(line, 'PREFILL = "ANSWER: "')
        self.assertIn("ANSWER", val)

    def test_definition_reply_includes_docstring(self):
        doc = (
            "client.py\n"
            "def is_looping(tail: str) -> bool:\n"
            '    """True if tail ends in a repeated unit."""\n'
            "    for n in range(1, 601):\n"
            "        pass\n"
        )
        defn = ds.find_definition("What does is_looping detect?", doc)
        reply = ds.definition_reply(doc, defn)
        self.assertIn("is_looping", reply)
        self.assertIn("repeated unit", reply)
        self.assertNotIn("def is_looping", reply)

    def test_definition_falls_back_to_comment(self):
        doc = (
            "mod.py\n"
            "def ping():\n"
            "    # return a heartbeat token\n"
            "    return 1\n"
        )
        defn = ds.find_definition("What does ping do?", doc)
        reply = ds.definition_reply(doc, defn)
        self.assertIn("heartbeat", reply)


class TestAbstainHelpers(unittest.TestCase):
    def test_weak_hit_abstains(self):
        self.assertTrue(ds.should_abstain(
            question="What is PREFILL?",
            doc="coffee is served hot",
            score=0,
            core="ANSWER",
        ))

    def test_tiny_digit_core_abstains(self):
        self.assertTrue(ds.should_abstain(
            question="What is the PREFILL string?",
            doc="PREFILL = 'ANSWER: '",
            score=5,
            core="2",
        ))


class TestProseLead(unittest.TestCase):
    def test_lead_skips_heading(self):
        doc = "README.md\n# deku\n\nSmall, quick local-LLM harness for MiniCPM.\nSource: file:README.md"
        lead = ds.prose_lead_sentence(doc)
        self.assertIsNotNone(lead)
        self.assertIn("harness", lead.lower())

    def test_lead_prefers_question_overlap(self):
        doc = (
            "README.md\n"
            "A small harness for MiniCPM on localhost.\n"
            "client.py    # CLI client (Python stdlib only — keep it dependency-free)\n"
            "bin/         # deku-server (mlx_lm.server wrapper)\n"
        )
        lead = ds.prose_lead_sentence(doc, "Where is the default mlx server wrapper?")
        self.assertIsNotNone(lead)
        self.assertIn("bin/", lead)

    def test_lead_prefers_overview_for_about(self):
        doc = (
            "README.md\n"
            "A small, quick local-LLM harness: MiniCPM5-1B on localhost.\n"
            "bin/         # deku-server (mlx_lm.server wrapper)\n"
        )
        lead = ds.prose_lead_sentence(doc, "What is this project about?")
        self.assertIsNotNone(lead)
        self.assertIn("harness", lead.lower())
        self.assertNotIn("bin/", lead)

    def test_lead_prefers_minicpm_for_models(self):
        doc = (
            "README.md\n"
            "A small, quick local-LLM harness: MiniCPM5-1B (4-bit MLX) resident on localhost.\n"
            "- Resident server: bin/deku-server wraps mlx_lm.server on localhost.\n"
        )
        lead = ds.prose_lead_sentence(doc, "What language models does deku run?")
        self.assertIsNotNone(lead)
        self.assertIn("MiniCPM", lead)
        self.assertNotIn("bin/", lead)

    def test_packs_same_file_neighbors(self):
        scored = [
            (10, {"path": "a.py", "snippet": "line1 has enough words here", "title": "a#0", "url": "a#0"}),
            (8, {"path": "a.py", "snippet": "line2 also has enough words here", "title": "a#1", "url": "a#1"}),
            (7, {"path": "b.py", "snippet": "other file content words", "title": "b#0", "url": "b#0"}),
        ]
        packed = ds.pack_hits(scored, n=2, mode="code")
        self.assertEqual([h["path"] for h in packed], ["a.py", "a.py"])

    def test_prose_mode_filters_code_paths(self):
        hits = [
            {"path": "harness/web_search.py", "snippet": "CEO of Apple wiki", "title": "w", "url": "w"},
            {"path": "README.md", "snippet": "MiniCPM5-1B harness on localhost via mlx", "title": "r", "url": "r"},
        ]
        scored = ds.rank_chunks_scored(
            "What language models does deku run?", hits, k=2
        )
        self.assertEqual(scored[0][1]["path"], "README.md")


if __name__ == "__main__":
    unittest.main()
