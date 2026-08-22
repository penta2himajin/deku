import sys
import unittest
from pathlib import Path

from deku import hier_summary as hs
from deku import multi_hop as mh
from deku import url_read as ur


class TestChunk(unittest.TestCase):
    def test_short_stays_one(self):
        self.assertEqual(len(hs.chunk_text("hello world " * 10)), 1)

    def test_long_splits(self):
        paras = [f"Paragraph {i}. " + ("word " * 40) for i in range(8)]
        text = "\n\n".join(paras)
        chunks = hs.chunk_text(text, size=200, max_chunks=6)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertLessEqual(len(chunks), 6)


class TestExtractiveHier(unittest.TestCase):
    def test_offline_summary_uses_source_sentences(self):
        text = (
            "Alpha is a red fruit that grows on trees in summer. "
            "Beta is a blue mineral found only in deep mines. "
            "Gamma is a small bird that nests near rivers in spring.\n\n"
            "Delta markets sell Alpha and Beta together in coastal towns. "
            "Epsilon researchers study Gamma migration every autumn."
        )
        # Force multiple chunks with a tiny window.
        chunks = hs.chunk_text(text, size=120, max_chunks=8)
        self.assertGreaterEqual(len(chunks), 2)
        got = hs.summarize(text, live=False)
        self.assertEqual(got.status, "ok")
        self.assertTrue(got.answer)
        # Offline path is extractive — tokens should appear in source.
        for tok in ("Alpha", "Beta"):
            if tok.lower() in (got.answer or "").lower():
                self.assertIn(tok.lower(), text.lower())


class TestUrlSummarize(unittest.TestCase):
    def test_summarize_mode_offline(self):
        html = (
            b"<html><body>"
            + b"".join(
                f"<p>Section {i} discusses ports and harbours in detail with many words here.</p>".encode()
                for i in range(12)
            )
            + b"</body></html>"
        )
        got = ur.run(
            "Summarize https://example.com/ports",
            fetch=lambda u, **kw: html,
            live_answer=False,
        )
        self.assertEqual(got.status, "ok")
        self.assertEqual(got.detail.get("mode"), "hier_summary")
        self.assertTrue(got.answer)


class TestMultiHop(unittest.TestCase):
    def test_decompose_and(self):
        subs = mh.decompose(
            "Who is the CEO of Apple and what is the capital of France?"
        )
        self.assertGreaterEqual(len(subs), 2)
        self.assertTrue(any("Apple" in s for s in subs))
        self.assertTrue(any("France" in s or "capital" in s.lower() for s in subs))
        # Leading wh-word must be capitalized (MiniCPM / templates).
        self.assertTrue(all(s[:1].isupper() for s in subs))

    def test_run_composes(self):
        def fake_run(q, **kw):
            r = type("R", (), {})()
            if "CEO" in q or ("Apple" in q and "headquarter" not in q.lower()):
                r.status, r.answer, r.document, r.hits = "ok", "Tim Cook.", "doc", []
            else:
                r.status, r.answer, r.document, r.hits = (
                    "ok", "Cupertino.", "doc2", []
                )
            return r

        got = mh.run(
            "Who is the CEO of Apple and where is Apple headquartered?",
            web_run=fake_run,
        )
        self.assertEqual(got.status, "ok")
        self.assertIn("Tim Cook", got.answer or "")
        self.assertIn("Cupertino", got.answer or "")

    def test_failed_hop_abstains(self):
        def fake_run(q, **kw):
            r = type("R", (), {})()
            r.hits = []
            r.document = ""
            r.detail = {}
            if "CEO" in q:
                r.status, r.answer = "ok", "Tim Cook."
                r.detail = {"core": "Tim Cook"}
            else:
                r.status, r.answer = "cannot_answer", None
            return r

        got = mh.run(
            "Who is the CEO of Apple and where is Apple headquartered?",
            web_run=fake_run,
        )
        self.assertEqual(got.status, "cannot_answer")
        self.assertIn("failed on", (got.answer or "").lower())

    def test_rewrite_followup_pronoun(self):
        self.assertEqual(
            mh.rewrite_followup("where was he born?", "Tim Cook"),
            "Where was Tim Cook born?",
        )
        self.assertEqual(
            mh.rewrite_followup("What is the capital of France?", "Tim Cook"),
            "What is the capital of France?",
        )

    def test_dependent_hop_feeds_prior_core(self):
        seen = []

        def fake_run(q, **kw):
            seen.append(q)
            r = type("R", (), {})()
            r.hits, r.document = [], ""
            if "CEO of Apple" in q or ("Apple" in q and "born" not in q.lower()):
                r.status = "ok"
                r.answer = "The CEO of Apple is Tim Cook."
                r.detail = {"core": "Tim Cook"}
            elif "Tim Cook" in q and "born" in q.lower():
                r.status = "ok"
                r.answer = "Tim Cook was born in Alabama."
                r.detail = {"core": "Alabama"}
            else:
                r.status, r.answer, r.detail = "cannot_answer", None, {}
            return r

        got = mh.run(
            "Who is the CEO of Apple and where was he born?",
            web_run=fake_run,
        )
        self.assertEqual(got.status, "ok")
        self.assertTrue(got.detail.get("dependent"))
        self.assertTrue(any("Tim Cook" in q and "born" in q.lower() for q in seen))
        # Integrated paragraph, not just a numbered list of unrelated facts.
        self.assertIn("Tim Cook", got.answer or "")
        self.assertIn("Alabama", got.answer or "")
        self.assertFalse((got.answer or "").strip().startswith("1."))

    def test_independent_still_numbered(self):
        def fake_run(q, **kw):
            r = type("R", (), {})()
            r.hits, r.document, r.detail = [], "", {}
            if "CEO" in q:
                r.status, r.answer = "ok", "The CEO of Apple is Tim Cook."
                r.detail = {"core": "Tim Cook"}
            else:
                r.status, r.answer = "ok", "Apple is headquartered in Cupertino."
                r.detail = {"core": "Cupertino"}
            return r

        got = mh.run(
            "Who is the CEO of Apple and where is Apple headquartered?",
            web_run=fake_run,
        )
        self.assertEqual(got.status, "ok")
        self.assertFalse(got.detail.get("dependent"))
        self.assertIn("1.", got.answer or "")
        self.assertIn("2.", got.answer or "")


if __name__ == "__main__":
    unittest.main()
