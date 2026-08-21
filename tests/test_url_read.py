import sys
import unittest
from pathlib import Path

from deku import url_read as ur


class TestExtractUrl(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(
            ur.extract_url("Read https://example.com/x and summarize"),
            "https://example.com/x",
        )

    def test_strips_trailing_period(self):
        self.assertEqual(
            ur.extract_url("See https://example.com/page."),
            "https://example.com/page",
        )


class TestHtmlToText(unittest.TestCase):
    def test_strips_tags_keeps_words(self):
        raw = b"<html><script>x()</script><p>Tim Cook is the CEO of Apple Inc.</p></html>"
        text = ur.html_to_text(raw)
        self.assertIn("Tim Cook", text)
        self.assertNotIn("<p>", text)
        self.assertNotIn("x()", text)


class TestLexical(unittest.TestCase):
    def test_picks_relevant_sentence(self):
        doc = (
            "page\n"
            "Apples grow on trees in orchards worldwide.\n"
            "Tim Cook is the CEO of Apple Inc.\n"
            "Source: https://example.com"
        )
        ans = ur.lexical_answer("Who is the CEO of Apple?", doc)
        self.assertIsNotNone(ans)
        self.assertIn("Tim Cook", ans)

    def test_who_prefers_named_person(self):
        doc = (
            "page\n"
            "Apple designs and sells consumer electronics worldwide today.\n"
            "Tim Cook is the CEO of Apple Inc.\n"
            "Source: https://example.com"
        )
        ans = ur.lexical_answer("Who is the CEO according to the page?", doc)
        self.assertIn("Tim Cook", ans or "")

    def test_off_topic_page_no_lexical(self):
        doc = (
            "page\n"
            "The weather in Paris is mild this week according to locals.\n"
            "Source: https://example.com"
        )
        self.assertIsNone(ur.lexical_answer("Who is the CEO of Apple?", doc))


class TestGroundedReply(unittest.TestCase):
    def test_rejects_ungrounded_summary(self):
        doc = "page\nTim Cook is the CEO of Apple Inc.\nSource: https://example.com"
        reply = ur.finalize_reply(
            question="Who is the CEO of Apple?",
            doc=doc,
            core="Tim Cook",
            summary="Tim Cook founded Microsoft in 1975.",
        )
        self.assertIn("Tim Cook", reply or "")
        self.assertNotIn("Microsoft", reply or "")

    def test_off_topic_abstains(self):
        doc = "page\nThe weather in Paris is mild.\nSource: https://example.com"
        reply = ur.finalize_reply(
            question="Who is the CEO of Apple?",
            doc=doc,
            core=None,
            summary=None,
        )
        self.assertIsNone(reply)


class TestRunOffline(unittest.TestCase):
    def test_fetch_and_answer(self):
        html = (
            b"<html><body><p>The server listens on port 8765 for local clients.</p>"
            b"<p>Coffee is served in the hall.</p></body></html>"
        )
        got = ur.run(
            "What port does the server listen on according to https://example.com/p?",
            fetch=lambda u, **kw: html,
            live_answer=False,
        )
        self.assertEqual(got.status, "ok")
        self.assertEqual(got.url, "https://example.com/p")
        self.assertIn("8765", got.answer or "")

    def test_no_url_abstains(self):
        got = ur.run("Who is the CEO of Apple?", live_answer=False)
        self.assertEqual(got.status, "no_url")

    def test_fetch_error(self):
        def boom(u, **kw):
            raise OSError("down")

        got = ur.run(
            "Read https://example.com/x",
            fetch=boom,
            live_answer=False,
        )
        self.assertEqual(got.status, "fetch_error")


if __name__ == "__main__":
    unittest.main()
