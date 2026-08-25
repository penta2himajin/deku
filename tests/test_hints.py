"""Machine next_hint mapping for parent agents."""

from __future__ import annotations

import unittest
from unittest import mock

from deku import hints
from deku import route as rt


class TestHintMapping(unittest.TestCase):
    def test_ok_is_none(self):
        self.assertEqual(
            hints.next_hint_for(status="ok"),
            {"action": "none"},
        )

    def test_refuse_asks_in_scope(self):
        h = hints.next_hint_for(status="refused", reason="math")
        self.assertEqual(h["action"], "ask_in_scope_fact")
        self.assertEqual(h["reason"], "math")

    def test_partial_retries_failed(self):
        h = hints.next_hint_for(
            status="partial",
            failed=[{"query": "what is PREFILL", "tool": "dir_search"}],
        )
        self.assertEqual(h["action"], "retry_failed_clauses")
        self.assertEqual(h["clauses"], ["what is PREFILL"])

    def test_abstain_reasons(self):
        cases = [
            ("identifiers_missing_from_top_hit", "name_symbol_or_path"),
            ("weak_or_off_topic_hit", "narrow_or_rephrase"),
            ("no_prose_lead", "narrow_or_rephrase"),
            ("no_diff", "check_workdir"),
            ("no_commits_for_path", "provide_path"),
            ("no_url", "provide_url"),
            ("fetch_error", "retry_or_other_url"),
            ("no_grounded_core", "abstain_or_narrow"),
        ]
        for abstain, action in cases:
            with self.subTest(abstain=abstain):
                h = hints.next_hint_for(
                    status="cannot_answer",
                    tool="dir_search",
                    abstain_reason=abstain,
                )
                self.assertEqual(h["action"], action)
                self.assertEqual(h["abstain_reason"], abstain)
                self.assertEqual(h["failed_tool"], "dir_search")

    def test_skipped_offline(self):
        h = hints.next_hint_for(status="skipped_offline", tool="web_search")
        self.assertEqual(h["action"], "enable_live_or_serve")
        self.assertEqual(h["failed_tool"], "web_search")


class TestDispatchAttachesHint(unittest.TestCase):
    def test_dir_cannot_answer_gets_hint(self):
        fake = mock.Mock()
        fake.status = "cannot_answer"
        fake.answer = "I cannot answer from the available evidence."
        fake.query = "Where is PREFILL set?"
        fake.document = ""
        fake.hits = []
        fake.detail = {"abstain_reason": "identifiers_missing_from_top_hit"}

        with mock.patch("deku.dir_search.run", return_value=fake):
            got = rt.dispatch(
                "Where is PREFILL set?",
                live_answer=False,
                audience="agent",
            )
        self.assertEqual(got.tool, "dir_search")
        self.assertEqual(got.status, "cannot_answer")
        h = got.detail["next_hint"]
        self.assertEqual(h["action"], "name_symbol_or_path")
        env = rt.envelope(got)
        self.assertEqual(env["next_hint"]["action"], "name_symbol_or_path")

    def test_ok_dir_hint_none(self):
        got = rt.dispatch(
            "What is the PREFILL string?",
            live_answer=False,
            audience="agent",
            root=".",
        )
        self.assertEqual(got.status, "ok")
        self.assertEqual(got.detail.get("next_hint", {}).get("action"), "none")


if __name__ == "__main__":
    unittest.main()
