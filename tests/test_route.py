import sys
import unittest
from pathlib import Path

from deku import route as rt
from deku.route_cases import ROUTE_CASES


class TestGoldCases(unittest.TestCase):
    def test_rule_matches_all_gold(self):
        misses = []
        for question, want, kind in ROUTE_CASES:
            got = rt.rule_route(question).tool
            if got != want:
                misses.append((kind, want, got, question))
        self.assertEqual(misses, [], msg=misses)

    def test_hard_route_covers_hard_cases(self):
        for question, want, kind in ROUTE_CASES:
            if kind != "hard":
                continue
            hard = rt.hard_route(question)
            self.assertIsNotNone(hard, msg=question)
            self.assertEqual(hard.tool, want, msg=question)

    def test_hard_route_skips_soft_web(self):
        self.assertIsNone(rt.hard_route("Who is the CEO of Apple?"))


class TestExtractUrl(unittest.TestCase):
    def test_https_url(self):
        u = rt.extract_url("Summarize https://example.com/page.html for me")
        self.assertEqual(u, "https://example.com/page.html")

    def test_url_with_trailing_punct(self):
        u = rt.extract_url("See https://en.wikipedia.org/wiki/Apple_Inc.")
        self.assertEqual(u, "https://en.wikipedia.org/wiki/Apple_Inc")

    def test_no_url(self):
        self.assertIsNone(rt.extract_url("Who is the CEO of Apple?"))


class TestRuleRoute(unittest.TestCase):
    def test_url_forces_url_read(self):
        got = rt.rule_route("What does https://example.com/a say about ports?")
        self.assertEqual(got.tool, "url_read")
        self.assertEqual(got.url, "https://example.com/a")

    def test_math_refuses(self):
        self.assertEqual(rt.rule_route("What is 2+2?").tool, "refuse")

    def test_web_fact(self):
        self.assertEqual(rt.rule_route("Who is the CEO of Apple?").tool, "web_search")

    def test_dir_ident(self):
        self.assertEqual(rt.rule_route("What is the PREFILL string?").tool, "dir_search")

    def test_acronym_is_not_dir_ident(self):
        self.assertEqual(
            rt.rule_route("Who is the current CEO of LVMH?").tool, "web_search"
        )
        self.assertEqual(
            rt.rule_route("What is NASA?").tool, "web_search"
        )

    def test_mixed_web_dir_plans(self):
        got = rt.rule_route(
            "Who is the CEO of Apple and what is the PREFILL string?"
        )
        self.assertEqual(got.tool, "multi_hop")
        self.assertEqual(got.detail.get("plan_id"), "web+dir")

    def test_dir_overview(self):
        self.assertEqual(rt.rule_route("What is this project about?").tool, "dir_search")

    def test_git_log_cue(self):
        self.assertEqual(rt.rule_route("What changed in the last commit?").tool, "git_search")

    def test_diff_cue(self):
        self.assertEqual(
            rt.rule_route("What is in the unstaged diff for extract.py?").tool,
            "diff_search",
        )

    def test_chitchat_refuses(self):
        self.assertEqual(rt.rule_route("hello there").tool, "refuse")


class TestParseTool(unittest.TestCase):
    def test_known_tools(self):
        self.assertEqual(rt.parse_tool("TOOL: url_read"), "url_read")
        self.assertEqual(rt.parse_tool("git_search\n"), "git_search")

    def test_unknown_is_refuse(self):
        self.assertEqual(rt.parse_tool("maybe search the web"), "refuse")


class TestDispatch(unittest.TestCase):
    def test_refuse_short_circuits(self):
        got = rt.dispatch("What is 2+2?", router="rule")
        self.assertEqual(got.tool, "refuse")
        self.assertEqual(got.status, "refused")
        self.assertTrue(got.answer)

    def test_url_read_dispatches(self):
        html = b"<html><title>T</title><body><p>Tim Cook is the CEO of Apple.</p></body></html>"
        got = rt.dispatch(
            "Who is the CEO according to https://example.com/apple?",
            router="rule",
            url_fetch=lambda u, **kw: html,
            seed=0,
            live_answer=False,
        )
        self.assertEqual(got.tool, "url_read")
        self.assertEqual(got.status, "ok")
        self.assertIn("Tim Cook", got.answer or "")


if __name__ == "__main__":
    unittest.main()
