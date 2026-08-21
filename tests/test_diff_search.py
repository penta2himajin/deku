import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from deku import diff_search as dfs
from deku import route as rt


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _make_dirty_repo() -> Path:
    root = Path(tempfile.mkdtemp(prefix="deku-diff-"))
    _git(root, "init")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "Test User")
    (root / "extract.py").write_text("PREFILL = 'old'\nTEMP = 0.1\n")
    _git(root, "add", "extract.py")
    _git(root, "commit", "-m", "init")
    (root / "extract.py").write_text("PREFILL = 'ANSWER: '\nTEMP = 0.1\n")
    return root


class TestDiffScope(unittest.TestCase):
    def test_unstaged_scope(self):
        self.assertEqual(
            dfs.diff_scope("What is in the unstaged diff for extract.py?"),
            "unstaged",
        )

    def test_staged_scope(self):
        self.assertEqual(dfs.diff_scope("Show the staged diff"), "staged")

    def test_default_working(self):
        self.assertEqual(dfs.diff_scope("What changed in extract.py?"), "working")


class TestRun(unittest.TestCase):
    def test_unstaged_prefills(self):
        root = _make_dirty_repo()
        got = dfs.run(
            "What is in the unstaged diff for extract.py?",
            root=root,
            live_answer=False,
        )
        self.assertEqual(got.status, "ok")
        self.assertIn("PREFILL", got.answer or "")
        self.assertIn("ANSWER", got.answer or "")

    def test_clean_tree_abstains(self):
        root = _make_dirty_repo()
        _git(root, "checkout", "--", "extract.py")
        got = dfs.run("What is in the unstaged diff?", root=root, live_answer=False)
        self.assertIn(got.status, ("no_diff", "cannot_answer"))

    def test_ident_line_preferred(self):
        hits = [{
            "path": "extract.py",
            "snippet": (
                "diff --git a/extract.py b/extract.py\n"
                "--- a/extract.py\n+++ b/extract.py\n"
                "-PREFILL = 'old'\n"
                "+PREFILL = 'ANSWER: '\n"
                "+    if dec.tool == \"diff_search\":\n"
            ),
            "url": "diff:extract.py",
        }]
        ans = dfs.lexical_answer("What changed about PREFILL?", hits)
        self.assertIsNotNone(ans)
        self.assertIn("PREFILL", ans)
        self.assertIn("ANSWER", ans)
        self.assertNotIn("diff_search", ans)

    def test_open_diff_returns_hunk_block(self):
        hits = [{
            "path": "extract.py",
            "snippet": (
                "diff --git a/extract.py b/extract.py\n"
                "--- a/extract.py\n+++ b/extract.py\n"
                "-PREFILL = 'old'\n"
                "+PREFILL = 'ANSWER: '\n"
            ),
            "url": "diff:extract.py",
        }]
        ans = dfs.lexical_answer("What is in the unstaged diff?", hits)
        self.assertIsNotNone(ans)
        self.assertIn("extract.py", ans)
        self.assertIn("PREFILL", ans)


class TestRouteDispatch(unittest.TestCase):
    def test_diff_dispatch(self):
        root = _make_dirty_repo()
        got = rt.dispatch(
            "What is in the unstaged diff for extract.py?",
            router="rule",
            root=str(root),
            live_answer=False,
        )
        self.assertEqual(got.tool, "diff_search")
        self.assertEqual(got.status, "ok")
        self.assertIn("PREFILL", got.answer or "")


if __name__ == "__main__":
    unittest.main()
