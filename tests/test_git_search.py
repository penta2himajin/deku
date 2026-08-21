import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from deku import git_search as gs
from deku import route as rt


def _git(cwd: Path, *args: str) -> str:
    r = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return r.stdout


def _make_repo() -> Path:
    root = Path(tempfile.mkdtemp(prefix="deku-git-"))
    _git(root, "init")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "Test User")
    (root / "readme.txt").write_text("hello harness\n")
    _git(root, "add", "readme.txt")
    _git(root, "commit", "-m", "feat: add readme for harness")
    (root / "readme.txt").write_text("hello harness\nsecond line\n")
    _git(root, "add", "readme.txt")
    _git(root, "commit", "-m", "fix: extend readme")
    return root


class TestGitReadonly(unittest.TestCase):
    def test_rejects_write_verbs(self):
        with self.assertRaises(ValueError):
            gs.git_run(Path("."), ["commit", "-m", "x"])


class TestIndexLog(unittest.TestCase):
    def test_log_hits_include_subjects(self):
        root = _make_repo()
        hits = gs.index_log(root, limit=5)
        self.assertGreaterEqual(len(hits), 2)
        subjects = " ".join(h["snippet"] for h in hits)
        self.assertIn("feat: add readme", subjects)
        self.assertIn("fix: extend readme", subjects)


class TestRun(unittest.TestCase):
    def test_last_commit_message(self):
        root = _make_repo()
        got = gs.run("What is the last commit message?", root=root, live_answer=False)
        self.assertEqual(got.status, "ok")
        self.assertIn("fix: extend readme", got.answer or "")

    def test_who_authored_last(self):
        root = _make_repo()
        got = gs.run("Who authored the last commit?", root=root, live_answer=False)
        self.assertEqual(got.status, "ok")
        self.assertIn("Test User", got.answer or "")

    def test_files_changed_in_last_commit(self):
        root = _make_repo()
        got = gs.run(
            "What files changed in the last commit?",
            root=root,
            live_answer=False,
        )
        self.assertEqual(got.status, "ok")
        self.assertIn("readme.txt", got.answer or "")

    def test_compose_rejects_ungrounded(self):
        doc = "git show\nAuthor: Test User\nfix: extend readme\n readme.txt | 1 +\nSource: git:show"
        reply = gs.finalize_reply(
            question="Who authored the last commit?",
            doc=doc,
            hits=[{"author": "Test User", "sha": "abc", "subject": "fix: extend readme"}],
            show_text=doc,
            core="Test User",
            summary="Test User founded Apple in 1976.",
        )
        self.assertIn("Test User", reply or "")
        self.assertNotIn("Apple", reply or "")


class TestRouteDispatch(unittest.TestCase):
    def test_git_dispatch(self):
        root = _make_repo()
        got = rt.dispatch(
            "What is the last commit message?",
            router="rule",
            root=str(root),
            live_answer=False,
        )
        self.assertEqual(got.tool, "git_search")
        self.assertEqual(got.status, "ok")
        self.assertIn("fix: extend readme", got.answer or "")

    def test_path_last_commit_message(self):
        root = _make_repo()
        (root / "other.txt").write_text("x\n")
        _git(root, "add", "other.txt")
        _git(root, "commit", "-m", "chore: add other")
        # Touch readme again so path history differs from HEAD.
        (root / "readme.txt").write_text("hello harness\nsecond line\nthird\n")
        _git(root, "add", "readme.txt")
        _git(root, "commit", "-m", "docs: touch readme again")
        got = gs.run(
            "What is the commit message of the last commit that changed readme.txt?",
            root=root,
            live_answer=False,
        )
        self.assertEqual(got.status, "ok")
        self.assertEqual(got.detail.get("path"), "readme.txt")
        self.assertIn("docs: touch readme again", got.answer or "")
        self.assertIn("readme.txt", got.answer or "")
        # HEAD for other.txt alone would be chore — path filter must win.
        got2 = gs.run(
            "What is the commit message of the last commit that changed other.txt?",
            root=root,
            live_answer=False,
        )
        self.assertIn("chore: add other", got2.answer or "")


if __name__ == "__main__":
    unittest.main()
