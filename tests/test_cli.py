"""CLI ask smoke (offline refuse path)."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

from deku import cli


class TestAskCli(unittest.TestCase):
    def test_math_refuse_prints_message(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = cli.ask("What is 2+2?", live=False)
        self.assertEqual(code, 0)
        self.assertIn("math", buf.getvalue().lower())

    def test_json_shape(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = cli.ask("hello there", live=False, as_json=True)
        self.assertEqual(code, 0)
        text = buf.getvalue()
        self.assertIn('"tool": "refuse"', text)
        self.assertIn('"envelope": "slim"', text)
        self.assertNotIn('"detail"', text)

    def test_json_full_includes_detail(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = cli.ask("What is 2+2?", live=False, json_full=True)
        self.assertEqual(code, 0)
        text = buf.getvalue()
        self.assertIn('"envelope": "full"', text)
        self.assertIn('"detail"', text)


if __name__ == "__main__":
    unittest.main()
