"""Tests for deku.doctor (no network)."""

from __future__ import annotations

import unittest
from unittest import mock

from deku import doctor


class TestDoctor(unittest.TestCase):
    def test_main_returns_int(self) -> None:
        with mock.patch.object(doctor, "print"):
            code = doctor.main()
        self.assertIn(code, (0, 1))


if __name__ == "__main__":
    unittest.main()
