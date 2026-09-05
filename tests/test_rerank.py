"""HTTP MiniCPM-Reranker client (no torch in agent core)."""

from __future__ import annotations

import json
import os
import unittest
from unittest import mock

from deku import rerank as rr
from deku import web_search as ws


class TestRerankClient(unittest.TestCase):
    def test_disabled_when_url_empty(self):
        with mock.patch.dict(os.environ, {"DEKU_RERANK_URL": ""}, clear=False):
            self.assertFalse(rr.rerank_enabled())
            self.assertIsNone(
                rr.try_rerank_hits("Who is the CEO of Apple?", [
                    {"title": "A", "snippet": "a", "url": "u1"},
                    {"title": "B", "snippet": "b", "url": "u2"},
                ], k=2)
            )

    def test_orders_by_remote_scores(self):
        hits = [
            {"title": "Wrong", "snippet": "unrelated", "url": "w"},
            {"title": "Right", "snippet": "CEO of Apple", "url": "r"},
        ]

        def fake_urlopen(req, timeout=0):  # noqa: ARG001
            body = json.loads(req.data.decode())
            self.assertEqual(body["query"], "Who is the CEO of Apple?")
            self.assertEqual(len(body["documents"]), 2)
            resp = mock.MagicMock()
            resp.__enter__ = lambda s: s
            resp.__exit__ = mock.Mock(return_value=False)
            resp.read = lambda: json.dumps({"scores": [0.1, 0.9]}).encode()
            return resp

        with mock.patch.dict(
            os.environ, {"DEKU_RERANK_URL": "http://127.0.0.1:8091"}, clear=False
        ):
            with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
                scored = rr.try_rerank_hits(
                    "Who is the CEO of Apple?", hits, k=2
                )
        self.assertIsNotNone(scored)
        assert scored is not None
        self.assertEqual(scored[0][1]["url"], "r")
        self.assertGreater(scored[0][0], scored[1][0])

    def test_fallback_on_http_error(self):
        hits = [
            {"title": "Apple Inc.", "snippet": "Tim Cook is the CEO of Apple Inc.", "url": "corp"},
            {"title": "Apple", "snippet": "An apple is a fruit.", "url": "fruit"},
        ]

        with mock.patch.dict(
            os.environ, {"DEKU_RERANK_URL": "http://127.0.0.1:8091"}, clear=False
        ):
            with mock.patch(
                "urllib.request.urlopen", side_effect=OSError("down")
            ):
                got = ws.rank_hits("Who is the CEO of Apple?", hits, k=2)
        # Lexical fallback still prefers corp.
        self.assertEqual(got[0]["url"], "corp")

    def test_rank_hits_uses_rerank_when_up(self):
        hits = [
            {"title": "Apple", "snippet": "An apple is a fruit.", "url": "fruit"},
            {"title": "Apple Inc.", "snippet": "Tim Cook is the CEO.", "url": "corp"},
        ]

        def fake_urlopen(req, timeout=0):  # noqa: ARG001
            resp = mock.MagicMock()
            resp.__enter__ = lambda s: s
            resp.__exit__ = mock.Mock(return_value=False)
            # Prefer fruit (wrong) to prove rerank overrides lexical.
            resp.read = lambda: json.dumps({"scores": [0.99, 0.01]}).encode()
            return resp

        with mock.patch.dict(
            os.environ, {"DEKU_RERANK_URL": "http://127.0.0.1:8091"}, clear=False
        ):
            with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
                got = ws.rank_hits("Who is the CEO of Apple?", hits, k=2)
        self.assertEqual(got[0]["url"], "fruit")


if __name__ == "__main__":
    unittest.main()
