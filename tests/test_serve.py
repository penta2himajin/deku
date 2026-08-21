"""Tests for GGUF ensure + llama-server argv helpers (no real download)."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from deku import serve


class TestServeHelpers(unittest.TestCase):
    def test_default_gguf_name(self) -> None:
        self.assertEqual(serve.DEFAULT_GGUF, "MiniCPM5-1B-Q4_K_M.gguf")
        self.assertEqual(serve.DEFAULT_HF_REPO, "openbmb/MiniCPM5-1B-GGUF")

    def test_model_path_under_cache(self) -> None:
        cache = Path("/tmp/deku-models-test")
        self.assertEqual(
            serve.model_path(cache),
            cache / serve.DEFAULT_GGUF,
        )

    def test_ensure_gguf_skips_download_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            target = cache / serve.DEFAULT_GGUF
            target.write_bytes(b"fake-gguf")
            downloader = mock.Mock()
            path = serve.ensure_gguf(cache_dir=cache, download=downloader)
            self.assertEqual(path, target)
            downloader.assert_not_called()

    def test_ensure_gguf_downloads_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)

            def fake_download(repo: str, filename: str, dest: Path) -> None:
                self.assertEqual(repo, serve.DEFAULT_HF_REPO)
                self.assertEqual(filename, serve.DEFAULT_GGUF)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(b"downloaded")

            path = serve.ensure_gguf(cache_dir=cache, download=fake_download)
            self.assertTrue(path.is_file())
            self.assertEqual(path.read_bytes(), b"downloaded")

    def test_llama_server_argv_includes_jinja(self) -> None:
        model = Path("/models/MiniCPM5-1B-Q4_K_M.gguf")
        argv = serve.llama_server_argv(model, host="127.0.0.1", port=8080)
        self.assertEqual(argv[0], "llama-server")
        self.assertIn("--jinja", argv)
        self.assertIn("-m", argv)
        self.assertIn(str(model), argv)
        self.assertIn("--host", argv)
        self.assertIn("127.0.0.1", argv)
        self.assertIn("--port", argv)
        self.assertIn("8080", argv)

    def test_require_llama_server_missing(self) -> None:
        with mock.patch.object(shutil, "which", return_value=None):
            with self.assertRaises(serve.ServeError) as ctx:
                serve.require_llama_server()
            self.assertIn("llama-server", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
