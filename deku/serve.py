"""GGUF ensure + llama-server launch helpers.

Agent code must not import this for inference — only bin/deku-serve / CLI
convenience uses these functions. The harness talks HTTP via deku.llm only.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

DEFAULT_HF_REPO = "openbmb/MiniCPM5-1B-GGUF"
DEFAULT_GGUF = "MiniCPM5-1B-Q4_K_M.gguf"
DEFAULT_CACHE = Path(
    os.environ.get("DEKU_MODEL_DIR", Path.home() / ".cache" / "deku" / "models")
)
DEFAULT_HOST = os.environ.get("DEKU_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("DEKU_PORT", "8080"))

DownloadFn = Callable[[str, str, Path], None]


class ServeError(RuntimeError):
    """Raised when llama-server or the GGUF cannot be prepared."""


def model_path(cache_dir: Path | None = None) -> Path:
    return (cache_dir or DEFAULT_CACHE) / DEFAULT_GGUF


def _default_download(repo: str, filename: str, dest: Path) -> None:
    """Download via huggingface_hub (declared dependency), with CLI fallback."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        hf_hub_download = None  # type: ignore[assignment]

    if hf_hub_download is not None:
        downloaded = Path(
            hf_hub_download(
                repo_id=repo,
                filename=filename,
                local_dir=str(dest.parent),
            )
        )
        if downloaded != dest and downloaded.is_file():
            shutil.copy2(downloaded, dest)
        if dest.is_file():
            return

    cli = shutil.which("hf") or shutil.which("huggingface-cli")
    if not cli:
        raise ServeError(
            "huggingface_hub is not importable and neither `hf` nor "
            "`huggingface-cli` is on PATH; run `mise run sync` (or "
            f"`uv sync`) or place the GGUF at {dest}"
        )
    cmd = [
        cli,
        "download",
        repo,
        filename,
        "--local-dir",
        str(dest.parent),
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise ServeError(f"download failed: {' '.join(cmd)}") from e
    if not dest.is_file():
        found = list(dest.parent.rglob(filename))
        if not found:
            raise ServeError(f"download finished but {dest} is missing")
        if found[0] != dest:
            shutil.copy2(found[0], dest)


def ensure_gguf(
    *,
    cache_dir: Path | None = None,
    download: DownloadFn | None = None,
) -> Path:
    """Return path to the default GGUF, downloading once if needed."""
    path = model_path(cache_dir)
    if path.is_file():
        return path
    (download or _default_download)(DEFAULT_HF_REPO, DEFAULT_GGUF, path)
    if not path.is_file():
        raise ServeError(f"GGUF still missing after download: {path}")
    return path


def require_llama_server() -> str:
    path = shutil.which("llama-server")
    if not path:
        raise ServeError(
            "llama-server not found on PATH; install llama.cpp "
            "(e.g. `brew install llama.cpp`) and retry"
        )
    return path


def llama_server_argv(
    model: Path,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    binary: str = "llama-server",
) -> list[str]:
    return [
        binary,
        "-m",
        str(model),
        "--host",
        host,
        "--port",
        str(port),
        "--jinja",
    ]


def main(argv: list[str] | None = None) -> None:
    """CLI entry: ensure GGUF, then exec llama-server (replaces this process)."""
    _ = argv  # reserved for future flags
    try:
        binary = require_llama_server()
        gguf = ensure_gguf()
        cmd = llama_server_argv(
            gguf, host=DEFAULT_HOST, port=DEFAULT_PORT, binary=binary
        )
    except ServeError as e:
        print(f"deku-serve: {e}", file=sys.stderr)
        raise SystemExit(1) from e
    print(f"deku-serve: exec {' '.join(cmd)}", flush=True)
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
