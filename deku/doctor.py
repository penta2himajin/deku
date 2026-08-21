"""Environment doctor for local / OSS checkouts."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

from deku import __version__
from deku import serve


def _ok(label: str, detail: str) -> None:
    print(f"ok   {label}: {detail}")


def _warn(label: str, detail: str) -> None:
    print(f"warn {label}: {detail}")


def _fail(label: str, detail: str) -> None:
    print(f"FAIL {label}: {detail}")


def main() -> int:
    print(f"deku {__version__} doctor")
    failed = 0

    py = sys.version.split()[0]
    if sys.version_info >= (3, 11):
        _ok("python", f"{py} ({sys.executable})")
    else:
        _fail("python", f"{py} (need >= 3.11)")
        failed += 1

    if shutil.which("uv"):
        _ok("uv", shutil.which("uv") or "")
    else:
        _warn("uv", "not on PATH (optional; `mise install` provides it)")

    if shutil.which("mise"):
        _ok("mise", shutil.which("mise") or "")
    else:
        _warn("mise", "not on PATH (optional; see https://mise.jdx.dev)")

    llama = shutil.which("llama-server")
    if llama:
        _ok("llama-server", llama)
    else:
        _fail(
            "llama-server",
            "not on PATH — `brew install llama.cpp` (macOS) or build llama.cpp",
        )
        failed += 1

    if importlib.util.find_spec("huggingface_hub") is not None:
        _ok("huggingface_hub", "importable (GGUF download ready)")
    else:
        _fail("huggingface_hub", "not installed — run `mise run sync` / `uv sync`")
        failed += 1

    gguf = serve.model_path()
    if gguf.is_file():
        size_mb = gguf.stat().st_size / (1024 * 1024)
        _ok("gguf", f"{gguf} ({size_mb:.0f} MiB)")
    else:
        _warn("gguf", f"missing at {gguf} (will download on `mise run serve`)")

    root = Path(__file__).resolve().parents[1]
    if (root / "uv.lock").is_file():
        _ok("uv.lock", "present")
    else:
        _warn("uv.lock", "missing — run `uv lock` and commit for reproducible installs")

    if failed:
        print(f"\n{failed} required check(s) failed.")
        return 1
    print("\nReady for `mise run serve` (or `uv run deku-serve`).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
