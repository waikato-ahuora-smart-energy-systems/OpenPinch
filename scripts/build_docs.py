"""Build the local Sphinx documentation tree."""

from __future__ import annotations

import subprocess
from pathlib import Path
from shutil import which


def main() -> int:
    """Build the HTML documentation tree into ``docs/_build/html``."""
    repo_root = Path(__file__).resolve().parents[1]
    build_dir = repo_root / "docs" / "_build" / "html"
    build_dir.parent.mkdir(parents=True, exist_ok=True)
    uv_bin = which("uv") or "/opt/homebrew/bin/uv"

    proc = subprocess.run(
        [
            uv_bin,
            "run",
            "--group",
            "dev",
            "python",
            "-m",
            "sphinx",
            "-b",
            "html",
            "docs",
            str(build_dir),
        ],
        cwd=repo_root,
    )
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
