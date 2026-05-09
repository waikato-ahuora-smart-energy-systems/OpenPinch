"""Format the repository with Ruff."""

from __future__ import annotations

import subprocess
from pathlib import Path
from shutil import which


def main() -> int:
    """Run the repository formatter through the local ``uv`` toolchain."""
    repo_root = Path(__file__).resolve().parents[1]
    uv_bin = which("uv") or "/opt/homebrew/bin/uv"

    proc = subprocess.run(
        [
            uv_bin,
            "run",
            "--group",
            "dev",
            "ruff",
            "format",
            ".",
        ],
        cwd=repo_root,
    )
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
