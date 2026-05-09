"""Update the project Python runtime and lockfile to the latest compatible versions."""

from __future__ import annotations

import subprocess
from pathlib import Path
from shutil import which


def _run(cmd: list[str], *, cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=cwd)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    uv_bin = which("uv") or "/opt/homebrew/bin/uv"
    python_minor = "3.14"

    _run(
        [
            uv_bin,
            "python",
            "install",
            "--managed-python",
            "--upgrade",
            python_minor,
        ],
        cwd=repo_root,
    )
    _run(
        [
            uv_bin,
            "python",
            "pin",
            python_minor,
        ],
        cwd=repo_root,
    )
    _run(
        [
            uv_bin,
            "lock",
            "--upgrade",
            "--python",
            python_minor,
        ],
        cwd=repo_root,
    )
    _run(
        [
            uv_bin,
            "sync",
            "--python",
            python_minor,
            "--all-extras",
            "--group",
            "dev",
        ],
        cwd=repo_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
