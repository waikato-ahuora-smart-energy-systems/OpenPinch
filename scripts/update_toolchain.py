"""Update the project Python runtime and lockfile to the latest compatible versions."""

from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path
from shutil import which


def _run(cmd: list[str], *, cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=cwd)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def _read_python_version(repo_root: Path) -> str:
    """Read the pinned Python version from ``pyproject.toml``."""
    with (repo_root / "pyproject.toml").open("rb") as handle:
        data = tomllib.load(handle)

    requires_python = data["project"]["requires-python"]
    if not requires_python.startswith(">="):
        raise ValueError(
            "Expected project.requires-python to start with '>=' for toolchain sync."
        )
    return requires_python.removeprefix(">=")


def _python_minor_from_version(version: str) -> str:
    """Return the ``major.minor`` selector accepted by ``uv python install``."""
    parts = version.split(".")
    if len(parts) < 2 or not all(part.isdigit() for part in parts[:2]):
        raise ValueError(f"Expected a Python version like '3.14.2', got {version!r}.")
    return ".".join(parts[:2])


def _read_python_minor(repo_root: Path) -> str:
    """Read the Python minor selector from ``pyproject.toml``."""
    return _python_minor_from_version(_read_python_version(repo_root))


def main() -> int:
    """Refresh the pinned Python toolchain, lockfile, and synced dev env."""
    repo_root = Path(__file__).resolve().parents[1]
    uv_bin = which("uv") or "/opt/homebrew/bin/uv"
    python_version = _read_python_version(repo_root)
    python_minor = _read_python_minor(repo_root)

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
            python_version,
        ],
        cwd=repo_root,
    )
    _run(
        [
            uv_bin,
            "lock",
            "--upgrade",
            "--python",
            python_version,
        ],
        cwd=repo_root,
    )
    _run(
        [
            uv_bin,
            "sync",
            "--python",
            python_version,
            "--all-extras",
            "--group",
            "dev",
        ],
        cwd=repo_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
