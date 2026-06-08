"""Build the local Sphinx documentation tree."""

from __future__ import annotations

import argparse
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path
from shutil import which


def _build_command(build_dir: Path) -> list[str] | None:
    """Return the preferred command for building the docs in this environment."""
    sphinx_args = [
        "-b",
        "html",
        "--fail-on-warning",
        "--keep-going",
        "docs",
        str(build_dir),
    ]
    if find_spec("sphinx") is not None:
        return [
            sys.executable,
            "-m",
            "sphinx",
            *sphinx_args,
        ]

    uv_bin = which("uv")
    if uv_bin is None:
        return None

    return [
        uv_bin,
        "run",
        "--group",
        "dev",
        "python",
        "-m",
        "sphinx",
        *sphinx_args,
    ]


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for docs builds."""
    parser = argparse.ArgumentParser(
        prog="build_docs.py",
        description="Build the OpenPinch HTML documentation tree.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination directory for the built HTML tree.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Build the HTML documentation tree into ``docs/_build/html`` by default."""
    repo_root = Path(__file__).resolve().parents[1]
    parser = build_parser()
    args = parser.parse_args(argv)

    build_dir = (
        args.output_dir
        if args.output_dir is not None
        else repo_root / "docs" / "_build" / "html"
    )
    build_dir.parent.mkdir(parents=True, exist_ok=True)
    command = _build_command(build_dir)
    if command is None:
        print(
            "Unable to build docs: install 'sphinx' in the active interpreter or "
            "make 'uv' available on PATH.",
            file=sys.stderr,
        )
        return 1

    proc = subprocess.run(command, cwd=repo_root)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
