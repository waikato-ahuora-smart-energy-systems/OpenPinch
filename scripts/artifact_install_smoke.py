"""Smoke-check an installed OpenPinch wheel without importing the checkout."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Checkout path that the installed package must not resolve from.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Exercise the installed package, CLI, and packaged resources."""
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()

    import OpenPinch
    from OpenPinch import list_notebooks, list_sample_cases, read_sample_case

    package_path = Path(OpenPinch.__file__).resolve()
    if package_path.is_relative_to(repo_root):
        raise AssertionError(f"Imported OpenPinch from checkout: {package_path}")

    notebooks = list_notebooks()
    sample_cases = list_sample_cases()
    if not notebooks or not sample_cases:
        raise AssertionError("Installed wheel is missing packaged resources.")
    if not read_sample_case("basic_pinch.json"):
        raise AssertionError("Installed wheel could not read basic_pinch.json.")

    subprocess.run(
        [sys.executable, "-m", "OpenPinch", "notebook", "--help"],
        check=True,
        cwd=repo_root.parent,
    )
    print(f"Installed artifact smoke passed for {package_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
