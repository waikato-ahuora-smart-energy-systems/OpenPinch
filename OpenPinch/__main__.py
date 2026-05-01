"""Command-line entry point for running an OpenPinch analysis."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="openpinch",
        description="Run an OpenPinch analysis and export the results workbook.",
    )
    parser.add_argument(
        "problem_path",
        type=Path,
        help="Path to the source workbook or JSON problem file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("results"),
        help="Directory for exported results (default: ./results).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute the OpenPinch CLI."""
    from . import PinchProblem

    args = build_parser().parse_args(argv)

    problem = PinchProblem()
    problem.load(args.problem_path)
    problem.target()
    problem.export_to_Excel(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
