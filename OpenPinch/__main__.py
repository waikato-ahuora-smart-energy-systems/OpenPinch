"""Command-line helpers for packaged OpenPinch notebook assets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .resources import (
    copy_notebook,
    list_notebooks,
)

_COMMANDS = {"notebook"}


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="openpinch",
        description="Copy packaged OpenPinch example notebooks.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Raise full tracebacks instead of concise CLI errors.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    notebook_parser = subparsers.add_parser(
        "notebook",
        help="Copy one or more packaged example notebooks.",
    )
    notebook_parser.add_argument(
        "--name",
        choices=list_notebooks(),
        default=None,
        help="Notebook to copy. Omit to copy the full series.",
    )
    notebook_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("."),
        help="Destination filepath or directory.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute the OpenPinch CLI."""
    parser = build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)

    args = parser.parse_args(argv)
    try:
        if args.command == "notebook":
            return _notebook_command(args)
    except Exception as exc:
        if args.debug:
            raise
        parser.exit(status=1, message=f"OpenPinch error: {exc}\n")
    return 0


def _notebook_command(args: argparse.Namespace) -> int:
    if args.name is not None:
        destination = copy_notebook(args.name, args.output)
        print(f"Copied notebook to {destination}")
        return 0

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in list_notebooks():
        copy_notebook(name, output_dir / name)
    print(f"Copied {len(list_notebooks())} notebook(s) to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
