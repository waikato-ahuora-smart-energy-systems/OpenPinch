"""Command-line entry points for running OpenPinch workflows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .resources import (
    copy_notebook,
    copy_sample_case,
    list_notebooks,
    list_sample_cases,
)

_COMMANDS = {"run", "graph", "validate", "sample", "notebook"}


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="openpinch",
        description="Run OpenPinch analyses, export graphs, and copy learning assets.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Raise full tracebacks instead of concise CLI errors.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Run a pinch analysis and optionally export results.",
    )
    _add_problem_input_argument(run_parser)
    run_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Directory for exported Excel results.",
    )
    run_parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path for the serialized TargetOutput JSON.",
    )
    run_parser.add_argument(
        "--graph-output",
        type=Path,
        default=None,
        help="Optional directory for exported graph HTML files.",
    )
    run_parser.add_argument(
        "--detailed-summary",
        action="store_true",
        help="Print the wide summary table instead of the compact one.",
    )
    run_parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Launch the Streamlit dashboard after running the analysis.",
    )

    graph_parser = subparsers.add_parser(
        "graph",
        help="Run a pinch analysis and export graph HTML files.",
    )
    _add_problem_input_argument(graph_parser)
    graph_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("graphs"),
        help="Output directory for graph HTML files.",
    )
    graph_parser.add_argument(
        "--zone",
        type=str,
        default=None,
        help="Optional zone/target key to export.",
    )
    graph_parser.add_argument(
        "--graph-type",
        type=str,
        default=None,
        help="Optional graph selector such as 'gcc', 'composite', or 'tsp'.",
    )

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate a workbook, JSON file, or CSV bundle without targeting.",
    )
    _add_problem_input_argument(validate_parser)
    validate_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress success output and rely on the exit code.",
    )

    sample_parser = subparsers.add_parser(
        "sample",
        help="Copy a packaged sample case into the working directory.",
    )
    sample_parser.add_argument(
        "--name",
        choices=list_sample_cases(),
        default=list_sample_cases()[0],
        help="Sample case to copy.",
    )
    sample_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("."),
        help="Destination filepath or directory.",
    )

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
    from . import PinchProblem

    parser = build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in _COMMANDS and not argv[0].startswith("-"):
        argv = ["run", *argv]

    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return _run_command(PinchProblem, args)
        if args.command == "graph":
            return _graph_command(PinchProblem, args)
        if args.command == "validate":
            return _validate_command(PinchProblem, args)
        if args.command == "sample":
            return _sample_command(args)
        if args.command == "notebook":
            return _notebook_command(args)
    except Exception as exc:
        if args.debug:
            raise
        parser.exit(status=1, message=f"OpenPinch error: {exc}\n")
    return 0


def _add_problem_input_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "problem_path",
        type=Path,
        help="Path to the source workbook, JSON problem file, or CSV bundle directory.",
    )


def _run_command(problem_cls, args: argparse.Namespace) -> int:
    problem = problem_cls()
    problem.load(args.problem_path)
    if hasattr(problem, "validate"):
        problem.validate()
    problem.target()

    summary = (
        problem.summary_frame(detailed=args.detailed_summary)
        if hasattr(problem, "summary_frame")
        else None
    )
    if summary is not None:
        print(summary.to_string(index=False))

    if args.output is not None:
        output_path = problem.export_to_Excel(args.output)
        print(f"Wrote Excel summary to {output_path}")

    if args.json_output is not None:
        payload = problem.results.model_dump() if hasattr(problem.results, "model_dump") else problem.results
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"Wrote JSON summary to {args.json_output}")

    if args.graph_output is not None:
        written = problem.export_graphs(args.graph_output)
        print(f"Wrote {len(written)} graph file(s) to {args.graph_output}")

    if args.dashboard:
        problem.render_streamlit_dashboard()

    return 0


def _graph_command(problem_cls, args: argparse.Namespace) -> int:
    problem = problem_cls()
    problem.load(args.problem_path)
    if hasattr(problem, "validate"):
        problem.validate()
    problem.target()
    written = problem.export_graphs(
        args.output,
        zone_name=args.zone,
        graph_type=args.graph_type,
    )
    print(f"Wrote {len(written)} graph file(s) to {args.output}")
    return 0


def _validate_command(problem_cls, args: argparse.Namespace) -> int:
    problem = problem_cls()
    payload = problem.load(args.problem_path)
    validated = problem.validate()

    if not args.quiet:
        num_streams = len(validated.streams)
        num_utilities = len(validated.utilities)
        print(
            f"Validation successful: {num_streams} stream(s), "
            f"{num_utilities} utilit(y/ies) loaded from {args.problem_path}"
        )
    return 0


def _sample_command(args: argparse.Namespace) -> int:
    destination = copy_sample_case(args.name, args.output)
    print(f"Copied sample case to {destination}")
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
