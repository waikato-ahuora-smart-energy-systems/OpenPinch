"""Smoke-check an installed OpenPinch wheel without importing the checkout."""

from __future__ import annotations

import argparse
import importlib.util
import json
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
    from OpenPinch import PinchProblem, PinchWorkspace
    from OpenPinch.resources import list_notebooks, list_sample_cases, read_sample_case

    package_path = Path(OpenPinch.__file__).resolve()
    if package_path.is_relative_to(repo_root):
        raise AssertionError(f"Imported OpenPinch from checkout: {package_path}")

    notebooks = list_notebooks()
    sample_cases = list_sample_cases()
    if not notebooks or not sample_cases:
        raise AssertionError("Installed wheel is missing packaged resources.")
    sample = read_sample_case("basic_pinch.json")
    if not sample:
        raise AssertionError("Installed wheel could not read basic_pinch.json.")
    if "process_mvr.json" not in sample_cases:
        raise AssertionError("Installed wheel is missing process_mvr.json.")
    if "11_process_mvr_and_cascade.ipynb" not in notebooks:
        raise AssertionError("Installed wheel is missing the Process MVR tutorial.")

    problem = PinchProblem(json.loads(sample), project_name="Wheel contract")
    problem.target.direct_heat_integration()
    if problem.results.name != "Wheel contract" or not problem.results.targets:
        raise AssertionError("Installed wheel failed the PinchProblem workflow.")
    workspace = PinchWorkspace(json.loads(sample), project_name="Wheel contract")
    if workspace.list_cases() != ["baseline"]:
        raise AssertionError("Installed wheel failed the PinchWorkspace workflow.")

    if OpenPinch.__all__ != ["PinchProblem", "PinchWorkspace"]:
        raise AssertionError(f"Unexpected root exports: {OpenPinch.__all__}")
    forbidden_root_exports = {"TargetInput", "TargetOutput"}
    leaked = sorted(name for name in forbidden_root_exports if hasattr(OpenPinch, name))
    if leaked:
        raise AssertionError(f"Root package exposes unexpected aliases: {leaked}")

    retired_packages = (
        "OpenPinch.classes",
        "OpenPinch.lib",
        "OpenPinch.services",
        "OpenPinch.streamlit_webviewer",
        "OpenPinch.utils",
    )
    resolved = sorted(
        package
        for package in retired_packages
        if importlib.util.find_spec(package) is not None
    )
    if resolved:
        raise AssertionError(f"Installed wheel contains retired packages: {resolved}")

    subprocess.run(
        [sys.executable, "-m", "OpenPinch", "notebook", "--help"],
        check=True,
        cwd=repo_root.parent,
    )
    print(f"Installed artifact smoke passed for {package_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
