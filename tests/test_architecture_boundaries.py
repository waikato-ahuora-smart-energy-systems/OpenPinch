"""Static dependency rules for the owner-oriented package layers."""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

import OpenPinch
from OpenPinch.application.problem import PinchProblem
from OpenPinch.application.workspace import PinchWorkspace
from OpenPinch.domain.heat_exchanger import HeatExchanger
from OpenPinch.domain.heat_exchanger_network import HeatExchangerNetwork
from OpenPinch.domain.problem_table import ProblemTable
from OpenPinch.domain.stream import Stream
from OpenPinch.domain.stream_collection import StreamCollection
from OpenPinch.domain.value import Value
from OpenPinch.domain.zone import Zone

PACKAGE_DIR = Path(OpenPinch.__file__).parent


def _module_name(path: Path) -> str:
    relative = path.relative_to(PACKAGE_DIR).with_suffix("")
    parts = relative.parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(("OpenPinch", *parts))


def _import_target(path: Path, node: ast.ImportFrom) -> str:
    if node.level == 0:
        return node.module or ""
    module_name = _module_name(path)
    package = (
        module_name if path.name == "__init__.py" else module_name.rpartition(".")[0]
    )
    relative_name = "." * node.level + (node.module or "")
    return importlib.util.resolve_name(relative_name, package)


def _openpinch_import_roots(path: Path) -> set[str]:
    roots: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        targets: list[str] = []
        if isinstance(node, ast.ImportFrom):
            targets.append(_import_target(path, node))
        elif isinstance(node, ast.Import):
            targets.extend(alias.name for alias in node.names)
        for target in targets:
            if not target.startswith("OpenPinch."):
                continue
            parts = target.split(".")
            if len(parts) > 1:
                roots.add(parts[1])
    return roots


def _import_targets(path: Path) -> set[str]:
    targets: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            targets.add(_import_target(path, node))
        elif isinstance(node, ast.Import):
            targets.update(alias.name for alias in node.names)
    return targets


def test_domain_has_no_outward_package_dependencies() -> None:
    forbidden = {
        "adapters",
        "analysis",
        "application",
        "classes",
        "contracts",
        "presentation",
        "services",
        "streamlit_webviewer",
        "utils",
    }
    for path in (PACKAGE_DIR / "domain").rglob("*.py"):
        assert _openpinch_import_roots(path).isdisjoint(forbidden), path


def test_contracts_depend_only_on_domain_and_contract_peers() -> None:
    forbidden = {
        "adapters",
        "analysis",
        "application",
        "classes",
        "optimisation",
        "presentation",
        "services",
        "streamlit_webviewer",
        "utils",
    }
    for path in (PACKAGE_DIR / "contracts").rglob("*.py"):
        assert _openpinch_import_roots(path).isdisjoint(forbidden), path


def test_core_domain_classes_have_concrete_domain_owners() -> None:
    expected = {
        Value: "OpenPinch.domain.value",
        Stream: "OpenPinch.domain.stream",
        StreamCollection: "OpenPinch.domain.stream_collection",
        ProblemTable: "OpenPinch.domain.problem_table",
        Zone: "OpenPinch.domain.zone",
        HeatExchanger: "OpenPinch.domain.heat_exchanger",
        HeatExchangerNetwork: "OpenPinch.domain.heat_exchanger_network",
    }
    assert {owner: owner.__module__ for owner in expected} == expected


def test_application_use_cases_have_concrete_application_owners() -> None:
    assert PinchProblem.__module__ == "OpenPinch.application.problem"
    assert PinchWorkspace.__module__ == "OpenPinch.application.workspace"
    assert not (PACKAGE_DIR / "classes" / "pinch_problem.py").exists()
    assert not (PACKAGE_DIR / "classes" / "pinch_workspace.py").exists()
    assert not (PACKAGE_DIR / "classes" / "_pinch_problem").exists()
    assert not (PACKAGE_DIR / "classes" / "_pinch_workspace").exists()


def test_application_has_no_concrete_ui_filesystem_or_solver_imports() -> None:
    forbidden_roots = {
        "OpenPinch.classes",
        "OpenPinch.lib",
        "OpenPinch.optimisation.backends",
        "OpenPinch.streamlit_webviewer",
        "idaes",
        "pathlib",
        "plotly",
        "pyomo",
        "streamlit",
    }
    for path in (PACKAGE_DIR / "application").rglob("*.py"):
        targets = _import_targets(path)
        assert not {
            target
            for target in targets
            if any(
                target == forbidden or target.startswith(f"{forbidden}.")
                for forbidden in forbidden_roots
            )
        }, path


def test_domain_and_contracts_are_cold_importable() -> None:
    code = """
import builtins
real_import = builtins.__import__
blocked = {'gekko', 'idaes', 'kaleido', 'plotly', 'pyomo', 'streamlit', 'tespy'}
def guarded_import(name, *args, **kwargs):
    if name.split('.', 1)[0] in blocked:
        raise ModuleNotFoundError(name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import OpenPinch.contracts.input
import OpenPinch.contracts.output
import OpenPinch.contracts.synthesis.result
import OpenPinch.application.problem
import OpenPinch.application.workspace
import OpenPinch.domain.stream
import OpenPinch.domain.problem_table
import OpenPinch.domain.heat_exchanger_network
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PACKAGE_DIR.parent,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
