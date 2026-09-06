"""Fresh-process import contracts for package layers and optional features."""

from __future__ import annotations

import subprocess
import sys

import pytest

from tests.support.paths import REPOSITORY_ROOT

BLOCKED_OPTIONAL_PACKAGES = {
    "gekko",
    "idaes",
    "kaleido",
    "openpyxl",
    "plotly",
    "pyomo",
    "streamlit",
    "tespy",
    "wakepy",
}
IMPORT_CASES = {
    "root-workflows": ("OpenPinch",),
    "contracts": (
        "OpenPinch.contracts.hpr",
        "OpenPinch.contracts.hpr_performance_map",
        "OpenPinch.contracts.input",
        "OpenPinch.contracts.output",
        "OpenPinch.contracts.reporting",
        "OpenPinch.contracts.synthesis.result",
    ),
    "domain": (
        "OpenPinch.domain.stream",
        "OpenPinch.domain.problem_table",
        "OpenPinch.domain.heat_exchanger_network",
        "OpenPinch.domain.targets",
    ),
    "optimisation": (
        "OpenPinch.optimisation.models",
        "OpenPinch.optimisation.service",
    ),
    "application": (
        "OpenPinch.application._problem.accessors.target",
        "OpenPinch.application.problem",
        "OpenPinch.application.workspace",
    ),
    "dashboard-leaves": (
        "OpenPinch.presentation.dashboard.rendering",
        "OpenPinch.presentation.graphs.plotly",
        "OpenPinch.presentation.network_grid.service",
    ),
    "heat-pumps": (
        "OpenPinch.analysis.heat_pumps.optimisation_adapter",
        "OpenPinch.analysis.heat_pumps.performance_maps.generation",
        "OpenPinch.analysis.heat_pumps.performance_maps.target_basis",
        "OpenPinch.analysis.heat_pumps.performance_maps.target_records",
        "OpenPinch.analysis.heat_pumps.performance_maps.targeting",
    ),
    "heat-exchanger-networks": ("OpenPinch.analysis.heat_exchanger_networks.service",),
}


@pytest.mark.parametrize(
    "case_name",
    IMPORT_CASES,
    ids=IMPORT_CASES,
)
def test_layer_is_cold_importable_without_optional_packages(case_name: str) -> None:
    modules = IMPORT_CASES[case_name]
    code = f"""
import builtins
import importlib
import sys

blocked = {BLOCKED_OPTIONAL_PACKAGES!r}
modules = {modules!r}
real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    level = args[3] if len(args) > 3 else kwargs.get("level", 0)
    if level == 0 and name.split(".", 1)[0] in blocked:
        raise ModuleNotFoundError(name)
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
for module in modules:
    importlib.import_module(module)
loaded = sorted(blocked.intersection(sys.modules))
if loaded:
    raise AssertionError(f"optional packages imported eagerly: {{loaded}}")
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_root_exports_resolve_to_concrete_workflow_owners() -> None:
    code = """
import OpenPinch
from OpenPinch.application.problem import PinchProblem
from OpenPinch.application.workspace import PinchWorkspace

assert OpenPinch.__all__ == ["PinchProblem", "PinchWorkspace"]
assert OpenPinch.PinchProblem is PinchProblem
assert OpenPinch.PinchWorkspace is PinchWorkspace
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_default_vapour_targeting_selector_stays_cold_without_tespy() -> None:
    code = f"""
import builtins
import sys

blocked = {BLOCKED_OPTIONAL_PACKAGES!r}
real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    level = args[3] if len(args) > 3 else kwargs.get("level", 0)
    if level == 0 and name.split(".", 1)[0] in blocked:
        raise ModuleNotFoundError(name)
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
from OpenPinch import PinchProblem
from OpenPinch.application._problem.accessors.target import _TargetAccessor

_TargetAccessor._hpr = lambda self, **kwargs: kwargs["simulation_backend"]
assert PinchProblem().target.vapour_compression_heat_pump() == "coolprop"
assert PinchProblem().target.vapour_compression_refrigeration() == "coolprop"
if "tespy" in sys.modules:
    raise AssertionError("default targeting imported TESPy")
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
