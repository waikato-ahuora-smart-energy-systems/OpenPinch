"""End-to-end tests for the package-root application workflow."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from OpenPinch import PinchProblem, PinchWorkspace
from tests.support.paths import REPOSITORY_ROOT

EXAMPLE_INPUTS = REPOSITORY_ROOT / "examples" / "stream_data"


def _example_problem_filepaths() -> list[Path]:
    return sorted(EXAMPLE_INPUTS.glob("p_*.json"), key=lambda path: path.name)


def _scalar_value(value: dict[str, object]) -> float:
    return float(value["value"])


def test_root_workflow_preserves_validation_failure_shape() -> None:
    with pytest.raises(ValueError, match="Field 'streams' - Field required"):
        PinchProblem({})


def test_pinch_problem_returns_stable_target_output_structure() -> None:
    data = json.loads((EXAMPLE_INPUTS / "p_illustrative.json").read_text())
    problem = PinchProblem(data, project_name="Contract")

    problem.target.direct_heat_integration()
    dumped = problem.results.model_dump(mode="json")

    assert list(dumped) == ["name", "period_id", "targets", "graphs", "design"]
    assert dumped["name"] == "Contract"
    assert len(dumped["targets"]) == 1
    assert dumped["targets"][0]["name"] == "Contract/Direct Integration"
    assert _scalar_value(dumped["targets"][0]["Qh"]) == pytest.approx(749.9999950000001)
    assert _scalar_value(dumped["targets"][0]["Qc"]) == pytest.approx(1000.0)
    assert _scalar_value(dumped["targets"][0]["Qr"]) == pytest.approx(5150.000005)


def test_workspace_uses_case_vocabulary() -> None:
    data = json.loads((EXAMPLE_INPUTS / "p_illustrative.json").read_text())
    workspace = PinchWorkspace(data, project_name="Contract")

    workspace.case("baseline").target.direct_heat_integration()
    scenario = workspace.scenario("tight", dt_cont_multiplier=0.8)
    scenario.target.direct_heat_integration()

    assert workspace.list_cases() == ["baseline", "tight"]
    assert not workspace.compare_cases("baseline", "tight").empty


@pytest.mark.parametrize(
    "problem_path",
    _example_problem_filepaths(),
    ids=lambda path: path.name,
)
def test_problem_pipeline_solves_every_shipped_example(problem_path: Path) -> None:
    project_name = problem_path.stem.removeprefix("p_")
    problem = PinchProblem(
        json.loads(problem_path.read_text()), project_name=project_name
    )
    problem.target.direct_heat_integration()
    actual = problem.results.model_dump(mode="json")

    assert actual["name"] == project_name
    assert actual["targets"]
    for actual_target in actual["targets"]:
        for field in ("Qh", "Qc", "Qr"):
            value = _scalar_value(actual_target[field])
            assert math.isfinite(value)
            assert value >= 0.0


def test_root_import_does_not_require_optional_feature_packages() -> None:
    blocked = {
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
    script = f"""
import builtins
blocked = {blocked!r}
real_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.', 1)[0] in blocked:
        raise ModuleNotFoundError(name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from OpenPinch import PinchProblem, PinchWorkspace
assert PinchProblem and PinchWorkspace
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
