"""External contract tests for :mod:`OpenPinch.main`."""

from __future__ import annotations

import inspect
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from OpenPinch.main import pinch_analysis_service
from tests.support.paths import REPOSITORY_ROOT

EXAMPLE_INPUTS = REPOSITORY_ROOT / "examples" / "stream_data"


def _example_problem_filepaths() -> list[Path]:
    """Return all supported example inputs in deterministic name order."""
    return sorted(EXAMPLE_INPUTS.glob("p_*.json"), key=lambda path: path.name)


def _scalar_value(value: dict[str, object]) -> float:
    """Read the numeric value from the external value-with-unit structure."""
    return float(value["value"])


def test_pinch_analysis_service_signature_is_stable() -> None:
    """Protect the only supported external Python call contract."""
    signature = inspect.signature(pinch_analysis_service)
    parameters = list(signature.parameters.values())

    assert [(parameter.name, parameter.kind) for parameter in parameters] == [
        ("data", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        ("project_name", inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ]
    assert parameters[0].annotation is Any
    assert parameters[1].annotation is str
    assert parameters[1].default == "Project"
    assert signature.return_annotation.__name__ == "TargetOutput"


def test_pinch_analysis_service_preserves_validation_failure_shape() -> None:
    """Invalid caller data fails before any targeting work begins."""
    with pytest.raises(ValidationError) as caught:
        pinch_analysis_service({})

    errors = caught.value.errors(include_url=False)
    assert [(error["type"], error["loc"]) for error in errors] == [
        ("missing", ("streams",)),
    ]


def test_pinch_analysis_service_returns_stable_output_structure() -> None:
    """Pin representative values and serialized field ordering at the boundary."""
    data = json.loads((EXAMPLE_INPUTS / "p_illustrative.json").read_text())

    result = pinch_analysis_service(data, project_name="Contract")
    dumped = result.model_dump(mode="json")

    assert type(result).__name__ == "TargetOutput"
    assert list(dumped) == ["name", "period_id", "targets", "graphs", "design"]
    assert dumped["name"] == "Contract"
    assert dumped["period_id"] is None
    assert dumped["design"] is None
    assert len(dumped["targets"]) == 2
    assert list(dumped["targets"][0])[:6] == [
        "name",
        "period_idx",
        "period_id",
        "degree_of_integration",
        "Qh",
        "Qc",
    ]
    assert dumped["targets"][0]["name"] == "Contract/Direct Integration"
    assert _scalar_value(dumped["targets"][0]["Qh"]) == pytest.approx(749.9999950000001)
    assert _scalar_value(dumped["targets"][0]["Qc"]) == pytest.approx(1000.0)
    assert _scalar_value(dumped["targets"][0]["Qr"]) == pytest.approx(5150.000005)
    assert list(dumped["graphs"]) == [
        "Contract/Direct Integration",
        "Plant/Direct Integration",
    ]


def test_pinch_analysis_service_accepts_mapping_and_default_project() -> None:
    """A caller mapping and the default project label remain supported."""
    data = json.loads((EXAMPLE_INPUTS / "p_illustrative.json").read_text())

    result = pinch_analysis_service(data)

    assert result.name == "Project"
    assert result.targets[0].name == "Project/Direct Integration"


@pytest.mark.parametrize(
    "optimiser_identifier",
    ["dual_annealing", "cmaes", "bo", "rbf_surrogate"],
)
def test_pinch_analysis_service_accepts_canonical_optimiser_identifiers(
    optimiser_identifier: str,
) -> None:
    data = json.loads((EXAMPLE_INPUTS / "p_illustrative.json").read_text())
    data["options"] = {"HPR_BB_MINIMISER": optimiser_identifier}

    result = pinch_analysis_service(data, project_name="Canonical Optimiser")

    assert result.name == "Canonical Optimiser"


@pytest.mark.parametrize(
    "problem_path",
    _example_problem_filepaths(),
    ids=lambda path: path.name,
)
def test_pinch_analysis_pipeline_solves_every_shipped_example(
    problem_path: Path,
) -> None:
    """Every shipped example returns complete, finite caller-visible targets."""
    project_name = problem_path.stem.removeprefix("p_")
    data = json.loads(problem_path.read_text())
    actual = pinch_analysis_service(data=data, project_name=project_name).model_dump(
        mode="json"
    )

    assert actual["name"] == project_name
    assert actual["targets"]
    assert isinstance(actual["graphs"], dict)
    for actual_target in actual["targets"]:
        assert actual_target["name"].endswith("/Direct Integration")
        for field in ("Qh", "Qc", "Qr"):
            value = _scalar_value(actual_target[field])
            assert math.isfinite(value)
            assert value >= 0.0


def test_main_import_does_not_require_optional_feature_packages() -> None:
    """The external entry point remains importable in a core-only install."""
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
from OpenPinch.main import pinch_analysis_service
assert callable(pinch_analysis_service)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
