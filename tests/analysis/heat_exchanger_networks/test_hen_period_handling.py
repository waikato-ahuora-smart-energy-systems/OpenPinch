"""How HEN synthesis treats the periods of a multiperiod case today.

Characterisation tests: they pin current behaviour before the design entry
points are merged. Every call builds its solver models over every period of
the zone; ``period_id`` only picks the cached targets reused and labels the
run, so it does not restrict the design to that period.
"""

from __future__ import annotations

import json

import pytest

from OpenPinch.analysis.heat_exchanger_networks.errors import WorkflowContractError
from OpenPinch.analysis.heat_exchanger_networks.execution import executor
from OpenPinch.analysis.heat_exchanger_networks.reporting.ranking import (
    network_structure_signature,
)
from OpenPinch.analysis.heat_exchanger_networks.solver import backend
from OpenPinch.analysis.heat_exchanger_networks.solver.dependencies import (
    MissingSynthesisDependencyError,
    MissingSynthesisSolverError,
)
from OpenPinch.application.problem import PinchProblem
from tests.support.paths import REPOSITORY_ROOT

FOUR_STREAM_JSON = (
    REPOSITORY_ROOT
    / "tests/fixtures/openhens/Four-stream-Yee-and-Grossmann-1990-1.json"
)
PERIODS = ("base", "peak")


def _highs_available() -> bool:
    try:
        backend.require_solver_backend("highs")
    except (MissingSynthesisDependencyError, MissingSynthesisSolverError):
        return False
    return True


def _two_period_problem() -> PinchProblem:
    """Yee and Grossmann's four streams, with the first stream 25 % larger at peak."""

    payload = json.loads(FOUR_STREAM_JSON.read_text(encoding="utf-8"))
    for index, record in enumerate(payload["streams"]):
        for key, value in list(record.items()):
            if isinstance(value, dict) and isinstance(value.get("value"), int | float):
                scale = (
                    1.25
                    if index == 0 and key in ("heat_flow", "heat_capacity_flowrate")
                    else 1.0
                )
                record[key] = {
                    "unit": value["unit"],
                    "values": [value["value"], value["value"] * scale],
                }
    options = {
        key: value
        for key, value in payload["options"].items()
        if not key.startswith("HENS_SOLVER")
        and key not in ("HENS_DT_CONT_MULTIPLIERS", "HENS_OUTPUT_FOLDER")
    }
    payload["options"] = options | {
        "PROBLEM_PERIOD_IDS": list(PERIODS),
        "PROBLEM_PERIOD_WEIGHTS": [0.7, 0.3],
        "HENS_OUTPUT_FORMATS": [],
        "HENS_MAX_PARALLEL": 1,
    }
    return PinchProblem(source=payload, project_name="two-period")


_CALLS = {
    "no period_id": lambda problem: problem.design.heat_exchanger_network(
        width=0, depth=0
    ),
    "period_id base": lambda problem: problem.design.heat_exchanger_network(
        width=0, depth=0, period_id="base"
    ),
    "multiperiod method": lambda problem: (
        problem.design.multiperiod_heat_exchanger_network(
            width=0, depth=0, prerequisites=True
        )
    ),
}


@pytest.mark.parametrize("call", list(_CALLS))
def test_every_design_call_builds_its_solver_arrays_over_all_periods(monkeypatch, call):
    seen = []
    real = executor.problem_to_solver_arrays

    def record_and_stop(problem, dTmin):
        arrays = real(problem, dTmin)
        seen.append(tuple(str(p) for p in arrays.arrays["period_ids"]))
        # Stop the task here: every task then fails, so the run does too.
        raise RuntimeError("arrays recorded")

    monkeypatch.setattr(executor, "problem_to_solver_arrays", record_and_stop)

    with pytest.raises(WorkflowContractError):
        _CALLS[call](_two_period_problem())

    assert seen and set(seen) == {PERIODS}


@pytest.mark.synthesis
@pytest.mark.skipif("not _highs_available()", reason="needs highspy")
def test_a_period_id_and_the_multiperiod_method_design_the_same_network():
    # With every call designing over all periods, the period label changes
    # nothing about the network the heuristic pinch design method finds.
    structures = {
        call: network_structure_signature(run(_two_period_problem()).selected_network)
        for call, run in _CALLS.items()
    }

    assert len(set(structures.values())) == 1, structures
