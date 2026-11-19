"""How HEN synthesis treats the periods of a multiperiod case.

Synthesis on the all-periods basis builds its solver models over every
period of the zone. A ``period_id`` is a basis period, and the default median
basis a synthetic ``median`` period: synthesis runs on that one period, and
the network is then fitted to every period.
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
    "all-periods basis": lambda problem: problem.design.heat_exchanger_network(
        width=0, depth=0, design_basis="all"
    ),
}
# Synthesis on a basis period sees that period alone; it used to see every
# period, as period_id only labelled the run. With no period_id the default
# median basis synthesises on its synthetic period.
_SYNTHESIS_PERIODS = {
    "no period_id": ("median",),
    "period_id base": ("base",),
    "all-periods basis": PERIODS,
}


@pytest.mark.parametrize("call", list(_CALLS))
def test_synthesis_builds_its_solver_arrays_over_its_basis_periods(monkeypatch, call):
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

    assert seen and set(seen) == {_SYNTHESIS_PERIODS[call]}


@pytest.mark.synthesis
@pytest.mark.skipif("not _highs_available()", reason="needs highspy")
def test_a_stored_all_periods_basis_designs_as_the_argument_does():
    stored = _two_period_problem()
    stored.update_options({"HENS_DESIGN_BASIS": "all"})

    structures = {
        network_structure_signature(
            stored.design.heat_exchanger_network(width=0, depth=0).selected_network
        ),
        network_structure_signature(
            _CALLS["all-periods basis"](_two_period_problem()).selected_network
        ),
    }

    assert len(structures) == 1, structures


def test_the_median_basis_reads_heat_flows_of_streams_given_by_cp():
    """A stream given by its heat capacity flowrate has no ``heat_flow`` in
    the case's source; the basis takes the prepared streams' heat flows, so
    each period still runs and the median case has every stream's load."""
    source = _two_period_problem().to_problem_json()
    for record in source["streams"]:
        record["heat_flow"] = None
    problem = PinchProblem(source=source, project_name="cp-only")

    basis = problem.design.design_basis()

    assert {period for cluster in basis.clusters for period in cluster.period_ids} == (
        set(PERIODS)
    )
    assert all(
        stream.included and stream.quantities["heat_flow"] > 0
        for stream in basis.streams
    )


def test_the_median_basis_matches_cp_streams_by_their_full_zone():
    """Streams given by CP below a nested zone take the heat flow of the
    prepared stream in that zone, even when another branch has a stream of
    the same name."""
    source = _two_period_problem().to_problem_json()
    source["zone_tree"] = {
        "name": "Site",
        "type": "Site",
        "children": [
            {
                "name": branch,
                "type": "Process Zone",
                "children": [
                    {"name": "Area", "type": "Process Zone", "children": None}
                ],
            }
            for branch in ("P1", "P2")
        ],
    }
    streams = []
    for branch, scale in (("P1", 1.0), ("P2", 2.0)):
        for record in source["streams"]:
            copy = json.loads(json.dumps(record))
            copy["zone"] = f"Site/{branch}/Area"
            copy["heat_flow"] = None
            cp = copy["heat_capacity_flowrate"]
            cp["values"] = [value * scale for value in cp["values"]]
            streams.append(copy)
    source["streams"] = streams
    problem = PinchProblem(source=source, project_name="nested-cp")

    basis = problem.design.design_basis()

    loads = {
        (stream.zone, stream.name): stream.quantities["heat_flow"]
        for stream in basis.streams
    }
    assert len(loads) == 8
    for name in ("Raw Milk", "HT Flash", "Milk Concentrate", "CIP Water"):
        assert loads[("Site/P1/Area", name)] > 0
        assert loads[("Site/P2/Area", name)] == pytest.approx(
            2 * loads[("Site/P1/Area", name)]
        )
