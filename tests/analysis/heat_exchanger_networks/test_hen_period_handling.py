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


def _dairy_with_an_idle_period() -> PinchProblem:
    """The multiperiod dairy case with its condensate cooling off in the
    first period."""
    from importlib.resources import files

    source = (
        files("OpenPinch.tutorials.sample_cases") / "dairy_chilling_multiperiod.json"
    )
    payload = json.loads(source.read_text(encoding="utf-8"))
    for record in payload["streams"]:
        if record["name"] == "Condensate cooling":
            record["heat_flow"]["values"] = [0, 200, 250]
    payload["options"].update({"HENS_OUTPUT_FORMATS": [], "HENS_MAX_PARALLEL": 1})
    return PinchProblem(source=payload, project_name="idle-period")


@pytest.mark.solver
@pytest.mark.skipif(not _highs_available(), reason="HiGHS is not installed")
@pytest.mark.parametrize(
    "method",
    [
        {"HENS_PDM_MILP": True, "HENS_HEURISTIC_PDM_NETWORKS": 0},
        {"HENS_PDM_MILP": False},
    ],
    ids=["milp", "heuristic"],
)
def test_a_stream_idle_in_one_period_does_not_break_the_pinch_design(method):
    """A stream with no flow in one period is still above and below the
    pinch by temperature; its matches carry no duty in that period, and
    the two sides of the pinch join into one network."""
    view = _dairy_with_an_idle_period().design.pinch_design(
        options={
            "HENS_DESIGN_BASIS": "all",
            "HENS_DT_CONT_MULTIPLIERS": [1.0],
            **method,
        }
    )

    network = view.selected_network
    condensate = [
        e for e in network.exchangers if e.source_stream.endswith("Condensate cooling")
    ]
    assert condensate
    assert all(
        e.period_states[0].duty == pytest.approx(0.0, abs=1e-6) for e in condensate
    )


@pytest.mark.solver
@pytest.mark.skipif(not _highs_available(), reason="HiGHS is not installed")
def test_stage_packing_counts_a_matchs_duty_in_every_period():
    """Stage packing reads every period's duty: a match that carries load
    only in later periods is not forced off, so the packed MILP pinch
    design of a multiperiod case solves as the unpacked one does."""
    from importlib.resources import files

    source = (
        files("OpenPinch.tutorials.sample_cases")
        / "crude_preheat_train_multiperiod.json"
    )
    problem = PinchProblem(
        source=json.loads(source.read_text(encoding="utf-8")), project_name="crude"
    )

    view = problem.design.pinch_design(
        options={
            "HENS_DESIGN_BASIS": "all",
            "HENS_PDM_MILP": True,
            "HENS_HEURISTIC_PDM_NETWORKS": 0,
            "HENS_DT_CONT_MULTIPLIERS": [1.0],
            "HENS_STAGE_PACKING": "pdm",
            "HENS_OUTPUT_FORMATS": [],
            "HENS_MAX_PARALLEL": 1,
        }
    )

    assert view.selected_network.exchangers


def test_a_small_active_duty_is_copied_with_its_approaches():
    """The idle-period skip applies only to duty the side treats as none:
    below the side's own tolerance, not the joined model's coarser one. A
    small active match the joined model cannot run is refused, not copied
    without its approaches."""
    from types import SimpleNamespace

    from OpenPinch.analysis.heat_exchanger_networks.models._pinch_design import (
        amalgamation,
    )

    target = SimpleNamespace(
        N_periods=1,
        tol=1e-3,
        z=[[[_Slot()]]],
        Q_r_by_period=[[[[_Slot()]]]],
        theta_1_by_period=[[[[5.0]]]],
        theta_2_by_period=[[[[5.0]]]],
    )
    source = SimpleNamespace(
        N_periods=1,
        tol=1e-4,
        non_isothermal_model=False,
        z=[[[1.0]]],
        Q_r_by_period=[[[[5e-4]]]],
        theta_1_by_period=[[[[47.5]]]],
        theta_2_by_period=[[[[47.5]]]],
    )

    with pytest.raises(ValueError, match="disallowed match"):
        amalgamation._copy_recovery_match(None, target, source, 0, 0, 0, 0)


class _Slot:
    """A Pyomo-shaped variable slot."""

    value = None

    def set_value(self, value, skip_validation=False):
        self.value = value


@pytest.mark.parametrize(
    "method", ["pinch_design", "thermal_derivative", "network_evolution"]
)
def test_a_standalone_method_refuses_a_period_id_on_a_multiperiod_case(method):
    """The standalone methods design over every period; a ``period_id``
    used to be taken as a label only, so the stored targets described one
    period and the design all of them. It is refused, naming the ways to
    design one period."""
    with pytest.raises(ValueError, match="design_basis='period'"):
        getattr(_two_period_problem().design, method)(period_id="base")


@pytest.mark.parametrize(
    "method", ["pinch_design", "thermal_derivative", "network_evolution"]
)
def test_a_period_id_in_the_options_is_refused_too(method):
    with pytest.raises(ValueError, match="design_basis='period'"):
        getattr(_two_period_problem().design, method)(options={"period_id": "base"})
