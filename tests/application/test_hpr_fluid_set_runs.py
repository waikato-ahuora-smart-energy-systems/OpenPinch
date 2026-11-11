"""Vapour compression targets run the fluid sets a refrigerant screen proposes."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from OpenPinch import PinchProblem
from OpenPinch.application._problem.accessors.target import _TargetAccessor
from OpenPinch.contracts.hpr import HPRTopologyIdentifier
from OpenPinch.contracts.hpr_fluid_set_runs import HprFluidSetRunsResult
from OpenPinch.contracts.hpr_refrigerant_screen import HprRefrigerantScreenResult
from OpenPinch.domain.value import Value

T_CRIT = {"Ammonia": 132.4, "n-Butane": 152.0, "Isobutane": 134.7, "Water": 374.0}


def _quantity(value, unit="degC"):
    return {"value": value, "unit": unit}


def _row(fluid, rank):
    return {
        "fluid": fluid,
        "T_crit": _quantity(T_CRIT[fluid]),
        "p_evap": _quantity(4.3, "bar"),
        "p_cond": _quantity(46.0, "bar"),
        "pressure_ratio": 10.7,
        "suction_vhc": _quantity(4000.0 - rank, "kJ/m^3"),
        "verdict": "practical",
        "rank": rank,
    }


def _stage(T_evap, T_cond, fluids, kind="vapour_compression"):
    return {
        "kind": kind,
        "T_evap": _quantity(T_evap),
        "T_cond": _quantity(T_cond),
        "rows": [_row(fluid, rank) for rank, fluid in enumerate(fluids, 1)],
    }


def _layout(method, conds, evaps, stages, n_fluid_sets):
    return {
        "method": method,
        "T_cond_levels": [_quantity(t) for t in conds],
        "T_evap_levels": [_quantity(t) for t in evaps],
        "stages": stages,
        "n_fluid_sets": n_fluid_sets,
    }


SINGLE_HOT = _layout(
    "single_stage_vapour_compression",
    [120.0],
    [0.0],
    [_stage(0.0, 120.0, ["Ammonia", "n-Butane"])],
    2,
)
SINGLE_COLD = _layout(
    "single_stage_vapour_compression",
    [85.0],
    [0.0],
    [_stage(0.0, 85.0, ["Ammonia", "n-Butane"])],
    2,
)
CASCADE = _layout(
    "cascade_vapour_compression",
    [120.0, 85.0],
    [0.0],
    [_stage(80.0, 120.0, ["n-Butane"]), _stage(0.0, 85.0, ["Ammonia", "Isobutane"])],
    2,
)
MVR = _layout(
    "vapour_compression_mvr",
    [120.0, 85.0],
    [0.0],
    [
        _stage(80.0, 100.0, ["Water"], kind="mvr"),
        _stage(100.0, 120.0, ["Water"], kind="mvr"),
        _stage(0.0, 85.0, ["Ammonia", "Isobutane"]),
    ],
    2,
)


def _screen(*layouts, is_heat_pumping=True) -> HprRefrigerantScreenResult:
    return HprRefrigerantScreenResult.model_validate(
        {
            "source": "explicit_levels",
            "topology": "cascade",
            "is_heat_pumping": is_heat_pumping,
            "T_evap_levels": [_quantity(0.0)],
            "T_cond_levels": [_quantity(120.0), _quantity(85.0)],
            "dt_cont": _quantity(5.0, "delta_degC"),
            "dt_cascade_hx": _quantity(5.0, "delta_degC"),
            "max_pressure_ratio": 15.0,
            "min_suction_pressure": _quantity(0.5, "bar"),
            "critical_margin": _quantity(5.0, "delta_degC"),
            "max_stage_lift": _quantity(70.0, "delta_degC"),
            "layouts": list(layouts or (SINGLE_HOT, SINGLE_COLD, CASCADE, MVR)),
        }
    )


@pytest.fixture
def calls(monkeypatch):
    """Record each ``_hpr`` call; the TAC falls with each call made."""
    observed: list[dict] = []

    def fake_hpr(self, **kwargs):
        observed.append(kwargs)
        return SimpleNamespace(
            hpr_success=True,
            hpr_total_annualized_cost=Value(1000.0 - len(observed), "$/y"),
        )

    monkeypatch.setattr(_TargetAccessor, "_hpr", fake_hpr)
    return observed


def _refrigerants(call):
    return call["extra_configuration"]["HPR_REFRIGERANTS"]


def test_a_screen_runs_every_matching_fluid_set_once(calls):
    runs = PinchProblem().target.vapour_compression_heat_pump(refrigerants=_screen())

    assert isinstance(runs, HprFluidSetRunsResult)
    # The two single-stage layouts propose the same sets: each runs once.
    assert [run.fluids for run in runs.runs] == [
        ("Ammonia",),
        ("n-Butane",),
        ("n-Butane", "Ammonia"),
        ("n-Butane", "Isobutane"),
    ]
    assert [_refrigerants(call) for call in calls] == [
        ["Ammonia"],
        ["n-Butane"],
        ["n-Butane", "Ammonia"],
        ["n-Butane", "Isobutane"],
    ]
    assert [(call["condensers"], call["evaporators"]) for call in calls] == [
        (1, 1),
        (1, 1),
        (2, 1),
        (2, 1),
    ]
    assert [run.method for run in runs.runs] == [
        HPRTopologyIdentifier.SINGLE_STAGE_VAPOUR_COMPRESSION,
        HPRTopologyIdentifier.SINGLE_STAGE_VAPOUR_COMPRESSION,
        HPRTopologyIdentifier.CASCADE_VAPOUR_COMPRESSION,
        HPRTopologyIdentifier.CASCADE_VAPOUR_COMPRESSION,
    ]
    assert len(runs.runs[0].layouts) == 2


def test_the_best_run_has_the_lowest_total_annualized_cost(calls):
    runs = PinchProblem().target.vapour_compression_heat_pump(refrigerants=_screen())

    assert [run.total_annualized_cost for run in runs.runs] == [999, 998, 997, 996]
    assert runs.best is runs.runs[-1]
    assert runs.best.target is not None


def test_failed_runs_are_recorded_and_never_best(monkeypatch):
    def fake_hpr(self, **kwargs):
        fluids = kwargs["extra_configuration"]["HPR_REFRIGERANTS"]
        if fluids == ["Ammonia"]:
            raise ValueError("no feasible point")
        return SimpleNamespace(
            hpr_success=fluids != ["n-Butane"],
            hpr_total_annualized_cost=Value(
                1.0 if fluids == ["n-Butane"] else 50.0, "$/y"
            ),
        )

    monkeypatch.setattr(_TargetAccessor, "_hpr", fake_hpr)

    runs = PinchProblem().target.vapour_compression_heat_pump(
        refrigerants=_screen(SINGLE_HOT, CASCADE)
    )

    failed, unsuccessful, *solved = runs.runs
    assert failed.success is False
    assert failed.target is None
    assert "no feasible point" in failed.error
    assert unsuccessful.success is False
    assert unsuccessful.error is None
    assert runs.best in solved
    assert runs.best.total_annualized_cost == 50.0


def test_no_successful_run_leaves_no_best(monkeypatch):
    def fake_hpr(self, **kwargs):
        raise RuntimeError("solver failed")

    monkeypatch.setattr(_TargetAccessor, "_hpr", fake_hpr)

    runs = PinchProblem().target.vapour_compression_heat_pump(
        refrigerants=_screen(SINGLE_HOT)
    )

    assert len(runs.runs) == 2
    assert runs.best is None


def test_stage_counts_select_the_layouts(calls):
    runs = PinchProblem().target.vapour_compression_heat_pump(
        refrigerants=_screen(), condensers=1, evaporators=1
    )

    assert {run.method for run in runs.runs} == {
        HPRTopologyIdentifier.SINGLE_STAGE_VAPOUR_COMPRESSION
    }


def test_one_layout_runs_only_its_fluid_sets(calls):
    layout = _screen().layouts[2]

    runs = PinchProblem().target.vapour_compression_heat_pump(refrigerants=layout)

    assert [run.fluids for run in runs.runs] == list(layout.fluid_sets())
    assert runs.runs[0].layouts == (layout,)


def test_one_fluid_set_runs_as_a_plain_target(calls):
    fluid_set = _screen().layouts[2].fluid_sets()[0]

    target = PinchProblem().target.vapour_compression_heat_pump(refrigerants=fluid_set)

    assert target.hpr_success
    assert [_refrigerants(call) for call in calls] == [fluid_set]


def test_a_refrigerant_list_still_returns_the_target(calls):
    target = PinchProblem().target.vapour_compression_heat_pump(
        refrigerants=["Ammonia"]
    )

    assert target.hpr_success
    assert _refrigerants(calls[0]) == ["Ammonia"]


def test_parallel_targets_reject_a_screen_with_no_parallel_layout(calls):
    with pytest.raises(ValueError, match="parallel_vapour_compression"):
        PinchProblem().target.vapour_compression_heat_pump(
            refrigerants=_screen(), is_cascade_cycle=False
        )

    assert calls == []


def test_refrigeration_rejects_a_heat_pumping_screen(calls):
    with pytest.raises(ValueError, match="heat pump"):
        PinchProblem().target.vapour_compression_refrigeration(refrigerants=_screen())

    assert calls == []


def test_refrigeration_runs_a_refrigeration_screen(calls):
    runs = PinchProblem().target.vapour_compression_refrigeration(
        refrigerants=_screen(SINGLE_HOT, is_heat_pumping=False)
    )

    assert [run.fluids for run in runs.runs] == [("Ammonia",), ("n-Butane",)]
    assert all(call["is_heat_pump"] is False for call in calls)


def test_mvr_heat_pumps_run_the_mvr_layouts(calls):
    runs = PinchProblem().target.mvr_heat_pump(refrigerants=_screen())

    assert [run.fluids for run in runs.runs] == [("Ammonia",), ("Isobutane",)]
    assert [run.method for run in runs.runs] == [
        HPRTopologyIdentifier.VAPOUR_COMPRESSION_MVR
    ] * 2
    call = calls[0]
    assert (call["condensers"], call["evaporators"]) == (1, 1)
    assert call["extra_configuration"]["HPR_MVR_COUNT"] == 2
    assert runs.runs[0].mvr_stages == 2


def test_mvr_heat_pumps_take_a_refrigerant_list(calls):
    PinchProblem().target.mvr_heat_pump(refrigerants=["Ammonia"])

    assert _refrigerants(calls[0]) == ["Ammonia"]


def test_a_sweep_keeps_the_other_arguments(calls):
    PinchProblem().target.vapour_compression_heat_pump(
        refrigerants=_screen(SINGLE_HOT),
        load_fraction=0.25,
        maximum_evaluations=20,
    )

    assert all(call["load_fraction"] == 0.25 for call in calls)
    assert all(call["maximum_evaluations"] == 20 for call in calls)


def test_a_real_screen_runs_through_the_coolprop_target():
    pytest.importorskip("CoolProp")
    problem = PinchProblem("basic_pinch.json")
    # One fluid set: each run is a full search, slow in Pyodide.
    screen = problem.target.hpr_refrigerant_screen(
        T_evap=[40.0],
        T_cond=[100.0],
        topology="cascade",
        refrigerants=["n-Butane"],
    )

    runs = problem.target.vapour_compression_heat_pump(
        refrigerants=screen,
        load_fraction=0.25,
        maximum_restarts=1,
        maximum_iterations=2,
        maximum_evaluations=20,
        options={"COSTING_HPR_CAPITAL_RECOVERY_ENABLED": False},
    )

    assert [run.fluids for run in runs.runs] == list(screen.layouts[0].fluid_sets())
    for run in runs.runs:
        assert (run.target is None) == (run.error is not None)
    assert runs.best is not None
    settings = json.loads(runs.best.target.provenance.effective_settings_json)
    assert settings["HPR_REFRIGERANTS"] == list(runs.best.fluids)
    assert json.loads(runs.model_dump_json())["runs"][0]["fluids"]


def test_a_split_layout_runs_a_condenser_on_each_added_level(calls):
    split = _layout(
        "cascade_vapour_compression",
        [120.0, 60.0],
        [0.0],
        [_stage(55.0, 120.0, ["n-Butane"]), _stage(0.0, 60.0, ["Ammonia"])],
        1,
    ) | {"added_levels": [_quantity(60.0)]}

    runs = PinchProblem().target.vapour_compression_heat_pump(
        refrigerants=_screen(split)
    )

    assert [run.fluids for run in runs.runs] == [("n-Butane", "Ammonia")]
    assert (calls[0]["condensers"], calls[0]["evaporators"]) == (2, 1)


def _committing_hpr(succeeds):
    """An ``_hpr`` that commits its run, as a target transaction does."""

    def fake_hpr(self, **kwargs):
        fluids = kwargs["extra_configuration"]["HPR_REFRIGERANTS"]
        self._problem._results = SimpleNamespace(fluids=list(fluids))
        return SimpleNamespace(
            hpr_success=succeeds(fluids),
            hpr_total_annualized_cost=Value(1.0, "$/y"),
        )

    return fake_hpr


def test_an_unsuccessful_run_leaves_the_last_successful_one(monkeypatch):
    monkeypatch.setattr(
        _TargetAccessor, "_hpr", _committing_hpr(lambda f: f == ["Ammonia"])
    )
    problem = PinchProblem()

    runs = problem.target.vapour_compression_heat_pump(refrigerants=_screen(SINGLE_HOT))

    assert [run.success for run in runs.runs] == [True, False]
    assert problem.results.fluids == ["Ammonia"]


def test_no_successful_run_leaves_the_problem_as_it_was(monkeypatch):
    monkeypatch.setattr(_TargetAccessor, "_hpr", _committing_hpr(lambda f: False))
    problem = PinchProblem()

    problem.target.vapour_compression_heat_pump(refrigerants=_screen(SINGLE_HOT))

    assert problem.results is None
