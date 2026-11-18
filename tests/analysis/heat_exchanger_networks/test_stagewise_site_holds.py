"""Stage-wise holds for a site design: levels per process, no cross matches.

A site network keeps each process's utility levels at the duties its pinch
design targeted, and recovers heat only within a process. The stage-wise
model takes both as inputs, and evolution children inherit them.
"""

from __future__ import annotations

import pytest

from OpenPinch.analysis.heat_exchanger_networks.models._base.execution import _value
from OpenPinch.analysis.heat_exchanger_networks.models._stagewise import (
    evolution as stagewise_evolution,
)
from OpenPinch.analysis.heat_exchanger_networks.models._stagewise.level_targets import (
    UtilityLevelTarget,
)
from OpenPinch.analysis.heat_exchanger_networks.models.stagewise import (
    StageWiseModel,
)
from OpenPinch.analysis.heat_exchanger_networks.solver.arrays import (
    problem_to_solver_arrays,
)
from tests.analysis.heat_exchanger_networks.test_staged_utilities import (
    D_TMIN,
    _highs_available,
    _hot_utility_duties,
    _ipopt_available,
    _quantity,
    _utility,
)
from OpenPinch.application.problem import PinchProblem

pytestmark = pytest.mark.skipif(
    "not _highs_available()", reason="needs highspy and pyomo"
)


def _stream(zone, name, supply, target, heat_flow):
    return {
        "zone": f"Site/{zone}",
        "name": name,
        "t_supply": _quantity(supply, "degC"),
        "t_target": _quantity(target, "degC"),
        "heat_flow": _quantity(heat_flow, "kW"),
        "dt_cont": _quantity(D_TMIN / 2, "delta_degC"),
        "htc": _quantity(1.0, "kW/m^2/delta_degC"),
    }


def _site(streams, utilities) -> PinchProblem:
    zones = sorted({stream["zone"].split("/")[1] for stream in streams})
    return PinchProblem(
        source={
            "streams": streams,
            "utilities": utilities,
            "zone_tree": {
                "name": "Site",
                "type": "Site",
                "children": [{"name": zone, "type": "Process Zone"} for zone in zones],
            },
        }
    )


def _heating_site() -> PinchProblem:
    """C1 (60 to 200 C) is heated by H2 low, H1 high, and 40 kW between.

    LPS (140 C) can serve the 40 kW between; it is priced above HPS, so by
    price HPS takes it.
    """
    return _site(
        [
            _stream("Plant", "H1", 220.0, 150.0, 70.0),
            _stream("Plant", "H2", 100.0, 70.0, 30.0),
            _stream("Plant", "C1", 60.0, 200.0, 140.0),
        ],
        [
            _utility("LPS", "Hot", 140.0, 139.9, 60.0),
            _utility("HPS", "Hot", 250.0, 249.9, 40.0),
            _utility("CW", "Cold", 20.0, 25.0, 5.0),
        ],
    )


def _model(problem, **kwargs) -> StageWiseModel:
    arguments = {
        "name": "site-holds",
        "framework": "ESM",
        "solver": "highs",
        "solver_arrays": problem_to_solver_arrays(problem, D_TMIN),
        "stages": 2,
        "dTmin": D_TMIN,
        "z_restriction": None,
        "min_dqda": 0.0,
        "minimisation_goal": "utility costs",
        "non_isothermal_model": False,
        "integers": True,
        "tol": 1e-6,
    }
    return StageWiseModel(**(arguments | kwargs))


def _lps_held(model) -> tuple[UtilityLevelTarget, ...]:
    (c1,) = range(model.J)
    return (UtilityLevelTarget("hot", (0,), (c1,), (40.0,)),)


def test_by_price_the_dearer_level_goes_unused():
    model = _model(_heating_site(), staged_utilities=True)
    model.optimise(False)

    assert model.mSuccess == 1
    assert _hot_utility_duties(model)["LPS"] == pytest.approx(0.0, abs=1e-6)


def test_a_level_target_holds_the_levels_duty():
    problem = _heating_site()
    probe = _model(problem, staged_utilities=True)
    model = _model(problem, staged_utilities=True, level_targets=_lps_held(probe))
    model.optimise(False)

    assert model.mSuccess == 1
    duties = _hot_utility_duties(model)
    assert duties["LPS"] == pytest.approx(40.0, rel=1e-6)
    # The band (1e-7 relative) lets 4e-6 kW go to the cheaper level.
    assert duties["HPS"] == pytest.approx(0.0, abs=1e-5)


def test_a_level_target_counts_only_its_streams():
    """Two processes each heat a stream; only one process's LPS is held."""
    problem = _site(
        [
            _stream("A", "C1", 100.0, 130.0, 30.0),
            _stream("B", "C2", 100.0, 130.0, 30.0),
            _stream("A", "H1", 90.0, 80.0, 10.0),
        ],
        [
            _utility("LPS", "Hot", 140.0, 139.9, 60.0),
            _utility("HPS", "Hot", 250.0, 249.9, 40.0),
            _utility("CW", "Cold", 20.0, 25.0, 5.0),
        ],
    )
    probe = _model(problem)
    cold = list(probe.solver_arrays.stream_identities["cold_process_streams"])
    a = next(j for j, name in enumerate(cold) if name.endswith(".C1"))
    target = (UtilityLevelTarget("hot", (0,), (a,), (30.0,)),)

    model = _model(problem, level_targets=target)
    model.optimise(False)

    assert model.mSuccess == 1
    lps, hps = 0, 1
    by_stream = model.Q_h_by_period[0]
    assert _value(by_stream[a][lps]) == pytest.approx(30.0, rel=1e-6)
    b = 1 - a
    assert _value(by_stream[b][hps]) == pytest.approx(30.0, rel=1e-6)


def _two_plants() -> PinchProblem:
    """Plant A's H1 could heat plant B's C2; a site design may not match them."""
    return _site(
        [
            _stream("A", "H1", 200.0, 100.0, 50.0),
            _stream("B", "C2", 80.0, 150.0, 50.0),
        ],
        [
            _utility("HPS", "Hot", 250.0, 249.9, 40.0),
            _utility("CW", "Cold", 20.0, 25.0, 5.0),
        ],
    )


def test_streams_of_one_group_only_match_each_other():
    problem = _two_plants()
    pooled = _model(problem, minimisation_goal="total utility")
    pooled.optimise(False)
    assert _value(pooled.Q_r_by_period[0][0][0][0]) > 1.0

    model = _model(
        problem, minimisation_goal="total utility", stream_groups=(("A",), ("B",))
    )
    model.optimise(False)

    assert model.mSuccess == 1
    assert model.z_feasible[0][0] == [0, 0]
    for k in range(model.S):
        assert _value(model.Q_r_by_period[0][0][0][k]) == pytest.approx(0.0, abs=1e-6)
    assert model.get_max_benefit_HX_candidates(5) == []


@pytest.mark.skipif("not _ipopt_available()", reason="needs ipopt")
def test_an_evolution_child_keeps_its_parents_holds():
    problem = _heating_site()
    probe = _model(problem, staged_utilities=True)
    held = _lps_held(probe)
    parent = _model(
        problem,
        staged_utilities=True,
        level_targets=held,
        stream_groups=(("P", "P"), ("P",)),
    )
    parent.optimise(False)
    position = next(
        (i, j, k)
        for i in range(parent.I)
        for j in range(parent.J)
        for k in range(parent.S)
        if _value(parent.Q_r_by_period[0][i][j][k]) > 1e-6
    )

    child = stagewise_evolution._build_and_solve_evolution(
        parent,
        "plus",
        print_output=False,
        unit=0,
        prev_case=parent,
        position=position,
        z_allowed=parent.z_allowed,
    )

    assert child.level_targets == held
    assert child.stream_groups == (("P", "P"), ("P",))
    assert child.mSuccess == 1
    assert _hot_utility_duties(child)["LPS"] == pytest.approx(40.0, rel=1e-4)


def test_searches_share_solved_children_only_with_the_same_holds():
    problem = _heating_site()
    probe = _model(problem, staged_utilities=True)
    held = _model(problem, staged_utilities=True, level_targets=_lps_held(probe))
    also_held = _model(problem, staged_utilities=True, level_targets=_lps_held(probe))
    free = _model(problem, staged_utilities=True)
    grouped = _model(
        problem, staged_utilities=True, stream_groups=(("P", "Q"), ("P",))
    )
    store: dict = {}

    shared = stagewise_evolution._shared_solved_topologies(held, store)

    assert stagewise_evolution._shared_solved_topologies(also_held, store) is shared
    assert stagewise_evolution._shared_solved_topologies(free, store) is not shared
    free_store = stagewise_evolution._shared_solved_topologies(free, store)
    assert stagewise_evolution._shared_solved_topologies(grouped, store) is not (
        free_store
    )
