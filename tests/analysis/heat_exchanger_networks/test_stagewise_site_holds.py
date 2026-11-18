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
from OpenPinch.application.problem import PinchProblem
from tests.analysis.heat_exchanger_networks.test_staged_utilities import (
    D_TMIN,
    _highs_available,
    _hot_utility_duties,
    _ipopt_available,
    _quantity,
    _utility,
)

pytestmark = pytest.mark.skipif(
    not _highs_available(), reason="needs highspy and pyomo"
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


def _heating_site(*, hot_water: bool = False) -> PinchProblem:
    """C1 (60 to 200 C) is heated by H2 low, H1 high, and 40 kW between.

    LPS (140 C) can serve the 40 kW between; it is priced above HPS, so by
    price HPS takes it. Hot water (90 to 70 C) could heat C1's cold end but
    not the span between.
    """
    water = [_utility("HW", "Hot", 90.0, 70.0, 10.0)] if hot_water else []
    return _site(
        [
            _stream("Plant", "H1", 220.0, 150.0, 70.0),
            _stream("Plant", "H2", 100.0, 70.0, 30.0),
            _stream("Plant", "C1", 60.0, 200.0, 140.0),
        ],
        [
            _utility("LPS", "Hot", 140.0, 139.9, 60.0),
            _utility("HPS", "Hot", 250.0, 249.9, 40.0),
            *water,
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


@pytest.mark.skipif(not _ipopt_available(), reason="needs ipopt")
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
    grouped = _model(problem, staged_utilities=True, stream_groups=(("P", "Q"), ("P",)))
    store: dict = {}

    shared = stagewise_evolution._shared_solved_topologies(held, store)

    assert stagewise_evolution._shared_solved_topologies(also_held, store) is shared
    assert stagewise_evolution._shared_solved_topologies(free, store) is not shared
    free_store = stagewise_evolution._shared_solved_topologies(free, store)
    assert stagewise_evolution._shared_solved_topologies(grouped, store) is not (
        free_store
    )


def _two_heated_plants() -> PinchProblem:
    """Plants A and B each heat a stream from 100 to 130 C on LPS or HPS."""
    return _site(
        [
            _stream("A", "C1", 100.0, 130.0, 30.0),
            _stream("B", "C2", 100.0, 130.0, 30.0),
            _stream("A", "H1", 200.0, 190.0, 10.0),
            _stream("B", "H2", 200.0, 190.0, 10.0),
        ],
        [
            _utility("LPS", "Hot", 140.0, 139.9, 60.0),
            _utility("HPS", "Hot", 250.0, 249.9, 40.0),
            _utility("CW", "Cold", 20.0, 25.0, 5.0),
        ],
    )


def test_a_site_problem_holds_each_processs_levels_by_name():
    from OpenPinch.analysis.heat_exchanger_networks.models.problem import (
        InternalHeatExchangerNetworkProblem,
    )

    problem = _two_heated_plants()
    arrays = problem_to_solver_arrays(problem, D_TMIN)
    cold = list(arrays.stream_identities["cold_process_streams"])
    hot = list(arrays.stream_identities["hot_process_streams"])
    a, b = (name.rsplit(".", 1)[0] for name in sorted(cold))
    internal = InternalHeatExchangerNetworkProblem(
        solver_arrays=arrays,
        name="site",
        framework="ESM",
        solver="highs",
        dTmin=D_TMIN,
        stages=2,
        minimisation_goal="utility costs",
        integers=True,
        tol=1e-6,
        staged_utilities=True,
        site_processes={
            a: {"Hot Utility.LPS": [30.0], "Hot Utility.HPS": [0.0]},
            b: {"Hot Utility.LPS": [0.0], "Hot Utility.HPS": [30.0]},
        },
    )

    internal.load_model()
    case = internal.case

    assert case.stream_groups == (
        tuple(name.rsplit(".", 1)[0] for name in hot),
        tuple(name.rsplit(".", 1)[0] for name in cold),
    )
    c1 = cold.index(f"{a}.C1")
    c2 = cold.index(f"{b}.C2")
    assert UtilityLevelTarget("hot", (0,), (c1,), (30.0,)) in case.level_targets
    assert UtilityLevelTarget("hot", (1,), (c2,), (30.0,)) in case.level_targets
    assert internal.utility_level_targets == {
        "Hot Utility.LPS": [30.0],
        "Hot Utility.HPS": [30.0],
    }
    case.optimise(False)
    assert case.mSuccess == 1
    assert _value(case.Q_h_by_period[0][c1][0]) == pytest.approx(30.0, rel=1e-6)
    assert _value(case.Q_h_by_period[0][c2][1]) == pytest.approx(30.0, rel=1e-6)


def test_a_site_seed_carries_its_process_targets_into_the_evolution_task():
    from OpenPinch.analysis.heat_exchanger_networks.execution.executor import (
        LocalSynthesisExecutor,
    )
    from OpenPinch.analysis.heat_exchanger_networks.execution.settings import (
        workflow_settings_from_problem,
    )
    from OpenPinch.analysis.heat_exchanger_networks.targeting.network_evolution_method import (
        build_seeded_network_evolution_method_tasks,
    )
    from OpenPinch.domain.heat_exchanger_network import HeatExchangerNetwork
    from tests.analysis.heat_exchanger_networks.test_network_evolution_method import (
        _recovery_exchanger,
    )

    problem = _two_heated_plants()
    arrays = problem_to_solver_arrays(problem, D_TMIN)
    a, b = sorted(
        {
            name.rsplit(".", 1)[0]
            for name in arrays.stream_identities["cold_process_streams"]
        }
    )
    processes = {
        a: {"Hot Utility.LPS": [30.0]},
        b: {"Hot Utility.HPS": [30.0]},
    }
    seed = HeatExchangerNetwork(
        exchangers=(_recovery_exchanger(f"{a}.H1", f"{a}.C1", 1),),
        run_id="seed-run",
        method="heuristic_pinch_design_method",
        stage_count=1,
        source_metadata={"site_processes": processes},
    )

    (task,) = build_seeded_network_evolution_method_tasks(
        workflow_settings_from_problem(problem), [seed]
    )
    built = LocalSynthesisExecutor(print_output=False)._build_problem(
        task, problem=problem, parent_outcomes={}
    )

    assert task.settings["site_processes"] == processes
    assert built.site_processes == processes
    assert built.staged_utilities


@pytest.mark.skipif(not _ipopt_available(), reason="needs ipopt")
def test_a_seeded_site_task_places_its_staged_units_before_relaxing():
    """A relaxed model fixes staged binaries; a seed without a parent first
    places them with a MILP on its recovery topology and levels."""
    from OpenPinch.analysis.heat_exchanger_networks.models.problem import (
        InternalHeatExchangerNetworkProblem,
    )

    problem = _heating_site(hot_water=True)
    arrays = problem_to_solver_arrays(problem, D_TMIN)
    (plant,) = {
        name.rsplit(".", 1)[0]
        for name in arrays.stream_identities["cold_process_streams"]
    }
    placed = _model(
        problem,
        staged_utilities=True,
        level_targets=_lps_held(_model(problem, staged_utilities=True)),
    )
    placed.optimise(False)
    recovery = [
        [
            [1 if _value(placed.z[i][j][k]) > 0.5 else 0 for k in range(placed.S)]
            for j in range(placed.J)
        ]
        for i in range(placed.I)
    ]
    internal = InternalHeatExchangerNetworkProblem(
        solver_arrays=arrays,
        name="site-seed",
        framework="ESM",
        solver="ipopt-pyomo",
        dTmin=D_TMIN,
        stages=2,
        z_restriction=[recovery, None, None],
        minimisation_goal="total cost",
        non_isothermal_model=True,
        integers=False,
        tol=1e-3,
        staged_utilities=True,
        site_processes={
            plant: {
                "Hot Utility.LPS": [40.0],
                "Hot Utility.HPS": [0.0],
                "Hot Utility.HW": [0.0],
            }
        },
    )

    internal.load_model()
    internal.case.optimise(False)

    case = internal.case
    assert case.mSuccess == 1
    lps, hw = 0, 2
    for u, duty in ((lps, 40.0), (hw, 0.0)):
        held = sum(_value(row[u]) for row in case.Q_h_by_period[0]) + sum(
            _value(by_k) for row in case.Q_hs_by_period[0] for by_k in row[u]
        )
        assert held == pytest.approx(duty, rel=1e-4, abs=1e-4)


def _tiny_match_case(model_cost: float):
    """One 0.011 kW match, as a pinch design leaves in a site seed."""
    from types import SimpleNamespace

    return SimpleNamespace(
        S=1,
        I=1,
        J=1,
        z_allowed=[[[1]]],
        Q_r=[[[0.011]]],
        theta_1=[[[96.0]]],
        theta_2=[[[95.7]]],
        U_r=[[0.5]],
        A_coeff=[943.9],
        A_exp=[0.6],
        recovery_area_cost_filtered=[[model_cost]],
    )


def _smoothed_cost(case) -> float:
    """The model's own area cost: ``A_coeff * (area + 1e-3) ** A_exp``."""
    theta_1, theta_2 = case.theta_1[0][0][0], case.theta_2[0][0][0]
    mean = (theta_1 * theta_2 * (theta_1 + theta_2) / 2 + 1e-3) ** (1 / 3)
    area = case.Q_r[0][0][0] / (case.U_r[0][0] * mean)
    return case.A_coeff[0] * (area + 1e-3) ** case.A_exp[0]


def test_the_area_check_compares_the_models_smoothed_cost_on_a_tiny_match():
    """The model adds 1e-3 m2 to each area before the cost exponent, so a
    tiny match's cost exceeds its exact cost by up to
    ``A_coeff * 1e-3 ** A_exp`` (about 15 here), well over the check's 1.0
    absolute tolerance. The check recomputes the model's smoothed cost, so a
    cost off by that much still fails."""
    from OpenPinch.analysis.heat_exchanger_networks.models._stagewise import (
        verification,
    )

    case = _tiny_match_case(0.0)
    smoothed = _smoothed_cost(case)

    assert verification._check_area_costs(_tiny_match_case(smoothed)) is True
    bound = case.A_coeff[0] * 1e-3 ** case.A_exp[0]
    assert verification._check_area_costs(_tiny_match_case(smoothed + bound)) is False


def test_a_seed_match_the_placement_leaves_idle_stays_in_use():
    """H1 is spent in stage 1, so the placement gives its seed match in
    stage 2 no duty; post-processing reads that as off, but the seed keeps
    it."""
    from OpenPinch.analysis.heat_exchanger_networks.models.problem import (
        InternalHeatExchangerNetworkProblem,
    )

    problem = _heating_site()
    arrays = problem_to_solver_arrays(problem, D_TMIN)
    hot = list(arrays.stream_identities["hot_process_streams"])
    (plant,) = {name.rsplit(".", 1)[0] for name in hot}
    h1, h2 = hot.index(f"{plant}.H1"), hot.index(f"{plant}.H2")
    recovery = [[[0, 0]], [[0, 0]]]
    recovery[h1][0] = [1, 1]
    recovery[h2][0] = [0, 1]
    internal = InternalHeatExchangerNetworkProblem(
        solver_arrays=arrays,
        name="site-seed",
        framework="ESM",
        solver="ipopt-pyomo",
        dTmin=D_TMIN,
        stages=2,
        z_restriction=[recovery, None, None],
        minimisation_goal="total cost",
        non_isothermal_model=True,
        integers=False,
        tol=1e-3,
        staged_utilities=True,
        site_processes={plant: {"Hot Utility.LPS": [40.0], "Hot Utility.HPS": [0.0]}},
    )

    internal.load_model()

    assert _value(internal.case.z[h1][0][1]) == pytest.approx(1.0)


def test_two_heaters_at_one_stream_end_split_the_stream():
    """LPS and HPS share C1's end in parallel, each on its duty's share of
    C1's flow over the whole end span, as the model sized them."""
    from OpenPinch.analysis.heat_exchanger_networks.extraction.service import (
        extract_heat_exchanger_network,
    )
    from OpenPinch.analysis.heat_exchanger_networks.reporting.verification import (
        verify_network_feasibility,
    )
    from OpenPinch.domain.enums import HeatExchangerKind

    problem = _site(
        [
            _stream("Plant", "C1", 100.0, 130.0, 30.0),
            _stream("Plant", "H1", 90.0, 80.0, 10.0),
        ],
        [
            _utility("LPS", "Hot", 140.0, 139.9, 60.0),
            _utility("HPS", "Hot", 250.0, 249.9, 40.0),
            _utility("CW", "Cold", 20.0, 25.0, 5.0),
        ],
    )
    model = _model(
        problem,
        stages=1,
        level_targets=(
            UtilityLevelTarget("hot", (0,), (0,), (10.0,)),
            UtilityLevelTarget("hot", (1,), (0,), (20.0,)),
        ),
    )
    model.optimise(False)
    assert model.mSuccess == 1

    network = extract_heat_exchanger_network(
        model, model.solver_arrays, run_id="series", method="ESM", stage_count=1
    )

    heaters = {
        exchanger.source_stream: exchanger.state()
        for exchanger in network.exchangers
        if exchanger.kind is HeatExchangerKind.HOT_UTILITY
    }
    for name, share in (("Hot Utility.LPS", 1 / 3), ("Hot Utility.HPS", 2 / 3)):
        assert heaters[name].sink_split_fraction == pytest.approx(share, rel=1e-6)
        assert heaters[name].sink_inlet_temperature == pytest.approx(100.0 + 273.15)
        assert heaters[name].sink_outlet_temperature == pytest.approx(130.0 + 273.15)
    assert verify_network_feasibility(network) == ()
