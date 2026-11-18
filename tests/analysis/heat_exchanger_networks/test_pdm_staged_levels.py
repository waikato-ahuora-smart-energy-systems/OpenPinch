"""Pinch design sides hold each utility level at its targeted duty.

With ``staged_utilities`` a side offers heaters and coolers between its
stages, with one more stage than it would have, so a level can serve the
pinch end, and holds each utility's duty at the duty targeting gives it,
whatever the prices.
"""

from __future__ import annotations

import pytest

from OpenPinch.analysis.heat_exchanger_networks.models._base.execution import _value
from OpenPinch.analysis.heat_exchanger_networks.models._stagewise import (
    staged_utilities as stagewise_staged,
)
from OpenPinch.analysis.heat_exchanger_networks.models.pinch_decomposition import (
    PinchDecompModel,
)
from OpenPinch.analysis.heat_exchanger_networks.models.problem import (
    InternalHeatExchangerNetworkProblem,
)
from OpenPinch.analysis.heat_exchanger_networks.reporting.verification import (
    verify_network_feasibility,
)
from OpenPinch.analysis.heat_exchanger_networks.solver import backend
from OpenPinch.analysis.heat_exchanger_networks.solver import (
    pinch_design_decomposition as decomposition_module,
)
from OpenPinch.analysis.heat_exchanger_networks.solver.arrays import (
    problem_to_solver_arrays,
)
from OpenPinch.analysis.heat_exchanger_networks.solver.dependencies import (
    MissingSynthesisDependencyError,
    MissingSynthesisSolverError,
)
from OpenPinch.application.problem import PinchProblem
from OpenPinch.domain.enums import HeatExchangerKind

D_TMIN = 10.0


def _highs_available() -> bool:
    try:
        backend.require_solver_backend("highs")
    except (MissingSynthesisDependencyError, MissingSynthesisSolverError):
        return False
    return True


pytestmark = pytest.mark.skipif(
    "not _highs_available()", reason="needs highspy and pyomo"
)


def _quantity(value, unit):
    return {"value": value, "unit": unit}


def _stream(name, supply, target, heat_flow):
    return {
        "zone": "Site/Plant",
        "name": name,
        "t_supply": _quantity(supply, "degC"),
        "t_target": _quantity(target, "degC"),
        "heat_flow": _quantity(heat_flow, "kW"),
        "dt_cont": _quantity(D_TMIN / 2, "delta_degC"),
        "htc": _quantity(1.0, "kW/m^2/delta_degC"),
    }


def _utility(name, kind, supply, target, price):
    return {
        "name": name,
        "type": kind,
        "t_supply": _quantity(supply, "degC"),
        "t_target": _quantity(target, "degC"),
        "dt_cont": _quantity(2.5, "delta_degC"),
        "price": _quantity(price, "$/MWh"),
        "htc": _quantity(1.0, "kW/m^2/delta_degC"),
        "heat_flow": None,
    }


def _problem(utilities, options: dict | None = None) -> PinchProblem:
    """H1 (220 to 150 C) and H2 (100 to 40 C) heat C1 (60 to 200 C).

    The pinch is at 95 C shifted. Above it C1 runs from 90 to 200 C: H1 can
    heat it from 130 C up, so 40 kW from 90 to 130 C, at the pinch end of
    the side, is utility. Below it H2 heats C1 to 90 C and cooling water
    takes the other 30 kW.
    """
    return PinchProblem(
        source={
            "streams": [
                _stream("H1", 220.0, 150.0, 70.0),
                _stream("H2", 100.0, 40.0, 60.0),
                _stream("C1", 60.0, 200.0, 140.0),
            ],
            "utilities": utilities,
            "zone_tree": {
                "name": "Site",
                "type": "Site",
                "children": [{"name": "Plant", "type": "Process Zone"}],
            },
        }
        | ({"options": options} if options else {})
    )


def _side(problem: PinchProblem, location: str, *, staged_utilities: bool):
    model = PinchDecompModel(
        name=f"{location}-staged",
        framework="PDM",
        solver="highs",
        solver_arrays=problem_to_solver_arrays(problem, D_TMIN),
        dTmin=D_TMIN,
        z_restriction=None,
        min_dqda=0,
        minimisation_goal="hot utility" if location == "above" else "cold utility",
        non_isothermal_model=False,
        integers=True,
        tol=1e-3,
        pinch_loc=location,
        pinch_decomposition=decomposition_module.build_pinch_design_decomposition(
            problem, D_TMIN, pinch_location=location
        ),
        stage_selection="automated",
        **({"staged_utilities": True} if staged_utilities else {}),
    )
    model.optimise(False)
    assert model.mSuccess == 1
    return model


def _duties(model, side: str, names) -> dict[str, float]:
    """Each utility's duty (kW) on a side, stream-end and staged together."""
    grid = model.Q_h_by_period[0] if side == "hot" else model.Q_c_by_period[0]
    staged = model.Q_hs_by_period[0] if side == "hot" else model.Q_cs_by_period[0]
    duties = dict.fromkeys(names, 0.0)
    for row in grid:
        for u, duty in enumerate(row):
            duties[names[u]] += _value(duty)
    for row in staged:
        for u, by_k in enumerate(row):
            duties[names[u]] += sum(_value(duty) for duty in by_k)
    return duties


def _targets(problem: PinchProblem) -> dict[str, float]:
    result = problem.target.direct_heat_integration()
    return {
        utility.name: float(utility.heat_flow)
        for utility in (*result.hot_utilities, *result.cold_utilities)
    }


HEATING = [
    # LPS dearer than HPS: by price alone the side heats at its hot end.
    _utility("LPS", "Hot", 140.0, 139.9, 50.0),
    _utility("HPS", "Hot", 250.0, 249.9, 40.0),
    _utility("CW", "Cold", 20.0, 25.0, 5.0),
]


def test_without_staging_the_above_side_heats_by_price():
    problem = _problem(HEATING)

    duties = _duties(
        _side(problem, "above", staged_utilities=False), "hot", ("LPS", "HPS")
    )

    assert duties["HPS"] == pytest.approx(40.0, rel=1e-6)


def test_the_above_side_holds_each_hot_level_at_its_target():
    problem = _problem(HEATING)
    targets = _targets(problem)
    assert targets["LPS"] == pytest.approx(40.0, rel=1e-6)

    model = _side(problem, "above", staged_utilities=True)

    duties = _duties(model, "hot", ("LPS", "HPS"))
    assert duties["LPS"] == pytest.approx(targets["LPS"], rel=1e-6)
    assert duties["HPS"] == pytest.approx(targets["HPS"], abs=1e-6)
    assert model.n_hu_units == 1


def _cooling_problem() -> PinchProblem:
    """H1 (200 to 60 C) heats C2 (160 to 210 C) and C1 (50 to 120 C).

    H1 heats C2 to 190 C and HPS the last 20 kW. H1 has 40 kW to spare
    between C2 and C1, from 170 to 130 C, which raises LP steam (LPG,
    120 C) rather than going to cooling water.
    """
    return PinchProblem(
        source={
            "streams": [
                _stream("H1", 200.0, 60.0, 140.0),
                _stream("C1", 50.0, 120.0, 70.0),
                _stream("C2", 160.0, 210.0, 50.0),
            ],
            "utilities": [
                _utility("HPS", "Hot", 250.0, 249.9, 40.0),
                # LPG dearer than CW: by price alone the side cools at its end.
                _utility("LPG", "Cold", 120.0, 120.1, 10.0),
                _utility("CW", "Cold", 20.0, 25.0, 5.0),
            ],
            "zone_tree": {
                "name": "Site",
                "type": "Site",
                "children": [{"name": "Plant", "type": "Process Zone"}],
            },
        }
    )


def test_the_below_side_holds_each_cold_level_at_its_target():
    problem = _cooling_problem()
    targets = _targets(problem)
    assert targets["LPG"] > 1.0

    model = _side(problem, "below", staged_utilities=True)

    duties = _duties(model, "cold", ("LPG", "CW"))
    for name in ("LPG", "CW"):
        assert duties[name] == pytest.approx(targets[name], rel=1e-6, abs=1e-6)


def _pdm_network(problem: PinchProblem, **kwargs):
    pdm = InternalHeatExchangerNetworkProblem(
        solver_arrays=problem_to_solver_arrays(problem, D_TMIN),
        name="staged-pdm",
        framework="PDM",
        solver="highs",
        dTmin=D_TMIN,
        minimisation_goal="total utility",
        pinch_decompositions={
            location: decomposition_module.build_pinch_design_decomposition(
                problem, D_TMIN, pinch_location=location
            )
            for location in ("above", "below")
        },
        staged_utilities=True,
        **kwargs,
    )
    case = pdm.get_solution(print_output=False)
    assert case is not None, getattr(pdm, "solution_failure_reason", None)
    return pdm.extract_network(run_id="staged-pdm")


def _network_duties(network) -> dict[str, float]:
    duties: dict[str, float] = {}
    for exchanger in network.exchangers:
        if exchanger.kind is HeatExchangerKind.RECOVERY:
            continue
        utility = (
            exchanger.source_stream
            if exchanger.kind is HeatExchangerKind.HOT_UTILITY
            else exchanger.sink_stream
        ).split(".")[-1]
        duties[utility] = duties.get(utility, 0.0) + exchanger.state().duty
    return duties


@pytest.mark.parametrize(
    "make_problem",
    [lambda: _problem(HEATING), _cooling_problem, lambda: _utility_pinch_problem()],
    ids=["heating", "cooling", "utility pinch"],
)
def test_a_pdm_network_meets_each_level_target_and_verifies(make_problem):
    problem = make_problem()
    targets = _targets(problem)

    network = _pdm_network(problem)

    duties = _network_duties(network)
    for name, target in targets.items():
        assert duties.get(name, 0.0) == pytest.approx(target, rel=1e-6, abs=1e-6)
    assert verify_network_feasibility(network) == ()


def _couenne_available() -> bool:
    try:
        backend.require_solver_backend("couenne")
    except (MissingSynthesisDependencyError, MissingSynthesisSolverError):
        return False
    return True


@pytest.mark.solver
@pytest.mark.skipif("not _couenne_available()", reason="needs couenne")
def test_a_stage_wise_task_takes_on_its_pdm_parents_staged_units():
    problem = _problem(HEATING)
    pdm = InternalHeatExchangerNetworkProblem(
        solver_arrays=problem_to_solver_arrays(problem, D_TMIN),
        name="staged-pdm",
        framework="PDM",
        solver="highs",
        dTmin=D_TMIN,
        minimisation_goal="total utility",
        pinch_decompositions={
            location: decomposition_module.build_pinch_design_decomposition(
                problem, D_TMIN, pinch_location=location
            )
            for location in ("above", "below")
        },
        staged_utilities=True,
    )
    assert pdm.get_solution(print_output=False) is not None
    tdm = InternalHeatExchangerNetworkProblem(
        solver_arrays=problem_to_solver_arrays(problem, D_TMIN),
        name="staged-tdm",
        framework="TDM",
        solver="couenne",
        dTmin=D_TMIN,
        minimisation_goal="total cost",
        parent=pdm,
    )

    tdm.load_model()

    assert tdm.case.staged_utilities
    assert stagewise_staged.staged_units_on(
        tdm.case
    ) == stagewise_staged.staged_units_on(pdm.case)
    assert stagewise_staged.staged_units_on(tdm.case)


def test_a_side_offers_staged_heaters_above_the_pinch_and_coolers_below():
    problem = _problem(HEATING)

    above = _side(problem, "above", staged_utilities=True)
    below = _side(problem, "below", staged_utilities=True)

    def offered(allowed):
        return any(flag for row in allowed for by_k in row for flag in by_k)

    assert offered(above.z_hs_allowed) and not offered(above.z_cs_allowed)
    assert offered(below.z_cs_allowed) and not offered(below.z_hs_allowed)


def _steam_raising_problem() -> PinchProblem:
    """H1 (200 to 195 C, 32 kW) raises LP steam and helps H2 heat C1.

    Only H1 is hot enough to raise the steam (LPG, 150 C), and it must keep
    26 kW for it: H2 (150 to 40 C) heats C1 (100 to 120 C) from its top,
    and H1 only the last 6 kW. A design that ticks C1 off with H1 alone
    leaves too little for the steam.
    """
    return PinchProblem(
        source={
            "streams": [
                _stream("H1", 200.0, 195.0, 32.0),
                _stream("H2", 150.0, 40.0, 11.0),
                _stream("C1", 100.0, 120.0, 10.0),
            ],
            "utilities": [
                _utility("HPS", "Hot", 250.0, 249.9, 40.0),
                _utility("LPG", "Cold", 150.0, 150.1, 10.0),
                _utility("CW", "Cold", 20.0, 25.0, 5.0),
            ],
            "zone_tree": {
                "name": "Site",
                "type": "Site",
                "children": [{"name": "Plant", "type": "Process Zone"}],
            },
        }
    )


def _utility_pinch_problem() -> PinchProblem:
    """H1 (200 to 199 C, 100 kW) raises LP steam; H2 heats C1's cold end.

    C1 (50 to 190 C) meets H1 at the pinch. H2 (120 to 40 C) can heat it
    to 110 C, so H1 heats it only above 110 C and raises 92 kW of steam
    (LPG, 180 C). The steam leaves no heat to spare at 115 C shifted, a
    utility pinch: a pinch match that ticks C1 off with H1 alone crosses it.
    """
    return PinchProblem(
        source={
            "streams": [
                _stream("H1", 200.0, 199.0, 100.0),
                _stream("H2", 120.0, 40.0, 8.0),
                _stream("C1", 50.0, 190.0, 14.0),
            ],
            "utilities": [
                _utility("HPS", "Hot", 250.0, 249.9, 40.0),
                _utility("LPG", "Cold", 180.0, 180.1, 10.0),
                _utility("CW", "Cold", 20.0, 25.0, 5.0),
            ],
            "zone_tree": {
                "name": "Site",
                "type": "Site",
                "children": [{"name": "Plant", "type": "Process Zone"}],
            },
        }
    )


@pytest.mark.parametrize(
    "make_problem",
    [
        lambda: _problem(HEATING),
        _cooling_problem,
        _steam_raising_problem,
        _utility_pinch_problem,
    ],
    ids=["heating", "cooling", "steam raising", "utility pinch"],
)
def test_heuristic_proposals_hold_each_level_at_its_target(make_problem, caplog):
    """The heuristic designs each level's duty into its proposals.

    Each level is matched like a process stream that must be used, so the
    process matches leave the level its share of the partners, nearest the
    pinch where its temperature serves.
    """
    problem = make_problem()
    targets = _targets(problem)

    with caplog.at_level("WARNING"):
        network = _pdm_network(problem, heuristic_network_limit=3)

    assert "utility levels" not in caplog.text
    duties = _network_duties(network)
    for name, target in targets.items():
        assert duties.get(name, 0.0) == pytest.approx(target, rel=1e-6, abs=1e-6)
    assert verify_network_feasibility(network) == ()


def test_a_side_that_cannot_hold_its_levels_falls_back_with_a_warning(caplog):
    problem = _problem(HEATING)
    decompositions = {
        location: decomposition_module.build_pinch_design_decomposition(
            problem, D_TMIN, pinch_location=location
        )
        for location in ("above", "below")
    }
    # Levels the above side cannot meet: none of its 40 kW on any level.
    above = decompositions["above"]
    decompositions["above"] = above.model_copy(
        update={
            "period_targets": tuple(
                target.model_copy(
                    update={
                        "hot_utility_level_targets": tuple(
                            0.0 for _ in target.hot_utility_level_targets
                        )
                    }
                )
                for target in above.period_targets
            )
        }
    )
    pdm = InternalHeatExchangerNetworkProblem(
        solver_arrays=problem_to_solver_arrays(problem, D_TMIN),
        name="staged-pdm",
        framework="PDM",
        solver="highs",
        dTmin=D_TMIN,
        minimisation_goal="total utility",
        pinch_decompositions=decompositions,
        staged_utilities=True,
    )

    with caplog.at_level("WARNING"):
        case = pdm.get_solution(print_output=False)

    assert case is not None, getattr(pdm, "solution_failure_reason", None)
    assert "cannot hold its utility levels" in caplog.text
    network = pdm.extract_network(run_id="staged-pdm")
    duties = _network_duties(network)
    assert duties.get("LPS", 0.0) + duties.get("HPS", 0.0) == pytest.approx(
        40.0, rel=1e-6
    )
    assert verify_network_feasibility(network) == ()


def test_a_stream_absent_from_a_side_has_no_staged_units_there():
    # H1 (220 to 150 C) lies wholly above the pinch.
    below = _side(_problem(HEATING), "below", staged_utilities=True)

    (h1,) = [i for i in range(below.I) if not below.z_i_active[i]]
    assert not any(flag for by_k in below.z_cs_allowed[h1] for flag in by_k)


def test_an_amalgamated_mid_temperature_without_a_staged_unit_is_the_boundarys():
    problem = _problem(HEATING)
    pdm = InternalHeatExchangerNetworkProblem(
        solver_arrays=problem_to_solver_arrays(problem, D_TMIN),
        name="staged-pdm",
        framework="PDM",
        solver="highs",
        dTmin=D_TMIN,
        minimisation_goal="total utility",
        pinch_decompositions={
            location: decomposition_module.build_pinch_design_decomposition(
                problem, D_TMIN, pinch_location=location
            )
            for location in ("above", "below")
        },
        staged_utilities=True,
    )
    case = pdm.get_solution(print_output=False)

    on = stagewise_staged.staged_units_on(case)
    for side, mids, boundaries in (
        ("hot", case.T_c_mid_by_period, case.T_c_by_period),
        ("cold", case.T_h_mid_by_period, case.T_h_by_period),
    ):
        for s, row in enumerate(mids[0]):
            for k in range(1, case.S):
                if any(
                    unit[0] == side and unit[1] == s and unit[3] == k for unit in on
                ):
                    continue
                assert _value(row[k]) == pytest.approx(
                    _value(boundaries[0][s][k]), abs=1e-9
                )


def test_the_workflow_holds_levels_only_with_more_than_one_utility_on_a_side():
    from OpenPinch.analysis.heat_exchanger_networks.execution import executor

    levels = decomposition_module.build_pinch_design_decomposition(
        _problem(HEATING), D_TMIN, pinch_location="above"
    )
    single = decomposition_module.build_pinch_design_decomposition(
        _problem(
            [
                _utility("HPS", "Hot", 250.0, 249.9, 40.0),
                _utility("CW", "Cold", 20.0, 25.0, 5.0),
            ]
        ),
        D_TMIN,
        pinch_location="above",
    )

    assert executor._has_utility_levels(levels)
    assert not executor._has_utility_levels(single)


@pytest.mark.parametrize("zone", ["Food Plant[S]", "Chemical Plant[S]"])
def test_heuristic_designs_hold_the_zonal_sites_levels(zone, caplog):
    """The heuristic meets each level of two ``zonal_site`` processes.

    Food Plant's condensing stream must keep most of its heat to raise steam;
    Chemical Plant's steam raising leaves a utility pinch below the process
    pinch that no match may cross.
    """
    from tests.analysis.heat_exchanger_networks.test_pdm_level_targets import (
        _process,
        _targeted,
    )

    problem = _process(zone)
    # Keyed "Hot Utility.<name>" or "Cold Utility.<name>": a steam level can
    # be both a hot and a cold utility.
    targets = _targeted(problem, zone)

    with caplog.at_level("WARNING"):
        network = _pdm_network(problem, heuristic_network_limit=3)

    assert "utility levels" not in caplog.text
    duties: dict[str, float] = {}
    for exchanger in network.exchangers:
        if exchanger.kind is HeatExchangerKind.RECOVERY:
            continue
        utility = (
            exchanger.source_stream
            if exchanger.kind is HeatExchangerKind.HOT_UTILITY
            else exchanger.sink_stream
        )
        duties[utility] = duties.get(utility, 0.0) + exchanger.state().duty
    for name, target in targets.items():
        # The heuristic counts a load within 0.1% as met.
        assert duties.get(name, 0.0) == pytest.approx(target, rel=1e-3, abs=1e-3)
    assert verify_network_feasibility(network) == ()


@pytest.mark.parametrize(
    ("placement", "held"), [(None, True), ("temperature", True), ("price", False)]
)
def test_the_workflow_places_utilities_by_temperature_or_by_price(placement, held):
    """By temperature, the pinch design holds each level at its target; by
    price, it picks the levels by cost."""
    from OpenPinch.analysis.heat_exchanger_networks.execution.executor import (
        LocalSynthesisExecutor,
    )
    from OpenPinch.analysis.heat_exchanger_networks.execution.settings import (
        workflow_settings_from_problem,
    )
    from OpenPinch.analysis.heat_exchanger_networks.targeting.pinch_design_method import (
        build_pinch_design_method_tasks,
    )

    options = {"HENS_SOLVER_PDM": "highs"}
    if placement is not None:
        options["HENS_UTILITY_PLACEMENT"] = placement
    problem = _problem(HEATING, options)
    task = build_pinch_design_method_tasks(workflow_settings_from_problem(problem))[0]

    built = LocalSynthesisExecutor(print_output=False)._build_problem(
        task, problem=problem, parent_outcomes={}
    )

    assert built.staged_utilities is held
