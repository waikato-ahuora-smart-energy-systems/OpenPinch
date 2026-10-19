"""HEN synthesis with more than one hot or cold utility (audit H6).

The superstructure gives every heater and cooler a utility index: each
(stream, utility) pair has its own duty, binary, approach constraints, cost
and area. The models used to index every utility array at 0, so a second
cold utility was never used, even where the first could not reach a target.
"""

from __future__ import annotations

import pytest

from OpenPinch.analysis.heat_exchanger_networks.extraction.service import (
    extract_heat_exchanger_network,
)
from OpenPinch.analysis.heat_exchanger_networks.models.stagewise import (
    StageWiseModel,
)
from OpenPinch.analysis.heat_exchanger_networks.reporting.verification import (
    verify_network_feasibility,
)
from OpenPinch.analysis.heat_exchanger_networks.solver.arrays import (
    problem_to_solver_arrays,
)
from OpenPinch.application.problem import PinchProblem
from OpenPinch.domain.enums import HeatExchangerKind

D_TMIN = 4.0


def _value(value, unit):
    return {"value": value, "unit": unit}


def _two_cold_utility_problem(hot_target: float, *, glycol_first: bool) -> PinchProblem:
    """H1 cools from 100 C to ``hot_target``; C1 heats from 10 to 40 C.

    Cooling water (20 to 25 C, 5 $/MWh) is cheap but cannot cool below
    about 24 C; glycol (-5 to 0 C, 60 $/MWh) can.
    """

    def stream(name, supply, target, heat_flow):
        return {
            "zone": "Site/Plant",
            "name": name,
            "t_supply": _value(supply, "degC"),
            "t_target": _value(target, "degC"),
            "heat_flow": _value(heat_flow, "kW"),
            "dt_cont": _value(D_TMIN / 2, "delta_degC"),
            "htc": _value(1.0, "kW/m^2/delta_degC"),
        }

    def utility(name, kind, supply, target, price):
        return {
            "name": name,
            "type": kind,
            "t_supply": _value(supply, "degC"),
            "t_target": _value(target, "degC"),
            "dt_cont": _value(D_TMIN / 2, "delta_degC"),
            "price": _value(price, "$/MWh"),
            "htc": _value(1.0, "kW/m^2/delta_degC"),
            "heat_flow": None,
        }

    cold_utilities = [
        utility("CW", "Cold", 20.0, 25.0, 5.0),
        utility("Glycol", "Cold", -5.0, 0.0, 60.0),
    ]
    if glycol_first:
        cold_utilities.reverse()
    return PinchProblem(
        source={
            "options": {
                "COSTING_ANNUAL_OP_TIME": 1000.0,
                "COSTING_DISCOUNT_RATE": 0.0,
                "COSTING_SERVICE_LIFE": 1.0,
                "COSTING_HX_UNIT_COST": 100.0,
                "COSTING_HX_AREA_COEFF": 1.0,
            },
            "streams": [
                stream("H1", 100.0, hot_target, 10.0 * (100.0 - hot_target)),
                stream("C1", 10.0, 40.0, 150.0),
            ],
            "utilities": [
                utility("Steam", "Hot", 150.0, 149.9, 40.0),
                *cold_utilities,
            ],
            "zone_tree": {
                "name": "Site",
                "type": "Site",
                "children": [{"name": "Plant", "type": "Process Zone"}],
            },
        }
    )


def _solved_model(hot_target: float, *, glycol_first: bool):
    pytest.importorskip("gekko")
    arrays = problem_to_solver_arrays(
        _two_cold_utility_problem(hot_target, glycol_first=glycol_first), D_TMIN
    )
    model = StageWiseModel(
        name=f"two-cold-utilities-{hot_target}",
        framework="TDM",
        solver="apopt",
        solver_arrays=arrays,
        stages=1,
        dTmin=D_TMIN,
        z_restriction=None,
        min_dqda=0.0,
        minimisation_goal="total cost",
        non_isothermal_model=False,
        # Relaxed match binaries: APOPT's branch and bound fails on the glycol
        # case, which the relaxed model, IPOPT and Couenne all solve.
        integers=False,
        tol=1e-6,
    )
    model.optimise(False)
    assert model.mSuccess == 1
    return model, arrays


def _cooler_duties(network) -> dict[str, float]:
    return {
        exchanger.sink_stream.split(".")[-1]: exchanger.state().duty
        for exchanger in network.exchangers
        if exchanger.kind is HeatExchangerKind.COLD_UTILITY and exchanger.state().active
    }


@pytest.mark.synthesis
@pytest.mark.parametrize("glycol_first", [False, True])
@pytest.mark.parametrize(
    ("hot_target", "expected_utility"),
    [(30.0, "CW"), (5.0, "Glycol")],
)
def test_apopt_stagewise_cools_with_the_cheapest_feasible_utility(
    hot_target, expected_utility, glycol_first
):
    model, arrays = _solved_model(hot_target, glycol_first=glycol_first)

    network = extract_heat_exchanger_network(model, arrays, run_id="two-utilities")

    # C1 takes 150 kW from H1; the rest of H1 goes to one cold utility.
    cooling = 10.0 * (100.0 - hot_target) - 150.0
    duties = _cooler_duties(network)
    assert duties == {expected_utility: pytest.approx(cooling, abs=1e-3)}
    assert verify_network_feasibility(network) == ()


def test_default_backup_utilities_stay_out_of_synthesis():
    import json

    from tests.support.paths import FIXTURES_ROOT

    # Cooling water cannot take all of this case's cooling, so targeting adds
    # a default cold utility "CU" as a backup. With every utility now in the
    # superstructure, it would become a free way to cool anything.
    fixture = FIXTURES_ROOT / "hen_audit" / "utility_cross_cooling_water.json"
    problem = PinchProblem(source=json.loads(fixture.read_text()))

    axis_maps = problem_to_solver_arrays(problem, D_TMIN).axis_maps

    assert axis_maps["hot_utilities"] == {"Hot Utility.HPS": 0}
    assert axis_maps["cold_utilities"] == {"Cold Utility.CW": 0}


@pytest.mark.solver
def test_dairy_chilling_synthesis_cools_chilled_streams_with_glycol():
    # Every chilling stream ends at 2 to 5 C, below cooling water (20 to
    # 25 C); synthesis used only the first cold utility and so failed.
    from dataclasses import replace

    from OpenPinch.analysis.heat_exchanger_networks.execution.executor import (
        LocalSynthesisExecutor,
    )
    from OpenPinch.analysis.heat_exchanger_networks.execution.settings import (
        workflow_settings_from_problem,
    )
    from OpenPinch.analysis.heat_exchanger_networks.solver.dependencies import (
        MissingSynthesisDependencyError,
        MissingSynthesisSolverError,
        require_solver_binary,
    )
    from OpenPinch.analysis.heat_exchanger_networks.targeting.open_hens_method import (
        execute_open_hens_method,
    )

    try:
        for binary in ("couenne", "ipopt"):
            require_solver_binary(binary, purpose="multi-utility regression")
    except (MissingSynthesisDependencyError, MissingSynthesisSolverError) as exc:
        pytest.skip(str(exc))
    problem = PinchProblem("dairy_chilling_multiperiod.json")
    problem.target.all_heat_integration()
    settings = replace(
        workflow_settings_from_problem(problem),
        approach_temperatures=(5.0,),
        max_parallel=1,
    )

    result = execute_open_hens_method(
        problem, settings, executor=LocalSynthesisExecutor(print_output=False)
    )

    network = result.accepted_result.network
    assert verify_network_feasibility(network) == ()
    coolers = {
        exchanger.source_stream.split(".")[-1]: exchanger.sink_stream.split(".")[-1]
        for exchanger in network.exchangers
        if exchanger.kind is HeatExchangerKind.COLD_UTILITY
        and any(state.active for state in exchanger.period_states)
    }
    for chilled in (
        "Raw milk chilling",
        "Ice-water return",
        "Pasteurised milk final cooling",
    ):
        assert coolers.get(chilled) == "Glycol", coolers
