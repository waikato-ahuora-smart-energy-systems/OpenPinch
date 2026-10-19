"""Heater and cooler approach temperatures in HEN synthesis (audit C3).

The cross case has cooling water at 320 to 325 K and a hot stream H1 that
must leave at 322 K. A cooler on H1 can then only start above
325 K + dT_cu; with no approach constraint on the utility exchanger, the
models used to cool H1 to 324 K by recovery and finish it with a crossed
cooler (hot-end dT -1 K) that was given a fallback LMTD area.
"""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

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
from tests.support.paths import FIXTURES_ROOT

CROSS_CASE = FIXTURES_ROOT / "hen_audit" / "utility_cross_cooling_water.json"
D_TMIN = 2.0


def _cross_problem() -> PinchProblem:
    return PinchProblem(source=json.loads(CROSS_CASE.read_text()))


def _utility_terminal_differences(model, n: int = 0):
    """Yield (kind, index, hot-end dT, cold-end dT, required) per active unit."""
    for i in range(model.I):
        if model._active_binary_value(model.Q_c_by_period[n][i]) > model.tol:
            yield (
                "cooler",
                i,
                model._active_binary_value(model.T_h_by_period[n][i][model.S])
                - model.T_cu_out_period[n][0],
                model.T_h_out_period[n][i] - model.T_cu_in_period[n][0],
                model.dT_cu_period[n][i],
            )
    for j in range(model.J):
        if model._active_binary_value(model.Q_h_by_period[n][j]) > model.tol:
            yield (
                "heater",
                j,
                model.T_hu_in_period[n][0] - model.T_c_out_period[n][j],
                model.T_hu_out_period[n][0]
                - model._active_binary_value(model.T_c_by_period[n][j][0]),
                model.dT_hu_period[n][j],
            )


def _fixed_topology(arrays):
    """H1-C1 and H2-C2 in stage 0, with a heater and cooler on every stream."""
    hot = arrays.axis_maps["hot_process_streams"]
    cold = arrays.axis_maps["cold_process_streams"]
    pairs = {(hot[f"Process A.H{n}"], cold[f"Process A.C{n}"]) for n in (1, 2)}
    z = [
        [[1 if (i, j) in pairs and k == 0 else 0 for k in range(2)] for j in range(2)]
        for i in range(2)
    ]
    return [z, [1, 1], [1, 1]]


def _solved_fixed_topology_model(non_isothermal_model: bool):
    pytest.importorskip("gekko")
    arrays = problem_to_solver_arrays(_cross_problem(), D_TMIN)
    model = StageWiseModel(
        name=f"utility-cross-noniso-{non_isothermal_model}",
        framework="TDM",
        solver="apopt",
        solver_arrays=arrays,
        stages=2,
        dTmin=D_TMIN,
        z_restriction=_fixed_topology(arrays),
        min_dqda=0.0,
        minimisation_goal="hot utility",
        non_isothermal_model=non_isothermal_model,
        integers=False,
        tol=1e-6,
    )

    model.optimise(False)
    assert model.mSuccess == 1
    return model, arrays


@pytest.mark.synthesis
@pytest.mark.parametrize("non_isothermal_model", [False, True])
def test_apopt_stagewise_utility_exchangers_keep_their_approach_temperature(
    non_isothermal_model,
):
    model, _arrays = _solved_fixed_topology_model(non_isothermal_model)

    units = list(_utility_terminal_differences(model))
    assert any(kind == "cooler" for kind, *_ in units)
    for kind, index, hot_end, cold_end, required in units:
        assert min(hot_end, cold_end) >= required - 1e-4, (kind, index)


@pytest.mark.synthesis
def test_apopt_extracted_network_records_utility_approaches():
    model, arrays = _solved_fixed_topology_model(non_isothermal_model=True)

    network = extract_heat_exchanger_network(model, arrays, run_id="utility")

    metadata = network.source_metadata
    # [period][hot utility][cold stream] and [period][hot stream][cold utility]
    assert metadata["hot_utility_minimum_approach_by_period"] == [
        [list(model.dT_hu_period[0])]
    ]
    assert metadata["cold_utility_minimum_approach_by_period"] == [
        [[value] for value in model.dT_cu_period[0]]
    ]
    assert verify_network_feasibility(network) == ()


@pytest.mark.solver
def test_open_hens_method_networks_have_no_utility_cross():
    from OpenPinch.analysis.heat_exchanger_networks.errors import (
        WorkflowContractError,
    )
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
    from OpenPinch.domain.enums import HeatExchangerKind

    try:
        for binary in ("couenne", "ipopt"):
            require_solver_binary(binary, purpose="utility approach regression")
    except (MissingSynthesisDependencyError, MissingSynthesisSolverError) as exc:
        pytest.skip(str(exc))
    problem = _cross_problem()
    problem.target.all_heat_integration()
    settings = replace(workflow_settings_from_problem(problem), max_parallel=1)

    # The only feasible networks cool H1 with cooling water before recovery
    # (330 to 325 K, then 325 to 322 K against C1). The SYNHEAT superstructure
    # puts coolers after the last stage, so synthesis can only reject the case
    # or, if a later superstructure can place the cooler, solve it uncrossed.
    try:
        result = execute_open_hens_method(
            problem, settings, executor=LocalSynthesisExecutor(print_output=False)
        )
    except WorkflowContractError as exc:
        assert "no successful task outcomes" in str(exc)
        return

    networks = [
        outcome.network
        for outcome in result.outcomes
        if outcome.status == "success" and outcome.network is not None
    ]
    assert networks
    for network in networks:
        assert verify_network_feasibility(network) == ()
        for exchanger in network.exchangers:
            if exchanger.kind is HeatExchangerKind.RECOVERY:
                continue
            state = exchanger.period_states[0]
            if not state.active:
                continue
            hot_end = state.source_inlet_temperature - state.sink_outlet_temperature
            cold_end = state.source_outlet_temperature - state.sink_inlet_temperature
            assert min(hot_end, cold_end) >= D_TMIN - 1e-3, exchanger.exchanger_id


def _recovery_only_problem() -> PinchProblem:
    """H1 150 to 60 C and C1 40 to 130 C balance exactly by recovery.

    Steam at 135 C cannot heat C1 to 130 C with a 10 K approach.
    """

    def value(number, unit):
        return {"value": number, "unit": unit}

    def stream(name, supply, target):
        return {
            "zone": "Site/Plant",
            "name": name,
            "t_supply": value(supply, "degC"),
            "t_target": value(target, "degC"),
            "heat_flow": value(900.0, "kW"),
            "dt_cont": value(5.0, "delta_degC"),
            "htc": value(1.0, "kW/m^2/delta_degC"),
        }

    def utility(name, kind, supply, target):
        return {
            "name": name,
            "type": kind,
            "t_supply": value(supply, "degC"),
            "t_target": value(target, "degC"),
            "dt_cont": value(5.0, "delta_degC"),
            "price": value(10.0, "$/MWh"),
            "htc": value(1.0, "kW/m^2/delta_degC"),
            "heat_flow": None,
        }

    return PinchProblem(
        source={
            "streams": [stream("H1", 150.0, 60.0), stream("C1", 40.0, 130.0)],
            "utilities": [
                utility("Steam", "Hot", 135.0, 134.9),
                utility("CW", "Cold", 20.0, 40.0),
            ],
            "zone_tree": {
                "name": "Site",
                "type": "Site",
                "children": [{"name": "Plant", "type": "Process Zone"}],
            },
        }
    )


@pytest.mark.synthesis
def test_apopt_relaxed_model_builds_with_an_unusable_heater():
    # Stage reduction and evolution build relaxed models whose utility
    # binaries are fixed at 1. A heater whose fixed end cannot meet its
    # approach then carries no duty; construction used to raise on it.
    pytest.importorskip("gekko")
    arrays = problem_to_solver_arrays(_recovery_only_problem(), 10.0)
    model = StageWiseModel(
        name="recovery-only-relaxed",
        framework="TDM",
        solver="apopt",
        solver_arrays=arrays,
        stages=1,
        dTmin=10.0,
        z_restriction=[[[[1]]], None, None],
        min_dqda=0.0,
        minimisation_goal="hot utility",
        non_isothermal_model=False,
        integers=False,
        tol=1e-6,
    )

    model.optimise(False)

    assert model.mSuccess == 1
    assert model.Q_r_total == pytest.approx(900.0, abs=1e-3)
    assert model.Q_hu_total == pytest.approx(0.0, abs=1e-3)
    assert model.Q_cu_total == pytest.approx(0.0, abs=1e-3)


def test_stage_reduction_keeps_only_the_heaters_and_coolers_in_use(monkeypatch):
    # The reduced model is relaxed, so every allowed heater and cooler has its
    # binary fixed at 1 and both approach constraints. Restricting them to the
    # ones the solved case uses keeps an unused unit from constraining it.
    from OpenPinch.analysis.heat_exchanger_networks.models import problem as unit
    from OpenPinch.analysis.heat_exchanger_networks.models.problem import (
        InternalHeatExchangerNetworkProblem,
    )

    class _Built(Exception):
        pass

    built: dict = {}

    def record(**kwargs):
        built.update(kwargs)
        raise _Built

    monkeypatch.setattr(unit, "StageWiseModel", record)
    case = SimpleNamespace(
        mSuccess=1,
        name="case",
        framework="TDM",
        solver_arrays=object(),
        dTmin=10.0,
        min_dqda=0.0,
        minimisation_goal="total cost",
        non_isothermal_model=False,
        tol=1e-3,
        I=2,
        J=1,
        S=2,
        # Stage 2 is empty, so the case is reduced to one stage.
        Q_r=[[[[10.0], [0.0]]], [[[5.0], [0.0]]]],
        Q_h=[[3.0]],
        Q_c=[[0.0], [4.0]],
    )

    with pytest.raises(_Built):
        InternalHeatExchangerNetworkProblem(
            solver_arrays=object(), framework="TDM", stages=2
        ).remove_unused_stages(case)

    assert built["stages"] == 1
    assert built["z_restriction"] == [[[[1]], [[1]]], [1], [0, 1]]


def test_stage_reduction_keeps_heaters_and_coolers_used_in_any_period(monkeypatch):
    # Q_h and Q_c are period 0's duties; a unit used only in a later period
    # must stay available in the reduced model.
    from OpenPinch.analysis.heat_exchanger_networks.models import problem as unit
    from OpenPinch.analysis.heat_exchanger_networks.models.problem import (
        InternalHeatExchangerNetworkProblem,
    )

    class _Built(Exception):
        pass

    built: dict = {}

    def record(**kwargs):
        built.update(kwargs)
        raise _Built

    monkeypatch.setattr(unit, "StageWiseModel", record)
    hot_utility_by_period = [[[0.0]], [[3.0]]]
    cold_utility_by_period = [[[0.0], [4.0]], [[2.0], [0.0]]]
    case = SimpleNamespace(
        mSuccess=1,
        name="case",
        framework="TDM",
        solver_arrays=object(),
        dTmin=10.0,
        min_dqda=0.0,
        minimisation_goal="total cost",
        non_isothermal_model=False,
        tol=1e-3,
        I=2,
        J=1,
        S=2,
        Q_r=[[[[10.0], [0.0]]], [[[5.0], [0.0]]]],
        Q_h=hot_utility_by_period[0],
        Q_c=cold_utility_by_period[0],
        Q_h_by_period=hot_utility_by_period,
        Q_c_by_period=cold_utility_by_period,
    )

    with pytest.raises(_Built):
        InternalHeatExchangerNetworkProblem(
            solver_arrays=object(), framework="TDM", stages=2
        ).remove_unused_stages(case)

    assert built["z_restriction"][1:] == [[1], [1, 1]]
