"""Regression tests for the HEN audit fixes (area 5)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from OpenPinch.analysis.heat_exchanger_networks.execution.task_builders import (
    has_recovery_topology,
)
from OpenPinch.domain.enums import HeatExchangerKind


def _exchanger(kind, *, active=True, stage=1):
    return SimpleNamespace(
        kind=kind,
        match_allowed=True,
        stage=stage,
        period_states=(SimpleNamespace(active=active),),
    )


def test_utility_only_outcome_cannot_seed_later_methods():
    utility_only = SimpleNamespace(
        network=SimpleNamespace(
            exchangers=(_exchanger(HeatExchangerKind.HOT_UTILITY, stage=None),)
        )
    )
    with_recovery = SimpleNamespace(
        network=SimpleNamespace(exchangers=(_exchanger(HeatExchangerKind.RECOVERY),))
    )

    assert has_recovery_topology(utility_only) is False
    assert has_recovery_topology(with_recovery) is True
    assert has_recovery_topology(SimpleNamespace(network=None)) is False


# 5.6 Per-period stream heat balances


def _two_period_network(*, cold_utility_p1: float, by_period: bool = True):
    from OpenPinch.domain.enums import StreamID
    from OpenPinch.domain.heat_exchanger import HeatExchanger
    from OpenPinch.domain.heat_exchanger_network import HeatExchangerNetwork

    def states(*duties):
        return tuple(
            {"period_id": f"p{index}", "period_idx": index, "duty": duty}
            for index, duty in enumerate(duties)
        )

    # H1 (CP 10, then 5; 500 -> 400) gives 1000 then 500; C1 (CP 10, then 6;
    # 300 -> 350) takes 500 then 300. The rest of H1 goes to cooling.
    metadata = {
        "hot_stream_heat_capacity_flowrates": [10.0],
        "cold_stream_heat_capacity_flowrates": [10.0],
        "hot_stream_supply_temperatures": [500.0],
        "hot_stream_target_temperatures": [400.0],
        "cold_stream_supply_temperatures": [300.0],
        "cold_stream_target_temperatures": [350.0],
    }
    if by_period:
        metadata |= {
            "hot_stream_heat_capacity_flowrates_by_period": [[10.0], [5.0]],
            "cold_stream_heat_capacity_flowrates_by_period": [[10.0], [6.0]],
            "hot_stream_supply_temperatures_by_period": [[500.0], [500.0]],
            "hot_stream_target_temperatures_by_period": [[400.0], [400.0]],
            "cold_stream_supply_temperatures_by_period": [[300.0], [300.0]],
            "cold_stream_target_temperatures_by_period": [[350.0], [350.0]],
        }
    return HeatExchangerNetwork(
        exchangers=(
            HeatExchanger(
                exchanger_id="R1",
                kind=HeatExchangerKind.RECOVERY,
                source_stream="H1",
                sink_stream="C1",
                source_stream_role=StreamID.Process,
                sink_stream_role=StreamID.Process,
                stage=1,
                period_states=states(500.0, 300.0),
            ),
            HeatExchanger(
                exchanger_id="CU1",
                kind=HeatExchangerKind.COLD_UTILITY,
                source_stream="H1",
                sink_stream="CU",
                source_stream_role=StreamID.Process,
                sink_stream_role=StreamID.Utility,
                period_states=states(500.0, cold_utility_p1),
            ),
        ),
        solver_axis_metadata={
            "axis_maps": {
                "hot_process_streams": {"H1": 0},
                "cold_process_streams": {"C1": 0},
            }
        },
        source_metadata=metadata,
    )


def test_multiperiod_stream_heat_balances_are_checked_per_period():
    from OpenPinch.analysis.heat_exchanger_networks.reporting.verification import (
        _check_stream_heat_balances,
    )

    balanced = _two_period_network(cold_utility_p1=200.0)
    short = _two_period_network(cold_utility_p1=100.0)

    for period_id in ("p0", "p1"):
        assert _check_stream_heat_balances(balanced, period_id) == []
    assert _check_stream_heat_balances(short, "p0") == []
    (failure,) = _check_stream_heat_balances(short, "p1")
    assert "hot stream H1" in failure and "in period 'p1'" in failure


def test_networks_without_period_stream_data_are_not_checked_per_period():
    from OpenPinch.analysis.heat_exchanger_networks.reporting.verification import (
        _check_stream_heat_balances,
    )

    # Period 1 loads differ from the single-period values; without per-period
    # data the check must not compare them.
    legacy = _two_period_network(cold_utility_p1=200.0, by_period=False)

    assert _check_stream_heat_balances(legacy, "p1") == []


def test_extraction_exports_stream_data_per_period():
    import numpy as np

    from OpenPinch.analysis.heat_exchanger_networks.extraction.service import (
        _period_matrix,
        _segment_parent_total_duties_by_period,
    )

    arrays = SimpleNamespace(
        arrays={
            "T_h_in_period": np.array([[500.0, 450.0], [490.0, 440.0]]),
            # [period][stream][segment]
            "hot_segment_duty_period": np.array(
                [[[100.0, 50.0], [20.0, 0.0]], [[80.0, 40.0], [10.0, 5.0]]]
            ),
        }
    )
    model = SimpleNamespace(T_h_out_period=np.array([[400.0, 350.0], [390.0, 340.0]]))

    assert _period_matrix(model, arrays, "T_h_out_period") == [
        [400.0, 350.0],
        [390.0, 340.0],
    ]
    assert _period_matrix(model, arrays, "T_h_in_period") == [
        [500.0, 450.0],
        [490.0, 440.0],
    ]
    assert _period_matrix(model, arrays, "T_c_in_period") == []
    assert _segment_parent_total_duties_by_period(arrays, "hot") == [
        [150.0, 20.0],
        [120.0, 15.0],
    ]
    assert _segment_parent_total_duties_by_period(arrays, "cold") == []


# HEN costs on the same $/y basis as area targeting


def test_hen_costs_are_annualised_like_area_targeting():
    import json

    import numpy as np

    from OpenPinch.analysis.economics import compute_capital_recovery_factor
    from OpenPinch.analysis.heat_exchanger_networks.solver.arrays import (
        problem_to_solver_arrays,
    )
    from OpenPinch.application.problem import PinchProblem
    from tests.support.paths import FIXTURES_ROOT

    fixture = FIXTURES_ROOT / "openhens" / "Four-stream-Yee-and-Grossmann-1990-1.json"
    benchmark = json.loads(fixture.read_text())
    annual = json.loads(fixture.read_text())
    annual["options"].update(
        {
            "COSTING_ANNUAL_OP_TIME": 8300.0,
            "COSTING_DISCOUNT_RATE": 0.07,
            "COSTING_SERVICE_LIFE": 20.0,
        }
    )

    as_quoted = problem_to_solver_arrays(PinchProblem(source=benchmark), 14.0).arrays
    annualised = problem_to_solver_arrays(PinchProblem(source=annual), 14.0).arrays

    # The benchmark's own basis ($/kW/y and $/y) passes through unchanged.
    np.testing.assert_allclose(as_quoted["hu_cost_period"], [[80.0]])
    np.testing.assert_allclose(as_quoted["cu_cost_period"], [[15.0]])
    np.testing.assert_allclose(as_quoted["A_coeff"], [150.0])
    np.testing.assert_allclose(as_quoted["unit_cost"], [5500.0])

    crf = compute_capital_recovery_factor(0.07, 20.0)
    np.testing.assert_allclose(annualised["hu_cost_period"], [[80.0 * 8.3]])
    np.testing.assert_allclose(annualised["cu_cost_period"], [[15.0 * 8.3]])
    for name, quoted in (("A_coeff", 150.0), ("unit_cost", 5500.0)):
        np.testing.assert_allclose(annualised[name], [quoted * crf])
    np.testing.assert_allclose(annualised["A_exp"], as_quoted["A_exp"])


# verify() compares the solver's utility cost with a recomputation


@pytest.mark.synthesis
def test_apopt_stagewise_verify_compares_solver_utility_cost():
    pytest.importorskip("gekko")
    import json

    from OpenPinch.analysis.heat_exchanger_networks.models._stagewise import (
        verification,
    )
    from OpenPinch.analysis.heat_exchanger_networks.models.stagewise import (
        StageWiseModel,
    )
    from OpenPinch.analysis.heat_exchanger_networks.solver.arrays import (
        problem_to_solver_arrays,
    )
    from OpenPinch.application.problem import PinchProblem
    from tests.support.paths import FIXTURES_ROOT

    fixture = FIXTURES_ROOT / "hen_audit" / "utility_cross_cooling_water.json"
    model = StageWiseModel(
        name="verify-utility-cost",
        framework="TDM",
        solver="apopt",
        solver_arrays=problem_to_solver_arrays(
            PinchProblem(source=json.loads(fixture.read_text())), 2.0
        ),
        stages=1,
        dTmin=2.0,
        z_restriction=None,
        min_dqda=0.0,
        minimisation_goal="total cost",
        non_isothermal_model=False,
        integers=False,
        tol=1e-6,
    )
    model.optimise(False)
    assert model.mSuccess == 1
    assert verification._check_utility_costs(model) is True

    # A solver whose cost expression disagrees with its duties must fail,
    # even though post-processing recomputed hu_cost_total from the duties.
    solved = model.hu_cost_solver_by_period[0]
    solved.VALUE.value = [2.0 * verification._value(solved)]

    assert verification._check_utility_costs(model) is False


# First-audit H10: the first stage has no cold flow-on, whatever S is


@pytest.mark.parametrize(("stages", "cold_flow_on_stages"), [(1, []), (2, [1])])
def test_only_stages_after_the_first_get_cold_gamma_flow_on(
    monkeypatch, stages, cold_flow_on_stages
):
    from OpenPinch.analysis.heat_exchanger_networks.models._base import alpha

    cold_calls: list[int] = []
    monkeypatch.setattr(
        alpha,
        "_cold_gamma_flow_on",
        lambda model, i, j, k: cold_calls.append(k) or 1.0,
    )
    monkeypatch.setattr(alpha, "_hot_gamma_flow_on", lambda model, i, j, k: 1.0)
    model = SimpleNamespace(
        I=1,
        J=1,
        S=stages,
        z_allowed=[[[1] * stages]],
        gamma_h=[[[0.0] * stages]],
        gamma_c=[[[0.0] * stages]],
    )
    equations = SimpleNamespace(Equation=lambda expression: expression)

    alpha._set_gamma_equations(model, equations)

    # A single stage is both the last and the first: gamma_c used to take
    # flow-on from stage k - 1 = -1, the stage itself.
    assert cold_calls == cold_flow_on_stages


# M7: the accepted network is the cheapest feasible one from any method


def _outcome(method: str, task_id: str, total_annual_cost: float, status="success"):
    return SimpleNamespace(
        status=status,
        task=SimpleNamespace(method=method, task_id=task_id),
        objective_value=total_annual_cost,
    )


def test_accepted_outcome_is_the_cheapest_success_from_any_method():
    from OpenPinch.analysis.heat_exchanger_networks.results.assembly import (
        _accepted_outcome,
    )

    outcomes = (
        _outcome("pinch_design_method", "pdm", 1530.0),
        _outcome("thermal_derivative_method", "tdm", 1610.0),
        _outcome("network_evolution_method", "evm", 4819.5),
        _outcome("network_evolution_method", "evm-failed", 10.0, status="failed"),
    )

    accepted = _accepted_outcome(outcomes)

    assert accepted.task.task_id == "pdm"


def test_ranked_networks_compare_every_method():
    from OpenPinch.analysis.heat_exchanger_networks.reporting import ranking
    from OpenPinch.domain.heat_exchanger_network import HeatExchangerNetwork

    feasible = HeatExchangerNetwork()
    result = SimpleNamespace(
        method="pinch_design_method",
        ranked_networks=tuple(
            SimpleNamespace(**vars(outcome), network=feasible)
            for outcome in (
                _outcome("network_evolution_method", "evm", 4819.5),
                _outcome("pinch_design_method", "pdm", 1530.0),
                _outcome("thermal_derivative_method", "tdm", 1610.0),
            )
        ),
    )

    ranked = ranking._ranked_candidate_outcomes(result)

    assert [outcome.task.task_id for outcome in ranked] == ["pdm", "tdm", "evm"]


# M6: multiperiod stage reduction reads every period


def _two_period_stage_case():
    # [period][hot][cold][stage]: stage 2 is used only in period 2, and the
    # second hot stream's stage-1 match only in period 2. Stage 3 is empty.
    recovery = [
        [[[10.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]],
        [[[10.0, 400.0, 0.0]], [[5.0, 0.0, 0.0]]],
    ]
    return SimpleNamespace(
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
        S=3,
        Q_r_by_period=[
            [[[[q] for q in stages] for stages in hot] for hot in period]
            for period in recovery
        ],
        Q_h=[[[0.0]]],
        Q_c=[[[0.0]], [[0.0]]],
    )


def test_stage_reduction_keeps_stages_and_matches_used_in_any_period(monkeypatch):
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
    case = _two_period_stage_case()
    case.Q_r = case.Q_r_by_period[0]

    with pytest.raises(_Built):
        InternalHeatExchangerNetworkProblem(
            solver_arrays=object(), framework="TDM", stages=3
        ).remove_unused_stages(case)

    assert built["stages"] == 2
    assert built["z_restriction"][0] == [[[1, 1]], [[1, 0]]]


# M5: isothermal split branches report their own outlets and split fractions


def test_isothermal_split_branches_report_mixing_outlets_and_split_fractions():
    from OpenPinch.analysis.heat_exchanger_networks.extraction.period_state import (
        _recovery_period_state,
    )

    # Stage 1: H1 (CP 10) cools 400 -> 300 K, 600 kW to C1 and 400 kW to C2,
    # which each leave the stage at its mixing temperature.
    model = SimpleNamespace(
        T_h_by_period=[[[400.0, 300.0]]],
        T_c_by_period=[[[390.0, 290.0], [360.0, 280.0]]],
        f_h_period=[[10.0]],
        f_c_period=[[6.0, 5.0]],
        z=[[[[1.0]], [[1.0]]]],
        theta_1_by_period=[[[[10.0], [40.0]]]],
        theta_2_by_period=[[[[10.0], [20.0]]]],
    )
    q_by_period = [[[[600.0], [400.0]]]]

    states = [
        _recovery_period_state(
            model,
            period_id="0",
            period_idx=0,
            hot_index=0,
            cold_index=j,
            stage_index=0,
            q_values=None,
            q_values_by_period=q_by_period,
            tolerance=1e-6,
        )
        for j in (0, 1)
    ]

    # Each branch leaves at the hot stream's stage outlet, not at
    # 400 - Q / CP as if it carried the whole stream (340 and 360 K).
    assert [state.source_outlet_temperature for state in states] == [300.0, 300.0]
    assert [state.source_split_fraction for state in states] == pytest.approx(
        [0.6, 0.4]
    )
    # Each cold stream has one match here: the whole stream, outlet at its
    # stage boundary.
    assert [state.sink_outlet_temperature for state in states] == [390.0, 360.0]
    assert [state.sink_split_fraction for state in states] == pytest.approx([1.0, 1.0])


def test_stage_reduction_keeps_the_original_when_the_reduced_network_fails(
    monkeypatch,
):
    # Reduction only simplifies a solved network: a reduced solve that fails
    # verification must not replace it.
    from OpenPinch.analysis.heat_exchanger_networks.models import problem as unit
    from OpenPinch.analysis.heat_exchanger_networks.models.problem import (
        InternalHeatExchangerNetworkProblem,
    )

    class _Variable:
        def __init__(self):
            self.lower, self.upper = -1e9, 1e9
            self.VALUE = SimpleNamespace(value=None)

    class _Reduced:
        def __init__(self, **kwargs):
            stages = kwargs["stages"]

            def grid(*shape):
                if not shape:
                    return _Variable()
                return [grid(*shape[1:]) for _ in range(shape[0])]

            self.Q_r = grid(2, 1, stages)
            self.z = grid(2, 1, stages)
            self.theta_1 = grid(2, 1, stages)
            self.theta_2 = grid(2, 1, stages)
            self.T_h = grid(2, stages + 1)
            self.T_c = grid(1, stages + 1)
            self.Q_h = grid(1, 1)
            self.z_hu = grid(1, 1)
            self.Q_c = grid(2, 1)
            self.z_cu = grid(2, 1)

        def optimise(self, print_output=False):
            self.mSuccess = 1

        def verify(self):
            return False, ["temperature"]

    monkeypatch.setattr(unit, "StageWiseModel", _Reduced)
    case = _two_period_stage_case()
    case.Q_r = case.Q_r_by_period[0]
    case.T_h = [[[400.0]] * 4] * 2
    case.T_c = [[[300.0]] * 4]
    case.theta_1 = case.theta_2 = [[[[10.0]] * 3]] * 2
    case.TAC_model = case.TAC = case.solve_time = 1.0

    reduced = InternalHeatExchangerNetworkProblem(
        solver_arrays=object(), framework="TDM", stages=3
    ).remove_unused_stages(case)

    assert reduced is case
