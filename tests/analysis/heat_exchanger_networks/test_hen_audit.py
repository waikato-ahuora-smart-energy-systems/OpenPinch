"""Regression tests for the HEN audit fixes (area 5)."""

from __future__ import annotations

from types import SimpleNamespace

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
