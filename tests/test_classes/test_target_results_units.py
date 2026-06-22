"""Unit-aware target-result serialization regressions."""

from __future__ import annotations

import pytest

from OpenPinch.classes.problem_table import ProblemTable
from OpenPinch.classes.stream_collection import StreamCollection
from OpenPinch.classes.value import Value
from OpenPinch.lib.config import Configuration
from OpenPinch.lib.enums import PT, TT
from OpenPinch.lib.schemas.targets import (
    DirectHeatPumpTarget,
    DirectIntegrationTarget,
    IndirectHeatPumpTarget,
    TotalSiteTarget,
)


def _problem_table() -> ProblemTable:
    return ProblemTable({PT.T: [120.0, 60.0]})


def _config() -> Configuration:
    return Configuration()


def _make_di_target() -> DirectIntegrationTarget:
    return DirectIntegrationTarget(
        zone_name="Plant",
        type=TT.DI.value,
        config=_config(),
        pt=_problem_table(),
        pt_real=_problem_table(),
        hot_utility_target=10.0,
        cold_utility_target=5.0,
        heat_recovery_target=15.0,
    )


def _make_ts_target() -> TotalSiteTarget:
    return TotalSiteTarget(
        zone_name="Plant",
        type=TT.TS.value,
        config=_config(),
        pt=_problem_table(),
        hot_utility_target=10.0,
        cold_utility_target=5.0,
        heat_recovery_target=15.0,
    )


def _make_dhp_target() -> DirectHeatPumpTarget:
    return DirectHeatPumpTarget(
        zone_name="Plant",
        type=TT.DHP.value,
        config=_config(),
        pt=_problem_table(),
        hot_utilities=StreamCollection(),
        cold_utilities=StreamCollection(),
        hot_utility_target=10.0,
        cold_utility_target=5.0,
        heat_recovery_target=15.0,
        hpr_cycle="stub",
        hpr_utility_total=0.0,
        hpr_work=0.0,
        hpr_external_utility=0.0,
        hpr_ambient_hot=0.0,
        hpr_ambient_cold=0.0,
        hpr_cop=0.0,
        hpr_eta_he=0.0,
        hpr_success=True,
        hpr_hot_streams=StreamCollection(),
        hpr_cold_streams=StreamCollection(),
        hpr_details={},
    )


def _make_ihp_target() -> IndirectHeatPumpTarget:
    return IndirectHeatPumpTarget(
        zone_name="Plant",
        type=TT.IHP.value,
        config=_config(),
        pt=_problem_table(),
        hot_utilities=StreamCollection(),
        cold_utilities=StreamCollection(),
        hot_utility_target=10.0,
        cold_utility_target=5.0,
        heat_recovery_target=15.0,
        hpr_cycle="stub",
        hpr_utility_total=0.0,
        hpr_work=0.0,
        hpr_external_utility=0.0,
        hpr_ambient_hot=0.0,
        hpr_ambient_cold=0.0,
        hpr_cop=0.0,
        hpr_eta_he=0.0,
        hpr_success=True,
        hpr_hot_streams=StreamCollection(),
        hpr_cold_streams=StreamCollection(),
        hpr_details={},
    )


def _assert_scalar_payload(payload, expected_value: float, expected_unit: str) -> None:
    assert isinstance(payload, Value)
    assert payload.value == pytest.approx(expected_value)
    assert payload.unit == expected_unit


@pytest.mark.parametrize(
    "factory",
    [
        _make_di_target,
        _make_ts_target,
        _make_dhp_target,
        _make_ihp_target,
    ],
)
def test_target_results_wrap_common_summary_metrics_with_units(factory):
    target = factory()
    target.utility_cost = 20.0
    target.hot_pinch = 140.0
    target.cold_pinch = 110.0
    target.degree_of_int = 0.5
    target.exergy_sources = 18.0
    target.exergy_sinks = 14.0
    target.ETE = 0.8
    target.exergy_req_min = 6.0
    target.exergy_des_min = 4.0

    results = target.to_target_results()

    _assert_scalar_payload(results.Qh, 10.0, "kW")
    _assert_scalar_payload(results.Qc, 5.0, "kW")
    _assert_scalar_payload(results.Qr, 15.0, "kW")
    _assert_scalar_payload(results.utility_cost, 20.0, "$/h")
    _assert_scalar_payload(results.degree_of_integration, 50.0, "%")
    _assert_scalar_payload(results.pinch_temp.hot_temp, 140.0, "degC")
    _assert_scalar_payload(results.pinch_temp.cold_temp, 110.0, "degC")
    _assert_scalar_payload(results.exergy_sources, 18.0, "kW")
    _assert_scalar_payload(results.exergy_sinks, 14.0, "kW")
    _assert_scalar_payload(results.ETE, 80.0, "%")
    _assert_scalar_payload(results.exergy_req_min, 6.0, "kW")
    _assert_scalar_payload(results.exergy_des_min, 4.0, "kW")


def test_direct_integration_results_include_unit_aware_cost_and_area_metrics():
    target = _make_di_target()
    target.work_target = 8.0
    target.turbine_efficiency_target = 0.42
    target.area = 120.0
    target.num_units = 6
    target.capital_cost = 1000.0
    target.total_cost = 120.0

    results = target.to_target_results()

    _assert_scalar_payload(results.work_target, 8.0, "kW")
    _assert_scalar_payload(results.turbine_efficiency_target, 42.0, "%")
    _assert_scalar_payload(results.area, 120.0, "m^2")
    _assert_scalar_payload(results.capital_cost, 1000.0, "$")
    _assert_scalar_payload(results.total_cost, 120.0, "$/y")
    assert results.num_units == 6


@pytest.mark.parametrize("factory", [_make_dhp_target, _make_ihp_target])
def test_heat_pump_results_include_unit_aware_hpr_metrics(factory):
    target = factory()
    target.work_target = 8.0
    target.turbine_efficiency_target = 0.42
    target.hpr_utility_total = [10.0, 12.0]
    target.hpr_work = [1.0, 2.0]
    target.hpr_external_utility = 3.0
    target.hpr_ambient_hot = 4.0
    target.hpr_ambient_cold = 5.0
    target.hpr_cop = 3.5
    target.hpr_eta_he = 0.25
    target.hpr_operating_cost = Value(1000.0, "$/y")
    target.hpr_capital_cost = Value(5000.0, "$")
    target.hpr_annualized_capital_cost = Value(600.0, "$/y")
    target.hpr_total_annualized_cost = Value(1600.0, "$/y")
    target.hpr_compressor_capital_cost = Value(3000.0, "$")
    target.hpr_heat_exchanger_capital_cost = Value(2000.0, "$")

    results = target.to_target_results()

    _assert_scalar_payload(results.work_target, 8.0, "kW")
    _assert_scalar_payload(results.turbine_efficiency_target, 42.0, "%")
    assert isinstance(results.hpr_utility_total, Value)
    assert results.hpr_utility_total.values == [10.0, 12.0]
    assert results.hpr_utility_total.unit == "kW"
    assert isinstance(results.hpr_work, Value)
    assert results.hpr_work.values == [1.0, 2.0]
    assert results.hpr_work.unit == "kW"
    _assert_scalar_payload(results.hpr_external_utility, 3.0, "kW")
    _assert_scalar_payload(results.hpr_ambient_hot, 4.0, "kW")
    _assert_scalar_payload(results.hpr_ambient_cold, 5.0, "kW")
    _assert_scalar_payload(results.hpr_cop, 3.5, "-")
    _assert_scalar_payload(results.hpr_eta_he, 25.0, "%")
    _assert_scalar_payload(results.hpr_operating_cost, 1000.0, "$/y")
    _assert_scalar_payload(results.hpr_capital_cost, 5000.0, "$")
    _assert_scalar_payload(results.hpr_annualized_capital_cost, 600.0, "$/y")
    _assert_scalar_payload(results.hpr_total_annualized_cost, 1600.0, "$/y")
    _assert_scalar_payload(results.hpr_compressor_capital_cost, 3000.0, "$")
    _assert_scalar_payload(results.hpr_heat_exchanger_capital_cost, 2000.0, "$")


def test_target_results_apply_configured_output_unit_overrides():
    target = DirectIntegrationTarget(
        zone_name="Plant",
        type=TT.DI.value,
        config=Configuration(
            options={
                "OUTPUT_UNIT_HEAT_FLOW": "MW",
                "OUTPUT_UNIT_WORK": "MW",
                "OUTPUT_UNIT_TEMPERATURE": "K",
                "OUTPUT_UNIT_PERCENT": "-",
                "OUTPUT_UNIT_AREA": "ft^2",
            }
        ),
        pt=_problem_table(),
        pt_real=_problem_table(),
        hot_utility_target=1500.0,
        cold_utility_target=500.0,
        heat_recovery_target=2000.0,
    )
    target.degree_of_int = 0.5
    target.hot_pinch = 140.0
    target.cold_pinch = 110.0
    target.work_target = 250.0
    target.turbine_efficiency_target = 0.42
    target.area = 120.0

    results = target.to_target_results()

    _assert_scalar_payload(results.Qh, 1.5, "MW")
    _assert_scalar_payload(results.Qc, 0.5, "MW")
    _assert_scalar_payload(results.Qr, 2.0, "MW")
    _assert_scalar_payload(results.work_target, 0.25, "MW")
    _assert_scalar_payload(results.pinch_temp.hot_temp, 413.15, "K")
    _assert_scalar_payload(results.pinch_temp.cold_temp, 383.15, "K")
    _assert_scalar_payload(results.degree_of_integration, 0.5, "-")
    _assert_scalar_payload(results.turbine_efficiency_target, 0.42, "-")
    _assert_scalar_payload(results.area, 1291.669, "ft^2")
