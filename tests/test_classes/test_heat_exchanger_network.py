"""Tests for OpenPinch-native heat exchanger network domain objects."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from OpenPinch.classes import (
    HeatExchanger,
    HeatExchangerKind,
    HeatExchangerNetwork,
    HeatExchangerStreamRole,
)
from OpenPinch.lib.enums import HeatExchangerNetworkLabel as HEN


def _recovery_exchanger(
    source: str = "H1",
    sink: str = "C1",
    *,
    stage: int = 1,
    duty: float = 100.0,
) -> HeatExchanger:
    return HeatExchanger(
        exchanger_id=f"{source}-{sink}-{stage}",
        kind=HeatExchangerKind.RECOVERY,
        source_stream=source,
        sink_stream=sink,
        source_stream_role=HeatExchangerStreamRole.PROCESS,
        sink_stream_role=HeatExchangerStreamRole.PROCESS,
        stage=stage,
        duty=duty,
        area=25.0,
        approach_temperatures=(12.0, 8.0),
        source_inlet_temperature=180.0,
        source_outlet_temperature=120.0,
        sink_inlet_temperature=60.0,
        sink_outlet_temperature=110.0,
        capital_cost=5000.0,
        operating_cost=200.0,
        total_annual_cost=5200.0,
        solver_metadata={"array": "Q_r", "i": 0, "j": 0, "k": stage - 1},
    )


def _hot_utility_exchanger() -> HeatExchanger:
    return HeatExchanger(
        kind=HeatExchangerKind.HOT_UTILITY,
        source_stream="Steam",
        sink_stream="C1",
        source_stream_role=HeatExchangerStreamRole.UTILITY,
        sink_stream_role=HeatExchangerStreamRole.PROCESS,
        duty=30.0,
        area=8.0,
    )


def _cold_utility_exchanger() -> HeatExchanger:
    return HeatExchanger(
        kind=HeatExchangerKind.COLD_UTILITY,
        source_stream="H1",
        sink_stream="CoolingWater",
        source_stream_role=HeatExchangerStreamRole.PROCESS,
        sink_stream_role=HeatExchangerStreamRole.UTILITY,
        duty=20.0,
        area=5.0,
    )


def test_heat_exchanger_serialization_round_trip_excludes_solver_metadata():
    exchanger = _recovery_exchanger()

    round_tripped = HeatExchanger.model_validate_json(exchanger.model_dump_json())

    assert exchanger.source_mid_temperature == pytest.approx(150.0)
    assert exchanger.sink_mid_temperature == pytest.approx(85.0)
    assert round_tripped == exchanger.model_copy(
        update={"solver_metadata": {}, "source_metadata": {}}
    )
    assert "solver_metadata" not in exchanger.model_dump()
    assert "source_metadata" not in exchanger.model_dump()


def test_heat_exchanger_preserves_explicit_mid_temperatures():
    exchanger = HeatExchanger(
        exchanger_id="explicit-midpoints",
        kind=HeatExchangerKind.RECOVERY,
        source_stream="H1",
        sink_stream="C1",
        source_stream_role=HeatExchangerStreamRole.PROCESS,
        sink_stream_role=HeatExchangerStreamRole.PROCESS,
        stage=1,
        duty=100.0,
        source_inlet_temperature=180.0,
        source_outlet_temperature=120.0,
        source_mid_temperature=155.0,
        sink_inlet_temperature=60.0,
        sink_outlet_temperature=110.0,
        sink_mid_temperature=82.0,
    )

    assert exchanger.source_mid_temperature == pytest.approx(155.0)
    assert exchanger.sink_mid_temperature == pytest.approx(82.0)


def test_heat_exchanger_does_not_infer_mid_temperature_from_one_endpoint():
    exchanger = HeatExchanger(
        exchanger_id="partial-midpoints",
        kind=HeatExchangerKind.RECOVERY,
        source_stream="H1",
        sink_stream="C1",
        source_stream_role=HeatExchangerStreamRole.PROCESS,
        sink_stream_role=HeatExchangerStreamRole.PROCESS,
        stage=1,
        duty=100.0,
        source_inlet_temperature=180.0,
        sink_outlet_temperature=110.0,
    )

    assert exchanger.source_mid_temperature is None
    assert exchanger.sink_mid_temperature is None

    exchanger.source_outlet_temperature = 120.0

    assert exchanger.source_mid_temperature == pytest.approx(150.0)


def test_heat_exchanger_direction_semantics_are_enforced():
    with pytest.raises(ValidationError, match="recovery exchangers"):
        HeatExchanger(
            kind=HeatExchangerKind.RECOVERY,
            source_stream="Steam",
            sink_stream="C1",
            source_stream_role=HeatExchangerStreamRole.UTILITY,
            sink_stream_role=HeatExchangerStreamRole.PROCESS,
            stage=1,
            duty=10.0,
        )

    with pytest.raises(ValidationError, match="recovery exchangers must include"):
        HeatExchanger(
            kind=HeatExchangerKind.RECOVERY,
            source_stream="H1",
            sink_stream="C1",
            source_stream_role=HeatExchangerStreamRole.PROCESS,
            sink_stream_role=HeatExchangerStreamRole.PROCESS,
            duty=10.0,
        )


def test_heat_exchanger_network_labelled_access_and_totals():
    network = HeatExchangerNetwork(
        exchangers=(
            _recovery_exchanger(),
            _hot_utility_exchanger(),
            _cold_utility_exchanger(),
        ),
        run_id="run-1",
        task_id="task-1",
        stage_count=2,
        objective_value=12345.0,
        solver_axis_metadata={
            "hot_process_streams": {"H1": 2},
            "cold_process_streams": {"C1": 1},
        },
    )

    assert (
        network.exchanger_between(
            source_stream="H1",
            sink_stream="C1",
            stage=1,
        )
        == network.exchangers[0]
    )
    assert network.labelled_value(
        HEN.RECOVERY_DUTY,
        source_stream="H1",
        sink_stream="C1",
        stage=1,
    ) == pytest.approx(100.0)
    assert network.labelled_value(
        HEN.HOT_RECOVERY_OUTLET_TEMPERATURE,
        source_stream="H1",
        sink_stream="C1",
        stage=1,
    ) == pytest.approx(120.0)
    assert network.labelled_value(
        HEN.MATCH_ACTIVE,
        source_stream="H1",
        sink_stream="C1",
        stage=1,
    )
    assert network.total_duty(kind=HeatExchangerKind.RECOVERY) == pytest.approx(100.0)
    assert network.total_duty(stream="H1") == pytest.approx(120.0)
    assert network.total(HEN.HOT_UTILITY_DUTY, stream="C1") == pytest.approx(30.0)
    assert len(network.exchangers_involving_stream("C1")) == 2
    assert "solver_axis_metadata" not in network.model_dump()


def test_network_serialization_round_trip_preserves_public_result_fields():
    network = HeatExchangerNetwork(
        exchangers=(_recovery_exchanger(),),
        run_id="run-1",
        task_id="task-1",
        summary_metrics={"recovery_units": 1, "verified": True},
        solver_axis_metadata={"hot_process_streams": {"H1": 0}},
    )

    round_tripped = HeatExchangerNetwork.model_validate_json(network.model_dump_json())

    assert round_tripped.model_dump() == network.model_dump()
    assert round_tripped.summary_metrics == {"recovery_units": 1, "verified": True}


def test_labelled_access_is_stable_when_solver_axes_and_rows_reorder():
    first = HeatExchangerNetwork(
        exchangers=(
            _recovery_exchanger("H1", "C1", stage=1, duty=100.0),
            _recovery_exchanger("H2", "C2", stage=2, duty=80.0),
        ),
        solver_axis_metadata={
            "hot_process_streams": {"H1": 0, "H2": 1},
            "cold_process_streams": {"C1": 0, "C2": 1},
            "stages": {"1": 0, "2": 1},
        },
    )
    reordered = HeatExchangerNetwork(
        exchangers=(
            _recovery_exchanger("H2", "C2", stage=2, duty=80.0),
            _recovery_exchanger("H1", "C1", stage=1, duty=100.0),
        ),
        solver_axis_metadata={
            "hot_process_streams": {"H2": 0, "H1": 1},
            "cold_process_streams": {"C2": 0, "C1": 1},
            "stages": {"2": 0, "1": 1},
        },
    )

    for network in (first, reordered):
        assert network.labelled_value(
            HEN.RECOVERY_DUTY,
            source_stream="H1",
            sink_stream="C1",
            stage=1,
        ) == pytest.approx(100.0)
        assert network.total_duty(stream="H2") == pytest.approx(80.0)
