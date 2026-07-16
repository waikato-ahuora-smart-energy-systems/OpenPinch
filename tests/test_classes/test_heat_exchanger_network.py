"""Tests for OpenPinch-native heat exchanger network domain objects."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

import OpenPinch.services.heat_exchanger_network_controllability as controllability
import OpenPinch.services.network_grid_diagram as grid_diagram
from OpenPinch.classes import (
    HeatExchanger,
    HeatExchangerKind,
    HeatExchangerNetwork,
    HeatExchangerStreamRole,
)
from OpenPinch.classes._heat_exchanger.period_state import HeatExchangerPeriodState
from OpenPinch.lib.enums import HeatExchangerNetworkLabel as HEN


def _recovery_exchanger(
    source: str = "H1",
    sink: str = "C1",
    *,
    stage: int = 1,
    duty: float = 100.0,
    period_states: tuple[HeatExchangerPeriodState, ...] | None = None,
) -> HeatExchanger:
    return HeatExchanger(
        exchanger_id=f"{source}-{sink}-{stage}",
        kind=HeatExchangerKind.RECOVERY,
        source_stream=source,
        sink_stream=sink,
        source_stream_role=HeatExchangerStreamRole.PROCESS,
        sink_stream_role=HeatExchangerStreamRole.PROCESS,
        stage=stage,
        period_states=period_states
        or (
            HeatExchangerPeriodState(
                period_id="base",
                period_idx=0,
                duty=duty,
                approach_temperatures=(12.0, 8.0),
                source_inlet_temperature=180.0,
                source_outlet_temperature=120.0,
                sink_inlet_temperature=60.0,
                sink_outlet_temperature=110.0,
            ),
        ),
        area=25.0,
        capital_cost=5000.0,
        solver_metadata={"array": "Q_r", "i": 0, "j": 0, "k": stage - 1},
    )


def _hot_utility_exchanger() -> HeatExchanger:
    return HeatExchanger(
        kind=HeatExchangerKind.HOT_UTILITY,
        source_stream="Steam",
        sink_stream="C1",
        source_stream_role=HeatExchangerStreamRole.UTILITY,
        sink_stream_role=HeatExchangerStreamRole.PROCESS,
        period_states=(
            HeatExchangerPeriodState(period_id="base", period_idx=0, duty=30.0),
        ),
        area=8.0,
    )


def _cold_utility_exchanger() -> HeatExchanger:
    return HeatExchanger(
        kind=HeatExchangerKind.COLD_UTILITY,
        source_stream="H1",
        sink_stream="CoolingWater",
        source_stream_role=HeatExchangerStreamRole.PROCESS,
        sink_stream_role=HeatExchangerStreamRole.UTILITY,
        period_states=(
            HeatExchangerPeriodState(period_id="base", period_idx=0, duty=20.0),
        ),
        area=5.0,
    )


def test_heat_exchanger_serialization_round_trip_excludes_solver_metadata():
    exchanger = _recovery_exchanger()

    round_tripped = HeatExchanger.model_validate_json(exchanger.model_dump_json())

    assert exchanger.state().source_mid_temperature == pytest.approx(150.0)
    assert exchanger.state().sink_mid_temperature == pytest.approx(85.0)
    assert round_tripped == exchanger.model_copy(
        update={"solver_metadata": {}, "source_metadata": {}}
    )
    assert "solver_metadata" not in exchanger.model_dump()
    assert "source_metadata" not in exchanger.model_dump()


def test_heat_exchanger_rejects_retired_scalar_operating_fields():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        HeatExchanger(
            kind=HeatExchangerKind.HOT_UTILITY,
            source_stream="Steam",
            sink_stream="C1",
            source_stream_role=HeatExchangerStreamRole.UTILITY,
            sink_stream_role=HeatExchangerStreamRole.PROCESS,
            period_states=(
                HeatExchangerPeriodState(
                    period_id="base",
                    period_idx=0,
                    duty=10.0,
                ),
            ),
            duty=10.0,
        )


def test_heat_exchanger_does_not_infer_mid_temperature_from_one_endpoint():
    exchanger = HeatExchanger(
        exchanger_id="partial-midpoints",
        kind=HeatExchangerKind.RECOVERY,
        source_stream="H1",
        sink_stream="C1",
        source_stream_role=HeatExchangerStreamRole.PROCESS,
        sink_stream_role=HeatExchangerStreamRole.PROCESS,
        stage=1,
        period_states=(
            HeatExchangerPeriodState(
                period_id="base",
                period_idx=0,
                duty=100.0,
                source_inlet_temperature=180.0,
                sink_outlet_temperature=110.0,
            ),
        ),
    )

    assert exchanger.state().source_mid_temperature is None
    assert exchanger.state().sink_mid_temperature is None

    exchanger.period_states[0].source_outlet_temperature = 120.0

    assert exchanger.state().source_mid_temperature == pytest.approx(150.0)


def test_heat_exchanger_direction_semantics_are_enforced():
    with pytest.raises(ValidationError, match="recovery exchangers"):
        HeatExchanger(
            kind=HeatExchangerKind.RECOVERY,
            source_stream="Steam",
            sink_stream="C1",
            source_stream_role=HeatExchangerStreamRole.UTILITY,
            sink_stream_role=HeatExchangerStreamRole.PROCESS,
            stage=1,
            period_states=(
                HeatExchangerPeriodState(period_id="base", period_idx=0, duty=10.0),
            ),
        )

    with pytest.raises(ValidationError, match="recovery exchangers must include"):
        HeatExchanger(
            kind=HeatExchangerKind.RECOVERY,
            source_stream="H1",
            sink_stream="C1",
            source_stream_role=HeatExchangerStreamRole.PROCESS,
            sink_stream_role=HeatExchangerStreamRole.PROCESS,
            period_states=(
                HeatExchangerPeriodState(period_id="base", period_idx=0, duty=10.0),
            ),
        )


def test_heat_exchanger_validators_reject_invalid_edge_values():
    assert HeatExchanger._validate_identity(None) is None
    assert HeatExchangerPeriodState._validate_finite_temperature(None) is None

    with pytest.raises(ValidationError, match="non-empty strings"):
        _recovery_exchanger(source=" ")
    with pytest.raises(ValidationError, match="positive integer"):
        _recovery_exchanger(stage=0)
    with pytest.raises(ValidationError, match="finite and non-negative"):
        HeatExchangerPeriodState(period_id="base", period_idx=0, duty=-1.0)
    with pytest.raises(ValidationError, match="temperatures must be finite"):
        HeatExchangerPeriodState(
            period_id="base",
            period_idx=0,
            duty=10.0,
            source_inlet_temperature=float("inf"),
        )
    with pytest.raises(ValidationError, match="approach temperatures"):
        HeatExchangerPeriodState(
            period_id="base",
            period_idx=0,
            duty=10.0,
            approach_temperatures=(-1.0,),
        )
    with pytest.raises(ValidationError, match="split fractions"):
        HeatExchangerPeriodState(
            period_id="base",
            period_idx=0,
            duty=10.0,
            source_split_fraction=1.1,
        )
    with pytest.raises(ValidationError, match="must be distinct"):
        _recovery_exchanger(source="H1", sink="H1")


def test_heat_exchanger_network_validators_reject_invalid_metadata():
    with pytest.raises(ValidationError, match="non-empty strings"):
        HeatExchangerNetwork(run_id=" ")
    with pytest.raises(ValidationError, match="positive integer"):
        HeatExchangerNetwork(stage_count=0)
    with pytest.raises(ValidationError, match="finite and non-negative"):
        HeatExchangerNetwork(objective_value=float("inf"))
    with pytest.raises(ValidationError, match="summary metric names"):
        HeatExchangerNetwork(summary_metrics={"": 1.0})
    with pytest.raises(ValidationError, match="summary metric values"):
        HeatExchangerNetwork(summary_metrics={"cost": float("inf")})


def test_heat_exchanger_network_delegates_grid_and_controllability(monkeypatch):
    network = HeatExchangerNetwork(exchangers=(_recovery_exchanger(),))
    calls = {}

    def fake_build_grid_diagram(target_network, **kwargs):
        calls["grid_network"] = target_network
        calls["grid_kwargs"] = kwargs
        return {"grid": "ok"}

    def fake_quantify(target_network, **kwargs):
        calls["controllability_network"] = target_network
        calls["controllability_kwargs"] = kwargs
        return {"rank": 1}

    monkeypatch.setattr(grid_diagram, "build_grid_diagram", fake_build_grid_diagram)
    monkeypatch.setattr(
        controllability,
        "quantify_heat_exchanger_network_controllability",
        fake_quantify,
    )

    assert network.build_grid_diagram(stream_line_width=7.0) == {"grid": "ok"}
    assert network.quantify_controllability(mode="steady") == {"rank": 1}
    assert calls["grid_network"] is network
    assert calls["grid_kwargs"] == {
        "period_id": "base",
        "stream_line_width": 7.0,
        "temperature_scaled": False,
    }
    assert calls["controllability_network"] is network
    assert calls["controllability_kwargs"] == {
        "mode": "steady",
        "period_id": "base",
    }


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
    assert network.total(HEN.RECOVERY_AREA) == pytest.approx(25.0)
    assert network.total_area(kind="cold_utility") == pytest.approx(5.0)
    assert network.labelled_value(
        HEN.RECOVERY_AREA,
        source_stream="H1",
        sink_stream="C1",
        stage=1,
    ) == pytest.approx(25.0)
    assert network.labelled_value(
        HEN.COLD_RECOVERY_OUTLET_TEMPERATURE,
        source_stream="H1",
        sink_stream="C1",
        stage=1,
    ) == pytest.approx(110.0)
    assert network.labelled_value(
        HEN.MATCH_ALLOWED,
        source_stream="H1",
        sink_stream="C1",
        stage=1,
    )
    assert (
        network.labelled_value(
            HEN.RECOVERY_DUTY,
            source_stream="missing",
            sink_stream="C1",
        )
        is None
    )
    assert len(network.exchangers_involving_stream("C1")) == 2
    assert len(network.exchangers_involving_stream("missing")) == 0
    assert network.exchanger_between(source_stream="missing", sink_stream="C1") is None
    assert "solver_axis_metadata" not in network.model_dump()


def test_multiperiod_queries_require_explicit_period_identity():
    exchanger = _recovery_exchanger(
        period_states=(
            HeatExchangerPeriodState(
                period_id="base",
                period_idx=0,
                duty=0.0,
                active=False,
                source_inlet_temperature=180.0,
                source_outlet_temperature=180.0,
                sink_inlet_temperature=60.0,
                sink_outlet_temperature=60.0,
            ),
            HeatExchangerPeriodState(
                period_id="peak",
                period_idx=1,
                duty=125.0,
                active=True,
                approach_temperatures=(15.0, 10.0),
                source_split_fraction=0.5,
                sink_split_fraction=0.4,
                source_inlet_temperature=190.0,
                source_outlet_temperature=130.0,
                sink_inlet_temperature=70.0,
                sink_outlet_temperature=120.0,
            ),
        )
    )
    network = HeatExchangerNetwork(exchangers=(exchanger,))

    assert (
        HeatExchanger.model_validate_json(exchanger.model_dump_json()).model_dump()
        == exchanger.model_dump()
    )
    assert exchanger.state("peak").duty == pytest.approx(125.0)
    assert exchanger.state("peak").source_split_fraction == pytest.approx(0.5)
    assert exchanger.state("peak").sink_split_fraction == pytest.approx(0.4)
    assert network.total_duty(period_id="base") == pytest.approx(0.0)
    assert network.total_duty(period_id="peak") == pytest.approx(125.0)
    assert network.labelled_value(
        HEN.MATCH_ACTIVE,
        source_stream="H1",
        sink_stream="C1",
        stage=1,
        period_id="peak",
    )
    with pytest.raises(ValueError, match="period_id is required"):
        exchanger.state()
    with pytest.raises(ValueError, match="period_id is required"):
        network.total_duty()
    with pytest.raises(ValueError, match="period_id is required"):
        network.build_grid_diagram()
    with pytest.raises(ValueError, match="period_id is required"):
        network.quantify_controllability()


def test_period_states_reject_out_of_order_or_duplicate_identities():
    with pytest.raises(ValidationError, match="ordered by contiguous period_idx"):
        _recovery_exchanger(
            period_states=(
                HeatExchangerPeriodState(period_id="peak", period_idx=1, duty=1.0),
            )
        )

    with pytest.raises(ValidationError, match="same ordered period_ids"):
        HeatExchangerNetwork(
            exchangers=(
                _recovery_exchanger("H1", "C1"),
                _recovery_exchanger(
                    "H2",
                    "C2",
                    period_states=(
                        HeatExchangerPeriodState(
                            period_id="peak", period_idx=0, duty=2.0
                        ),
                    ),
                ),
            )
        )
    with pytest.raises(ValidationError, match="unique period_id"):
        _recovery_exchanger(
            period_states=(
                HeatExchangerPeriodState(period_id="base", period_idx=0, duty=1.0),
                HeatExchangerPeriodState(period_id="base", period_idx=1, duty=2.0),
            )
        )


def test_heat_exchanger_network_rejects_ambiguous_or_incompatible_labels():
    duplicate_network = HeatExchangerNetwork(
        exchangers=(
            _recovery_exchanger("H1", "C1", stage=1, duty=100.0),
            _recovery_exchanger("H1", "C1", stage=1, duty=80.0),
        )
    )
    with pytest.raises(ValueError, match="multiple exchangers"):
        duplicate_network.exchanger_between(source_stream="H1", sink_stream="C1")

    network = HeatExchangerNetwork(exchangers=(_hot_utility_exchanger(),))
    with pytest.raises(ValueError, match="label cannot be used"):
        network.total(HEN.RECOVERY_DUTY, kind=HeatExchangerKind.HOT_UTILITY)
    with pytest.raises(ValueError, match="only valid for recovery"):
        network.labelled_value(
            HEN.HOT_RECOVERY_OUTLET_TEMPERATURE,
            source_stream="Steam",
            sink_stream="C1",
            kind=HeatExchangerKind.HOT_UTILITY,
        )
    with pytest.raises(ValueError, match="not a numeric total label"):
        network.total(HEN.MATCH_ACTIVE)


def test_heat_exchanger_network_totals_honour_active_stream_stage_and_none_values():
    inactive = _recovery_exchanger("H2", "C2", stage=2, duty=80.0)
    inactive.period_states[0].active = False
    no_area = _recovery_exchanger("H3", "C3", stage=3, duty=10.0)
    no_area.area = None
    network = HeatExchangerNetwork(
        exchangers=(_recovery_exchanger("H1", "C1"), inactive, no_area)
    )

    assert network.total_duty(active_only=True) == pytest.approx(110.0)
    assert network.total_duty(active_only=False) == pytest.approx(190.0)
    assert network.total_duty(stage=2, active_only=False) == pytest.approx(80.0)
    assert network.total_duty(stream="H2", active_only=True) == pytest.approx(0.0)
    assert network.total_area(active_only=False) == pytest.approx(50.0)


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
