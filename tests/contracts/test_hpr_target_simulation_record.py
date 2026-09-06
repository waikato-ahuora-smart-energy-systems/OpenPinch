"""Contract tests for detached winning HPR target simulation records."""

from __future__ import annotations

import math

import pytest
from hypothesis import given, seed
from hypothesis import strategies as st
from pydantic import ValidationError

from OpenPinch.contracts.hpr import HprTargetSimulationRecord
from tests.strategies.hpr_targeting import (
    EXPLICIT_MOLAR_MIXTURES,
    INVALID_RECORD_OVERRIDES,
    PURE_FLUIDS,
    REGISTERED_BLENDS,
    hpr_target_simulation_records,
    hpr_target_values,
    hpr_thermal_profiles,
)


def _record(**overrides) -> HprTargetSimulationRecord:
    values = {
        "simulation_backend": "coolprop",
        "mode": "heat_pump",
        "cycle_id": "single_stage_vapour_compression",
        "model_id": "openpinch-vapour-compression-v1",
        "refrigerant_spec": "HEOS::R32[0.5]&R125[0.5]",
        "nominal_evaporating_temperature": 5.0,
        "nominal_condensing_temperature": 55.0,
        "nominal_useful_duty": 500.0,
        "source_approach_temperature": 3.0,
        "sink_approach_temperature": 4.0,
        "compressor_isentropic_efficiency": 0.72,
        "superheat": 5.0,
        "subcooling": 2.0,
        "internal_hx_gas_temperature_change": 0.0,
        "evaporator_count": 1,
        "condenser_count": 1,
        "period_id": "winter",
        "engine_version": "8.0.0",
        "power_boundary": "compressor_only",
        "assumptions": {"anchor": "dew_bubble", "auxiliaries": []},
    }
    values.update(overrides)
    return HprTargetSimulationRecord.model_validate(values)


def test_record_is_frozen_strict_and_json_compatible() -> None:
    record = _record()

    assert record.model_validate_json(record.model_dump_json()) == record
    with pytest.raises(ValidationError, match="frozen"):
        record.mode = "refrigeration"
    with pytest.raises(ValidationError, match="extra_forbidden"):
        HprTargetSimulationRecord.model_validate(
            record.model_dump(mode="python") | {"engine_object": "forbidden"}
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("simulation_backend", "other"),
        ("mode", "other"),
        ("cycle_id", ""),
        ("model_id", " "),
        ("refrigerant_spec", ""),
        ("nominal_evaporating_temperature", math.inf),
        ("nominal_condensing_temperature", math.nan),
        ("nominal_useful_duty", 0.0),
        ("source_approach_temperature", -1.0),
        ("sink_approach_temperature", -1.0),
        ("compressor_isentropic_efficiency", 1.1),
        ("evaporator_count", 0),
        ("condenser_count", 0),
        ("engine_version", ""),
        ("power_boundary", "total_auxiliaries"),
        ("assumptions", {}),
    ],
)
def test_record_rejects_invalid_fields(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        _record(**{field: value})


def test_record_requires_positive_temperature_lift() -> None:
    with pytest.raises(ValidationError, match="temperature lift"):
        _record(
            nominal_evaporating_temperature=60.0, nominal_condensing_temperature=50.0
        )


@seed(20260715)
@given(hpr_target_simulation_records())
def test_record_json_round_trip_property(record: HprTargetSimulationRecord) -> None:
    assert (
        HprTargetSimulationRecord.model_validate_json(record.model_dump_json())
        == record
    )


@seed(20260715)
@given(hpr_target_simulation_records())
def test_record_round_trip_preserves_fluid_and_backend(
    record: HprTargetSimulationRecord,
) -> None:
    rebuilt = HprTargetSimulationRecord.model_validate(record.model_dump(mode="json"))

    assert rebuilt.refrigerant_spec == record.refrigerant_spec
    assert rebuilt.simulation_backend == record.simulation_backend
    assert rebuilt.assumptions == record.assumptions


@seed(20260715)
@given(hpr_target_simulation_records())
def test_record_is_detached_from_serialized_mutation(
    record: HprTargetSimulationRecord,
) -> None:
    serialized = record.model_dump(mode="json")
    serialized["assumptions"]["mutated"] = True

    assert "mutated" not in record.assumptions


@seed(20260715)
@given(INVALID_RECORD_OVERRIDES)
def test_invalid_record_strategy_is_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        _record(**overrides)


@seed(20260715)
@given(hpr_target_values())
def test_target_value_strategy_preserves_energy_closure(
    values: dict[str, float],
) -> None:
    assert values["useful_duty"] == pytest.approx(
        values["source_duty"] + values["work"]
    )
    assert values["cop"] == pytest.approx(values["useful_duty"] / values["work"])


@seed(20260715)
@given(hpr_thermal_profiles())
def test_profile_strategy_preserves_direction_and_duties(
    profiles: tuple[dict[str, float | str], dict[str, float | str]],
) -> None:
    source, sink = profiles
    assert source["outlet_temperature"] < source["inlet_temperature"]
    assert sink["outlet_temperature"] > sink["inlet_temperature"]
    assert source["duty"] >= 0.0
    assert sink["duty"] > 0.0


@pytest.mark.parametrize(
    "strategy",
    [PURE_FLUIDS, REGISTERED_BLENDS, EXPLICIT_MOLAR_MIXTURES],
)
@seed(20260715)
@given(data=st.data())
def test_fluid_category_strategies_produce_valid_records(strategy, data) -> None:
    fluid = data.draw(strategy)
    assert _record(refrigerant_spec=fluid).refrigerant_spec == fluid
