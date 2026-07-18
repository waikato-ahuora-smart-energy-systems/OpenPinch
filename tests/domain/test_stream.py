"""Regression tests for stream period handling with collection-owned metadata."""

import json

import numpy as np
import pytest

from OpenPinch.domain._stream.value_state import resolve_period_weights
from OpenPinch.domain.enums import FluidPhase, StreamType
from OpenPinch.domain.stream import Stream
from OpenPinch.domain.value import Value
from tests.support.paths import FIXTURES_ROOT

FIXTURE_PATH = FIXTURES_ROOT / "value_and_stream_edge_cases.json"


def _stream_fixture(name: str) -> dict:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    wire_to_runtime = {
        "t_supply": "supply_temperature",
        "t_target": "target_temperature",
        "p_supply": "supply_pressure",
        "p_target": "target_pressure",
        "h_supply": "supply_enthalpy",
        "h_target": "target_enthalpy",
        "dt_cont": "delta_t_contribution",
        "dt_cont_multiplier": "delta_t_contribution_multiplier",
        "htc": "heat_transfer_coefficient",
    }
    return {
        wire_to_runtime.get(key, key): value
        for key, value in fixture["stream_cases"][name].items()
    }


def test_scalar_stream_initialisation_computes_derived_fields():
    stream = Stream(
        name="Hot1",
        supply_temperature=300.0,
        target_temperature=200.0,
        heat_flow=5000.0,
        delta_t_contribution=10.0,
        heat_transfer_coefficient=2.0,
        price=50.0,
    )

    assert stream.stream_type == StreamType.Hot.value
    assert float(stream.minimum_temperature) == pytest.approx(200.0)
    assert float(stream.maximum_temperature) == pytest.approx(300.0)
    assert float(stream.shifted_minimum_temperature) == pytest.approx(190.0)
    assert float(stream.shifted_maximum_temperature) == pytest.approx(290.0)
    assert float(stream.heat_capacity_flowrate) == pytest.approx(50.0)
    assert float(stream.resistance_capacity_product) == pytest.approx(25.0)


def test_stream_accepts_optional_fluid_metadata():
    stream = Stream(
        name="Refrigerant",
        supply_temperature=30.0,
        target_temperature=80.0,
        heat_flow=100.0,
        fluid_name="HEOS::R32[0.697615]&R125[0.302385]",
        fluid_phase="VLE",
    )

    assert stream.name == "Refrigerant"
    assert stream.fluid_name == "HEOS::R32[0.697615]&R125[0.302385]"
    assert stream.fluid_phase == "vle"

    stream.fluid_name = " Water "
    stream.fluid_phase = "gas"
    assert stream.fluid_name == "Water"
    assert stream.fluid_phase == "gas"

    stream.fluid_phase = FluidPhase.liq
    assert stream.fluid_phase == "liq"


def test_stream_rejects_invalid_fluid_phase():
    with pytest.raises(ValueError, match="fluid_phase must be one of"):
        Stream(
            name="Invalid",
            supply_temperature=30.0,
            target_temperature=80.0,
            heat_flow=100.0,
            fluid_phase="plasma",
        )


def test_stream_rejects_invalid_coolprop_fluid_name():
    with pytest.raises(ValueError, match="Invalid CoolProp fluid_name"):
        Stream(
            name="InvalidFluid",
            supply_temperature=30.0,
            target_temperature=80.0,
            heat_flow=100.0,
            fluid_name="NotARealCoolPropFluid",
        )


def test_derived_stream_fields_are_read_only():
    stream = Stream(
        name="Hot1",
        supply_temperature=300.0,
        target_temperature=200.0,
        heat_flow=5000.0,
        delta_t_contribution=10.0,
        heat_transfer_coefficient=2.0,
        price=50.0,
    )

    with pytest.raises(AttributeError):
        stream.heat_capacity_flowrate = 10.0

    with pytest.raises(AttributeError):
        stream.minimum_temperature = 150.0

    with pytest.raises(TypeError, match="Stream-owned Value is read-only"):
        stream.minimum_temperature["0"] = 150.0
    with pytest.raises(TypeError, match="Stream-owned Value is read-only"):
        stream.heat_capacity_flowrate["0"] = 10.0
    assert float(stream.minimum_temperature) == pytest.approx(200.0)
    assert float(stream.heat_capacity_flowrate) == pytest.approx(50.0)


def test_period_weight_resolution_pads_missing_values_and_rejects_invalid_shapes():
    np.testing.assert_allclose(
        resolve_period_weights(["base", "peak", "idle"], [0.25]),
        [0.25, 1.0, 1.0],
    )

    with pytest.raises(ValueError, match="at most 2"):
        resolve_period_weights(["base", "peak"], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="non-negative"):
        resolve_period_weights(["base", "peak"], [1.0, -1.0])
    with pytest.raises(ValueError, match="finite"):
        resolve_period_weights(["base", "peak"], [1.0, np.nan])
    with pytest.raises(ValueError, match="positive"):
        resolve_period_weights(["base", "peak"], [0.0, 0.0])


def test_period_stream_resolves_named_periods_from_context():
    stream = Stream(
        name="Period-valued",
        supply_temperature={
            "values": [300.0, 280.0],
            "period_ids": ["base", "peak"],
            "weights": [0.25, 0.75],
            "unit": "degC",
        },
        target_temperature={
            "values": [200.0, 180.0],
            "period_ids": ["base", "peak"],
            "weights": [0.25, 0.75],
            "unit": "degC",
        },
        heat_flow={
            "values": [5000.0, 4000.0],
            "period_ids": ["base", "peak"],
            "weights": [0.25, 0.75],
            "unit": "kW",
        },
        delta_t_contribution=10.0,
        heat_transfer_coefficient=2.0,
    )
    stream.set_period_context(
        period_ids={"base": "0", "peak": "1"},
        weights={"base": 0.25, "peak": 0.75},
        num_periods=2,
    )

    assert stream.resolve_attr("supply_temperature", period_id="peak") == pytest.approx(
        280.0
    )
    assert stream.supply_temperature[1].value == pytest.approx(280.0)
    np.testing.assert_allclose(stream.supply_temperature[None].value, [300.0, 280.0])
    with pytest.raises(TypeError):
        float(stream.supply_temperature)
    assert stream.supply_temperature > 250.0


def test_stream_broadcasts_scalar_updates_over_existing_state_context():
    stream = Stream(
        name="Period-valued",
        supply_temperature={
            "values": [300.0, 280.0],
            "period_ids": ["0", "1"],
            "unit": "degC",
        },
        target_temperature={
            "values": [200.0, 180.0],
            "period_ids": ["0", "1"],
            "unit": "degC",
        },
        heat_flow={"values": [5000.0, 4000.0], "period_ids": ["0", "1"], "unit": "kW"},
        delta_t_contribution=10.0,
        heat_transfer_coefficient=2.0,
    )

    stream.heat_flow = 1000.0

    np.testing.assert_allclose(
        stream.heat_flow.period_values, np.array([1000.0, 1000.0])
    )
    assert stream.resolve_attr("heat_flow", period_id="1") == pytest.approx(1000.0)


def test_stream_set_attr_for_state_updates_selected_position():
    stream = Stream(
        name="Period-valued",
        supply_temperature={
            "values": [300.0, 280.0],
            "period_ids": ["0", "1"],
            "unit": "degC",
        },
        target_temperature={
            "values": [200.0, 180.0],
            "period_ids": ["0", "1"],
            "unit": "degC",
        },
        heat_flow={"values": [5000.0, 4000.0], "period_ids": ["0", "1"], "unit": "kW"},
        delta_t_contribution=10.0,
        heat_transfer_coefficient=2.0,
    )

    stream.set_attr_for_period("heat_flow", 3500.0, period_id="1")

    np.testing.assert_allclose(
        stream.heat_flow.period_values, np.array([5000.0, 3500.0])
    )


def test_stream_accessor_assignment_updates_selected_position():
    stream = Stream(
        name="Period-valued",
        supply_temperature={
            "values": [300.0, 280.0],
            "period_ids": ["0", "1"],
            "unit": "degC",
        },
        target_temperature={
            "values": [200.0, 180.0],
            "period_ids": ["0", "1"],
            "unit": "degC",
        },
        heat_flow={"values": [5000.0, 4000.0], "period_ids": ["0", "1"], "unit": "kW"},
        delta_t_contribution=10.0,
        heat_transfer_coefficient=2.0,
    )

    with pytest.raises(TypeError, match="Stream-owned Value is read-only"):
        stream.heat_flow["1"] = 3500.0
    stream.set_value_attr_at_idx("heat_flow", 3500.0, idx="1")

    np.testing.assert_allclose(
        stream.heat_flow.period_values, np.array([5000.0, 3500.0])
    )


def test_stream_accessor_assignment_with_none_updates_default_period():
    stream = Stream(
        name="Period-valued",
        supply_temperature={
            "values": [300.0, 280.0],
            "period_ids": ["0", "1"],
            "unit": "degC",
        },
        target_temperature={
            "values": [200.0, 180.0],
            "period_ids": ["0", "1"],
            "unit": "degC",
        },
        heat_flow={"values": [5000.0, 4000.0], "period_ids": ["0", "1"], "unit": "kW"},
        delta_t_contribution=10.0,
        heat_transfer_coefficient=2.0,
    )

    with pytest.raises(TypeError, match="Stream-owned Value is read-only"):
        stream.heat_flow[None] = 3500.0
    stream.set_value_attr_at_idx("heat_flow", 3500.0, idx=None)

    np.testing.assert_allclose(
        stream.heat_flow.period_values, np.array([3500.0, 4000.0])
    )


def test_scalar_stream_accessor_assignment_with_none_updates_value():
    stream = Stream(
        name="Scalar",
        supply_temperature=300.0,
        target_temperature=200.0,
        heat_flow=5000.0,
        delta_t_contribution=10.0,
        heat_transfer_coefficient=2.0,
    )

    with pytest.raises(TypeError, match="Stream-owned Value is read-only"):
        stream.heat_flow[None] = 3500.0
    stream.set_value_attr_at_idx("heat_flow", 3500.0, idx=None)

    assert stream.heat_flow.value == pytest.approx(3500.0)


def test_stream_equal_temperature_fixtures_resolve_hot_cold_and_neutral_states():
    cold = Stream(**_stream_fixture("cold"))
    hot = Stream(**_stream_fixture("hot"))
    neutral = Stream(**_stream_fixture("neutral"))

    assert cold.stream_type == StreamType.Cold.value
    assert cold.target_temperature.value == pytest.approx(40.01)
    assert cold.shifted_minimum_temperature.value == pytest.approx(45.0)
    assert cold.heat_transfer_resistance.value == pytest.approx(0.0)

    assert hot.stream_type == StreamType.Hot.value
    assert hot.target_temperature.value == pytest.approx(89.99)
    assert hot.shifted_maximum_temperature.value == pytest.approx(86.0)

    assert neutral.stream_type == StreamType.Neutral.value
    assert neutral.heat_capacity_flowrate.value == pytest.approx(0.0)
    assert neutral.resistance_capacity_product.value == pytest.approx(0.0)


def test_stream_descriptive_properties_and_setters_cover_public_surface():
    stream = Stream(**_stream_fixture("utility"))

    stream.name = "Renamed"
    stream.is_process_stream = False
    stream.is_active = False
    stream.delta_t_contribution_multiplier = 2.0

    assert stream.name == "Renamed"
    assert stream.is_process_stream is False
    assert stream.num_periods == 1
    assert stream.period_ids == {"0": 0}
    np.testing.assert_allclose(stream.weights, np.array([1.0]))
    assert stream.stream_type == StreamType.Cold.value
    assert stream.is_active is False
    assert stream.supply_temperature.value == pytest.approx(20.0)
    assert stream.target_temperature.value == pytest.approx(70.0)
    assert stream.minimum_temperature.value == pytest.approx(20.0)
    assert stream.maximum_temperature.value == pytest.approx(70.0)
    assert stream.shifted_minimum_temperature.value == pytest.approx(26.0)
    assert stream.shifted_maximum_temperature.value == pytest.approx(76.0)
    assert stream.entropic_mean_temperature.value == pytest.approx(
        stream.entropic_mean_temperature.value
    )
    assert stream.supply_pressure.value == pytest.approx(120.0)
    assert stream.target_pressure.value == pytest.approx(100.0)
    assert stream.supply_enthalpy.value == pytest.approx(10.0)
    assert stream.target_enthalpy.value == pytest.approx(20.0)
    assert stream.delta_t_contribution.value == pytest.approx(3.0)
    assert stream.delta_t_contribution_multiplier == pytest.approx(2.0)
    assert stream.effective_delta_t_contribution.value == pytest.approx(6.0)
    assert stream.heat_transfer_coefficient.value == pytest.approx(2.0)
    assert stream.heat_transfer_resistance.value == pytest.approx(0.5)
    assert stream.utility_cost.value == pytest.approx(0.25)
    assert stream.resistance_capacity_product.value == pytest.approx(0.5)

    stream.supply_temperature = 25.0
    stream.target_temperature = 75.0
    stream.supply_pressure = 130.0
    stream.target_pressure = 110.0
    stream.supply_enthalpy = 11.0
    stream.target_enthalpy = 21.0
    stream.delta_t_contribution = 1.5
    stream.heat_flow = 80.0
    stream.heat_transfer_coefficient = 4.0
    stream.price = 6.0

    assert stream.supply_temperature.value == pytest.approx(25.0)
    assert stream.target_temperature.value == pytest.approx(75.0)
    assert stream.supply_pressure.value == pytest.approx(130.0)
    assert stream.target_pressure.value == pytest.approx(110.0)
    assert stream.supply_enthalpy.value == pytest.approx(11.0)
    assert stream.target_enthalpy.value == pytest.approx(21.0)
    assert stream.delta_t_contribution.value == pytest.approx(1.5)
    assert stream.heat_flow.value == pytest.approx(80.0)
    assert stream.heat_transfer_coefficient.value == pytest.approx(4.0)
    assert stream.price.value == pytest.approx(6.0)


def test_stream_rejects_retired_compact_runtime_names():
    stream = Stream(**_stream_fixture("hot"))
    retired = Stream._RETIRED_PUBLIC_ATTRS

    for name in retired:
        assert not hasattr(stream, name)
        with pytest.raises(AttributeError, match="descriptive runtime name"):
            setattr(stream, name, 1.0)
        with pytest.raises(AttributeError, match="has no attribute"):
            stream.set_value_attr(name, 1.0)

    with pytest.raises(TypeError, match="unexpected keyword argument 't_supply'"):
        Stream(t_supply=150.0)


def test_stream_multiplier_lock_warns_and_preserves_derived_state():
    stream = Stream(**_stream_fixture("cold"))
    original_shifted_minimum = stream.shifted_minimum_temperature.value

    stream.delta_t_contribution_multiplier_locked = True
    assert stream.delta_t_contribution_multiplier_locked is True
    with pytest.warns(UserWarning, match="Attempted to change"):
        stream.delta_t_contribution_multiplier = 10.0

    assert stream.delta_t_contribution_multiplier == pytest.approx(1.0)
    assert stream.shifted_minimum_temperature.value == pytest.approx(
        original_shifted_minimum
    )


def test_stream_can_infer_missing_supply_temperature_from_target():
    stream = Stream(target_temperature=80.0, heat_flow=0.0)

    assert stream.supply_temperature.value == pytest.approx(80.0)
    assert stream.stream_type == StreamType.Neutral.value

    default_stream = Stream()
    assert default_stream.supply_temperature.value == pytest.approx(15.0)
    assert default_stream.target_temperature.value == pytest.approx(15.0)
    assert default_stream.heat_flow.value == pytest.approx(0.0)
    assert default_stream.heat_transfer_coefficient.value == pytest.approx(1.0)
    assert default_stream.price.value == pytest.approx(0.0)

    default_stream._dt_cont = None
    default_stream._heat_flow = None
    default_stream._htc = None
    default_stream._price = None
    default_stream._calculate_missing_properties()

    assert default_stream.delta_t_contribution.value == pytest.approx(0.0)
    assert default_stream.heat_flow.value == pytest.approx(0.0)
    assert default_stream.heat_transfer_coefficient.value == pytest.approx(1.0)
    assert default_stream.price.value == pytest.approx(0.0)


def test_stream_value_setters_validate_mutability_and_period_lengths():
    stream = Stream(
        name="Period-valued",
        supply_temperature={
            "values": [300.0, 280.0],
            "period_ids": ["0", "1"],
            "unit": "degC",
        },
        target_temperature={
            "values": [200.0, 180.0],
            "period_ids": ["0", "1"],
            "unit": "degC",
        },
        heat_flow={"values": [5000.0, 4000.0], "period_ids": ["0", "1"], "unit": "kW"},
        delta_t_contribution=10.0,
        heat_transfer_coefficient=2.0,
    )

    with pytest.raises(ValueError, match="not a mutable state property"):
        stream.set_value_attr_at_idx("heat_capacity_flowrate", 1.0)
    with pytest.raises(ValueError, match="Weights length"):
        stream.heat_transfer_coefficient = {
            "values": [1.0, 2.0, 3.0],
            "unit": "kW/m^2/delta_degC",
        }
    stream._t_target = Value([200.0, 180.0, 160.0], unit="degC")
    with pytest.raises(ValueError, match="unequal period counts"):
        stream._validate_num_periods()
    stream._t_target = Value([200.0, 180.0], unit="degC")
    stream._validate_num_periods()

    stream.set_value_attr("supply_pressure", None)
    assert stream.supply_pressure is None

    class DumpableValue:
        def model_dump(self, *, mode: str):
            assert mode == "python"
            return {"value": 115.0, "unit": "kPa"}

    stream.supply_pressure = DumpableValue()
    assert stream.supply_pressure.value == pytest.approx(115.0)
    stream.set_value_attr("supply_pressure", None)

    stream.set_value_attr_at_idx("supply_pressure", 101.0, idx=1)
    np.testing.assert_allclose(
        stream.supply_pressure.period_values, np.array([0.0, 101.0])
    )


def test_stream_fluid_metadata_empty_values_normalise_to_none():
    stream = Stream(supply_temperature=30.0, target_temperature=60.0, heat_flow=10.0)

    stream.fluid_name = " "
    stream.fluid_phase = ""

    assert stream.fluid_name is None
    assert stream.fluid_phase is None


def test_stream_invert_rejects_process_streams_and_flips_utility_streams():
    utility = Stream(**_stream_fixture("utility"))
    process = Stream(**_stream_fixture("cold"))

    with pytest.raises(ValueError, match="Process streams cannot be inverted"):
        process.invert()

    utility.invert()

    assert utility.is_process_stream is True
    assert utility.supply_temperature.value == pytest.approx(70.0)
    assert utility.target_temperature.value == pytest.approx(20.0)
    assert utility.supply_pressure.value == pytest.approx(100.0)
    assert utility.target_pressure.value == pytest.approx(120.0)
    assert utility.supply_enthalpy.value == pytest.approx(20.0)
    assert utility.target_enthalpy.value == pytest.approx(10.0)


def test_stream_period_context_helpers_validate_ids_and_lengths():
    stream = Stream(
        name="Period-valued",
        supply_temperature={
            "values": [300.0, 280.0],
            "period_ids": ["0", "1"],
            "unit": "degC",
        },
        target_temperature={
            "values": [200.0, 180.0],
            "period_ids": ["0", "1"],
            "unit": "degC",
        },
        heat_flow={"values": [5000.0, 4000.0], "period_ids": ["0", "1"], "unit": "kW"},
        delta_t_contribution=10.0,
        heat_transfer_coefficient=2.0,
    )

    period_ids, weights = stream._get_period_context()
    assert period_ids == {"0": 0, "1": 1}
    np.testing.assert_allclose(weights, np.array([1.0, 1.0]))

    assert stream.get_period_index() == 0
    assert stream.resolve_attr("heat_flow", period_id="1") == pytest.approx(4000.0)
    assert stream.resolve_attr("name") == "Period-valued"
    with pytest.raises(ValueError, match="Unknown period_id"):
        stream.get_period_index("missing")
    with pytest.raises(ValueError, match="Expected at most 1 period weight"):
        stream.set_period_context(["base"], [0.5, 0.5], num_periods=2)


def test_multiperiod_pressure_preserves_all_core_field_derived_broadcasting():
    stream = Stream(
        name="Pressure periods",
        supply_temperature=200.0,
        target_temperature=100.0,
        supply_pressure=[200.0, 180.0],
        target_pressure=[150.0, 140.0],
        heat_flow=50.0,
    )

    assert stream.num_periods == 2
    assert stream.heat_capacity_flowrate.num_periods == 2
    np.testing.assert_allclose(stream.heat_capacity_flowrate.period_values, [0.5, 0.5])
    np.testing.assert_allclose(stream.minimum_temperature.period_values, [100.0, 100.0])
