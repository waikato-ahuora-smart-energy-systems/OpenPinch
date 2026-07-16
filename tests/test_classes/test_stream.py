"""Regression tests for stream period handling with collection-owned metadata."""

import json
from pathlib import Path

import numpy as np
import pytest

from OpenPinch.classes._stream_value_state import resolve_period_weights
from OpenPinch.classes.stream import Stream
from OpenPinch.classes.value import Value
from OpenPinch.lib.enums import ST, FluidPhase

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "value_and_stream_edge_cases.json"
)


def _stream_fixture(name: str) -> dict:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    return fixture["stream_cases"][name]


def test_scalar_stream_initialisation_computes_derived_fields():
    stream = Stream(
        name="Hot1",
        t_supply=300.0,
        t_target=200.0,
        heat_flow=5000.0,
        dt_cont=10.0,
        htc=2.0,
        price=50.0,
    )

    assert stream.type == ST.Hot.value
    assert float(stream.t_min) == pytest.approx(200.0)
    assert float(stream.t_max) == pytest.approx(300.0)
    assert float(stream.t_min_star) == pytest.approx(190.0)
    assert float(stream.t_max_star) == pytest.approx(290.0)
    assert float(stream.CP) == pytest.approx(50.0)
    assert float(stream.rCP) == pytest.approx(25.0)


def test_stream_accepts_optional_fluid_metadata():
    stream = Stream(
        name="Refrigerant",
        t_supply=30.0,
        t_target=80.0,
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
            t_supply=30.0,
            t_target=80.0,
            heat_flow=100.0,
            fluid_phase="plasma",
        )


def test_stream_rejects_invalid_coolprop_fluid_name():
    with pytest.raises(ValueError, match="Invalid CoolProp fluid_name"):
        Stream(
            name="InvalidFluid",
            t_supply=30.0,
            t_target=80.0,
            heat_flow=100.0,
            fluid_name="NotARealCoolPropFluid",
        )


def test_derived_stream_fields_are_read_only():
    stream = Stream(
        name="Hot1",
        t_supply=300.0,
        t_target=200.0,
        heat_flow=5000.0,
        dt_cont=10.0,
        htc=2.0,
        price=50.0,
    )

    with pytest.raises(AttributeError):
        stream.CP = 10.0

    with pytest.raises(AttributeError):
        stream.t_min = 150.0

    with pytest.raises(TypeError, match="Stream-owned Value is read-only"):
        stream.t_min["0"] = 150.0
    with pytest.raises(TypeError, match="Stream-owned Value is read-only"):
        stream.CP["0"] = 10.0
    assert float(stream.t_min) == pytest.approx(200.0)
    assert float(stream.CP) == pytest.approx(50.0)


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
        t_supply={
            "values": [300.0, 280.0],
            "period_ids": ["base", "peak"],
            "weights": [0.25, 0.75],
            "unit": "degC",
        },
        t_target={
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
        dt_cont=10.0,
        htc=2.0,
    )
    stream.set_period_context(
        period_ids={"base": "0", "peak": "1"},
        weights={"base": 0.25, "peak": 0.75},
        num_periods=2,
    )

    assert stream.resolve_attr("t_supply", period_id="peak") == pytest.approx(280.0)
    assert stream.t_supply[1].value == pytest.approx(280.0)
    np.testing.assert_allclose(stream.t_supply[None].value, [300.0, 280.0])
    with pytest.raises(TypeError):
        float(stream.t_supply)
    assert stream.t_supply > 250.0


def test_stream_broadcasts_scalar_updates_over_existing_state_context():
    stream = Stream(
        name="Period-valued",
        t_supply={"values": [300.0, 280.0], "period_ids": ["0", "1"], "unit": "degC"},
        t_target={"values": [200.0, 180.0], "period_ids": ["0", "1"], "unit": "degC"},
        heat_flow={"values": [5000.0, 4000.0], "period_ids": ["0", "1"], "unit": "kW"},
        dt_cont=10.0,
        htc=2.0,
    )

    stream.heat_flow = 1000.0

    np.testing.assert_allclose(
        stream.heat_flow.period_values, np.array([1000.0, 1000.0])
    )
    assert stream.resolve_attr("heat_flow", period_id="1") == pytest.approx(1000.0)


def test_stream_set_attr_for_state_updates_selected_position():
    stream = Stream(
        name="Period-valued",
        t_supply={"values": [300.0, 280.0], "period_ids": ["0", "1"], "unit": "degC"},
        t_target={"values": [200.0, 180.0], "period_ids": ["0", "1"], "unit": "degC"},
        heat_flow={"values": [5000.0, 4000.0], "period_ids": ["0", "1"], "unit": "kW"},
        dt_cont=10.0,
        htc=2.0,
    )

    stream.set_attr_for_period("heat_flow", 3500.0, period_id="1")

    np.testing.assert_allclose(
        stream.heat_flow.period_values, np.array([5000.0, 3500.0])
    )


def test_stream_accessor_assignment_updates_selected_position():
    stream = Stream(
        name="Period-valued",
        t_supply={"values": [300.0, 280.0], "period_ids": ["0", "1"], "unit": "degC"},
        t_target={"values": [200.0, 180.0], "period_ids": ["0", "1"], "unit": "degC"},
        heat_flow={"values": [5000.0, 4000.0], "period_ids": ["0", "1"], "unit": "kW"},
        dt_cont=10.0,
        htc=2.0,
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
        t_supply={"values": [300.0, 280.0], "period_ids": ["0", "1"], "unit": "degC"},
        t_target={"values": [200.0, 180.0], "period_ids": ["0", "1"], "unit": "degC"},
        heat_flow={"values": [5000.0, 4000.0], "period_ids": ["0", "1"], "unit": "kW"},
        dt_cont=10.0,
        htc=2.0,
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
        t_supply=300.0,
        t_target=200.0,
        heat_flow=5000.0,
        dt_cont=10.0,
        htc=2.0,
    )

    with pytest.raises(TypeError, match="Stream-owned Value is read-only"):
        stream.heat_flow[None] = 3500.0
    stream.set_value_attr_at_idx("heat_flow", 3500.0, idx=None)

    assert stream.heat_flow.value == pytest.approx(3500.0)


def test_stream_equal_temperature_fixtures_resolve_hot_cold_and_neutral_states():
    cold = Stream(**_stream_fixture("cold"))
    hot = Stream(**_stream_fixture("hot"))
    neutral = Stream(**_stream_fixture("neutral"))

    assert cold.type == ST.Cold.value
    assert cold.t_target.value == pytest.approx(40.01)
    assert cold.t_min_star.value == pytest.approx(45.0)
    assert cold.htr.value == pytest.approx(0.0)

    assert hot.type == ST.Hot.value
    assert hot.t_target.value == pytest.approx(89.99)
    assert hot.t_max_star.value == pytest.approx(86.0)

    assert neutral.type == ST.Neutral.value
    assert neutral.CP.value == pytest.approx(0.0)
    assert neutral.rCP.value == pytest.approx(0.0)


def test_stream_readable_aliases_and_basic_setters_cover_public_surface():
    stream = Stream(**_stream_fixture("utility"))

    stream.name = "Renamed"
    stream.is_process_stream = False
    stream.is_active = False
    stream.dt_cont_multiplier = 2.0

    assert stream.name == "Renamed"
    assert stream.is_process_stream is False
    assert stream.num_periods == 1
    assert stream.period_ids == {"0": 0}
    np.testing.assert_allclose(stream.weights, np.array([1.0]))
    assert stream.stream_type == ST.Cold.value
    assert stream.is_active is False
    assert stream.supply_temperature.value == pytest.approx(20.0)
    assert stream.target_temperature.value == pytest.approx(70.0)
    assert stream.minimum_temperature.value == pytest.approx(20.0)
    assert stream.maximum_temperature.value == pytest.approx(70.0)
    assert stream.shifted_minimum_temperature.value == pytest.approx(26.0)
    assert stream.shifted_maximum_temperature.value == pytest.approx(76.0)
    assert stream.entropic_mean_temperature.value == pytest.approx(
        stream.t_entr_mean.value
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

    stream.t_supply = 25.0
    stream.t_target = 75.0
    stream.p_supply = 130.0
    stream.p_target = 110.0
    stream.h_supply = 11.0
    stream.h_target = 21.0
    stream.dt_cont = 1.5
    stream.heat_flow = 80.0
    stream.htc = 4.0
    stream.price = 6.0

    assert stream.t_supply.value == pytest.approx(25.0)
    assert stream.t_target.value == pytest.approx(75.0)
    assert stream.p_supply.value == pytest.approx(130.0)
    assert stream.p_target.value == pytest.approx(110.0)
    assert stream.h_supply.value == pytest.approx(11.0)
    assert stream.h_target.value == pytest.approx(21.0)
    assert stream.dt_cont.value == pytest.approx(1.5)
    assert stream.heat_flow.value == pytest.approx(80.0)
    assert stream.htc.value == pytest.approx(4.0)
    assert stream.price.value == pytest.approx(6.0)


def test_stream_multiplier_lock_warns_and_preserves_derived_state():
    stream = Stream(**_stream_fixture("cold"))
    original_shifted_minimum = stream.t_min_star.value

    stream.dt_cont_multiplier_locked = True
    assert stream.dt_cont_multiplier_locked is True
    with pytest.warns(UserWarning, match="Attempted to change"):
        stream.dt_cont_multiplier = 10.0

    assert stream.dt_cont_multiplier == pytest.approx(1.0)
    assert stream.t_min_star.value == pytest.approx(original_shifted_minimum)


def test_stream_can_infer_missing_supply_temperature_from_target():
    stream = Stream(t_target=80.0, heat_flow=0.0)

    assert stream.t_supply.value == pytest.approx(80.0)
    assert stream.type == ST.Neutral.value

    default_stream = Stream()
    assert default_stream.t_supply.value == pytest.approx(15.0)
    assert default_stream.t_target.value == pytest.approx(15.0)
    assert default_stream.heat_flow.value == pytest.approx(0.0)
    assert default_stream.htc.value == pytest.approx(1.0)
    assert default_stream.price.value == pytest.approx(0.0)

    default_stream._dt_cont = None
    default_stream._heat_flow = None
    default_stream._htc = None
    default_stream._price = None
    default_stream._calculate_missing_properties()

    assert default_stream.dt_cont.value == pytest.approx(0.0)
    assert default_stream.heat_flow.value == pytest.approx(0.0)
    assert default_stream.htc.value == pytest.approx(1.0)
    assert default_stream.price.value == pytest.approx(0.0)


def test_stream_value_setters_validate_mutability_and_period_lengths():
    stream = Stream(
        name="Period-valued",
        t_supply={"values": [300.0, 280.0], "period_ids": ["0", "1"], "unit": "degC"},
        t_target={"values": [200.0, 180.0], "period_ids": ["0", "1"], "unit": "degC"},
        heat_flow={"values": [5000.0, 4000.0], "period_ids": ["0", "1"], "unit": "kW"},
        dt_cont=10.0,
        htc=2.0,
    )

    with pytest.raises(ValueError, match="not a mutable state property"):
        stream.set_value_attr_at_idx("CP", 1.0)
    with pytest.raises(ValueError, match="Weights length"):
        stream.htc = {"values": [1.0, 2.0, 3.0], "unit": "kW/m^2/delta_degC"}
    stream._t_target = Value([200.0, 180.0, 160.0], unit="degC")
    with pytest.raises(ValueError, match="unequal period counts"):
        stream._validate_num_periods()
    stream._t_target = Value([200.0, 180.0], unit="degC")
    stream._validate_num_periods()

    stream.set_value_attr("p_supply", None)
    assert stream.p_supply is None

    class DumpableValue:
        def model_dump(self, *, mode: str):
            assert mode == "python"
            return {"value": 115.0, "unit": "kPa"}

    stream.p_supply = DumpableValue()
    assert stream.p_supply.value == pytest.approx(115.0)
    stream.set_value_attr("p_supply", None)

    stream.set_value_attr_at_idx("p_supply", 101.0, idx=1)
    np.testing.assert_allclose(stream.p_supply.period_values, np.array([0.0, 101.0]))


def test_stream_fluid_metadata_empty_values_normalise_to_none():
    stream = Stream(t_supply=30.0, t_target=60.0, heat_flow=10.0)

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
    assert utility.t_supply.value == pytest.approx(70.0)
    assert utility.t_target.value == pytest.approx(20.0)
    assert utility.p_supply.value == pytest.approx(100.0)
    assert utility.p_target.value == pytest.approx(120.0)
    assert utility.h_supply.value == pytest.approx(20.0)
    assert utility.h_target.value == pytest.approx(10.0)


def test_stream_period_context_helpers_validate_ids_and_lengths():
    stream = Stream(
        name="Period-valued",
        t_supply={"values": [300.0, 280.0], "period_ids": ["0", "1"], "unit": "degC"},
        t_target={"values": [200.0, 180.0], "period_ids": ["0", "1"], "unit": "degC"},
        heat_flow={"values": [5000.0, 4000.0], "period_ids": ["0", "1"], "unit": "kW"},
        dt_cont=10.0,
        htc=2.0,
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
        t_supply=200.0,
        t_target=100.0,
        p_supply=[200.0, 180.0],
        p_target=[150.0, 140.0],
        heat_flow=50.0,
    )

    assert stream.num_periods == 2
    assert stream.CP.num_periods == 2
    np.testing.assert_allclose(stream.CP.period_values, [0.5, 0.5])
    np.testing.assert_allclose(stream.t_min.period_values, [100.0, 100.0])


def test_stream_private_helpers_cover_defensive_edges():
    stream = Stream(**_stream_fixture("cold"))

    assert stream._copy_value(None) is None
    assert stream._coerce_to_value(None, "_p_supply") is None
    assert stream._resolve_attr_name("_heat_flow") == "_heat_flow"
    assert stream._resolve_attr_name("stream_type") == "stream_type"
    assert stream._resolve_attr_name("numeric_revision") == "_numeric_revision"
    with pytest.raises(AttributeError, match="no attribute"):
        stream._resolve_attr_name("not_a_stream_field")

    assert Stream._normalise_period_ids(None) is None
    assert Stream._normalise_period_ids(["base", "peak"]) == {"base": 0, "peak": 1}
    assert Stream._normalise_period_ids({"base": "0"}) == {"base": 0}
    assert Stream._normalise_weights(None, expected_len=1) is None
    np.testing.assert_allclose(
        Stream._normalise_weights([1.0, 3.0], expected_len=2),
        np.array([0.25, 0.75]),
    )
    np.testing.assert_allclose(
        Stream._normalise_weights([0.0, 0.0], expected_len=2),
        np.array([0.0, 0.0]),
    )
    with pytest.raises(ValueError, match="weights length"):
        Stream._normalise_weights([1.0, 2.0], expected_len=1)

    with pytest.raises(ValueError, match="inconsistent"):
        stream._value_array(Value([1.0, 2.0], unit="kW"), size=3)
    np.testing.assert_allclose(stream._value_array(None, size=2), np.array([0.0, 0.0]))
