"""Regression tests for stream state handling with collection-owned metadata."""

import numpy as np
import pytest

from OpenPinch.classes.stream import Stream
from OpenPinch.lib.enums import ST


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

    # Derived-value accessors return defensive copies, so mutating them does not
    # write back onto the stream.
    stream.t_min["0"] = 150.0
    stream.CP["0"] = 10.0
    assert float(stream.t_min) == pytest.approx(200.0)
    assert float(stream.CP) == pytest.approx(50.0)


def test_stateful_stream_resolves_named_states_from_context():
    stream = Stream(
        name="Stateful",
        t_supply={
            "values": [300.0, 280.0],
            "state_ids": ["base", "peak"],
            "weights": [0.25, 0.75],
            "unit": "degC",
        },
        t_target={
            "values": [200.0, 180.0],
            "state_ids": ["base", "peak"],
            "weights": [0.25, 0.75],
            "unit": "degC",
        },
        heat_flow={
            "values": [5000.0, 4000.0],
            "state_ids": ["base", "peak"],
            "weights": [0.25, 0.75],
            "unit": "kW",
        },
        dt_cont=10.0,
        htc=2.0,
    )
    stream.set_state_context(
        state_ids={"base": "0", "peak": "1"},
        weights={"base": 0.25, "peak": 0.75},
        num_states=2,
    )

    assert stream.resolve_attr("t_supply", state_id="peak") == pytest.approx(280.0)
    assert stream.t_supply[1].value == pytest.approx(280.0)
    np.testing.assert_allclose(stream.t_supply[None].value, [300.0, 280.0])
    with pytest.raises(TypeError):
        float(stream.t_supply)
    assert stream.t_supply > 250.0


def test_stream_broadcasts_scalar_updates_over_existing_state_context():
    stream = Stream(
        name="Stateful",
        t_supply={"values": [300.0, 280.0], "state_ids": ["0", "1"], "unit": "degC"},
        t_target={"values": [200.0, 180.0], "state_ids": ["0", "1"], "unit": "degC"},
        heat_flow={"values": [5000.0, 4000.0], "state_ids": ["0", "1"], "unit": "kW"},
        dt_cont=10.0,
        htc=2.0,
    )

    stream.heat_flow = 1000.0

    np.testing.assert_allclose(
        stream.heat_flow.state_values, np.array([1000.0, 1000.0])
    )
    assert stream.resolve_attr("heat_flow", state_id="1") == pytest.approx(1000.0)


def test_stream_set_attr_for_state_updates_selected_position():
    stream = Stream(
        name="Stateful",
        t_supply={"values": [300.0, 280.0], "state_ids": ["0", "1"], "unit": "degC"},
        t_target={"values": [200.0, 180.0], "state_ids": ["0", "1"], "unit": "degC"},
        heat_flow={"values": [5000.0, 4000.0], "state_ids": ["0", "1"], "unit": "kW"},
        dt_cont=10.0,
        htc=2.0,
    )

    stream.set_attr_for_state("heat_flow", 3500.0, state_id="1")

    np.testing.assert_allclose(
        stream.heat_flow.state_values, np.array([5000.0, 3500.0])
    )


def test_stream_accessor_assignment_updates_selected_position():
    stream = Stream(
        name="Stateful",
        t_supply={"values": [300.0, 280.0], "state_ids": ["0", "1"], "unit": "degC"},
        t_target={"values": [200.0, 180.0], "state_ids": ["0", "1"], "unit": "degC"},
        heat_flow={"values": [5000.0, 4000.0], "state_ids": ["0", "1"], "unit": "kW"},
        dt_cont=10.0,
        htc=2.0,
    )

    stream.heat_flow["1"] = 3500.0

    np.testing.assert_allclose(
        stream.heat_flow.state_values, np.array([5000.0, 3500.0])
    )


def test_stream_accessor_assignment_with_none_updates_default_state():
    stream = Stream(
        name="Stateful",
        t_supply={"values": [300.0, 280.0], "state_ids": ["0", "1"], "unit": "degC"},
        t_target={"values": [200.0, 180.0], "state_ids": ["0", "1"], "unit": "degC"},
        heat_flow={"values": [5000.0, 4000.0], "state_ids": ["0", "1"], "unit": "kW"},
        dt_cont=10.0,
        htc=2.0,
    )

    stream.heat_flow[None] = 3500.0

    np.testing.assert_allclose(
        stream.heat_flow.state_values, np.array([3500.0, 4000.0])
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

    stream.heat_flow[None] = 3500.0

    assert stream.heat_flow.value == pytest.approx(3500.0)
