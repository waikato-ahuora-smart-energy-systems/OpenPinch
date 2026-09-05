"""Generated properties for the HPR performance-map contract."""

from __future__ import annotations

import copy
import json
import random

import pytest
from hypothesis import given, seed, settings
from jsonschema import validators
from pydantic import ValidationError

from OpenPinch.contracts.hpr_performance_map import (
    HprPerformanceMap,
    HprPerformanceMapRequest,
)
from scripts import generate_hpr_performance_map_contract as contract_generator
from tests.strategies.hpr_performance_maps import (
    hpr_performance_map_payloads,
    hpr_request_coordinate_sets,
)

_SCHEMA = json.loads(contract_generator.render_contract_resources()["schema-1.0.json"])
_SCHEMA_VALIDATOR = validators.validator_for(_SCHEMA)(_SCHEMA)


@seed(20260905)
@settings(max_examples=60)
@given(payload=hpr_performance_map_payloads())
def test_valid_maps_round_trip_and_preserve_physical_invariants(payload) -> None:
    _SCHEMA_VALIDATOR.validate(payload)
    performance_map = HprPerformanceMap.model_validate(payload)

    restored = HprPerformanceMap.model_validate_json(performance_map.model_dump_json())

    assert restored == performance_map
    assert json.loads(performance_map.model_dump_json()) == performance_map.model_dump(
        mode="json"
    )
    for point in performance_map.points:
        assert point.q_sink == pytest.approx(
            point.q_source + point.electric_power,
            abs=performance_map.energy_balance_tolerance,
        )
        useful_duty = (
            point.q_sink
            if performance_map.reference_capacity_basis == "q_sink"
            else point.q_source
        )
        assert useful_duty == pytest.approx(
            point.load_fraction * performance_map.reference_capacity,
            abs=performance_map.energy_balance_tolerance,
        )
        assert point.cop == pytest.approx(
            useful_duty / point.electric_power,
            abs=performance_map.energy_balance_tolerance,
        )


@seed(20260905)
@settings(max_examples=50)
@given(payload=hpr_performance_map_payloads())
def test_serialization_is_detached_and_keeps_json_types(payload) -> None:
    performance_map = HprPerformanceMap.model_validate(payload)
    before = performance_map.model_dump(mode="json")
    detached = performance_map.model_dump(mode="json")

    detached["map_id"] = "changed"
    detached["points"].clear()
    detached["provenance"].clear()

    assert performance_map.model_dump(mode="json") == before
    json.dumps(before, allow_nan=False)


@seed(20260905)
@settings(max_examples=50)
@given(coordinates=hpr_request_coordinate_sets())
def test_request_canonicalization_is_permutation_independent(coordinates) -> None:
    sources, sinks, loads = coordinates
    shuffled_sources = list(sources)
    shuffled_sinks = list(sinks)
    shuffled_loads = list(loads)
    random.Random(20260905).shuffle(shuffled_sources)
    random.Random(20260906).shuffle(shuffled_sinks)
    random.Random(20260907).shuffle(shuffled_loads)

    canonical = HprPerformanceMapRequest(
        map_id="generated",
        source_temperatures=sources,
        sink_temperatures=sinks,
        load_fractions=loads,
    )
    permuted = HprPerformanceMapRequest(
        map_id="generated",
        source_temperatures=shuffled_sources,
        sink_temperatures=shuffled_sinks,
        load_fractions=shuffled_loads,
    )

    assert permuted == canonical
    assert canonical.source_temperatures == tuple(sorted(sources))
    assert canonical.sink_temperatures == tuple(sorted(sinks))
    assert canonical.load_fractions == tuple(sorted(loads))


@seed(20260905)
@settings(max_examples=40)
@given(payload=hpr_performance_map_payloads())
def test_energy_corruption_always_fails_closed(payload) -> None:
    corrupted = copy.deepcopy(payload)
    corrupted["points"][0]["q_sink"] += 1.0

    with pytest.raises(ValidationError):
        HprPerformanceMap.model_validate(corrupted)


@seed(20260905)
@settings(max_examples=40)
@given(payload=hpr_performance_map_payloads())
def test_point_order_is_strictly_increasing_within_every_curve(payload) -> None:
    performance_map = HprPerformanceMap.model_validate(payload)

    points_by_curve: dict[str, list[float]] = {}
    for point in performance_map.points:
        points_by_curve.setdefault(point.curve_id, []).append(point.load_fraction)

    assert all(
        load_fractions == sorted(set(load_fractions))
        for load_fractions in points_by_curve.values()
    )
