"""Examples for the plain HPR performance-map contract."""

from __future__ import annotations

import json
import math
from pathlib import Path
from time import perf_counter

import pytest
from jsonschema import validators
from pydantic import ValidationError

from OpenPinch.contracts.hpr_performance_map import (
    HprPerformanceMap,
    HprPerformanceMapRequest,
    HprPerformanceMapUnits,
    HprPerformancePoint,
)
from scripts import generate_hpr_performance_map_contract as contract_generator

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_RESOURCE_ROOT = (
    ROOT / "OpenPinch" / "data" / "contracts" / "hpr_performance_map"
)
EXPECTED_CONTRACT_RESOURCES = {
    "schema-1.0.json",
    "heat-pump-1.0.json",
    "refrigeration-1.0.json",
}


def _point(
    *,
    name: str,
    curve_id: str = "curve-20-60",
    source_temperature: float = 20.0,
    sink_temperature: float = 60.0,
    load_fraction: float = 0.5,
    q_source: float = 37.5,
    q_sink: float = 50.0,
    electric_power: float = 12.5,
    cop: float = 4.0,
) -> HprPerformancePoint:
    return HprPerformancePoint(
        name=name,
        curve_id=curve_id,
        source_temperature=source_temperature,
        sink_temperature=sink_temperature,
        load_fraction=load_fraction,
        q_source=q_source,
        q_sink=q_sink,
        electric_power=electric_power,
        cop=cop,
    )


def _heat_pump_map() -> HprPerformanceMap:
    return HprPerformanceMap(
        schema_version="1.0",
        map_id="hp-demo",
        mode="heat_pump",
        units=HprPerformanceMapUnits(
            source_temperature="degC",
            sink_temperature="degC",
            q_source="kW",
            q_sink="kW",
            electric_power="kW",
        ),
        reference_capacity=100.0,
        reference_capacity_basis="q_sink",
        interpolation_topology="ordered_part_load_curve",
        thermodynamic_backend="coolprop",
        model_id="single-stage-r134a",
        provenance={
            "generator": "OpenPinch",
            "versions": {"openpinch": "0.6.4", "coolprop": "7"},
            "assumptions": ["external service temperatures", "auxiliaries included"],
            "converged": True,
            "iteration_limit": 50,
            "note": None,
        },
        points=(
            _point(name="hp-050"),
            _point(
                name="hp-075",
                load_fraction=0.75,
                q_source=57.5,
                q_sink=75.0,
                electric_power=17.5,
                cop=75.0 / 17.5,
            ),
            _point(
                name="hp-100",
                load_fraction=1.0,
                q_source=78.0,
                q_sink=100.0,
                electric_power=22.0,
                cop=100.0 / 22.0,
            ),
        ),
        cop_convention="heating",
    )


def _refrigeration_map() -> HprPerformanceMap:
    return HprPerformanceMap(
        schema_version="1.0",
        map_id="refrigeration-demo",
        mode="refrigeration",
        units=HprPerformanceMapUnits(
            source_temperature="degC",
            sink_temperature="degC",
            q_source="kW",
            q_sink="kW",
            electric_power="kW",
        ),
        reference_capacity=80.0,
        reference_capacity_basis="q_source",
        interpolation_topology="ordered_part_load_curve",
        thermodynamic_backend="tespy",
        model_id="single-stage-r290",
        provenance={"generator": "OpenPinch", "tespy": {"version": "0.10"}},
        points=(
            _point(
                name="ref-050",
                curve_id="curve--5-35",
                source_temperature=-5.0,
                sink_temperature=35.0,
                q_source=40.0,
                q_sink=52.0,
                electric_power=12.0,
                cop=40.0 / 12.0,
            ),
            _point(
                name="ref-075",
                curve_id="curve--5-35",
                source_temperature=-5.0,
                sink_temperature=35.0,
                load_fraction=0.75,
                q_source=60.0,
                q_sink=76.0,
                electric_power=16.0,
                cop=60.0 / 16.0,
            ),
            _point(
                name="ref-100",
                curve_id="curve--5-35",
                source_temperature=-5.0,
                sink_temperature=35.0,
                load_fraction=1.0,
                q_source=80.0,
                q_sink=100.0,
                electric_power=20.0,
                cop=4.0,
            ),
        ),
        cop_convention="cooling",
    )


def _payload() -> dict[str, object]:
    return _heat_pump_map().model_dump(mode="json")


def test_heat_pump_and_refrigeration_use_mode_specific_conventions() -> None:
    heat_pump = _heat_pump_map()
    refrigeration = _refrigeration_map()

    assert heat_pump.reference_capacity_basis == "q_sink"
    assert heat_pump.cop_convention == "heating"
    assert refrigeration.reference_capacity_basis == "q_source"
    assert refrigeration.cop_convention == "cooling"
    assert heat_pump.points[0].q_sink == pytest.approx(
        heat_pump.points[0].load_fraction * heat_pump.reference_capacity
    )
    assert refrigeration.points[0].q_source == pytest.approx(
        refrigeration.points[0].load_fraction * refrigeration.reference_capacity
    )


def test_contract_has_the_exact_alpha_field_and_unit_sets() -> None:
    payload = _payload()

    assert set(payload) == {
        "schema_version",
        "map_id",
        "mode",
        "units",
        "reference_capacity",
        "reference_capacity_basis",
        "interpolation_topology",
        "thermodynamic_backend",
        "model_id",
        "provenance",
        "points",
        "cop_convention",
        "energy_balance_tolerance",
        "temperature_match_tolerance",
    }
    assert payload["units"] == {
        "source_temperature": "degC",
        "sink_temperature": "degC",
        "q_source": "kW",
        "q_sink": "kW",
        "electric_power": "kW",
    }


def test_json_round_trip_preserves_structured_provenance() -> None:
    source = _heat_pump_map()

    restored = HprPerformanceMap.model_validate_json(source.model_dump_json())

    assert restored == source
    assert restored.provenance["converged"] is True
    assert restored.provenance["iteration_limit"] == 50
    assert restored.provenance["note"] is None
    assert restored.provenance["assumptions"] == [
        "external service temperatures",
        "auxiliaries included",
    ]


def test_serialized_payload_is_detached_and_json_compatible() -> None:
    source = _heat_pump_map()
    first = source.model_dump(mode="json")

    first["provenance"]["versions"]["openpinch"] = "changed"  # type: ignore[index]
    first["points"][0]["q_sink"] = -1.0  # type: ignore[index]

    second = source.model_dump(mode="json")
    assert second["provenance"]["versions"]["openpinch"] == "0.6.4"  # type: ignore[index]
    assert second["points"][0]["q_sink"] == 50.0  # type: ignore[index]
    json.dumps(second, allow_nan=False)


def test_contract_values_are_frozen() -> None:
    performance_map = _heat_pump_map()

    with pytest.raises(ValidationError, match="frozen"):
        performance_map.map_id = "changed"  # type: ignore[misc]


def test_request_is_canonical_and_rejects_duplicate_coordinates() -> None:
    request = HprPerformanceMapRequest(
        map_id="requested-map",
        source_temperatures=(20.0, 5.0),
        sink_temperatures=(70.0, 50.0),
        load_fractions=(1.0, 0.5, 0.75),
        reference_capacity=100.0,
    )

    assert request.source_temperatures == (5.0, 20.0)
    assert request.sink_temperatures == (50.0, 70.0)
    assert request.load_fractions == (0.5, 0.75, 1.0)

    with pytest.raises(ValidationError, match="unique"):
        HprPerformanceMapRequest(
            map_id="duplicate",
            source_temperatures=(20.0, 20.0),
            sink_temperatures=(60.0,),
            load_fractions=(1.0,),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("map_id", "  ", "map_id must not be empty"),
        ("source_temperatures", 20.0, "must be a sequence"),
        ("sink_temperatures", (), "must not be empty"),
        ("source_temperatures", (math.inf,), "must be finite"),
        ("load_fractions", (0.0, 1.0), r"interval \(0, 1\]"),
    ],
)
def test_request_rejects_invalid_grid_values(
    field: str, value: object, message: str
) -> None:
    payload = {
        "map_id": "requested-map",
        "source_temperatures": (5.0, 20.0),
        "sink_temperatures": (50.0, 70.0),
        "load_fractions": (0.5, 0.75, 1.0),
    }
    payload[field] = value

    with pytest.raises(ValidationError, match=message):
        HprPerformanceMapRequest.model_validate(payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(schema_version="2.0"), "schema_version"),
        (lambda value: value.update(mode="heater"), "mode"),
        (
            lambda value: value.update(reference_capacity_basis="q_source"),
            "reference_capacity_basis",
        ),
        (lambda value: value.update(cop_convention="cooling"), "cop_convention"),
        (
            lambda value: value.update(interpolation_topology="convex_hull"),
            "interpolation_topology",
        ),
        (lambda value: value.update(reference_capacity=0.0), "reference_capacity"),
        (
            lambda value: value.update(energy_balance_tolerance=-1.0),
            "energy_balance_tolerance",
        ),
        (
            lambda value: value.update(temperature_match_tolerance=math.inf),
            "temperature_match_tolerance",
        ),
    ],
)
def test_invalid_map_level_values_fail_closed(mutation, message: str) -> None:
    payload = _payload()
    mutation(payload)

    with pytest.raises(ValidationError, match=message):
        HprPerformanceMap.model_validate(payload)


@pytest.mark.parametrize(
    "field",
    ["map_id", "thermodynamic_backend", "model_id"],
)
def test_required_map_identifiers_reject_blank_values(field: str) -> None:
    payload = _payload()
    payload[field] = " "

    with pytest.raises(ValidationError, match="must not be empty"):
        HprPerformanceMap.model_validate(payload)


@pytest.mark.parametrize(
    "units",
    [
        {
            "source_temperature": "K",
            "sink_temperature": "degC",
            "q_source": "kW",
            "q_sink": "kW",
            "electric_power": "kW",
        },
        {
            "source_temperature": "degC",
            "sink_temperature": "degC",
            "q_source": "kW",
            "q_sink": "kW",
        },
        {
            "source_temperature": "degC",
            "sink_temperature": "degC",
            "q_source": "kW",
            "q_sink": "kW",
            "electric_power": "kW",
            "cop": "dimensionless",
        },
    ],
)
def test_units_are_exact_and_closed(units: dict[str, str]) -> None:
    payload = _payload()
    payload["units"] = units

    with pytest.raises(ValidationError, match="unit"):
        HprPerformanceMap.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_temperature", math.nan, "source_temperature"),
        ("sink_temperature", math.inf, "sink_temperature"),
        ("load_fraction", 0.0, "load_fraction"),
        ("load_fraction", 1.1, "load_fraction"),
        ("q_source", -1.0, "q_source"),
        ("q_sink", -1.0, "q_sink"),
        ("electric_power", 0.0, "electric_power"),
        ("cop", 0.0, "cop"),
    ],
)
def test_invalid_point_domains_are_rejected(
    field: str, value: float, message: str
) -> None:
    payload = _payload()
    payload["points"][0][field] = value  # type: ignore[index]

    with pytest.raises(ValidationError, match=message):
        HprPerformanceMap.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("q_source", 35.0, r"q_sink = q_source \+ electric_power"),
        ("q_sink", 49.0, r"load_fraction \* reference_capacity"),
        ("cop", 3.0, "heating COP"),
    ],
)
def test_physical_inconsistency_rejects_the_complete_map(
    field: str, value: float, message: str
) -> None:
    payload = _payload()
    payload["points"][0][field] = value  # type: ignore[index]
    if field == "q_sink":
        payload["points"][0]["q_source"] = 36.5  # type: ignore[index]

    with pytest.raises(ValidationError, match=message):
        HprPerformanceMap.model_validate(payload)


def test_point_names_and_coordinates_are_unique() -> None:
    duplicate_name = _payload()
    duplicate_name["points"][1]["name"] = "hp-050"  # type: ignore[index]
    with pytest.raises(ValidationError, match="point names.*unique"):
        HprPerformanceMap.model_validate(duplicate_name)

    duplicate_coordinate = _payload()
    duplicate_coordinate["points"][1].update(  # type: ignore[index]
        source_temperature=20.0,
        sink_temperature=60.0,
        load_fraction=0.5,
    )
    with pytest.raises(ValidationError, match="coordinates.*unique"):
        HprPerformanceMap.model_validate(duplicate_coordinate)


def test_each_curve_has_one_temperature_pair_and_ascending_loads() -> None:
    mixed_temperature = _payload()
    mixed_temperature["points"][1]["source_temperature"] = 25.0  # type: ignore[index]
    with pytest.raises(ValidationError, match="one source/sink temperature pair"):
        HprPerformanceMap.model_validate(mixed_temperature)

    unordered = _payload()
    unordered["points"][0], unordered["points"][1] = (  # type: ignore[index]
        unordered["points"][1],
        unordered["points"][0],
    )
    with pytest.raises(ValidationError, match="strictly increasing"):
        HprPerformanceMap.model_validate(unordered)

    noncanonical_curves = _payload()
    point = dict(noncanonical_curves["points"][0])  # type: ignore[index]
    point.update(
        name="earlier-curve-point",
        curve_id="earlier-curve",
        source_temperature=10.0,
        sink_temperature=50.0,
    )
    noncanonical_curves["points"].append(point)  # type: ignore[union-attr]
    with pytest.raises(ValidationError, match="canonical temperature order"):
        HprPerformanceMap.model_validate(noncanonical_curves)


@pytest.mark.parametrize(
    "provenance",
    [
        [],
        {},
        {1: "non-string key"},
        {"bad": b"bytes"},
        {"bad": {1, 2}},
        {"bad": math.nan},
        {"bad": object()},
    ],
)
def test_provenance_is_nonempty_structured_json_without_string_coercion(
    provenance: object,
) -> None:
    payload = _payload()
    payload["provenance"] = provenance

    with pytest.raises(ValidationError, match="provenance"):
        HprPerformanceMap.model_validate(payload)


def test_extra_fields_are_rejected_at_every_contract_level() -> None:
    map_payload = _payload()
    map_payload["candidate_cost"] = 123.0
    with pytest.raises(ValidationError, match="candidate_cost"):
        HprPerformanceMap.model_validate(map_payload)

    point_payload = _payload()
    point_payload["points"][0]["period"] = "winter"  # type: ignore[index]
    with pytest.raises(ValidationError, match="period"):
        HprPerformanceMap.model_validate(point_payload)

    units_payload = _payload()
    units_payload["units"]["cop"] = "dimensionless"  # type: ignore[index]
    with pytest.raises(ValidationError, match="cop"):
        HprPerformanceMap.model_validate(units_payload)


def test_contract_resource_generator_has_the_exact_closed_catalog() -> None:
    generated = contract_generator.render_contract_resources()

    assert set(generated) == EXPECTED_CONTRACT_RESOURCES
    assert all(
        text.endswith("\n") and not text.endswith("\n\n") for text in generated.values()
    )
    assert all(len(text.encode("utf-8")) < 250_000 for text in generated.values())
    assert len(generated["heat-pump-1.0.json"].encode("utf-8")) < 100_000
    assert len(generated["refrigeration-1.0.json"].encode("utf-8")) < 100_000


@pytest.mark.parametrize(
    "name",
    ["heat-pump-1.0.json", "refrigeration-1.0.json"],
)
def test_generated_golden_fixture_is_a_representative_valid_map(name: str) -> None:
    text = contract_generator.render_contract_resources()[name]
    payload = json.loads(text)

    performance_map = HprPerformanceMap.model_validate(payload)

    assert len(performance_map.points) >= 3
    assert len({point.cop for point in performance_map.points}) >= 2
    assert [point.load_fraction for point in performance_map.points] == sorted(
        point.load_fraction for point in performance_map.points
    )
    assert payload == json.loads(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    )


def test_committed_contract_resources_match_authoritative_generation() -> None:
    generated = contract_generator.render_contract_resources()

    assert {
        path.name: path.read_text(encoding="utf-8")
        for path in CONTRACT_RESOURCE_ROOT.glob("*.json")
    } == generated
    contract_generator.check_contract_resources(CONTRACT_RESOURCE_ROOT)


def test_golden_fixtures_pass_independent_json_schema_validation() -> None:
    generated = contract_generator.render_contract_resources()
    schema = json.loads(generated["schema-1.0.json"])
    validator_type = validators.validator_for(schema)
    validator_type.check_schema(schema)
    validator = validator_type(schema)

    for name in ("heat-pump-1.0.json", "refrigeration-1.0.json"):
        validator.validate(json.loads(generated[name]))


def test_ten_thousand_point_map_validates_within_bounded_time() -> None:
    points = []
    reference_capacity = 100.0
    for curve_index in range(100):
        source_temperature = float(-20 + curve_index)
        sink_temperature = source_temperature + 40.0
        curve_id = f"curve-{curve_index:03d}"
        for load_index in range(1, 101):
            load_fraction = load_index / 100.0
            q_sink = reference_capacity * load_fraction
            electric_power = q_sink / 4.0
            points.append(
                {
                    "name": f"point-{curve_index:03d}-{load_index:03d}",
                    "curve_id": curve_id,
                    "source_temperature": source_temperature,
                    "sink_temperature": sink_temperature,
                    "load_fraction": load_fraction,
                    "q_source": q_sink - electric_power,
                    "q_sink": q_sink,
                    "electric_power": electric_power,
                    "cop": 4.0,
                }
            )
    payload = _payload()
    payload["points"] = points

    started = perf_counter()
    performance_map = HprPerformanceMap.model_validate(payload)
    elapsed = perf_counter() - started

    assert len(performance_map.points) == 10_000
    assert elapsed < 2.0, f"10,000-point validation took {elapsed:.3f} seconds"
