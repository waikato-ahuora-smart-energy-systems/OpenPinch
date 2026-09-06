"""Generate or check canonical HPR performance-map contract resources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from OpenPinch.contracts.hpr_performance_map import (
    HprPerformanceMap,
    HprPerformanceMapUnits,
    HprPerformancePoint,
)

ROOT = Path(__file__).resolve().parents[1]
RESOURCE_ROOT = ROOT / "OpenPinch" / "data" / "contracts" / "hpr_performance_map"
# Historical generating release for these schema 1.0 examples. Golden resources
# must not change when a different OpenPinch release is installed.
FIXTURE_GENERATOR_VERSION = "0.6.4"


def _units() -> HprPerformanceMapUnits:
    return HprPerformanceMapUnits(
        source_temperature="degC",
        sink_temperature="degC",
        q_source="kW",
        q_sink="kW",
        electric_power="kW",
    )


def _heat_pump_fixture() -> HprPerformanceMap:
    return HprPerformanceMap(
        schema_version="1.0",
        map_id="openpinch-golden-heat-pump-1.0",
        mode="heat_pump",
        units=_units(),
        reference_capacity=100.0,
        reference_capacity_basis="q_sink",
        interpolation_topology="ordered_part_load_curve",
        thermodynamic_backend="coolprop",
        model_id="reference-single-stage-heat-pump-v1",
        provenance={
            "generator": "OpenPinch contract fixture",
            "versions": {"openpinch": FIXTURE_GENERATOR_VERSION},
            "cycle": {
                "family": "vapour_compression_heat_pump",
                "refrigerant": "R134a",
            },
            "assumptions": [
                "external service temperatures",
                "electric power includes declared auxiliaries",
                "no hidden thermal losses",
            ],
            "fixture": True,
        },
        points=(
            HprPerformancePoint(
                name="hp-20C-60C-plr-050",
                curve_id="hp-20C-60C",
                source_temperature=20.0,
                sink_temperature=60.0,
                load_fraction=0.5,
                q_source=37.5,
                q_sink=50.0,
                electric_power=12.5,
                cop=4.0,
            ),
            HprPerformancePoint(
                name="hp-20C-60C-plr-075",
                curve_id="hp-20C-60C",
                source_temperature=20.0,
                sink_temperature=60.0,
                load_fraction=0.75,
                q_source=57.5,
                q_sink=75.0,
                electric_power=17.5,
                cop=75.0 / 17.5,
            ),
            HprPerformancePoint(
                name="hp-20C-60C-plr-100",
                curve_id="hp-20C-60C",
                source_temperature=20.0,
                sink_temperature=60.0,
                load_fraction=1.0,
                q_source=78.0,
                q_sink=100.0,
                electric_power=22.0,
                cop=100.0 / 22.0,
            ),
        ),
        cop_convention="heating",
    )


def _refrigeration_fixture() -> HprPerformanceMap:
    return HprPerformanceMap(
        schema_version="1.0",
        map_id="openpinch-golden-refrigeration-1.0",
        mode="refrigeration",
        units=_units(),
        reference_capacity=80.0,
        reference_capacity_basis="q_source",
        interpolation_topology="ordered_part_load_curve",
        thermodynamic_backend="coolprop",
        model_id="reference-single-stage-refrigeration-v1",
        provenance={
            "generator": "OpenPinch contract fixture",
            "versions": {"openpinch": FIXTURE_GENERATOR_VERSION},
            "cycle": {
                "family": "vapour_compression_refrigeration",
                "refrigerant": "R290",
            },
            "assumptions": [
                "external service temperatures",
                "electric power includes declared auxiliaries",
                "no hidden thermal losses",
            ],
            "fixture": True,
        },
        points=(
            HprPerformancePoint(
                name="ref--5C-35C-plr-050",
                curve_id="ref--5C-35C",
                source_temperature=-5.0,
                sink_temperature=35.0,
                load_fraction=0.5,
                q_source=40.0,
                q_sink=52.0,
                electric_power=12.0,
                cop=40.0 / 12.0,
            ),
            HprPerformancePoint(
                name="ref--5C-35C-plr-075",
                curve_id="ref--5C-35C",
                source_temperature=-5.0,
                sink_temperature=35.0,
                load_fraction=0.75,
                q_source=60.0,
                q_sink=76.0,
                electric_power=16.0,
                cop=3.75,
            ),
            HprPerformancePoint(
                name="ref--5C-35C-plr-100",
                curve_id="ref--5C-35C",
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


def _canonical_json(value: Any) -> str:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def render_contract_resources() -> dict[str, str]:
    """Return the complete canonical resource catalog as UTF-8 text."""
    return {
        "heat-pump-1.0.json": _canonical_json(
            _heat_pump_fixture().model_dump(mode="json")
        ),
        "refrigeration-1.0.json": _canonical_json(
            _refrigeration_fixture().model_dump(mode="json")
        ),
        "schema-1.0.json": _canonical_json(HprPerformanceMap.model_json_schema()),
    }


def check_contract_resources(resource_root: Path = RESOURCE_ROOT) -> None:
    """Raise when committed resources differ from authoritative generation."""
    generated = render_contract_resources()
    actual = {
        path.name: path.read_text(encoding="utf-8")
        for path in resource_root.glob("*.json")
    }
    if actual != generated:
        missing = sorted(set(generated).difference(actual))
        unexpected = sorted(set(actual).difference(generated))
        changed = sorted(
            name
            for name in set(actual).intersection(generated)
            if actual[name] != generated[name]
        )
        raise RuntimeError(
            "HPR contract resources are stale: "
            f"missing={missing!r}, unexpected={unexpected!r}, changed={changed!r}"
        )


def write_contract_resources(resource_root: Path = RESOURCE_ROOT) -> None:
    """Write the complete authoritative resource catalog."""
    resource_root.mkdir(parents=True, exist_ok=True)
    for name, text in render_contract_resources().items():
        (resource_root / name).write_text(text, encoding="utf-8", newline="")


def main(argv: list[str] | None = None) -> int:
    """Run explicit check or regeneration mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check", action="store_true")
    modes.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    if args.write:
        write_contract_resources()
    else:
        check_contract_resources()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
