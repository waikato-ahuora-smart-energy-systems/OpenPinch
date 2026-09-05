"""Strict loader for the OpenPinch-owned TESPy compressor characteristic."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from importlib.resources import files
from typing import Any

CHARACTERISTIC_RESOURCE_NAME = "openpinch-single-stage-compressor-v1.json"
_RESOURCE_PACKAGE = "OpenPinch.data.heat_pumps.performance_maps"
_EXPECTED_KEYS = frozenset(
    {
        "schema_version",
        "characteristic_set_id",
        "abscissa",
        "ordinate",
        "points",
    }
)


@dataclass(frozen=True, slots=True)
class TespyCompressorCharacteristic:
    """Immutable characteristic data and canonical resource identity."""

    schema_version: str
    characteristic_set_id: str
    abscissa: str
    ordinate: str
    points: tuple[tuple[float, float], ...]
    sha256: str


def _object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"compressor characteristic repeats key {key!r}")
        result[key] = value
    return result


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"compressor characteristic {label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"compressor characteristic {label} must be finite")
    return result


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def parse_tespy_compressor_characteristic(
    content: bytes,
) -> TespyCompressorCharacteristic:
    """Validate canonical resource bytes and return an immutable value."""
    try:
        payload = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"compressor characteristic contains nonfinite {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        raise ValueError("compressor characteristic is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict) or set(payload) != _EXPECTED_KEYS:
        raise ValueError("compressor characteristic keys do not match schema 1.0")
    if content != _canonical_bytes(payload):
        raise ValueError("compressor characteristic bytes are not canonical")
    if payload["schema_version"] != "1.0":
        raise ValueError("compressor characteristic schema version is unsupported")
    if payload["characteristic_set_id"] != "openpinch-single-stage-compressor-v1":
        raise ValueError("compressor characteristic identifier is unsupported")
    if payload["abscissa"] != "relative_mass_flow":
        raise ValueError("compressor characteristic abscissa is unsupported")
    if payload["ordinate"] != "relative_isentropic_efficiency":
        raise ValueError("compressor characteristic ordinate is unsupported")

    raw_points = payload["points"]
    if not isinstance(raw_points, list) or len(raw_points) < 2:
        raise ValueError("compressor characteristic needs at least two points")
    points: list[tuple[float, float]] = []
    for index, raw_point in enumerate(raw_points):
        if not isinstance(raw_point, list) or len(raw_point) != 2:
            raise ValueError("compressor characteristic points must be pairs")
        x = _number(raw_point[0], f"point {index} abscissa")
        y = _number(raw_point[1], f"point {index} ordinate")
        if points and x <= points[-1][0]:
            raise ValueError(
                "compressor characteristic abscissae must be strictly ascending"
            )
        if y <= 0.0:
            raise ValueError(
                "compressor characteristic efficiency factors must be positive"
            )
        points.append((x, y))

    return TespyCompressorCharacteristic(
        schema_version=payload["schema_version"],
        characteristic_set_id=payload["characteristic_set_id"],
        abscissa=payload["abscissa"],
        ordinate=payload["ordinate"],
        points=tuple(points),
        sha256=hashlib.sha256(content).hexdigest(),
    )


def read_tespy_compressor_characteristic_bytes() -> bytes:
    """Read the packaged canonical compressor characteristic bytes."""
    return files(_RESOURCE_PACKAGE).joinpath(CHARACTERISTIC_RESOURCE_NAME).read_bytes()


def load_tespy_compressor_characteristic() -> TespyCompressorCharacteristic:
    """Load and strictly validate the packaged compressor characteristic."""
    return parse_tespy_compressor_characteristic(
        read_tespy_compressor_characteristic_bytes()
    )


__all__ = [
    "CHARACTERISTIC_RESOURCE_NAME",
    "TespyCompressorCharacteristic",
    "load_tespy_compressor_characteristic",
    "parse_tespy_compressor_characteristic",
    "read_tespy_compressor_characteristic_bytes",
]
