"""Shared unit rules and coercion helpers for input preparation and reporting."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..classes.value import Value

if TYPE_CHECKING:
    from .config import Configuration


@dataclass(frozen=True)
class InputUnitRule:
    """Canonical runtime unit and fallback aliases for one input field."""

    canonical_unit: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class OutputUnitRule:
    """Source/runtime and default display units for one report metric."""

    user_unit: str | None
    default_unit: str | None
    aliases: tuple[str, ...] = ()


_DIMENSIONLESS_UNIT_TOKENS = {"", "-", "dimensionless", "1"}

INPUT_UNIT_RULES: dict[str, InputUnitRule] = {
    "t_supply": InputUnitRule("degC", aliases=("temperature",)),
    "t_target": InputUnitRule("degC", aliases=("temperature",)),
    "p_supply": InputUnitRule("kPa", aliases=("pressure",)),
    "p_target": InputUnitRule("kPa", aliases=("pressure",)),
    "h_supply": InputUnitRule("kJ/kg", aliases=("enthalpy",)),
    "h_target": InputUnitRule("kJ/kg", aliases=("enthalpy",)),
    "heat_flow": InputUnitRule("kW", aliases=("heat_flow",)),
    "dt_cont": InputUnitRule(
        "delta_degC",
        aliases=("delta_temperature", "temperature_difference"),
    ),
    "htc": InputUnitRule(
        "kW/m^2/delta_degC",
        aliases=("heat_transfer_coefficient",),
    ),
    "price": InputUnitRule("$/MWh", aliases=("price", "utility_price")),
}

OUTPUT_UNIT_RULES: dict[str, OutputUnitRule] = {
    "Qh": OutputUnitRule("kW", "kW", aliases=("heat_flow",)),
    "Qc": OutputUnitRule("kW", "kW", aliases=("heat_flow",)),
    "Qr": OutputUnitRule("kW", "kW", aliases=("heat_flow",)),
    "utility_heat_flow": OutputUnitRule("kW", "kW", aliases=("heat_flow",)),
    "cold_temp": OutputUnitRule("degC", "degC", aliases=("temperature",)),
    "hot_temp": OutputUnitRule("degC", "degC", aliases=("temperature",)),
    "degree_of_integration": OutputUnitRule(
        "dimensionless",
        "%",
        aliases=("percent", "fraction"),
    ),
    "utility_cost": OutputUnitRule("$/h", "$/h", aliases=("utility_cost",)),
    "work_target": OutputUnitRule("kW", "kW", aliases=("heat_flow",)),
    "turbine_efficiency_target": OutputUnitRule(
        "dimensionless",
        "%",
        aliases=("percent", "fraction"),
    ),
    "area": OutputUnitRule("m^2", "m^2", aliases=("area",)),
    "capital_cost": OutputUnitRule("$", "$", aliases=("capital_cost", "currency")),
    "total_cost": OutputUnitRule(
        "$/y",
        "$/y",
        aliases=("annual_cost", "currency"),
    ),
    "exergy_sources": OutputUnitRule("kW", "kW", aliases=("heat_flow",)),
    "exergy_sinks": OutputUnitRule("kW", "kW", aliases=("heat_flow",)),
    "ETE": OutputUnitRule(
        "dimensionless",
        "%",
        aliases=("percent", "fraction"),
    ),
    "exergy_req_min": OutputUnitRule("kW", "kW", aliases=("heat_flow",)),
    "exergy_des_min": OutputUnitRule("kW", "kW", aliases=("heat_flow",)),
    "hpr_utility_total": OutputUnitRule("kW", "kW", aliases=("heat_flow",)),
    "hpr_work": OutputUnitRule("kW", "kW", aliases=("heat_flow",)),
    "hpr_external_utility": OutputUnitRule("kW", "kW", aliases=("heat_flow",)),
    "hpr_ambient_hot": OutputUnitRule("kW", "kW", aliases=("heat_flow",)),
    "hpr_ambient_cold": OutputUnitRule("kW", "kW", aliases=("heat_flow",)),
    "hpr_cop": OutputUnitRule(
        "dimensionless",
        "dimensionless",
        aliases=("dimensionless", "cop"),
    ),
    "hpr_eta_he": OutputUnitRule(
        "dimensionless",
        "%",
        aliases=("percent", "fraction"),
    ),
    "hpr_operating_cost": OutputUnitRule(
        "$/y",
        "$/y",
        aliases=("annual_cost", "currency"),
    ),
    "hpr_capital_cost": OutputUnitRule(
        "$",
        "$",
        aliases=("capital_cost", "currency"),
    ),
    "hpr_annualized_capital_cost": OutputUnitRule(
        "$/y",
        "$/y",
        aliases=("annual_cost", "currency"),
    ),
    "hpr_total_annualized_cost": OutputUnitRule(
        "$/y",
        "$/y",
        aliases=("annual_cost", "currency"),
    ),
    "hpr_compressor_capital_cost": OutputUnitRule(
        "$",
        "$",
        aliases=("capital_cost", "currency"),
    ),
    "hpr_heat_exchanger_capital_cost": OutputUnitRule(
        "$",
        "$",
        aliases=("capital_cost", "currency"),
    ),
}

__all__ = [
    "INPUT_UNIT_RULES",
    "OUTPUT_UNIT_RULES",
    "coerce_output_value",
    "standardise_input_value",
]


def _config_units(
    config_or_overrides: Configuration | Mapping[str, Any] | None,
    *,
    attr_name: str,
) -> Mapping[str, Any]:
    if config_or_overrides is None:
        return {}
    if isinstance(config_or_overrides, Mapping):
        if attr_name in config_or_overrides and isinstance(
            config_or_overrides[attr_name], Mapping
        ):
            return config_or_overrides[attr_name]
        return config_or_overrides
    overrides = getattr(config_or_overrides, attr_name, None)
    return overrides if isinstance(overrides, Mapping) else {}


def _resolve_override(
    *,
    key: str,
    aliases: tuple[str, ...],
    overrides: Mapping[str, Any],
) -> str | None:
    for candidate in (key, *aliases):
        if candidate not in overrides:
            continue
        resolved = overrides[candidate]
        if resolved is None:
            continue
        text = str(resolved).strip()
        if text:
            return text
    return None


def _normalise_unit_text(unit: str | None) -> str | None:
    if unit is None:
        return None
    text = str(unit).strip()
    if not text:
        return None
    if text in {"-", "dimensionless", "1"}:
        return "dimensionless"
    return text


def _unit_from_mapping(value: Mapping[str, Any]) -> str | None:
    return _normalise_unit_text(value.get("unit"))


def _unit_from_value_like(value: Any) -> str | None:
    if isinstance(value, Value):
        return _normalise_unit_text(value.unit)
    if isinstance(value, Mapping):
        return _unit_from_mapping(value)
    if hasattr(value, "unit"):
        return _normalise_unit_text(getattr(value, "unit", None))
    return None


def _has_explicit_unit(value: Any) -> bool:
    unit = _unit_from_value_like(value)
    return unit is not None and unit not in _DIMENSIONLESS_UNIT_TOKENS


def _value_like_payload(value: Any) -> Any:
    if hasattr(value, "model_dump") and not isinstance(value, Mapping):
        return value.model_dump(mode="python")
    return value


def _value_magnitudes(value: Any) -> Any:
    payload = _value_like_payload(value)
    if isinstance(payload, Value):
        return payload.state_values if payload.num_states > 1 else payload.value
    if isinstance(payload, Mapping):
        if "values" in payload:
            return payload.get("values")
        if "value" in payload:
            return payload.get("value")
    if hasattr(payload, "values"):
        return getattr(payload, "values")
    if hasattr(payload, "value"):
        return getattr(payload, "value")
    return payload


def _ensure_value(value: Any, *, user_unit: str | None) -> Value:
    payload = _value_like_payload(value)
    if isinstance(payload, Value):
        if _has_explicit_unit(payload):
            return Value(payload)
        return Value(payload.to_dict(), unit=user_unit)
    if _has_explicit_unit(payload):
        return Value(payload)
    return Value(payload, unit=user_unit)


def standardise_input_value(
    value: Any,
    *,
    field_name: str,
    config: Configuration | Mapping[str, Any] | None = None,
) -> Value | None:
    """Convert one input value into the canonical runtime unit for ``field_name``."""
    if value is None:
        return None

    rule = INPUT_UNIT_RULES.get(field_name)
    if rule is None:
        return Value(value)

    overrides = _config_units(config, attr_name="INPUT_UNITS")
    source_value = value
    source_unit_text = _unit_from_value_like(value)
    if field_name == "dt_cont" and source_unit_text in {"degC", "K"}:
        delta_unit = "K" if source_unit_text == "K" else "delta_degC"
        source_value = Value(_value_magnitudes(value), unit=delta_unit)

    user_unit = _resolve_override(
        key=field_name,
        aliases=rule.aliases,
        overrides=overrides,
    )
    resolved = _ensure_value(
        source_value,
        user_unit=rule.canonical_unit if user_unit is None else user_unit,
    )
    return resolved.to(rule.canonical_unit)


def coerce_output_value(
    value: Any,
    *,
    metric_name: str,
    config: Configuration | Mapping[str, Any] | None = None,
) -> Value | None:
    """Convert one report metric into a ``Value`` using configured display units."""
    if value is None:
        return None

    rule = OUTPUT_UNIT_RULES.get(metric_name)
    if rule is None:
        return Value(value)

    overrides = _config_units(config, attr_name="OUTPUT_UNITS")
    display_unit = _resolve_override(
        key=metric_name,
        aliases=rule.aliases,
        overrides=overrides,
    )
    resolved = _ensure_value(value, user_unit=rule.user_unit)

    target_unit = rule.default_unit if display_unit is None else display_unit
    if target_unit is None:
        return resolved
    return resolved.to(target_unit)
