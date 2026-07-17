"""Shared resolution of public workflow arguments and their provenance."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from ...domain.configuration_fields import (
    CONFIG_FIELD_SPECS,
    USER_CONFIG_FIELD_SPECS,
    validate_internal_configuration_options,
)


class _MissingArgument:
    def __repr__(self) -> str:
        return "MISSING"


MISSING = _MissingArgument()


@dataclass(frozen=True)
class ArgumentSpec:
    """Map one descriptive public argument to a flat configuration option."""

    option_key: str
    default: Any = MISSING


@dataclass(frozen=True)
class EffectiveArguments:
    """Resolved invocation values with immutable provenance labels."""

    values: Mapping[str, Any]
    provenance: Mapping[str, str]


def resolve_effective_arguments(
    specs: Mapping[str, ArgumentSpec],
    *,
    named: Mapping[str, Any] | None = None,
    options: Mapping[str, Any] | None = None,
    config_values: Mapping[str, Any] | None = None,
) -> EffectiveArguments:
    """Resolve named, advanced, configured, and default values in precedence order."""
    named = dict(named or {})
    options = dict(options or {})
    config_values = dict(config_values or {})
    values: dict[str, Any] = {}
    provenance: dict[str, str] = {}

    for public_name, spec in specs.items():
        named_value = named.get(public_name, MISSING)
        if named_value is not MISSING:
            value, source = named_value, "named"
        elif public_name in options:
            value, source = options[public_name], "options"
        elif spec.option_key in options:
            value, source = options[spec.option_key], "options"
        elif spec.option_key in config_values:
            value, source = config_values[spec.option_key], "config"
        elif spec.default is not MISSING:
            value, source = spec.default, "default"
        else:
            continue
        values[public_name] = value
        provenance[public_name] = source

    return EffectiveArguments(
        values=MappingProxyType(values),
        provenance=MappingProxyType(provenance),
    )


def split_runtime_and_configuration_options(
    options: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Separate flat numerical configuration values from execution context."""
    runtime: dict[str, Any] = {}
    configuration: dict[str, Any] = {}
    for key, value in dict(options or {}).items():
        if key in USER_CONFIG_FIELD_SPECS:
            configuration[key] = value
        elif key in CONFIG_FIELD_SPECS:
            raise ValueError(
                f"{key} is selected by a descriptive workflow method and cannot "
                "be supplied through options."
            )
        else:
            runtime[key] = value
    return runtime, configuration


@contextmanager
def temporary_zone_configuration(root, overrides: Mapping[str, Any] | None):
    """Apply validated numerical overrides to one zone tree for one invocation."""
    validated = validate_internal_configuration_options(dict(overrides or {}))
    if not validated:
        yield
        return

    zones = []
    stack = [root]
    while stack:
        zone = stack.pop()
        zones.append(zone)
        stack.extend(reversed(list(zone.subzones.values())))
    original = [zone.config for zone in zones]
    try:
        for zone in zones:
            effective = deepcopy(zone.config)
            effective._values.update(deepcopy(validated))
            effective._build_groups(effective._values)
            zone.config = effective
        yield
    finally:
        for zone, config in zip(zones, original, strict=True):
            zone.config = config


__all__ = [
    "ArgumentSpec",
    "EffectiveArguments",
    "MISSING",
    "resolve_effective_arguments",
    "split_runtime_and_configuration_options",
    "temporary_zone_configuration",
]
