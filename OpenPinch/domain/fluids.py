"""CoolProp fluid-spec parsing helpers."""

from __future__ import annotations

import functools
import math
import re
from typing import Any


def build_coolprop_abstract_state(value: str | Any):
    """Build a CoolProp ``AbstractState`` from a single or mixture fluid spec."""
    if not isinstance(value, str):
        return value

    import CoolProp

    backend, fluid = value.split("::", 1) if "::" in value else ("HEOS", value)
    backend = backend.strip()
    fluid = fluid.strip()

    if "[" not in fluid and "]" not in fluid:
        return CoolProp.AbstractState(backend, fluid)

    fluid_names: list[str] = []
    mole_fractions: list[float] = []

    for component_spec in fluid.split("&"):
        match = re.fullmatch(r"\s*([^\[\]&]+)\[([^\]]+)\]\s*", component_spec)
        if match is None:
            raise ValueError(
                "Explicit fluid mixtures must use "
                "'<component>[<mole_fraction>]' syntax."
            )

        component_name, mole_fraction_text = match.groups()
        mole_fraction = float(mole_fraction_text)

        if not math.isfinite(mole_fraction) or mole_fraction < 0.0:
            raise ValueError("Fluid mole fractions must be finite and non-negative.")

        fluid_names.append(component_name.strip())
        mole_fractions.append(mole_fraction)

    mole_fraction_total = sum(mole_fractions)
    if mole_fraction_total <= 0.0:
        raise ValueError("Fluid mole fractions must sum to a positive value.")

    state = CoolProp.AbstractState(backend, "&".join(fluid_names))
    state.set_mole_fractions(
        [mole_fraction / mole_fraction_total for mole_fraction in mole_fractions]
    )
    return state


@functools.cache
def critical_point(value: str) -> tuple[float, float, float]:
    """Return a fluid spec's critical temperature (K), pressure (Pa) and density.

    A mixture's critical point is a costly CoolProp solve, so each spec is
    solved once per process.
    """
    import CoolProp

    state = build_coolprop_abstract_state(value)
    return (
        state.keyed_output(CoolProp.iT_critical),
        state.keyed_output(CoolProp.iP_critical),
        state.keyed_output(CoolProp.irhomass_critical),
    )


def validate_coolprop_fluid_name(value: str) -> None:
    """Validate a CoolProp fluid or mixture name by constructing its state object."""
    try:
        build_coolprop_abstract_state(value)
    except Exception as exc:
        raise ValueError(f"Invalid CoolProp fluid_name {value!r}: {exc}") from exc


@functools.cache
def coolprop_fluid_spelling(value: str) -> str:
    """Return ``value`` upper-cased, or as given when CoolProp rejects that.

    Upper-casing lets lower-case input such as ``r134a`` resolve, but CoolProp
    has no upper-case key for every fluid: it accepts ``R1233zd(E)`` and not
    ``R1233ZD(E)``. A spec CoolProp rejects either way is upper-cased, so the
    caller reports the failure as before.
    """
    upper = value.upper()
    for spelling in (upper, value):
        try:
            build_coolprop_abstract_state(spelling)
        except ValueError:
            continue
        return spelling
    return upper


_WATER_NAMES = frozenset({"WATER", "H2O", "R718"})


def is_water(value: str) -> bool:
    """Return whether ``value`` names water, with or without a ``HEOS::`` prefix."""
    name = str(value).strip().upper().removeprefix("HEOS::")
    return name in _WATER_NAMES


__all__ = [
    "build_coolprop_abstract_state",
    "coolprop_fluid_spelling",
    "is_water",
    "validate_coolprop_fluid_name",
]
