"""CoolProp fluid-spec parsing helpers."""

from __future__ import annotations

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


def validate_coolprop_fluid_name(value: str) -> None:
    """Validate a CoolProp fluid or mixture name by constructing its state object."""
    try:
        build_coolprop_abstract_state(value)
    except Exception as exc:
        raise ValueError(f"Invalid CoolProp fluid_name {value!r}: {exc}") from exc


__all__ = ["build_coolprop_abstract_state", "validate_coolprop_fluid_name"]
