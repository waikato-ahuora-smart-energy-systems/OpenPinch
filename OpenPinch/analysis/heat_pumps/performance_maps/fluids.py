"""Working-fluid parsing and capability checks for HPR map simulation."""

from __future__ import annotations

import math
import re
from functools import lru_cache

import CoolProp
from CoolProp.CoolProp import get_global_param_string

from ....domain.fluids import build_coolprop_abstract_state
from .models import HprWorkingFluidSpec

_EXPLICIT_COMPONENT = re.compile(r"\s*([^\[\]&]+)\[([^\]]+)\]\s*")


@lru_cache(maxsize=1)
def _registered_mixture_names() -> frozenset[str]:
    """Return normalized predefined-mixture tokens from the installed backend."""
    names = get_global_param_string("predefined_mixtures").split(",")
    normalized: set[str] = set()
    for name in names:
        token = name.strip()
        if not token:
            continue
        normalized.add(token.casefold())
        if token.casefold().endswith(".mix"):
            normalized.add(token[:-4].casefold())
    return frozenset(normalized)


def parse_hpr_working_fluid(value: str) -> HprWorkingFluidSpec:
    """Parse one existing OpenPinch fluid string without constructing an engine."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("working-fluid specification must be a nonempty string")

    backend_text, fluid_text = (
        value.split("::", 1) if "::" in value else ("HEOS", value)
    )
    backend = backend_text.strip().upper()
    fluid = fluid_text.strip()
    if not backend or not fluid:
        raise ValueError("working-fluid backend and fluid token must not be empty")

    if "[" not in fluid and "]" not in fluid:
        kind = (
            "registered_blend"
            if fluid.casefold() in _registered_mixture_names()
            else "pure"
        )
        return HprWorkingFluidSpec(
            source_spec=value,
            property_backend=backend,
            kind=kind,
            registered_name=fluid,
            components=() if kind == "registered_blend" else (fluid,),
            mole_fractions=() if kind == "registered_blend" else (1.0,),
            composition_basis=(
                "provider_defined" if kind == "registered_blend" else "not_applicable"
            ),
        )

    components: list[str] = []
    fractions: list[float] = []
    for component_text in fluid.split("&"):
        match = _EXPLICIT_COMPONENT.fullmatch(component_text)
        if match is None:
            raise ValueError(
                "explicit fluid mixtures must use component[mole_fraction] syntax"
            )
        component, fraction_text = match.groups()
        component = component.strip()
        fraction = float(fraction_text)
        if not component:
            raise ValueError("mixture component names must not be empty")
        if component in components:
            raise ValueError("mixture component names must be distinct")
        if not math.isfinite(fraction) or fraction < 0.0:
            raise ValueError("mole fractions must be finite and nonnegative")
        components.append(component)
        fractions.append(fraction)

    total = sum(fractions)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("mole fractions must sum to a positive finite value")
    normalized = tuple(fraction / total for fraction in fractions)
    return HprWorkingFluidSpec(
        source_spec=value,
        property_backend=backend,
        kind="explicit_molar_mixture",
        registered_name=None,
        components=tuple(components),
        mole_fractions=normalized,
        composition_basis="molar",
    )


def resolve_hpr_working_fluid(
    value: str,
    evaporating_temperature: float,
    condensing_temperature: float,
) -> HprWorkingFluidSpec:
    """Parse a fluid and prove its required dew/bubble saturation states."""
    fluid = parse_hpr_working_fluid(value)
    if fluid.property_backend == "REFPROP":
        raise ValueError("REFPROP is not a supported HPR property backend")

    try:
        state = build_coolprop_abstract_state(fluid.source_spec)
        state.update(CoolProp.QT_INPUTS, 1.0, evaporating_temperature + 273.15)
        state.p()
        state.update(CoolProp.QT_INPUTS, 0.0, condensing_temperature + 273.15)
        state.p()
    except Exception as exc:
        raise ValueError(
            "working fluid is unsupported at the required dew/bubble states"
        ) from exc
    return fluid


def to_tespy_fluid_token(fluid: HprWorkingFluidSpec) -> str:
    """Return the equivalent single-token TESPy fluid-wrapper specification."""
    if fluid.kind != "explicit_molar_mixture":
        return fluid.source_spec.strip()
    parts = "&".join(
        f"{component}[{fraction:.16g}]"
        for component, fraction in zip(
            fluid.components,
            fluid.mole_fractions,
            strict=True,
        )
    )
    return f"{fluid.property_backend}::{parts}|molar"


__all__ = [
    "parse_hpr_working_fluid",
    "resolve_hpr_working_fluid",
    "to_tespy_fluid_token",
]
