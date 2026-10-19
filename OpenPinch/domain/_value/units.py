"""Unit spelling normalization and cached registry lookup."""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=256)
def normalise_unit_text(unit: str | None) -> str | None:
    """Normalize accepted user-facing unit spellings for Pint."""
    if unit is None:
        return None
    text = str(unit).strip().replace("$", "USD")
    if text in {"", "-", "dimensionless", "1", "fraction"}:
        return "dimensionless"
    if text in {"USD/y", "USD/yr", "USD/year"}:
        return "USD/year"
    # Per-year rates with a prefix or other numerator, such as k$/y.
    text = re.sub(r"/(?:y|yr)$", "/year", text)
    if text in {"C", "°C"}:
        return "degC"
    if text == "degK":
        return "K"
    text = re.sub(r"(?<=[A-Za-z])2(?=($|[./*]))", "^2", text)
    text = re.sub(r"(?<=[A-Za-z])3(?=($|[./*]))", "^3", text)
    return text.replace(".K", "/K").replace(".degC", "/degC")


MAX_UNIT_TEXT_LENGTH = 64
_MAX_UNIT_PARENTHESIS_DEPTH = 2
# Unit text tokens: names, plain numbers, one bounded power per factor, the
# product and ratio operators, parentheses and spaces.
_UNIT_TOKEN = re.compile(
    r"(?P<space>\s+)"
    r"|(?P<name>[A-Za-z_$%°Δµμ][A-Za-z0-9_$%°Δµμ]*)"
    r"|(?P<number>[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d{1,3})?)"
    r"|(?P<power>(?:\^|\*\*)\s*[+-]?\d{1,2}(?:\.\d{1,3})?(?![\d.]))"
    r"|(?P<operator>[*/])"
    r"|(?P<open>\()"
    r"|(?P<close>\))"
)


class UnitTextError(ValueError):
    """Unit text outside the grammar OpenPinch accepts."""


def check_unit_text(text: str) -> str:
    """Return ``text`` when it is plain unit or quantity text, else raise.

    Pint evaluates unit text as an expression, so text is held to a bounded
    grammar first: at most ``MAX_UNIT_TEXT_LENGTH`` characters, products and
    ratios of names and numbers, one power of at most two digits on a name
    or a parenthesised group, and at most two levels of parentheses.
    """
    if len(text) > MAX_UNIT_TEXT_LENGTH:
        raise UnitTextError(
            f"A unit can have at most {MAX_UNIT_TEXT_LENGTH} characters."
        )
    depth = 0
    previous = None
    position = 0
    while position < len(text):
        match = _UNIT_TOKEN.match(text, position)
        if match is None:
            raise UnitTextError(f"Unsupported character in unit {text!r}.")
        kind = match.lastgroup
        position = match.end()
        if kind == "space":
            continue
        if kind == "power" and previous not in {"name", "close"}:
            raise UnitTextError(
                f"A power in unit {text!r} must follow a unit name or a group."
            )
        if kind == "open":
            depth += 1
            if depth > _MAX_UNIT_PARENTHESIS_DEPTH:
                raise UnitTextError(f"Too many nested groups in unit {text!r}.")
        if kind == "close":
            depth -= 1
            if depth < 0:
                raise UnitTextError(f"Unbalanced parentheses in unit {text!r}.")
        previous = kind
    if depth:
        raise UnitTextError(f"Unbalanced parentheses in unit {text!r}.")
    return text


def clean_unit_text(text: str) -> str:
    """Return stable OpenPinch unit spelling for serialization and display."""
    text = text.replace("USD", "$").replace("NZD", "$").replace(" ", "")
    text = text.replace("°", "deg")
    text = text.replace("Δdeg", "delta_deg").replace("Δ°C", "delta_degC")
    text = text.replace("**2", "^2").replace("**3", "^3")
    text = text.replace("$/a", "$/y").replace("$/year", "$/y")
    return "-" if text == "" else text


def unit_object(registry, unit: str):
    """Return one registry unit; registry-level caching remains authoritative."""
    return registry.Unit(unit)


def format_units(units) -> str:
    """Format units for display using stable OpenPinch spellings."""
    return clean_unit_text(format(units, "~"))


def serialise_units(units) -> str:
    """Format units for stable serialized output."""
    return clean_unit_text(format(units, "~"))


def unit_from_normalised(registry, unit: str):
    """Resolve an already-normalized unit through the owning registry."""
    return unit_object(registry, unit)


def quantity_is_dimensionless(quantity) -> bool:
    """Return whether a Pint quantity is dimensionless."""
    return str(quantity.units) == "dimensionless"


def same_dimensionality(quantity, unit: str, *, quantity_factory, registry) -> bool:
    """Safely compare quantity dimensionality with a normalized unit."""
    try:
        expected = quantity_factory(1.0, unit_object(registry, unit))
        return quantity.dimensionality == expected.dimensionality
    except Exception:
        return False


def validate_weights(weights, *, expected_len: int) -> np.ndarray | None:
    """Return a validated copy of optional passive period weights.

    The rules match the problem's period weights (``resolve_period_weights``):
    finite, non-negative, and with a positive sum.
    """
    if weights is None:
        return None
    values = np.array(weights, dtype=float, copy=True).reshape(-1)
    if values.size != expected_len:
        raise ValueError("weights length must match the number of periods.")
    if not np.isfinite(values).all():
        raise ValueError("Period weights must be finite.")
    if (values < 0.0).any():
        raise ValueError("Period weights must be non-negative.")
    if float(values.sum()) <= 0.0:
        raise ValueError("Period weights must have a positive sum.")
    return values


def normalise_weights(weights, *, expected_len: int) -> np.ndarray | None:
    """Validate and normalize optional passive period weights."""
    values = validate_weights(weights, expected_len=expected_len)
    if values is None:
        return None
    return values / float(values.sum())
