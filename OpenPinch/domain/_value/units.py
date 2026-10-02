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
