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
    text = text.replace("ΔdegC", "delta_degC").replace("Δ°C", "delta_degC")
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


def normalise_weights(weights, *, expected_len: int) -> np.ndarray | None:
    """Validate and normalize optional passive period weights."""
    if weights is None:
        return None
    values = np.asarray(weights, dtype=float).reshape(-1)
    if values.size != expected_len:
        raise ValueError("weights length must match the number of periods.")
    total = float(values.sum())
    if total > 0.0:
        values = values / total
    return values
