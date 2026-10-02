"""Internal temperature units of the utility-placement analysis.

The placement analysis (bounds, thermodynamics, replay) works in degC and
delta_degC. A request may use other temperature units (K, degF): its inputs are
converted to these on entry and its results back to its own units on exit.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from ...contracts.utility_placement import QuantityInterval, QuantityValue
from ...domain.value import Value

INTERNAL_TEMPERATURE = "degC"
INTERNAL_TEMPERATURE_DIFFERENCE = "delta_degC"


def _converted(value: float, source: str, target: str) -> float:
    if source == target:
        return float(value)
    return float(Value(float(value), source).to(target).value)


def convert_to_request_units(model: Any, *, absolute: str, difference: str) -> Any:
    """Return ``model`` with internal temperatures in the request's units.

    Walks frozen contract models, tuples and lists, converting every
    ``QuantityValue`` and ``QuantityInterval`` labelled degC or delta_degC.
    """
    targets = {
        INTERNAL_TEMPERATURE: absolute,
        INTERNAL_TEMPERATURE_DIFFERENCE: difference,
    }
    if absolute == INTERNAL_TEMPERATURE and difference == (
        INTERNAL_TEMPERATURE_DIFFERENCE
    ):
        return model

    def convert(item: Any) -> Any:
        if isinstance(item, QuantityValue):
            target = targets.get(item.unit)
            if target is None:
                return item
            return item.model_copy(
                update={
                    "value": _converted(item.value, item.unit, target),
                    "unit": target,
                }
            )
        if isinstance(item, QuantityInterval):
            target = targets.get(item.unit)
            if target is None:
                return item
            return item.model_copy(
                update={
                    "lower": _converted(item.lower, item.unit, target),
                    "upper": _converted(item.upper, item.unit, target),
                    "unit": target,
                }
            )
        if isinstance(item, BaseModel):
            updates = {}
            for name in type(item).model_fields:
                current = getattr(item, name)
                new = convert(current)
                if new is not current:
                    updates[name] = new
            return item.model_copy(update=updates) if updates else item
        if isinstance(item, tuple):
            converted = tuple(convert(element) for element in item)
            changed = any(a is not b for a, b in zip(converted, item))
            return converted if changed else item
        if isinstance(item, list):
            return [convert(element) for element in item]
        return item

    return convert(model)


__all__ = [
    "INTERNAL_TEMPERATURE",
    "INTERNAL_TEMPERATURE_DIFFERENCE",
    "convert_to_request_units",
]
