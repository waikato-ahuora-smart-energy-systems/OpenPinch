"""Low-level input-shape predicates used by the parent Value model."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from pint.errors import DimensionalityError


def is_value_with_unit(data: Any) -> bool:
    """Return whether an object exposes value and unit attributes."""
    return hasattr(data, "value") and hasattr(data, "unit")


def is_bool_like(data: Any) -> bool:
    """Return whether input is a Python or NumPy boolean scalar."""
    return isinstance(data, (bool, np.bool_))


def is_numeric_scalar(data: Any) -> bool:
    """Return whether input is numeric scalar data excluding booleans."""
    return isinstance(data, (int, float, np.integer, np.floating)) and not is_bool_like(
        data
    )


def is_array_like_input(data: Any) -> bool:
    """Return whether input can represent a one-dimensional value sequence."""
    if data is None or isinstance(data, (str, bytes, Mapping)):
        return False
    if is_bool_like(data) or np.isscalar(data):
        return False
    try:
        list(data)
    except TypeError:
        return False
    return True


def is_serialized_scalar_data(data: Mapping[Any, Any], *, keys: set[str]) -> bool:
    """Return whether a mapping has the scalar serialization shape."""
    return set(data).issubset(keys) and "value" in data


def is_serialized_period_data(data: Mapping[Any, Any], *, keys: set[str]) -> bool:
    """Return whether a mapping has the period serialization shape."""
    return set(data).issubset(keys) and "values" in data


class ValueCoercion:
    """Normalize supported scalar, period, mapping, and quantity inputs."""

    def __init__(
        self,
        owner,
        *,
        value_type: type,
        quantity_factory,
        serialized_scalar_keys: set[str],
        serialized_period_keys: set[str],
    ):
        self._owner = owner
        self._value_type = value_type
        self._quantity_factory = quantity_factory
        self._serialized_scalar_keys = serialized_scalar_keys
        self._serialized_period_keys = serialized_period_keys

    def __getattr__(self, name):
        return getattr(self._owner, name)

    def _coerce_input(
        self,
        data,
        unit: str | None,
    ) -> tuple[Any, np.ndarray]:
        normalised_data = self._normalise_input_object(data)

        if isinstance(normalised_data, self._value_type):
            quantity = normalised_data._quantity
            weights = normalised_data.weights
        elif isinstance(normalised_data, Mapping):
            quantity, weights = self._coerce_mapping_input(normalised_data, unit)
            return quantity, weights
        elif hasattr(normalised_data, "units"):
            quantity = normalised_data
            weights = None
        elif self._is_array_like_input(normalised_data):
            quantity = self._quantity_from_values(normalised_data, unit)
            weights = None
        elif is_value_with_unit(normalised_data):
            quantity, weights = self._coerce_object_with_unit(normalised_data, unit)
            return quantity, weights
        elif normalised_data is None:
            quantity = self._quantity_from_scalar(0.0, unit)
            weights = None
        else:
            quantity = self._quantity_from_scalar(normalised_data, unit)
            weights = None

        quantity = self._coerce_quantity_to_unit(quantity, unit)
        return quantity, weights

    def _normalise_input_object(self, data):
        if hasattr(data, "model_dump") and not isinstance(data, Mapping):
            return data.model_dump(mode="python")
        return data

    def _coerce_mapping_input(
        self,
        data: Mapping[Any, Any],
        unit: str | None,
    ) -> tuple[Any, np.ndarray]:
        if self._is_serialized_period_data(data):
            quantity = self._quantity_from_values(
                data.get("values"),
                data.get("unit") or unit,
            )
            weights = data.get("weights")
        elif self._is_serialized_scalar_data(data):
            quantity = self._missing_or_zero_quantity(
                data.get("value"),
                data.get("unit") or unit,
            )
            weights = data.get("weights")
        else:
            quantity = self._quantity_from_values(list(data.values()), unit)
            weights = None

        return quantity, weights

    def _coerce_object_with_unit(
        self,
        data,
        unit: str | None,
    ) -> tuple[Any, np.ndarray]:
        source_unit = getattr(data, "unit", None) or unit
        if hasattr(data, "values"):
            quantity = self._quantity_from_values(data.values, source_unit)
            weights = getattr(data, "weights", None)
        else:
            quantity = self._missing_or_zero_quantity(
                getattr(data, "value", None), source_unit
            )
            weights = getattr(data, "weights", None)
        weights_arr = self._normalise_weights(
            weights,
            expected_len=np.asarray(quantity.magnitude, dtype=float).reshape(-1).size,
        )
        return quantity, weights_arr

    def _quantity_from_scalar(self, data, unit: str | None):
        if is_bool_like(data):
            raise TypeError("Boolean values are not supported.")
        resolved_unit = self._normalise_unit_input(unit)
        magnitude = np.asarray([data], dtype=float)
        return (
            self._quantity_factory(magnitude, self._unit_from_normalised(resolved_unit))
            if resolved_unit
            else self._quantity_factory(magnitude)
        )

    def _missing_or_zero_quantity(self, data, unit: str | None):
        resolved_unit = self._normalise_unit_input(unit)
        magnitude = np.asarray([np.nan if data is None else data], dtype=float)
        return (
            self._quantity_factory(magnitude, self._unit_from_normalised(resolved_unit))
            if resolved_unit
            else self._quantity_factory(magnitude)
        )

    def _quantity_from_values(self, data, unit: str | None):
        values_list = list(data)
        magnitudes = self._coerce_magnitude_array(
            values_list,
            expected_len=len(values_list),
            label="values",
        )
        resolved_unit = self._normalise_unit_input(unit)
        return (
            self._quantity_factory(
                magnitudes, self._unit_from_normalised(resolved_unit)
            )
            if resolved_unit
            else self._quantity_factory(magnitudes)
        )

    def _coerce_quantity_to_unit(self, quantity, unit: str | None):
        copied = self._quantity_factory(
            np.asarray(quantity.magnitude, dtype=float).reshape(-1),
            quantity.units,
        )
        if unit is None:
            return copied
        resolved_unit = self._normalise_unit_input(unit)
        unit_obj = self._unit_from_normalised(resolved_unit)
        try:
            return copied.to(unit_obj)
        except DimensionalityError, TypeError, ValueError:
            if self._quantity_is_dimensionless(copied) or self._same_dimensionality(
                copied, resolved_unit
            ):
                return self._quantity_factory(
                    np.asarray(copied.magnitude, dtype=float).reshape(-1), unit_obj
                )
            raise

    def _coerce_magnitude_array(
        self,
        data,
        *,
        expected_len: int,
        label: str,
        allow_scalar_broadcast: bool = False,
    ) -> np.ndarray:
        if is_bool_like(data):
            raise TypeError("Boolean values are not supported.")

        if (
            allow_scalar_broadcast
            and np.isscalar(data)
            and not isinstance(data, (str, bytes))
        ):
            return np.full(expected_len, float(data), dtype=float)

        if isinstance(data, (str, bytes)):
            values = [data]
        else:
            try:
                values = list(data)
            except TypeError as exc:
                raise TypeError(
                    f"{label} must be numeric scalar or 1-D array-like data."
                ) from exc

        if not values:
            raise ValueError(f"{label} cannot be empty.")
        if any(is_bool_like(value) for value in values):
            raise TypeError("Boolean values are not supported.")

        try:
            magnitudes = np.asarray(values, dtype=float).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{label} must contain numeric values.") from exc

        if len(magnitudes) != expected_len:
            raise ValueError(f"{label} length must match the number of periods.")

        return magnitudes

    @staticmethod
    def _is_numeric_scalar(other: Any) -> bool:
        return isinstance(
            other, (int, float, np.integer, np.floating)
        ) and not is_bool_like(other)

    @staticmethod
    def _is_array_like_input(data: Any) -> bool:
        if data is None or isinstance(data, (str, bytes, Mapping)):
            return False
        if is_bool_like(data) or np.isscalar(data):
            return False
        try:
            list(data)
        except TypeError:
            return False
        return True

    def _is_serialized_scalar_data(self, data: Mapping[Any, Any]) -> bool:
        return set(data).issubset(self._serialized_scalar_keys) and "value" in data

    def _is_serialized_period_data(self, data: Mapping[Any, Any]) -> bool:
        return set(data).issubset(self._serialized_period_keys) and ("values" in data)
