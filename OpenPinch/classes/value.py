"""Unit-aware scalar and discrete-state value wrapper powered by Pint quantities."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from pint import UnitRegistry

ureg = UnitRegistry()
Q_ = ureg.Quantity

_SERIALIZED_SCALAR_KEYS = {"value", "unit", "units"}
_SERIALIZED_STATEFUL_KEYS = {"values", "state_ids", "weights", "unit", "units"}


def _is_value_with_unit(data: Any) -> bool:
    """Return ``True`` for objects that look like ``ValueWithUnit`` instances."""
    return hasattr(data, "value") and hasattr(data, "units")


def _is_bool_like(data: Any) -> bool:
    """Return ``True`` for bool scalars including numpy bools."""
    return isinstance(data, (bool, np.bool_))


class Value:
    """Thin wrapper around a Pint ``Quantity`` with helpers for serialisation and arithmetic."""

    def __init__(
        self,
        data=None,
        unit: str = None,
        *,
        values=None,
        weights: Mapping[str, float] | list[float] | np.ndarray | None = None,
        state_id: list[str] | tuple[str, ...] | None = None,
        state_ids: list[str] | tuple[str, ...] | None = None,
    ):
        """Create a unit-aware scalar or discrete-state value."""
        if data is not None and values is not None:
            raise TypeError("Use either data or values, not both.")
        if state_id is not None and state_ids is not None:
            raise TypeError("Use either state_id or state_ids, not both.")

        if values is not None:
            data = values
        if state_id is not None:
            state_ids = state_id

        if isinstance(data, Mapping):
            if self._is_serialized_stateful_payload(data):
                self._init_stateful(
                    data["values"],
                    unit=data.get("unit") or data.get("units") or unit,
                    weights=data.get("weights", weights),
                    state_ids=data.get("state_ids"),
                )
                return
            if self._is_serialized_scalar_payload(data):
                if weights is not None or state_ids is not None:
                    raise TypeError(
                        "Scalar value payloads do not accept state_ids or weights."
                    )
                self._init_scalar(
                    data.get("value"),
                    data.get("unit") or data.get("units") or unit,
                    preserve_none_unit=(data.get("value") is None),
                )
                return
            self._init_stateful(data, unit=unit, weights=weights, state_ids=state_ids)
            return

        if _is_value_with_unit(data):
            if weights is not None or state_ids is not None:
                raise TypeError(
                    "ValueWithUnit inputs do not accept state_ids or weights."
                )
            self._init_from_value_with_unit(data, unit)
            return

        if self._is_array_like_input(data):
            if state_ids is None:
                raise TypeError(
                    "state_ids are required for array-backed stateful values."
                )
            self._init_stateful(data, unit=unit, weights=weights, state_ids=state_ids)
            return

        if state_ids is not None:
            self._init_stateful(data, unit=unit, weights=weights, state_ids=state_ids)
            return

        if weights is not None:
            raise TypeError("weights can only be used with stateful values.")

        if data is None:
            self._init_scalar(0.0, None, preserve_none_unit=True)
            return

        self._init_scalar(data, unit)

    @property
    def value(self):
        """Return the scalar magnitude or per-state magnitudes for stateful values."""
        if self._is_stateful():
            return self.state_values
        return self._weighted_magnitude()

    @property
    def weighted_value(self) -> float:
        """Return the weighted scalar view of the stored quantity."""
        return self._weighted_magnitude()

    @value.setter
    def value(self, data):
        if self._is_stateful():
            magnitudes = self._coerce_magnitude_array(
                data,
                expected_len=len(self._weights),
                allow_scalar_broadcast=True,
                label="value",
            )
            self._set_storage(
                Q_(magnitudes, self._quantity.units),
                self._copy_state_ids(),
                self._weights.copy(),
            )
            return

        if _is_bool_like(data):
            raise TypeError("Boolean values are not supported.")

        self._set_storage(
            Q_(np.asarray([data], dtype=float), self._quantity.units),
            None,
            np.asarray([1.0], dtype=float),
        )

    @property
    def state_values(self) -> np.ndarray:
        """Return the raw numpy magnitudes for each stored state."""
        return self._magnitude_array().copy()

    @property
    def state_ids(self) -> list[str] | None:
        """Return the state identifiers, or ``None`` for scalar values."""
        return self._copy_state_ids()

    @property
    def weights(self) -> np.ndarray:
        """Return the normalised state weights."""
        return self._weights.copy()

    @property
    def unit(self):
        """Return the unit in a human-friendly compact representation."""
        return self._format_units(self._quantity.units)

    @unit.setter
    def unit(self, unit_str):
        self._set_storage(
            Q_(self._magnitude_array(), unit_str),
            self._copy_state_ids(),
            self._weights.copy(),
        )

    def to(self, new_unit: str) -> "Value":
        """Return a copy converted to ``new_unit``."""
        return self._from_quantity(
            self._quantity.to(new_unit),
            state_ids=self._copy_state_ids(),
            weights=self._weights.copy(),
        )

    def add_states(
        self,
        state_id: list[str] | tuple[str, ...] | str,
        value,
        weight=None,
    ) -> None:
        """Append one or more states and renormalise weights.

        Parameters
        ----------
        state_id:
            One state identifier or a sequence of identifiers to append.
        value:
            One scalar or an array-like of magnitudes aligned to ``state_id``.
            Scalars broadcast across multiple state identifiers.
        weight:
            Optional final probability mass for the new state(s). Scalars
            broadcast across multiple state identifiers. When omitted, the new
            states receive uniform weight across the full combined state set
            while preserving the relative ratios between existing states.
        """
        if not self._is_stateful():
            raise TypeError("add_states requires an existing stateful Value.")

        new_state_ids = self._normalise_state_ids(
            [state_id] if isinstance(state_id, str) else state_id
        )
        if new_state_ids is None:
            raise TypeError("state_id is required.")

        existing_state_ids = self._copy_state_ids()
        overlapping_state_ids = sorted(set(existing_state_ids) & set(new_state_ids))
        if overlapping_state_ids:
            raise ValueError(
                f"Duplicate state_ids are not allowed: {overlapping_state_ids}."
            )

        new_values = self._coerce_magnitude_array(
            value,
            expected_len=len(new_state_ids),
            label="value",
            allow_scalar_broadcast=True,
        )
        new_weights = self._prepare_added_state_weights(len(new_state_ids), weight)

        combined_state_ids = existing_state_ids + new_state_ids
        combined_values = np.concatenate([self._magnitude_array(), new_values])
        remaining_mass = max(0.0, 1.0 - float(new_weights.sum()))
        combined_weights = np.concatenate([self._weights * remaining_mass, new_weights])

        self._set_storage(
            Q_(combined_values, self._quantity.units),
            combined_state_ids,
            combined_weights,
        )

    def __str__(self):
        if not self._is_stateful():
            return f"{self.value} {self.unit}"
        return (
            f"{self.value} {self.unit} "
            f"(states={self.state_ids}, values={self.state_values.tolist()}, "
            f"weights={self.weights.tolist()})"
        )

    def __repr__(self):
        if not self._is_stateful():
            return f"Value({self.value}, {repr(self.unit)})"
        return (
            "Value("
            f"values={self.state_values.tolist()}, "
            f"state_ids={self.state_ids!r}, "
            f"weights={self.weights.tolist()}, "
            f"unit={self.unit!r})"
        )

    def __float__(self):
        return float(self.weighted_value)

    def __int__(self):
        return int(self.weighted_value)

    def __round__(self, ndigits=None):
        return round(self.weighted_value, ndigits)

    def __eq__(self, other):
        try:
            if self._is_numeric_scalar(other):
                return self._weighted_magnitude() == other
            return self._weighted_quantity() == self._to_quantity(other)
        except Exception:
            return False

    def __lt__(self, other):
        if self._is_numeric_scalar(other):
            return self._weighted_magnitude() < other
        return self._weighted_quantity() < self._to_quantity(other)

    def __le__(self, other):
        if self._is_numeric_scalar(other):
            return self._weighted_magnitude() <= other
        return self._weighted_quantity() <= self._to_quantity(other)

    def __gt__(self, other):
        if self._is_numeric_scalar(other):
            return self._weighted_magnitude() > other
        return self._weighted_quantity() > self._to_quantity(other)

    def __ge__(self, other):
        if self._is_numeric_scalar(other):
            return self._weighted_magnitude() >= other
        return self._weighted_quantity() >= self._to_quantity(other)

    def __add__(self, other):
        return self._binary_operation(other, lambda left, right: left + right)

    def __radd__(self, other):
        return self._binary_operation(
            other, lambda left, right: left + right, reverse=True
        )

    def __sub__(self, other):
        return self._binary_operation(other, lambda left, right: left - right)

    def __rsub__(self, other):
        return self._binary_operation(
            other, lambda left, right: left - right, reverse=True
        )

    def __mul__(self, other):
        return self._binary_operation(other, lambda left, right: left * right)

    def __rmul__(self, other):
        return self._binary_operation(
            other, lambda left, right: left * right, reverse=True
        )

    def __truediv__(self, other):
        return self._binary_operation(other, lambda left, right: left / right)

    def __rtruediv__(self, other):
        return self._binary_operation(
            other, lambda left, right: left / right, reverse=True
        )

    def _to_quantity(self, other):
        """Return the weighted scalar quantity for comparisons."""
        if isinstance(other, Value):
            return other._weighted_quantity()
        return Q_(other)

    def _from_quantity(
        self,
        qty,
        *,
        state_ids: list[str] | None = None,
        weights: np.ndarray | None = None,
    ):
        """Build a new ``Value`` instance from a Pint quantity and state metadata."""
        instance = type(self).__new__(type(self))
        instance._set_storage(qty, state_ids, weights)
        return instance

    def _binary_operation(self, other, operator, reverse: bool = False):
        left, right = (other, self) if reverse else (self, other)
        left_qty, right_qty, state_ids, weights = self._align_operands(left, right)
        result = operator(left_qty, right_qty)
        return self._from_quantity(result, state_ids=state_ids, weights=weights)

    def _align_operands(self, left, right):
        left_value = left if isinstance(left, Value) else None
        right_value = right if isinstance(right, Value) else None

        left_qty = left_value._quantity if left_value is not None else Q_(left)
        right_qty = right_value._quantity if right_value is not None else Q_(right)

        if left_value is not None and right_value is not None:
            if left_value._is_stateful() and right_value._is_stateful():
                left_value._validate_compatible_states(right_value)
                return (
                    left_qty,
                    right_qty,
                    left_value._copy_state_ids(),
                    left_value._weights.copy(),
                )
            if left_value._is_stateful():
                return (
                    left_qty,
                    right_qty,
                    left_value._copy_state_ids(),
                    left_value._weights.copy(),
                )
            if right_value._is_stateful():
                return (
                    left_qty,
                    right_qty,
                    right_value._copy_state_ids(),
                    right_value._weights.copy(),
                )
            return left_qty, right_qty, None, np.asarray([1.0], dtype=float)

        if left_value is not None and left_value._is_stateful():
            return (
                left_qty,
                right_qty,
                left_value._copy_state_ids(),
                left_value._weights.copy(),
            )

        if right_value is not None and right_value._is_stateful():
            return (
                left_qty,
                right_qty,
                right_value._copy_state_ids(),
                right_value._weights.copy(),
            )

        return left_qty, right_qty, None, np.asarray([1.0], dtype=float)

    def _validate_compatible_states(self, other: "Value") -> None:
        if self._state_ids != other._state_ids:
            raise ValueError("Stateful arithmetic requires identical state_ids order.")
        if not np.allclose(self._weights, other._weights, rtol=1e-12, atol=1e-12):
            raise ValueError(
                "Stateful arithmetic requires identical normalised weights."
            )

    def _weighted_quantity(self):
        return Q_(self._weighted_magnitude(), self._quantity.units)

    def _weighted_magnitude(self) -> float:
        magnitudes = self._magnitude_array()
        return float(np.dot(self._weights, magnitudes))

    def _magnitude_array(self) -> np.ndarray:
        return np.asarray(self._quantity.magnitude, dtype=float).reshape(-1)

    def _is_stateful(self) -> bool:
        return self._state_ids is not None

    def _copy_state_ids(self) -> list[str] | None:
        return None if self._state_ids is None else list(self._state_ids)

    def _set_storage(self, quantity, state_ids, weights) -> None:
        magnitudes = np.asarray(quantity.magnitude, dtype=float).reshape(-1)
        state_ids = self._normalise_state_ids(state_ids)
        weights_array = self._normalise_weights(weights, len(magnitudes))

        if state_ids is None:
            if len(magnitudes) != 1:
                raise ValueError("Scalar values must contain exactly one magnitude.")
        elif len(state_ids) != len(magnitudes):
            raise ValueError(
                "state_ids length must match the number of stored magnitudes."
            )

        self._quantity = Q_(magnitudes, quantity.units)
        self._state_ids = state_ids
        self._weights = weights_array

    def _init_scalar(
        self, data, unit: str | None, preserve_none_unit: bool = False
    ) -> None:
        if _is_bool_like(data):
            raise TypeError("Boolean values are not supported.")

        if data is None:
            data = 0.0

        if preserve_none_unit and unit is None:
            quantity = Q_(np.asarray([0.0], dtype=float))
        else:
            quantity = (
                Q_(np.asarray([data], dtype=float), unit)
                if unit
                else Q_(np.asarray([data], dtype=float))
            )
        self._set_storage(quantity, None, np.asarray([1.0], dtype=float))

    def _init_from_value_with_unit(self, data, unit: str | None) -> None:
        quantity = Q_(data.value, data.units)
        if unit is not None:
            try:
                quantity.to(unit)
            except Exception:
                pass
        self._set_storage(
            Q_(np.asarray([quantity.magnitude], dtype=float), quantity.units),
            None,
            np.asarray([1.0], dtype=float),
        )

    def _init_stateful(
        self,
        data,
        *,
        unit: str | None,
        weights: Mapping[str, float] | list[float] | np.ndarray | None,
        state_ids: list[str] | tuple[str, ...] | None,
    ) -> None:
        if isinstance(data, Mapping):
            state_map = self._normalise_state_map(data)
            ordered_state_ids = self._state_ids_from_mapping(state_map, state_ids)
            magnitudes = self._coerce_magnitude_array(
                [state_map[state_id] for state_id in ordered_state_ids],
                expected_len=len(ordered_state_ids),
                label="values",
            )
        else:
            if state_ids is None:
                raise TypeError(
                    "state_ids are required for array-backed stateful values."
                )
            ordered_state_ids = self._normalise_state_ids(state_ids)
            if ordered_state_ids is None:
                raise TypeError(
                    "state_ids are required for array-backed stateful values."
                )
            magnitudes = self._coerce_magnitude_array(
                data,
                expected_len=len(ordered_state_ids),
                label="values",
            )

        quantity = Q_(magnitudes, unit) if unit else Q_(magnitudes)
        self._set_storage(
            quantity,
            ordered_state_ids,
            self._coerce_weights(weights, ordered_state_ids),
        )

    def _coerce_weights(self, weights, state_ids: list[str]) -> np.ndarray:
        if weights is None:
            return np.ones(len(state_ids), dtype=float) / len(state_ids)

        if isinstance(weights, Mapping):
            weight_map = self._normalise_state_map(weights)
            if set(weight_map) != set(state_ids):
                raise ValueError("Weight keys must match the provided state_ids.")
            raw_weights = [weight_map[state_id] for state_id in state_ids]
        else:
            raw_weights = weights

        return self._normalise_weights(raw_weights, len(state_ids))

    def _coerce_magnitude_array(
        self,
        data,
        *,
        expected_len: int,
        label: str,
        allow_scalar_broadcast: bool = False,
    ) -> np.ndarray:
        if _is_bool_like(data):
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
        if any(_is_bool_like(value) for value in values):
            raise TypeError("Boolean values are not supported.")

        try:
            magnitudes = np.asarray(values, dtype=float).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{label} must contain numeric values.") from exc

        if len(magnitudes) != expected_len:
            raise ValueError(f"{label} length must match the number of states.")

        return magnitudes

    def _normalise_weights(self, weights, expected_len: int) -> np.ndarray:
        if weights is None:
            weights_array = np.ones(expected_len, dtype=float)
        else:
            if _is_bool_like(weights):
                raise TypeError("Boolean weights are not supported.")
            try:
                weight_values = list(weights)
            except TypeError as exc:
                raise TypeError("weights must be 1-D array-like data.") from exc
            if any(_is_bool_like(weight) for weight in weight_values):
                raise TypeError("Boolean weights are not supported.")
            weights_array = np.asarray(weight_values, dtype=float).reshape(-1)

        if len(weights_array) != expected_len:
            raise ValueError("weights length must match the number of states.")
        if len(weights_array) == 0:
            raise ValueError("Stateful values cannot be empty.")
        if not np.all(np.isfinite(weights_array)):
            raise ValueError("weights must be finite.")
        if np.any(weights_array < 0.0):
            raise ValueError("weights must be non-negative.")

        total_weight = float(weights_array.sum())
        if total_weight <= 0.0:
            raise ValueError("weights must sum to a positive value.")

        return weights_array / total_weight

    def _prepare_added_state_weights(self, num_new_states: int, weight) -> np.ndarray:
        """Return final probability masses for newly appended states."""
        if num_new_states <= 0:
            raise ValueError("At least one new state is required.")

        num_existing_states = len(self._weights)
        if weight is None:
            total_new_mass = num_new_states / (num_existing_states + num_new_states)
            return np.full(num_new_states, total_new_mass / num_new_states, dtype=float)

        if _is_bool_like(weight):
            raise TypeError("Boolean weights are not supported.")

        if np.isscalar(weight) and not isinstance(weight, (str, bytes)):
            raw_weights = np.full(num_new_states, float(weight), dtype=float)
        else:
            try:
                raw_weight_values = list(weight)
            except TypeError as exc:
                raise TypeError(
                    "weight must be a scalar or 1-D array-like data."
                ) from exc
            if any(_is_bool_like(item) for item in raw_weight_values):
                raise TypeError("Boolean weights are not supported.")
            raw_weights = np.asarray(raw_weight_values, dtype=float).reshape(-1)

        if len(raw_weights) != num_new_states:
            raise ValueError("weight length must match the number of added states.")
        if not np.all(np.isfinite(raw_weights)):
            raise ValueError("weight must be finite.")
        if np.any(raw_weights < 0.0):
            raise ValueError("weight must be non-negative.")

        total_new_mass = float(raw_weights.sum())
        if total_new_mass > 1.0 + 1e-12:
            raise ValueError("Added state weights must sum to 1.0 or less.")

        return raw_weights

    def _state_ids_from_mapping(
        self,
        data: Mapping[str, Any],
        state_ids: list[str] | tuple[str, ...] | None,
    ) -> list[str]:
        if len(data) == 0:
            raise ValueError("Stateful values cannot be empty.")
        if state_ids is None:
            return list(data.keys())

        normalised_state_ids = self._normalise_state_ids(state_ids)
        if normalised_state_ids is None:
            raise TypeError("state_ids are required for array-backed stateful values.")
        if set(normalised_state_ids) != set(data):
            raise ValueError("state_ids must match the provided state-value keys.")
        return normalised_state_ids

    def _normalise_state_map(self, data: Mapping[Any, Any]) -> dict[str, Any]:
        state_map = {str(key): value for key, value in data.items()}
        if len(state_map) != len(data):
            raise ValueError("Duplicate state_ids are not allowed.")
        return state_map

    def _normalise_state_ids(
        self, state_ids: list[str] | tuple[str, ...] | None
    ) -> list[str] | None:
        if state_ids is None:
            return None
        if isinstance(state_ids, (str, bytes)):
            raise TypeError("state_ids must be a sequence of identifiers.")

        normalised_state_ids = [str(state_id) for state_id in state_ids]
        if len(normalised_state_ids) == 0:
            raise ValueError("state_ids cannot be empty.")
        if len(set(normalised_state_ids)) != len(normalised_state_ids):
            raise ValueError("Duplicate state_ids are not allowed.")
        return normalised_state_ids

    @staticmethod
    def _is_numeric_scalar(other: Any) -> bool:
        return isinstance(
            other, (int, float, np.integer, np.floating)
        ) and not _is_bool_like(other)

    @staticmethod
    def _is_array_like_input(data: Any) -> bool:
        if data is None or isinstance(data, (str, bytes, Mapping)):
            return False
        if _is_bool_like(data) or np.isscalar(data):
            return False
        try:
            list(data)
        except TypeError:
            return False
        return True

    @staticmethod
    def _is_serialized_scalar_payload(data: Mapping[Any, Any]) -> bool:
        return set(data).issubset(_SERIALIZED_SCALAR_KEYS) and "value" in data

    @staticmethod
    def _is_serialized_stateful_payload(data: Mapping[Any, Any]) -> bool:
        return set(data).issubset(_SERIALIZED_STATEFUL_KEYS) and (
            "values" in data or "state_ids" in data or "weights" in data
        )

    def _format_units(self, units) -> str:
        return format(units, "~").replace("°", "deg").replace(" ", "")

    def to_dict(self):
        """Serialise the value into a JSON-friendly dictionary."""
        if self._is_stateful():
            return {
                "values": self.state_values.tolist(),
                "state_ids": self.state_ids,
                "weights": self.weights.tolist(),
                "unit": self.unit,
            }
        return {"value": self.weighted_value, "unit": self.unit}

    @classmethod
    def from_dict(cls, data):
        """Instantiate from a scalar or stateful serialized mapping."""
        if not isinstance(data, Mapping):
            raise TypeError("data must be a mapping.")

        if cls._is_serialized_stateful_payload(data):
            return cls(
                data["values"],
                data.get("unit") or data.get("units"),
                weights=data.get("weights"),
                state_ids=data.get("state_ids"),
            )

        return cls(data["value"], data.get("unit") or data.get("units"))
