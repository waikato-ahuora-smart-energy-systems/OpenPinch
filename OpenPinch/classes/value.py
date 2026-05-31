"""Unit-aware scalar and discrete-state value wrapper powered by Pint quantities."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import numpy as np
from pint import UnitRegistry
from pint.errors import DimensionalityError

ureg = UnitRegistry()
try:
    ureg.define("USD = [currency]")
except Exception:
    pass
try:
    ureg.define("NZD = [currency]")
except Exception:
    pass
Q_ = ureg.Quantity

_SERIALIZED_SCALAR_KEYS = {"value", "unit"}
_SERIALIZED_STATEFUL_KEYS = {"values", "state_ids", "weights", "unit"}


def _is_value_with_unit(data: Any) -> bool:
    """Return ``True`` for objects that look like ``ValueWithUnit`` instances."""
    return hasattr(data, "value") and hasattr(data, "unit")


def _is_bool_like(data: Any) -> bool:
    """Return ``True`` for bool scalars including numpy bools."""
    return isinstance(data, (bool, np.bool_))


class Value:
    """Thin wrapper around a Pint ``Quantity`` with serialization helpers."""

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
        if weights is not None or state_id is not None or state_ids is not None:
            raise TypeError("State metadata belongs on StreamCollection, not Value.")
        if values is not None:
            data = values

        if hasattr(data, "model_dump") and not isinstance(data, Mapping):
            data = data.model_dump(mode="python")

        if isinstance(data, Value):
            self._init_from_value(data, unit)
            return

        if isinstance(data, Mapping):
            if self._is_serialized_stateful_payload(data):
                self._init_stateful(
                    data["values"],
                    unit=data.get("unit") or unit,
                )
                return
            if self._is_serialized_scalar_payload(data):
                if data.get("value") is None:
                    self._init_missing_scalar(data.get("unit") or unit)
                    return
                self._init_scalar(
                    data.get("value"),
                    data.get("unit") or unit,
                    preserve_none_unit=(data.get("value") is None),
                )
                return
            self._init_stateful(data, unit=unit)
            return

        if _is_value_with_unit(data):
            self._init_from_value_with_unit(data, unit)
            return

        if self._is_array_like_input(data):
            self._init_stateful(data, unit=unit)
            return

        if data is None:
            self._init_scalar(0.0, None, preserve_none_unit=True)
            return

        self._init_scalar(data, unit)

    @property
    def value(self):
        """Return the scalar magnitude or per-state magnitudes for stateful values."""
        if self._is_stateful():
            return self.state_values
        return self._mean_magnitude()

    @property
    def mean_value(self) -> float:
        """Return the mean scalar view of the stored quantity."""
        return self._mean_magnitude()

    def min_value(self) -> "Value":
        """Return the minimum stored magnitude as a scalar ``Value``."""
        min_magnitude = float(np.min(self._magnitude_array()))
        return self._from_quantity(
            Q_(np.asarray([min_magnitude], dtype=float), self._quantity.units),
        )

    def max_value(self) -> "Value":
        """Return the maximum stored magnitude as a scalar ``Value``."""
        max_magnitude = float(np.max(self._magnitude_array()))
        return self._from_quantity(
            Q_(np.asarray([max_magnitude], dtype=float), self._quantity.units),
        )

    @value.setter
    def value(self, data):
        """Set the scalar magnitude or per-state magnitudes in-place."""
        if self._is_stateful():
            magnitudes = self._coerce_magnitude_array(
                data,
                expected_len=len(self.state_values),
                allow_scalar_broadcast=True,
                label="value",
            )
            self._set_storage(Q_(magnitudes, self._quantity.units))
            return

        if _is_bool_like(data):
            raise TypeError("Boolean values are not supported.")

        self._set_storage(Q_(np.asarray([data], dtype=float), self._quantity.units))

    @property
    def state_values(self) -> np.ndarray:
        """Return the raw numpy magnitudes for each stored state."""
        return self._magnitude_array().copy()

    @property
    def unit(self):
        """Return the unit in a human-friendly compact representation."""
        return self._format_units(self._quantity.units)

    @unit.setter
    def unit(self, unit_str):
        """Convert the stored quantity to ``unit_str`` in-place."""
        self._set_storage(
            Q_(self._magnitude_array(), self._normalise_unit_input(unit_str)),
        )

    def to(self, new_unit: str) -> "Value":
        """Return a copy converted to ``new_unit``."""
        return self._from_quantity(
            self._quantity.to(self._normalise_unit_input(new_unit)),
        )

    def add_states(self, state_id, value, weight=None) -> None:
        """Append one or more states to the stored magnitude array."""
        if not self._is_stateful():
            raise TypeError("add_states requires an existing stateful Value.")
        del state_id, weight

        new_values = self._coerce_magnitude_array(
            value,
            expected_len=1
            if np.isscalar(value) and not isinstance(value, (str, bytes))
            else len(list(value)),
            label="value",
            allow_scalar_broadcast=True,
        )
        combined_values = np.concatenate([self._magnitude_array(), new_values])
        self._set_storage(Q_(combined_values, self._quantity.units))

    def __getitem__(self, state_id):
        """Return a scalar ``Value`` for one stored state."""
        if not self._is_stateful():
            return self._from_quantity(self._quantity)

        idx = self._resolve_state_index(state_id)

        return self._from_quantity(
            Q_(
                np.asarray([self._magnitude_array()[idx]], dtype=float),
                self._quantity.units,
            )
        )

    def __str__(self):
        if not self._is_stateful():
            return f"{self.value} {self.unit}"
        return f"{self.value} {self.unit}"

    def __repr__(self):
        if not self._is_stateful():
            return (
                f"Value({self.value}, "
                f"{repr(self._serialise_units(self._quantity.units))})"
            )
        return (
            "Value("
            f"values={self.state_values.tolist()}, "
            f"unit={self._serialise_units(self._quantity.units)!r})"
        )

    def __float__(self):
        return float(self.mean_value)

    def __int__(self):
        return int(self.mean_value)

    def __round__(self, ndigits=None):
        return round(self.mean_value, ndigits)

    def __eq__(self, other):
        try:
            if self._is_numeric_scalar(other):
                return self._mean_magnitude() == other
            return self._mean_quantity() == self._to_quantity(other)
        except DimensionalityError, TypeError, ValueError:
            return False

    def __lt__(self, other):
        if self._is_numeric_scalar(other):
            return self._mean_magnitude() < other
        return self._mean_quantity() < self._to_quantity(other)

    def __le__(self, other):
        if self._is_numeric_scalar(other):
            return self._mean_magnitude() <= other
        return self._mean_quantity() <= self._to_quantity(other)

    def __gt__(self, other):
        if self._is_numeric_scalar(other):
            return self._mean_magnitude() > other
        return self._mean_quantity() > self._to_quantity(other)

    def __ge__(self, other):
        if self._is_numeric_scalar(other):
            return self._mean_magnitude() >= other
        return self._mean_quantity() >= self._to_quantity(other)

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
        """Return the scalar quantity used for comparisons."""
        if isinstance(other, Value):
            return other._mean_quantity()
        return Q_(other)

    def _from_quantity(self, qty):
        """Build a new ``Value`` instance from a Pint quantity."""
        instance = type(self).__new__(type(self))
        instance._set_storage(qty)
        return instance

    def _binary_operation(self, other, operator, reverse: bool = False):
        left, right = (other, self) if reverse else (self, other)
        left_qty, right_qty = self._align_operands(left, right)
        result = operator(left_qty, right_qty)
        return self._from_quantity(result)

    def _align_operands(self, left, right):
        left_value = left if isinstance(left, Value) else None
        right_value = right if isinstance(right, Value) else None

        left_qty = left_value._quantity if left_value is not None else Q_(left)
        right_qty = right_value._quantity if right_value is not None else Q_(right)

        if left_value is not None and right_value is not None:
            if left_value._is_stateful() and right_value._is_stateful():
                left_value._validate_compatible_states(right_value)
                return left_qty, right_qty
            if left_value._is_stateful():
                return left_qty, right_qty
            if right_value._is_stateful():
                return left_qty, right_qty
            return left_qty, right_qty

        if left_value is not None and left_value._is_stateful():
            return left_qty, right_qty

        if right_value is not None and right_value._is_stateful():
            return left_qty, right_qty

        return left_qty, right_qty

    def _validate_compatible_states(self, other: "Value") -> None:
        if len(self._magnitude_array()) != len(other._magnitude_array()):
            raise ValueError("Stateful arithmetic requires identical state counts.")

    def _mean_quantity(self):
        return Q_(self._mean_magnitude(), self._quantity.units)

    def _mean_magnitude(self) -> float:
        return float(np.mean(self._magnitude_array()))

    def _magnitude_array(self) -> np.ndarray:
        return np.asarray(self._quantity.magnitude, dtype=float).reshape(-1)

    def _is_stateful(self) -> bool:
        return len(self._magnitude_array()) > 1

    def _set_storage(self, quantity) -> None:
        magnitudes = np.asarray(quantity.magnitude, dtype=float).reshape(-1)
        if len(magnitudes) == 0:
            raise ValueError("Values cannot be empty.")

        self._quantity = Q_(magnitudes, quantity.units)

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
                Q_(np.asarray([data], dtype=float), self._normalise_unit_input(unit))
                if unit
                else Q_(np.asarray([data], dtype=float))
            )
        self._set_storage(quantity)

    def _init_missing_scalar(self, unit: str | None) -> None:
        quantity = (
            Q_(np.asarray([np.nan], dtype=float), self._normalise_unit_input(unit))
            if unit
            else Q_(np.asarray([np.nan], dtype=float))
        )
        self._set_storage(quantity)

    def _init_from_value(self, data: "Value", unit: str | None) -> None:
        quantity = data._quantity
        if unit is not None:
            try:
                quantity = quantity.to(self._normalise_unit_input(unit))
            except DimensionalityError, TypeError, ValueError:
                if self._quantity_is_dimensionless(quantity):
                    quantity = Q_(
                        data._magnitude_array(), self._normalise_unit_input(unit)
                    )
        self._set_storage(
            Q_(np.asarray(quantity.magnitude, dtype=float).reshape(-1), quantity.units)
        )

    def _init_from_value_with_unit(self, data, unit: str | None) -> None:
        if data.value is None:
            self._init_missing_scalar(data.unit if data.unit is not None else unit)
            return
        quantity = Q_(data.value, self._normalise_unit_input(data.unit))
        if unit is not None:
            try:
                quantity = quantity.to(self._normalise_unit_input(unit))
            except DimensionalityError, TypeError, ValueError:
                if self._quantity_is_dimensionless(quantity):
                    quantity = Q_(
                        np.asarray([quantity.magnitude], dtype=float),
                        self._normalise_unit_input(unit),
                    )
        self._set_storage(
            Q_(np.asarray([quantity.magnitude], dtype=float), quantity.units)
        )

    def _init_stateful(
        self,
        data,
        *,
        unit: str | None,
    ) -> None:
        if isinstance(data, Mapping):
            state_map = self._normalise_state_map(data)
            magnitudes = self._coerce_magnitude_array(
                list(state_map.values()),
                expected_len=len(state_map),
                label="values",
            )
        else:
            values_list = list(data)
            magnitudes = self._coerce_magnitude_array(
                values_list,
                expected_len=len(values_list),
                label="values",
            )

        quantity = (
            Q_(magnitudes, self._normalise_unit_input(unit)) if unit else Q_(magnitudes)
        )
        self._set_storage(quantity)

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

    def _normalise_state_map(self, data: Mapping[Any, Any]) -> dict[str, Any]:
        state_map = {str(key): value for key, value in data.items()}
        if len(state_map) != len(data):
            raise ValueError("Duplicate state_ids are not allowed.")
        return state_map

    @staticmethod
    def _normalise_lookup_state_id(state_id) -> str | None:
        if state_id is None:
            return None
        if _is_bool_like(state_id):
            raise TypeError("state_id must not be boolean.")
        if isinstance(state_id, (np.integer, int)):
            return str(int(state_id))
        return str(state_id)

    def _resolve_state_index(self, state_id) -> int:
        lookup = self._normalise_lookup_state_id(state_id)
        if lookup is None:
            raise KeyError("state_id cannot be None.")
        try:
            idx = int(lookup)
        except (TypeError, ValueError) as exc:
            raise KeyError(f"Unknown state_id {state_id!r}.") from exc
        if idx < 0 or idx >= len(self.state_values):
            raise KeyError(f"Unknown state_id {state_id!r}.")
        return idx

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
        return (
            format(units, "~").replace("USD", "$").replace("NZD", "$").replace(" ", "")
        )

    @staticmethod
    def _serialise_units(units) -> str:
        return (
            format(units, "~")
            .replace("°", "deg")
            .replace("USD", "$")
            .replace("NZD", "$")
            .replace(" ", "")
        )

    @staticmethod
    def _normalise_unit_input(unit: str | None) -> str | None:
        if unit is None:
            return None
        text = str(unit).strip().replace("$", "USD")
        if text in {"C", "°C"}:
            return "degC"
        if text == "degK":
            return "K"
        text = re.sub(r"(?<=[A-Za-z])2(?=($|[./*]))", "^2", text)
        text = re.sub(r"(?<=[A-Za-z])3(?=($|[./*]))", "^3", text)
        text = text.replace(".K", "/K").replace(".degC", "/degC")
        return text

    @staticmethod
    def _quantity_is_dimensionless(quantity) -> bool:
        return str(quantity.units) == "dimensionless"

    def to_dict(self):
        """Serialise the value into a JSON-friendly dictionary."""
        if self._is_stateful():
            return {
                "values": self.state_values.tolist(),
                "unit": self._serialise_units(self._quantity.units),
            }
        if np.isnan(self.value):
            return {
                "value": None,
                "unit": self._serialise_units(self._quantity.units),
            }
        return {
            "value": self.value,
            "unit": self._serialise_units(self._quantity.units),
        }

    @classmethod
    def from_dict(cls, data):
        """Instantiate from a scalar or stateful serialized mapping."""
        if not isinstance(data, Mapping):
            raise TypeError("data must be a mapping.")

        if cls._is_serialized_stateful_payload(data):
            return cls(data["values"], data.get("unit"))

        return cls(data)
