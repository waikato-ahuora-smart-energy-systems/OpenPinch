"""Unit-aware scalar and discrete-state value wrapper powered by Pint quantities."""

from __future__ import annotations

import re
from collections.abc import Mapping
from functools import lru_cache
from typing import Any

import numpy as np
from pint import UnitRegistry, set_application_registry
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
set_application_registry(ureg)
Q_ = ureg.Quantity  # type: ignore

_SERIALIZED_SCALAR_KEYS = {"value", "unit"}
_SERIALIZED_STATEFUL_KEYS = {"values", "state_ids", "weights", "unit"}


@lru_cache(maxsize=256)
def _normalise_unit_text(unit: str | None) -> str | None:
    if unit is None:
        return None
    text = str(unit).strip().replace("$", "USD")
    if text in {"", "-", "dimensionless", "1"}:
        return "dimensionless"
    if text == "fraction":
        return "dimensionless"
    if text in {"USD/y", "USD/yr", "USD/year"}:
        return "USD/year"
    if text in {"C", "°C"}:
        return "degC"
    if text == "degK":
        return "K"
    text = re.sub(r"(?<=[A-Za-z])2(?=($|[./*]))", "^2", text)
    text = re.sub(r"(?<=[A-Za-z])3(?=($|[./*]))", "^3", text)
    text = text.replace(".K", "/K").replace(".degC", "/degC")
    return text


@lru_cache(maxsize=256)
def _unit_object(unit: str):
    return ureg.Unit(unit)


def _is_value_with_unit(data: Any) -> bool:
    """Return ``True`` for objects that look like ``ValueWithUnit`` instances."""
    return hasattr(data, "value") and hasattr(data, "unit")


def _is_bool_like(data: Any) -> bool:
    """Return ``True`` for bool scalars including numpy bools."""
    return isinstance(data, (bool, np.bool_))


class Value:
    """Thin wrapper around a Pint ``Quantity`` with serialization helpers."""

    def __init__(self, data=None, unit: str = None):
        """Create a scalar or stateful value from ``data`` and an optional ``unit``."""
        quantity, weights = self._coerce_input(data, unit)
        self._set_storage(quantity)
        self._weights = weights

    @property
    def value(self):
        """Return the scalar magnitude or per-state magnitudes for stateful values."""
        if not self._is_stateful():
            return self._quantity.magnitude[0]
        return self._quantity.magnitude.copy()

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
        else:
            if _is_bool_like(data):
                raise TypeError("Boolean values are not supported.")
            magnitudes = np.asarray([data], dtype=float)
        self._set_storage(Q_(magnitudes, self._quantity.units))

    @property
    def min(self) -> "Value":
        """Return the minimum stored magnitude as a scalar ``Value``."""
        return self._summary_value(np.min(self._quantity.magnitude))

    @property
    def max(self) -> "Value":
        """Return the maximum stored magnitude as a scalar ``Value``."""
        return self._summary_value(np.max(self._quantity.magnitude))

    @property
    def mean(self) -> "Value":
        """Return the arithmetic mean stored magnitude as a scalar ``Value``."""
        return self._summary_value(np.mean(self._quantity.magnitude))

    @property
    def weighted_mean(self) -> "Value":
        """Return the weighted mean stored magnitude as a scalar ``Value``."""
        return self._summary_value(
            np.average(self._quantity.magnitude, weights=self.weights)
        )

    @property
    def median(self) -> "Value":
        """Return the median stored magnitude as a scalar ``Value``."""
        return self._summary_value(np.median(self._quantity.magnitude))

    @property
    def state_values(self) -> np.ndarray:
        """Return the raw numpy magnitudes for each stored state."""
        return self._quantity.magnitude.copy()

    @property
    def values(self) -> list[float]:
        """Return magnitudes as a JSON-friendly list for stateful compatibility."""
        return [float(item) for item in self.state_values.tolist()]

    @property
    def weights(self) -> np.ndarray:
        """Return optional passive state weights carried with this value."""
        return self._weights

    @property
    def num_states(self) -> int:
        """Return the number of stored magnitudes."""
        return len(self._quantity.magnitude)

    @property
    def unit(self):
        """Return the unit in a human-friendly compact representation."""
        return self._format_units(self._quantity.units)

    @unit.setter
    def unit(self, unit_str):
        """Convert the stored quantity to ``unit_str`` in-place."""
        self._set_storage(self._quantity.to(self._normalise_unit_input(unit_str)))

    def to(self, new_unit: str) -> "Value":
        """Return a copy converted to ``new_unit``."""
        new_value = self._from_quantity(
            self._quantity.to(self._normalise_unit_input(new_unit)),
        )
        new_value._weights = self._weights.copy() if self._weights is not None else None
        return new_value

    def __getitem__(self, idx):
        """Return one selected state as an independent ``Value``."""
        if idx is None or not self._is_stateful():
            return Value(self)

        if isinstance(idx, slice):
            subset = self._quantity.magnitude[idx]
            result = Value(subset, unit=self.unit)
            if self._weights is not None:
                result._weights = np.asarray(self._weights[idx], dtype=float).reshape(
                    -1
                )
            return result

        resolved_idx = self._resolve_state_index(idx)
        return self._from_quantity(
            Q_(
                np.asarray([self._quantity.magnitude[resolved_idx]], dtype=float),
                self._quantity.units,
            )
        )

    def __iter__(self):
        if not self._is_stateful():
            raise TypeError("Scalar Value is not iterable.")
        return iter(self._quantity.magnitude)

    def __setitem__(self, idx, value):
        resolved_idx = self._resolve_state_index(idx)
        values = self._quantity.magnitude.copy()
        values[resolved_idx] = self._coerce_scalar_magnitude(value)
        self._set_storage(Q_(values, self._quantity.units))

    def __len__(self):
        """Return the number of states stored."""
        return len(self._quantity.magnitude)

    def __str__(self):
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
        if self._is_stateful():
            raise TypeError("Cannot convert stateful Value to float.")
        return float(self._quantity.magnitude[0])

    def __int__(self):
        if self._is_stateful():
            raise TypeError("Cannot convert stateful Value to int.")
        return int(self._quantity.magnitude[0])

    def __round__(self, ndigits=None):
        if self._is_stateful():
            raise TypeError("Cannot round stateful Value.")
        return round(self._quantity.magnitude[0], ndigits)

    def __array__(self, dtype=None, copy=None):
        if self._is_stateful():
            array = np.asarray(self.state_values, dtype=dtype)
        else:
            array = np.asarray(float(self), dtype=dtype)
        if copy is True:
            return array.copy()
        return array

    def __format__(self, format_spec):
        return format(float(self), format_spec)

    def __abs__(self):
        return abs(float(self))

    def __neg__(self):
        return -float(self)

    def __pos__(self):
        return +float(self)

    def __eq__(self, other):
        try:
            if self._is_numeric_scalar(other):
                return np.all(self._quantity.magnitude == other)
            if isinstance(other, Value):
                return np.all(self.to(other.unit).value == other.value)
            return False
        except DimensionalityError, TypeError, ValueError:
            return False

    def __lt__(self, other):
        try:
            if self._is_numeric_scalar(other):
                return np.all(self._quantity.magnitude < other)
            if isinstance(other, Value):
                return np.all(self.to(other.unit).value < other.value)
            return False
        except DimensionalityError, TypeError, ValueError:
            return False

    def __le__(self, other):
        try:
            if self._is_numeric_scalar(other):
                return np.all(self._quantity.magnitude <= other)
            if isinstance(other, Value):
                return np.all(self.to(other.unit).value <= other.value)
            return False
        except DimensionalityError, TypeError, ValueError:
            return False

    def __gt__(self, other):
        try:
            if self._is_numeric_scalar(other):
                return np.all(self._quantity.magnitude > other)
            if isinstance(other, Value):
                return np.all(self.to(other.unit).value > other.value)
            return False
        except DimensionalityError, TypeError, ValueError:
            return False

    def __ge__(self, other):
        try:
            if self._is_numeric_scalar(other):
                return np.all(self._quantity.magnitude >= other)
            if isinstance(other, Value):
                return np.all(self.to(other.unit).value >= other.value)
            return False
        except DimensionalityError, TypeError, ValueError:
            return False

    def __add__(self, other):
        if self._is_numeric_scalar(other):
            return self._from_quantity(self._quantity + Q_(other, self._quantity.units))
        return self._binary_operation(other, lambda left, right: left + right)

    def __radd__(self, other):
        if self._is_numeric_scalar(other):
            return self._from_quantity(Q_(other, self._quantity.units) + self._quantity)
        return self._binary_operation(
            other, lambda left, right: left + right, reverse=True
        )

    def __sub__(self, other):
        if self._is_numeric_scalar(other):
            return self._from_quantity(self._quantity - Q_(other, self._quantity.units))
        return self._binary_operation(other, lambda left, right: left - right)

    def __rsub__(self, other):
        if self._is_numeric_scalar(other):
            return self._from_quantity(Q_(other, self._quantity.units) - self._quantity)
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

    def _from_quantity(self, qty):
        """Build a new ``Value`` instance from a Pint quantity."""
        instance = type(self).__new__(type(self))
        instance._set_storage(qty)
        instance._weights = self.weights
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
                if len(left_value._quantity.magnitude) != len(
                    right_value._quantity.magnitude
                ):
                    raise ValueError(
                        "Stateful arithmetic requires identical state counts."
                    )
        return left_qty, right_qty

    def _resolve_state_index(self, idx: int | str | None) -> int:
        if not self._is_stateful():
            return 0
        if idx is None:
            return 0
        if isinstance(idx, str):
            try:
                return int(idx)
            except (TypeError, ValueError) as exc:
                raise KeyError(idx) from exc
        idx = int(idx)
        if idx < 0 or idx >= self.num_states:
            raise IndexError(idx)
        if idx >= self.num_states:
            idx = 0
        return idx

    def _is_stateful(self) -> bool:
        return len(self._quantity.magnitude) > 1

    def _set_storage(self, quantity) -> None:
        magnitudes = np.array(quantity.magnitude, dtype=float, copy=True).reshape(-1)
        if magnitudes.size == 0:
            raise ValueError("Values cannot be empty.")
        self._quantity = Q_(magnitudes, quantity.units)

    def _summary_value(self, magnitude: float) -> "Value":
        return self._from_quantity(
            Q_(np.asarray([float(magnitude)], dtype=float), self._quantity.units)
        )

    def _coerce_scalar_magnitude(self, value) -> float:
        if isinstance(value, Value):
            return float(value.to(self.unit).value)
        if hasattr(value, "units"):
            quantity = Q_(value).to(self._quantity.units)
            return float(np.asarray(quantity.magnitude, dtype=float).reshape(-1)[0])
        return float(value)

    def _coerce_input(
        self,
        data,
        unit: str | None,
    ) -> tuple[Any, np.ndarray]:
        payload = self._normalise_input_object(data)

        if isinstance(payload, Value):
            quantity = payload._quantity
            weights = payload.weights
        elif isinstance(payload, Mapping):
            quantity, weights = self._coerce_mapping_input(payload, unit)
            return quantity, weights
        elif hasattr(payload, "units"):
            quantity = payload
            weights = None
        elif self._is_array_like_input(payload):
            quantity = self._quantity_from_values(payload, unit)
            weights = None
        elif _is_value_with_unit(payload):
            quantity, weights = self._coerce_object_with_unit(payload, unit)
            return quantity, weights
        elif payload is None:
            quantity = self._quantity_from_scalar(0.0, unit)
            weights = None
        else:
            quantity = self._quantity_from_scalar(payload, unit)
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
        if self._is_serialized_stateful_payload(data):
            quantity = self._quantity_from_values(
                data.get("values"),
                data.get("unit") or unit,
            )
            weights = data.get("weights")
        elif self._is_serialized_scalar_payload(data):
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
        if _is_bool_like(data):
            raise TypeError("Boolean values are not supported.")
        resolved_unit = self._normalise_unit_input(unit)
        magnitude = np.asarray([data], dtype=float)
        return (
            Q_(magnitude, self._unit_from_normalised(resolved_unit))
            if resolved_unit
            else Q_(magnitude)
        )

    def _missing_or_zero_quantity(self, data, unit: str | None):
        resolved_unit = self._normalise_unit_input(unit)
        magnitude = np.asarray([np.nan if data is None else data], dtype=float)
        return (
            Q_(magnitude, self._unit_from_normalised(resolved_unit))
            if resolved_unit
            else Q_(magnitude)
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
            Q_(magnitudes, self._unit_from_normalised(resolved_unit))
            if resolved_unit
            else Q_(magnitudes)
        )

    def _coerce_quantity_to_unit(self, quantity, unit: str | None):
        copied = Q_(
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
                return Q_(
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
        return set(data).issubset(_SERIALIZED_STATEFUL_KEYS) and ("values" in data)

    def _format_units(self, units) -> str:
        return self._clean_unit_text(format(units, "~"))

    @staticmethod
    def _serialise_units(units) -> str:
        return Value._clean_unit_text(format(units, "~"))

    @staticmethod
    def _clean_unit_text(text: str) -> str:
        text = text.replace("USD", "$").replace("NZD", "$").replace(" ", "")
        text = text.replace("°", "deg")
        text = text.replace("ΔdegC", "delta_degC").replace("Δ°C", "delta_degC")
        text = text.replace("**2", "^2").replace("**3", "^3")
        text = text.replace("$/a", "$/y").replace("$/year", "$/y")
        return "-" if text == "" else text

    @staticmethod
    def _normalise_unit_input(unit: str | None) -> str | None:
        return _normalise_unit_text(unit)

    @staticmethod
    def _unit_from_normalised(unit: str):
        return _unit_object(unit)

    @staticmethod
    def _quantity_is_dimensionless(quantity) -> bool:
        return str(quantity.units) == "dimensionless"

    @staticmethod
    def _same_dimensionality(quantity, unit: str) -> bool:
        try:
            return quantity.dimensionality == Q_(1.0, _unit_object(unit)).dimensionality
        except Exception:
            return False

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
            return cls(data)
        return cls(data)
