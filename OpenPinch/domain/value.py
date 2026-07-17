"""Unit-aware scalar and multiperiod value wrapper powered by Pint quantities."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from pint import UnitRegistry, set_application_registry
from pint.errors import DimensionalityError

from ._value import coercion as _value_coercion
from ._value import units as _value_units
from ._value.coercion import is_bool_like as _is_bool_like
from ._value.units import unit_object as _registry_unit_object

ureg = UnitRegistry()
try:
    ureg.define("USD = [currency]")
except Exception:  # pragma: no cover - Pint may already know this unit.
    pass
try:
    ureg.define("NZD = [currency]")
except Exception:  # pragma: no cover - Pint may already know this unit.
    pass
set_application_registry(ureg)
Q_ = ureg.Quantity  # type: ignore

_SERIALIZED_SCALAR_KEYS = {"value", "unit"}
_SERIALIZED_PERIOD_KEYS = {"values", "period_ids", "weights", "unit"}


def _unit_object(unit: str):
    return _registry_unit_object(ureg, unit)


class Value:
    """Thin wrapper around a Pint ``Quantity`` with serialization helpers."""

    def __init__(self, data=None, unit: str = None):
        """Create a scalar or multiperiod value from data and optional unit."""
        quantity, weights = self._coerce_input(data, unit)
        self._set_storage(quantity)
        self._weights = weights
        self._read_only_reason: str | None = None

    @property
    def value(self):
        """Return scalar or per-period magnitudes for multiperiod values."""
        if not self._is_period_valued():
            return self._quantity.magnitude[0]
        return self._quantity.magnitude.copy()

    @value.setter
    def value(self, data):
        """Set the scalar magnitude or per-period magnitudes in-place."""
        self._assert_writable()
        if self._is_period_valued():
            magnitudes = self._coerce_magnitude_array(
                data,
                expected_len=len(self.period_values),
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
    def period_values(self) -> np.ndarray:
        """Return the raw numpy magnitudes for each stored period."""
        return self._quantity.magnitude.copy()

    @property
    def values(self) -> list[float]:
        """Return magnitudes as a JSON-friendly list for period-value compatibility."""
        return [float(item) for item in self.period_values.tolist()]

    @property
    def weights(self) -> np.ndarray:
        """Return optional passive period weights carried with this value."""
        return self._weights

    @property
    def num_periods(self) -> int:
        """Return the number of stored magnitudes."""
        return len(self._quantity.magnitude)

    @property
    def unit(self):
        """Return the unit in a human-friendly compact representation."""
        return self._format_units(self._quantity.units)

    @unit.setter
    def unit(self, unit_str):
        """Convert the stored quantity to ``unit_str`` in-place."""
        self._assert_writable()
        self._set_storage(self._quantity.to(self._normalise_unit_input(unit_str)))

    def to(self, new_unit: str) -> "Value":
        """Return a copy converted to ``new_unit``."""
        new_value = self._from_quantity(
            self._quantity.to(self._normalise_unit_input(new_unit)),
        )
        new_value._weights = self._weights.copy() if self._weights is not None else None
        return new_value

    def __getitem__(self, idx):
        """Return one selected period as an independent ``Value``."""
        if idx is None or not self._is_period_valued():
            return Value(self)

        if isinstance(idx, slice):
            subset = self._quantity.magnitude[idx]
            result = Value(subset, unit=self.unit)
            if self._weights is not None:
                result._weights = np.asarray(self._weights[idx], dtype=float).reshape(
                    -1
                )
            return result

        resolved_idx = self._resolve_period_index(idx)
        return self._from_quantity(
            Q_(
                np.asarray([self._quantity.magnitude[resolved_idx]], dtype=float),
                self._quantity.units,
            )
        )

    def __iter__(self):
        if not self._is_period_valued():
            raise TypeError("Scalar Value is not iterable.")
        return iter(self._quantity.magnitude)

    def __setitem__(self, idx, value):
        self._assert_writable()
        resolved_idx = self._resolve_period_index(idx)
        values = self._quantity.magnitude.copy()
        values[resolved_idx] = self._coerce_scalar_magnitude(value)
        self._set_storage(Q_(values, self._quantity.units))

    def __len__(self):
        """Return the number of periods stored."""
        return len(self._quantity.magnitude)

    def __str__(self):
        return f"{self.value} {self.unit}"

    def __repr__(self):
        if not self._is_period_valued():
            return (
                f"Value({self.value}, "
                f"{repr(self._serialise_units(self._quantity.units))})"
            )
        return (
            "Value("
            f"values={self.period_values.tolist()}, "
            f"unit={self._serialise_units(self._quantity.units)!r})"
        )

    def __float__(self):
        if self._is_period_valued():
            raise TypeError("Cannot convert multiperiod Value to float.")
        return float(self._quantity.magnitude[0])

    def __int__(self):
        if self._is_period_valued():
            raise TypeError("Cannot convert multiperiod Value to int.")
        return int(self._quantity.magnitude[0])

    def __round__(self, ndigits=None):
        if self._is_period_valued():
            raise TypeError("Cannot round multiperiod Value.")
        return round(self._quantity.magnitude[0], ndigits)

    def __array__(self, dtype=None, copy=None):
        if self._is_period_valued():
            array = np.asarray(self.period_values, dtype=dtype)
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
        instance._read_only_reason = None
        return instance

    def _make_read_only(self, reason: str) -> "Value":
        """Mark this value as an immutable domain-owned view."""
        self._read_only_reason = str(reason)
        return self

    def mutable_copy(self) -> "Value":
        """Return an independent writable copy of this value."""
        return Value(self)

    def _assert_writable(self) -> None:
        reason = getattr(self, "_read_only_reason", None)
        if reason is not None:
            raise TypeError(reason)

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
            if left_value._is_period_valued() and right_value._is_period_valued():
                if len(left_value._quantity.magnitude) != len(
                    right_value._quantity.magnitude
                ):
                    raise ValueError(
                        "Period-valued arithmetic requires identical period counts."
                    )
        return left_qty, right_qty

    def _resolve_period_index(self, idx: int | str | None) -> int:
        if not self._is_period_valued():
            return 0
        if idx is None:
            return 0
        if isinstance(idx, str):
            try:
                return int(idx)
            except (TypeError, ValueError) as exc:
                raise KeyError(idx) from exc
        idx = int(idx)
        if idx < 0 or idx >= self.num_periods:
            raise IndexError(idx)
        return idx

    def _is_period_valued(self) -> bool:
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

    def _coercion(self):
        return _value_coercion.ValueCoercion(
            self,
            value_type=Value,
            quantity_factory=Q_,
            serialized_scalar_keys=_SERIALIZED_SCALAR_KEYS,
            serialized_period_keys=_SERIALIZED_PERIOD_KEYS,
        )

    def _coerce_input(self, data, unit: str | None):
        return self._coercion()._coerce_input(data, unit)

    def _normalise_input_object(self, data):
        return self._coercion()._normalise_input_object(data)

    def _coerce_mapping_input(self, data, unit):
        return self._coercion()._coerce_mapping_input(data, unit)

    def _coerce_object_with_unit(self, data, unit):
        return self._coercion()._coerce_object_with_unit(data, unit)

    def _quantity_from_scalar(self, data, unit):
        return self._coercion()._quantity_from_scalar(data, unit)

    def _missing_or_zero_quantity(self, data, unit):
        return self._coercion()._missing_or_zero_quantity(data, unit)

    def _quantity_from_values(self, data, unit):
        return self._coercion()._quantity_from_values(data, unit)

    def _coerce_quantity_to_unit(self, quantity, unit):
        return self._coercion()._coerce_quantity_to_unit(quantity, unit)

    def _coerce_magnitude_array(self, data, **kwargs):
        return self._coercion()._coerce_magnitude_array(data, **kwargs)

    @staticmethod
    def _is_numeric_scalar(other: Any) -> bool:
        return _value_coercion.is_numeric_scalar(other)

    @staticmethod
    def _is_array_like_input(data: Any) -> bool:
        return _value_coercion.is_array_like_input(data)

    @staticmethod
    def _is_serialized_scalar_data(data: Mapping[Any, Any]) -> bool:
        return _value_coercion.is_serialized_scalar_data(
            data, keys=_SERIALIZED_SCALAR_KEYS
        )

    @staticmethod
    def _is_serialized_period_data(data: Mapping[Any, Any]) -> bool:
        return _value_coercion.is_serialized_period_data(
            data, keys=_SERIALIZED_PERIOD_KEYS
        )

    def _format_units(self, units) -> str:
        return _value_units.format_units(units)

    @staticmethod
    def _serialise_units(units) -> str:
        return _value_units.serialise_units(units)

    @staticmethod
    def _clean_unit_text(text: str) -> str:
        return _value_units.clean_unit_text(text)

    @staticmethod
    def _normalise_unit_input(unit: str | None) -> str | None:
        return _value_units.normalise_unit_text(unit)

    @staticmethod
    def _unit_from_normalised(unit: str):
        return _value_units.unit_from_normalised(ureg, unit)

    @staticmethod
    def _quantity_is_dimensionless(quantity) -> bool:
        return _value_units.quantity_is_dimensionless(quantity)

    @staticmethod
    def _same_dimensionality(quantity, unit: str) -> bool:
        return _value_units.same_dimensionality(
            quantity,
            unit,
            quantity_factory=Q_,
            registry=ureg,
        )

    @staticmethod
    def _normalise_weights(weights, *, expected_len: int) -> np.ndarray | None:
        return _value_units.normalise_weights(weights, expected_len=expected_len)

    def to_dict(self):
        """Serialise the value into a JSON-friendly dictionary."""
        if self._is_period_valued():
            return {
                "values": self.period_values.tolist(),
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
        """Instantiate from a scalar or multiperiod serialized mapping."""
        if not isinstance(data, Mapping):
            raise TypeError("data must be a mapping.")
        if cls._is_serialized_period_data(data):
            return cls(data)
        return cls(data)
