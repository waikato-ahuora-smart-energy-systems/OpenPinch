"""Stream-bound accessor for scalar-style and stateful attribute access."""

from __future__ import annotations

import operator

import numpy as np

from ..value import Value


class StreamValueAccessor:
    """Stream-bound proxy exposing value reads and optional stateful writes."""

    _stream_value_accessor = True

    def __init__(
        self,
        stream,
        attr_name: str,
        value: Value,
        *,
        writable: bool,
        state_ids: list[str] | None = None,
        weights: np.ndarray | None = None,
    ):
        self._stream = stream
        self._attr_name = attr_name
        self._value = value
        self._writable = writable
        self._state_ids = None if state_ids is None else list(state_ids)
        self._weights = None if weights is None else weights.copy()
        self._default_state_id = self._resolve_default_state_id(value, state_ids)

    @staticmethod
    def _resolve_default_state_id(
        value: Value,
        state_ids: list[str] | None,
    ) -> str | None:
        if len(value.state_values) <= 1:
            return None
        active_state_ids = (
            list(state_ids)
            if state_ids is not None
            else [str(idx) for idx in range(len(value.state_values))]
        )
        return "0" if "0" in active_state_ids else active_state_ids[0]

    def _coerce_other(self, other):
        if isinstance(other, Value):
            return float(other)
        if getattr(other, "_stream_value_accessor", False):
            return float(other)
        return other

    @property
    def value(self) -> float:
        """Default-state scalar magnitude."""
        return float(self)

    @property
    def unit(self) -> str:
        """Shared unit across all stored states."""
        return self._value.unit

    @property
    def state_ids(self) -> list[str] | None:
        """Ordered state identifiers for the wrapped value."""
        if self._state_ids is None:
            return None
        return list(self._state_ids)

    @property
    def state_values(self) -> np.ndarray:
        """Raw per-state magnitudes."""
        return self._value.state_values

    @property
    def weights(self) -> np.ndarray:
        """Normalised state weights."""
        if self._weights is None:
            count = len(self._value.state_values)
            return np.ones(count, dtype=float) / float(count)
        return self._weights.copy()

    @property
    def mean_value(self) -> float:
        """Mean scalar view of the wrapped value."""
        return self._value.mean_value

    @property
    def min(self) -> "StreamValueAccessor":
        """Minimum stored magnitude as a scalar accessor."""
        return type(self)(
            self._stream,
            self._attr_name,
            self._value.min_value(),
            writable=False,
        )

    @property
    def max(self) -> "StreamValueAccessor":
        """Maximum stored magnitude as a scalar accessor."""
        return type(self)(
            self._stream,
            self._attr_name,
            self._value.max_value(),
            writable=False,
        )

    @property
    def default_state_id(self) -> str | None:
        """State identifier used by the numeric scalar face."""
        return self._default_state_id

    @property
    def raw_value(self) -> Value:
        """Underlying value object as a copy."""
        return Value(self._value)

    def __getitem__(self, state_id) -> Value:
        """Return one scalar state from the wrapped value."""
        if len(self._value.state_values) <= 1:
            return Value(self._value)
        if state_id is None:
            state_id = self._default_state_id
            if state_id is None:
                return Value(self._value)
        active_state_ids = self.state_ids or [
            str(idx) for idx in range(len(self._value.state_values))
        ]
        return self._value[active_state_ids.index(str(state_id))]

    def __setitem__(self, state_id, new_value) -> None:
        """Write one state through the owning stream when permitted."""
        if not self._writable:
            raise TypeError(f"Stream attribute {self._attr_name!r} is read-only.")
        self._stream.set_attr_for_state(self._attr_name, new_value, state_id=state_id)

    def __float__(self):
        if self._default_state_id is None:
            return float(self._value.value)
        return float(self[self._default_state_id].value)

    def __int__(self):
        return int(float(self))

    def __round__(self, ndigits=None):
        return round(float(self), ndigits)

    def __array__(self, dtype=None):
        return np.asarray(float(self), dtype=dtype)

    def __bool__(self):
        return bool(float(self))

    def __str__(self):
        return f"{float(self)} {self.unit}"

    def __repr__(self):
        return (
            f"StreamValueAccessor(attr={self._attr_name!r}, value={float(self)!r}, "
            f"unit={self.unit!r}, writable={self._writable!r})"
        )

    def __format__(self, format_spec):
        return format(float(self), format_spec)

    def __eq__(self, other):
        return float(self) == self._coerce_other(other)

    def __lt__(self, other):
        return float(self) < self._coerce_other(other)

    def __le__(self, other):
        return float(self) <= self._coerce_other(other)

    def __gt__(self, other):
        return float(self) > self._coerce_other(other)

    def __ge__(self, other):
        return float(self) >= self._coerce_other(other)

    def _binary_op(self, other, op, *, reverse: bool = False):
        left, right = (
            (self._coerce_other(other), float(self))
            if reverse
            else (
                float(self),
                self._coerce_other(other),
            )
        )
        return op(left, right)

    def __add__(self, other):
        return self._binary_op(other, operator.add)

    def __radd__(self, other):
        return self._binary_op(other, operator.add, reverse=True)

    def __sub__(self, other):
        return self._binary_op(other, operator.sub)

    def __rsub__(self, other):
        return self._binary_op(other, operator.sub, reverse=True)

    def __mul__(self, other):
        return self._binary_op(other, operator.mul)

    def __rmul__(self, other):
        return self._binary_op(other, operator.mul, reverse=True)

    def __truediv__(self, other):
        return self._binary_op(other, operator.truediv)

    def __rtruediv__(self, other):
        return self._binary_op(other, operator.truediv, reverse=True)

    def __pow__(self, other):
        return self._binary_op(other, operator.pow)

    def __rpow__(self, other):
        return self._binary_op(other, operator.pow, reverse=True)

    def __neg__(self):
        return -float(self)

    def __pos__(self):
        return +float(self)

    def __abs__(self):
        return abs(float(self))
