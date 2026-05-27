"""Float-like stream-facing default-state view over stored :class:`Value` objects."""

from __future__ import annotations

import numpy as np

from .value import Value


class StreamValueView(float):
    """Float-like default-state view over a stored :class:`Value`."""

    _stream_value_view = True

    def __new__(cls, value: Value):
        default_state_id = cls._resolve_default_state_id(value)
        magnitude = cls._resolve_default_magnitude(value, default_state_id)
        instance = float.__new__(cls, magnitude)
        instance._value = value
        instance._default_state_id = default_state_id
        return instance

    @staticmethod
    def _resolve_default_state_id(value: Value) -> str | None:
        if value.state_ids is None:
            return None
        return "0" if "0" in value.state_ids else value.state_ids[0]

    @staticmethod
    def _resolve_default_magnitude(value: Value, state_id: str | None) -> float:
        if state_id is None:
            return float(value.value)
        return float(value[state_id].value)

    @property
    def value(self) -> float:
        """Default-state magnitude exposed by this stream-facing view."""
        return float(self)

    @property
    def unit(self) -> str:
        """Shared unit across all stored states."""
        return self._value.unit

    @property
    def units(self) -> str:
        """Compatibility alias for helpers expecting ``units``."""
        return self._value.unit

    @property
    def state_ids(self) -> list[str] | None:
        """Ordered state identifiers for the wrapped value."""
        return self._value.state_ids

    @property
    def state_values(self) -> np.ndarray:
        """Raw per-state magnitudes on the wrapped value."""
        return self._value.state_values

    @property
    def weights(self) -> np.ndarray:
        """Normalised state weights from the wrapped value."""
        return self._value.weights

    @property
    def weighted_value(self) -> float:
        """Weighted scalar view of the wrapped value."""
        return self._value.weighted_value

    @property
    def default_state_id(self) -> str | None:
        """Return the state identifier used by the numeric float face."""
        return self._default_state_id

    @property
    def raw_value(self) -> Value:
        """Return the underlying :class:`Value` object."""
        return Value(self._value)

    def __getitem__(self, state_id) -> Value:
        """Return one scalar state from the wrapped :class:`Value`."""
        if self._value.state_ids is None:
            return Value(self._value)
        return self._value[state_id]
