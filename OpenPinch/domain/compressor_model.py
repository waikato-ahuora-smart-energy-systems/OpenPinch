"""User-supplied compressor efficiency functions carried in configuration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CustomCompressorModel:
    """A compressor efficiency function with the name recorded in provenance.

    The function takes the compressor conditions and returns the isentropic
    efficiency. Settings JSON carries only the name, so a function cannot be
    restored from JSON.
    """

    function: Callable[[Any], float]
    name: str

    @classmethod
    def wrap(cls, function: Callable[[Any], float]) -> CustomCompressorModel:
        """Return ``function`` named by its ``__name__``, or as already wrapped."""
        if isinstance(function, cls):
            return function
        name = getattr(function, "__name__", None) or type(function).__name__
        return cls(function, name)

    def __call__(self, conditions: Any) -> float:
        return self.function(conditions)

    def to_dict(self) -> dict[str, str]:
        """Return the JSON form recorded in provenance."""
        return {"custom_compressor_model": self.name}
