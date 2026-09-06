"""Engine-neutral HPR point-simulator protocol."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from .models import (
    HprMapGenerationContext,
    HprOperatingPoint,
    HprPointSimulation,
    HprSimulatorMetadata,
)


@runtime_checkable
class HprPointSimulator(Protocol):
    """One fresh prepare/simulate/close engine session."""

    def prepare(self, context: HprMapGenerationContext) -> HprSimulatorMetadata: ...

    def simulate(self, point: HprOperatingPoint) -> HprPointSimulation: ...

    def close(self) -> None: ...


type HprPointSimulatorFactory = Callable[[str], HprPointSimulator]

__all__ = ["HprPointSimulator", "HprPointSimulatorFactory"]
