"""Deterministic test doubles for HPR point-simulator sessions."""

from __future__ import annotations

from dataclasses import dataclass, field

from OpenPinch.analysis.heat_pumps.performance_maps.errors import HprSimulatorFailure
from OpenPinch.analysis.heat_pumps.performance_maps.models import (
    HprMapGenerationContext,
    HprOperatingPoint,
    HprPointSimulation,
    HprSimulatorMetadata,
)


@dataclass(slots=True)
class FakeHprPointSimulator:
    """Observable fake with configurable lifecycle and point outcomes."""

    prepare_failure: HprSimulatorFailure | None = None
    point_failures: dict[int, HprSimulatorFailure] = field(default_factory=dict)
    point_results: dict[int, HprPointSimulation] = field(default_factory=dict)
    close_error: Exception | None = None
    events: list[tuple[str, int | None]] = field(default_factory=list)
    context: HprMapGenerationContext | None = None
    state: str = "fresh"

    def prepare(self, context: HprMapGenerationContext) -> HprSimulatorMetadata:
        if self.state != "fresh":
            raise RuntimeError("fake simulator can only be prepared once")
        self.events.append(("prepare", None))
        self.context = context
        if self.prepare_failure is not None:
            self.state = "fatal"
            raise self.prepare_failure
        self.state = "prepared"
        return HprSimulatorMetadata(
            backend=context.basis.simulation_backend,
            engine_version="fake-1.0",
            model_id=context.basis.model_id,
            characteristic_set_id=context.characteristic_set_id,
            design_converged=True,
            design_details={"engine": "fake"},
        )

    def simulate(self, point: HprOperatingPoint) -> HprPointSimulation:
        if self.state != "prepared":
            raise RuntimeError("fake simulator is not prepared")
        self.events.append(("simulate", point.ordinal))
        failure = self.point_failures.get(point.ordinal)
        if failure is not None:
            if failure.session_fatal:
                self.state = "fatal"
            raise failure
        if point.ordinal in self.point_results:
            return self.point_results[point.ordinal]
        assert self.context is not None
        useful_duty = point.requested_useful_duty
        compressor_power = useful_duty / 4.0
        if self.context.basis.mode == "heat_pump":
            q_source = useful_duty - compressor_power
            q_sink = useful_duty
        else:
            q_source = useful_duty
            q_sink = useful_duty + compressor_power
        return HprPointSimulation(
            q_source=q_source,
            q_sink=q_sink,
            compressor_power=compressor_power,
            converged=True,
            engine_details={"point": point.ordinal},
        )

    def close(self) -> None:
        if self.state == "closed":
            raise RuntimeError("fake simulator was closed twice")
        self.events.append(("close", None))
        self.state = "closed"
        if self.close_error is not None:
            raise self.close_error


__all__ = ["FakeHprPointSimulator"]
