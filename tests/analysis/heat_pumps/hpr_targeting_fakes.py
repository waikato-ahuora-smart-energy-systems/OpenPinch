"""Deterministic engine-neutral fakes for HPR targeting evaluator tests."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable

from OpenPinch.analysis.heat_pumps.performance_maps.targeting_models import (
    HprTargetEvaluatorError,
    HprTargetEvaluatorMetadata,
    HprTargetThermodynamicRequest,
    HprTargetThermodynamicResult,
    HprThermalProfilePoint,
)


class FakeHprTargetEvaluator:
    """Scriptable evaluator with explicit lifecycle and failure modes."""

    def __init__(
        self,
        behaviors: tuple[str, ...] = ("success",),
        *,
        cleanup_failure: bool = False,
        cop: float | Callable[[HprTargetThermodynamicRequest], float] = 4.0,
    ) -> None:
        self.behaviors = deque(behaviors)
        self.cleanup_failure = cleanup_failure
        self.cop = cop
        self.state = "created"
        self.open_calls = 0
        self.evaluate_calls = 0
        self.close_calls = 0
        self.requests: list[HprTargetThermodynamicRequest] = []

    def open(self) -> HprTargetEvaluatorMetadata:
        if self.state != "created":
            raise RuntimeError("fake evaluator can open only once")
        self.open_calls += 1
        self.state = "ready"
        return HprTargetEvaluatorMetadata(
            backend="tespy",
            engine_version="fake-1.0",
            model_id="openpinch-tespy-single-stage-v1",
            assumptions={"fake": True},
        )

    def evaluate(
        self,
        request: HprTargetThermodynamicRequest,
    ) -> HprTargetThermodynamicResult:
        if self.state != "ready":
            raise RuntimeError("fake evaluator is not ready")
        self.evaluate_calls += 1
        self.requests.append(request)
        self.state = "evaluating"
        behavior = self.behaviors.popleft() if self.behaviors else "success"
        if behavior == "local_failure":
            self.state = "ready"
            raise HprTargetEvaluatorError(
                "candidate_state_failure",
                "candidate did not converge",
                session_fatal=False,
                details={"candidate": request.candidate_id},
            )
        if behavior in {"fatal_failure", "restoration_failure"}:
            self.state = "fatal"
            raise HprTargetEvaluatorError(
                behavior,
                "evaluator state cannot be restored",
                session_fatal=True,
            )

        cop = self.cop(request) if callable(self.cop) else self.cop
        power = request.useful_duty / cop
        if request.mode == "heat_pump":
            q_sink = request.useful_duty
            q_source = q_sink - power
            cop = q_sink / power
        else:
            q_source = request.useful_duty
            q_sink = q_source + power
            cop = q_source / power
        self.state = "ready"
        return HprTargetThermodynamicResult(
            backend="tespy",
            model_id=request.model_id,
            working_fluid=request.working_fluid,
            converged=True,
            q_source=q_source,
            q_sink=q_sink,
            compressor_power=power,
            cop=cop,
            source_profile=(
                HprThermalProfilePoint(
                    temperature=request.evaporating_temperature + 5.0,
                    enthalpy=0.0,
                ),
                HprThermalProfilePoint(
                    temperature=request.evaporating_temperature,
                    enthalpy=q_source,
                ),
            ),
            sink_profile=(
                HprThermalProfilePoint(
                    temperature=request.condensing_temperature,
                    enthalpy=0.0,
                ),
                HprThermalProfilePoint(
                    temperature=request.condensing_temperature + 5.0,
                    enthalpy=q_sink,
                ),
            ),
            engine_version="fake-1.0",
            design_details={"fake": True},
        )

    def close(self) -> None:
        self.close_calls += 1
        if self.close_calls > 1:
            raise RuntimeError("fake evaluator closed more than once")
        if self.cleanup_failure:
            self.state = "cleanup_failed"
            raise HprTargetEvaluatorError(
                "cleanup_failure",
                "fake cleanup failed",
                session_fatal=True,
            )
        self.state = "closed"


__all__ = ["FakeHprTargetEvaluator"]
