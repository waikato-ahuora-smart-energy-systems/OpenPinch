"""Immutable engine-neutral values for HPR targeting thermodynamic evaluation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol

from ....contracts.hpr_performance_map import JsonValue
from .errors import sanitize_details
from .models import HprWorkingFluidSpec

type HprFrozenJson = (
    None
    | bool
    | int
    | float
    | str
    | tuple["HprFrozenJson", ...]
    | tuple[tuple[str, "HprFrozenJson"], ...]
)


def _freeze_json(value: JsonValue) -> HprFrozenJson:
    if isinstance(value, dict):
        return tuple(
            (str(key), _freeze_json(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _frozen_details(
    value: Mapping[str, object] | None,
) -> tuple[tuple[str, HprFrozenJson], ...]:
    sanitized = sanitize_details(value)
    return tuple(
        (key, _freeze_json(item))
        for key, item in sorted(sanitized.items(), key=lambda pair: pair[0])
    )


def _require_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True, slots=True)
class HprThermalProfilePoint:
    """One detached temperature/cumulative-enthalpy profile coordinate."""

    temperature: float
    enthalpy: float

    def __post_init__(self) -> None:
        _require_finite("profile temperature", self.temperature)
        _require_finite("profile enthalpy", self.enthalpy)


@dataclass(frozen=True, slots=True)
class HprTargetThermodynamicRequest:
    """Complete physical specification for one nominal optimizer candidate."""

    mode: Literal["heat_pump", "refrigeration"]
    cycle_id: Literal["single_stage_vapour_compression"]
    model_id: str
    working_fluid: HprWorkingFluidSpec
    evaporating_temperature: float
    condensing_temperature: float
    useful_duty: float
    source_approach_temperature: float
    sink_approach_temperature: float
    compressor_isentropic_efficiency: float
    superheat: float
    subcooling: float
    internal_hx_gas_temperature_change: float
    candidate_id: str = field(compare=False, hash=False)

    def __post_init__(self) -> None:
        if self.mode not in {"heat_pump", "refrigeration"}:
            raise ValueError("mode must be 'heat_pump' or 'refrigeration'")
        if self.cycle_id != "single_stage_vapour_compression":
            raise ValueError("unsupported HPR target cycle_id")
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id must be a nonempty string")
        if not isinstance(self.working_fluid, HprWorkingFluidSpec):
            raise TypeError("working_fluid must be HprWorkingFluidSpec")
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise ValueError("candidate_id must be a nonempty string")
        for name in (
            "evaporating_temperature",
            "condensing_temperature",
            "useful_duty",
            "source_approach_temperature",
            "sink_approach_temperature",
            "compressor_isentropic_efficiency",
            "superheat",
            "subcooling",
            "internal_hx_gas_temperature_change",
        ):
            _require_finite(name, getattr(self, name))
        if self.condensing_temperature <= self.evaporating_temperature:
            raise ValueError(
                "condensing temperature must exceed evaporating temperature"
            )
        if self.useful_duty <= 0.0:
            raise ValueError("useful_duty must be positive")
        for name in (
            "source_approach_temperature",
            "sink_approach_temperature",
            "superheat",
            "subcooling",
            "internal_hx_gas_temperature_change",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        if not 0.0 < self.compressor_isentropic_efficiency <= 1.0:
            raise ValueError(
                "compressor_isentropic_efficiency must be in the interval (0, 1]"
            )


@dataclass(frozen=True, slots=True)
class HprTargetThermodynamicResult:
    """Detached raw evaluator output before engine-neutral acceptance checks."""

    backend: Literal["coolprop", "tespy"]
    model_id: str
    working_fluid: HprWorkingFluidSpec
    converged: bool
    q_source: float
    q_sink: float
    compressor_power: float
    cop: float
    source_profile: tuple[HprThermalProfilePoint, ...]
    sink_profile: tuple[HprThermalProfilePoint, ...]
    engine_version: str
    design_details: Mapping[str, object] | tuple[tuple[str, HprFrozenJson], ...]

    def __post_init__(self) -> None:
        if self.backend not in {"coolprop", "tespy"}:
            raise ValueError("result backend must be 'coolprop' or 'tespy'")
        if not self.model_id.strip():
            raise ValueError("result model_id must not be empty")
        if not isinstance(self.working_fluid, HprWorkingFluidSpec):
            raise TypeError("result working_fluid must be HprWorkingFluidSpec")
        if not self.engine_version.strip():
            raise ValueError("engine_version must not be empty")
        object.__setattr__(self, "source_profile", tuple(self.source_profile))
        object.__setattr__(self, "sink_profile", tuple(self.sink_profile))
        if isinstance(self.design_details, Mapping):
            object.__setattr__(
                self,
                "design_details",
                _frozen_details(self.design_details),
            )


@dataclass(frozen=True, slots=True)
class HprTargetEvaluatorMetadata:
    """Stable evaluator-session identity returned when a session opens."""

    backend: Literal["coolprop", "tespy"]
    engine_version: str
    model_id: str
    assumptions: Mapping[str, object] | tuple[tuple[str, HprFrozenJson], ...]
    power_boundary: Literal["compressor_only"] = "compressor_only"

    def __post_init__(self) -> None:
        if self.backend not in {"coolprop", "tespy"}:
            raise ValueError("metadata backend must be 'coolprop' or 'tespy'")
        if not self.engine_version.strip() or not self.model_id.strip():
            raise ValueError("evaluator metadata identity must not be empty")
        if isinstance(self.assumptions, Mapping):
            object.__setattr__(self, "assumptions", _frozen_details(self.assumptions))


@dataclass(frozen=True, slots=True)
class HprTargetEvaluationFailure:
    """Immutable candidate-local or request-fatal evaluator diagnostic."""

    code: str
    message: str
    session_fatal: bool
    details: tuple[tuple[str, HprFrozenJson], ...] = ()


class HprTargetEvaluatorError(Exception):
    """Classified evaluator exception with bounded detached diagnostics."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        session_fatal: bool,
        details: Mapping[str, object] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        self.code = str(code).strip() or "target_evaluator_failure"
        self.stable_message = (str(message).strip() or "HPR evaluation failed")[:240]
        self.session_fatal = bool(session_fatal)
        self.details = _frozen_details(details)
        self.cause = cause
        super().__init__(self.stable_message)

    def to_failure(self) -> HprTargetEvaluationFailure:
        return HprTargetEvaluationFailure(
            code=self.code,
            message=self.stable_message,
            session_fatal=self.session_fatal,
            details=self.details,
        )


class HprTargetEvaluator(Protocol):
    """Sequential lifecycle implemented by one concrete targeting evaluator."""

    def open(self) -> HprTargetEvaluatorMetadata: ...

    def evaluate(
        self,
        request: HprTargetThermodynamicRequest,
    ) -> HprTargetThermodynamicResult: ...

    def close(self) -> None: ...


__all__ = [
    "HprFrozenJson",
    "HprTargetEvaluationFailure",
    "HprTargetEvaluator",
    "HprTargetEvaluatorError",
    "HprTargetEvaluatorMetadata",
    "HprTargetThermodynamicRequest",
    "HprTargetThermodynamicResult",
    "HprThermalProfilePoint",
]
