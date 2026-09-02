"""Frozen public values for inverse heat-recovery targeting."""

from __future__ import annotations

import math
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, field_validator


class _FrozenContract(BaseModel):
    """Strict immutable base for specialist heat-recovery values."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class HeatRecoveryDtMinStatus(StrEnum):
    """Termination states for an inverse heat-recovery calculation.

    ``AT_THERMODYNAMIC_LIMIT`` identifies a maximum-recovery request; its
    dt_min can be positive when a threshold problem has a recovery plateau.
    """

    SOLVED = "solved"
    AT_THERMODYNAMIC_LIMIT = "at_thermodynamic_limit"
    ZERO_RECOVERY_BOUNDARY = "zero_recovery_boundary"


class HeatRecoveryQuantity(_FrozenContract):
    """One finite thermal quantity with explicit unit metadata."""

    value: float
    unit: str

    @field_validator("value", mode="before")
    @classmethod
    def _validate_value(cls, value: object) -> float:
        if isinstance(value, bool):
            raise ValueError("value must be a finite number")
        result = float(value)  # type: ignore[arg-type]
        if not math.isfinite(result):
            raise ValueError("value must be finite")
        return 0.0 if result == 0.0 else result

    @field_validator("unit")
    @classmethod
    def _validate_unit(cls, value: str) -> str:
        unit = value.strip()
        if not unit:
            raise ValueError("unit must not be empty")
        return unit


class HeatRecoveryDtMinResult(_FrozenContract):
    """Diagnostic result from process-level global dt_min inversion."""

    scope: str
    period_id: str
    dt_min: HeatRecoveryQuantity
    requested_heat_recovery: HeatRecoveryQuantity
    achieved_heat_recovery: HeatRecoveryQuantity
    thermodynamic_limit: HeatRecoveryQuantity
    heat_recovery_residual: HeatRecoveryQuantity
    status: HeatRecoveryDtMinStatus
    iterations: int

    @field_validator("scope", "period_id")
    @classmethod
    def _validate_nonempty_text(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("value must not be empty")
        return text

    @field_validator("iterations", mode="before")
    @classmethod
    def _validate_iterations_type(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("iterations must be an integer")
        return value

    @field_validator("iterations")
    @classmethod
    def _validate_iterations_value(cls, value: int) -> int:
        if value < 0:
            raise ValueError("iterations must be non-negative")
        return value


__all__ = [
    "HeatRecoveryDtMinResult",
    "HeatRecoveryDtMinStatus",
    "HeatRecoveryQuantity",
]
