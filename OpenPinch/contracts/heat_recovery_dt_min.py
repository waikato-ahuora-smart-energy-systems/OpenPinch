"""Frozen public values for inverse heat-recovery targeting."""

from __future__ import annotations

import math
from decimal import Decimal
from enum import StrEnum
from numbers import Real

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from ..domain.value import Value

_RECOVERY_ABSOLUTE_TOLERANCE = 1e-6
_RECOVERY_RELATIVE_TOLERANCE = 1e-9


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
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value,
            (Real, Decimal, np.integer, np.floating),
        ):
            raise ValueError("value must be a finite number")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("value must be finite")
        return 0.0 if result == 0.0 else result

    @field_validator("unit", mode="before")
    @classmethod
    def _validate_unit(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("unit must be text")
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

    @staticmethod
    def _canonical(quantity: HeatRecoveryQuantity, unit: str, field: str) -> float:
        try:
            return float(Value(quantity.value, unit=quantity.unit).to(unit))
        except Exception as exc:
            raise ValueError(f"{field} must use units compatible with {unit}") from exc

    @model_validator(mode="after")
    def _validate_thermal_relationships(self) -> "HeatRecoveryDtMinResult":
        dt_min = self._canonical(self.dt_min, "delta_degC", "dt_min")
        requested = self._canonical(
            self.requested_heat_recovery,
            "kW",
            "requested_heat_recovery",
        )
        achieved = self._canonical(
            self.achieved_heat_recovery,
            "kW",
            "achieved_heat_recovery",
        )
        limit = self._canonical(
            self.thermodynamic_limit,
            "kW",
            "thermodynamic_limit",
        )
        residual = self._canonical(
            self.heat_recovery_residual,
            "kW",
            "heat_recovery_residual",
        )
        if dt_min < 0.0:
            raise ValueError("dt_min must be non-negative")
        for field, value in (
            ("requested_heat_recovery", requested),
            ("achieved_heat_recovery", achieved),
            ("thermodynamic_limit", limit),
        ):
            if value < 0.0:
                raise ValueError(f"{field} must be non-negative")

        scale = max(abs(requested), abs(achieved), abs(limit), abs(residual))
        tolerance = max(
            _RECOVERY_ABSOLUTE_TOLERANCE,
            _RECOVERY_RELATIVE_TOLERANCE * scale,
        )
        if requested > limit + tolerance:
            raise ValueError("requested heat recovery exceeds the thermodynamic limit")
        if achieved > limit + tolerance:
            raise ValueError("achieved heat recovery exceeds the thermodynamic limit")
        if requested > 0.0:
            meets_request = (
                achieved >= requested
                if requested <= _RECOVERY_ABSOLUTE_TOLERANCE
                else achieved + tolerance >= requested
            )
            if not meets_request:
                raise ValueError("achieved heat recovery does not meet the request")
        if abs(residual - (achieved - requested)) > tolerance:
            raise ValueError("heat recovery residual is inconsistent")

        at_limit = abs(requested - limit) <= tolerance
        if self.status is HeatRecoveryDtMinStatus.ZERO_RECOVERY_BOUNDARY:
            if requested != 0.0:
                raise ValueError("zero_recovery_boundary requires a zero request")
            if achieved > tolerance:
                raise ValueError("zero_recovery_boundary requires zero recovery")
        elif self.status is HeatRecoveryDtMinStatus.AT_THERMODYNAMIC_LIMIT:
            if not at_limit:
                raise ValueError("at_thermodynamic_limit requires a limit request")
        elif requested <= 0.0 or at_limit:
            raise ValueError("solved requires a positive non-limit request")
        return self


__all__ = [
    "HeatRecoveryDtMinResult",
    "HeatRecoveryDtMinStatus",
    "HeatRecoveryQuantity",
]
