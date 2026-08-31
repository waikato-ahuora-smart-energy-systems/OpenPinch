"""Typed failures for utility-placement analysis."""

from __future__ import annotations

import math
from enum import Enum
from typing import Any

from pydantic import BaseModel


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | int | bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError("Utility-placement error context must be finite")
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_safe(item) for item in value]
    raise TypeError(f"Unsupported utility-placement error context: {type(value)!r}")


class UtilityPlacementError(RuntimeError):
    """Root for every utility-placement-specific failure."""

    def __init__(
        self,
        *,
        code: str,
        message: str,
        field_path: str | None = None,
        template_key: Any = None,
        period_id: str | None = None,
        scope: str | None = None,
        method: str | None = None,
        seed: int | None = None,
        details: Any = None,
    ) -> None:
        normalized_code = code.strip()
        normalized_message = message.strip()
        if not normalized_code or not normalized_message:
            raise ValueError("Utility-placement errors require code and message")
        super().__init__(normalized_message)
        context = {"code": normalized_code, "message": normalized_message}
        optional = {
            "field_path": field_path,
            "template_key": template_key,
            "period_id": period_id,
            "scope": scope,
            "method": method,
            "seed": seed,
            "details": details,
        }
        context.update(
            (key, _json_safe(value))
            for key, value in optional.items()
            if value is not None
        )
        self.code = normalized_code
        self.context = context


class PlacementRequestValidationError(UtilityPlacementError, ValueError):
    """Invalid public request shape or option."""


class UtilityTemplateValidationError(UtilityPlacementError, ValueError):
    """Invalid utility-template inventory or metadata."""


class UtilityPlacementUnitError(UtilityPlacementError, ValueError):
    """Unknown or dimensionally incompatible unit value."""


class PlacementModelValidationError(UtilityPlacementError, ValueError):
    """Invalid detached envelope or pure model invariant."""


class EmptyPlacementFeasibleRegionError(PlacementModelValidationError):
    """Physical, caller, or ordering constraints have no intersection."""


class PlacementContextError(UtilityPlacementError):
    """Resolved scope or detached period context is unusable."""


class PlacementTargetingError(UtilityPlacementError):
    """Targeting failed independently of an ordinary candidate violation."""


class PlacementThermodynamicError(UtilityPlacementError):
    """Thermodynamic calculation or invariant failed at run level."""


class PlacementOptimisationError(UtilityPlacementError):
    """The solver-neutral optimization boundary failed."""


class NoFeasiblePlacementError(PlacementOptimisationError):
    """Bounded optimization produced no proven feasible placement."""


__all__ = [
    "EmptyPlacementFeasibleRegionError",
    "NoFeasiblePlacementError",
    "PlacementContextError",
    "PlacementModelValidationError",
    "PlacementOptimisationError",
    "PlacementRequestValidationError",
    "PlacementTargetingError",
    "PlacementThermodynamicError",
    "UtilityPlacementError",
    "UtilityPlacementUnitError",
    "UtilityTemplateValidationError",
]
