"""Typed internal and aggregate errors for HPR map generation."""

from __future__ import annotations

import math
from collections.abc import Mapping

from ....contracts.hpr_performance_map import JsonValue
from .models import HprOperatingPoint, HprSimulationDiagnostic

_MESSAGE_LIMIT = 240
_DETAIL_STRING_LIMIT = 160


def _bounded_message(value: object) -> str:
    message = str(value).strip() or "HPR map generation failed"
    return message[:_MESSAGE_LIMIT]


def _sanitize_json(value: object) -> JsonValue:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, str):
        return value[:_DETAIL_STRING_LIMIT]
    if isinstance(value, tuple | list):
        return [_sanitize_json(item) for item in value[:20]]
    if isinstance(value, Mapping):
        return {
            str(key)[:_DETAIL_STRING_LIMIT]: _sanitize_json(item)
            for key, item in list(value.items())[:20]
        }
    return type(value).__name__


def sanitize_details(value: Mapping[str, object] | None) -> dict[str, JsonValue]:
    """Return a bounded recursive JSON copy without engine representations."""
    if value is None:
        return {}
    return {
        str(key)[:_DETAIL_STRING_LIMIT]: _sanitize_json(item)
        for key, item in list(value.items())[:20]
    }


class HprSimulatorFailure(Exception):
    """Private-style adapter failure classified for coordinator recovery."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        session_fatal: bool,
        details: Mapping[str, object] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        self.code = code
        self.stable_message = _bounded_message(message)
        self.session_fatal = bool(session_fatal)
        self.details = sanitize_details(details)
        self.cause = cause
        super().__init__(self.stable_message)


class HprMapGenerationError(Exception):
    """Atomic map-generation failure with ordered immutable diagnostics."""

    def __init__(
        self,
        diagnostics: tuple[HprSimulationDiagnostic, ...],
    ) -> None:
        if not diagnostics:
            raise ValueError("map-generation diagnostics must not be empty")
        self.diagnostics = tuple(diagnostics)
        backends = sorted(
            {item.backend for item in diagnostics if item.backend is not None}
        )
        backend_text = ",".join(backends) if backends else "unresolved"
        super().__init__(
            f"HPR map generation failed for {backend_text}: "
            f"{len(diagnostics)} diagnostic(s)"
        )


def diagnostic_from_failure(
    failure: HprSimulatorFailure,
    *,
    backend: str | None,
    model_id: str | None,
    point: HprOperatingPoint | None = None,
) -> HprSimulationDiagnostic:
    """Convert one classified internal failure to a stable public diagnostic."""
    return HprSimulationDiagnostic(
        code=failure.code,
        backend=backend,
        model_id=model_id,
        point_ordinal=None if point is None else point.ordinal,
        curve_id=None if point is None else point.curve_id,
        source_temperature=None if point is None else point.source_temperature,
        sink_temperature=None if point is None else point.sink_temperature,
        load_fraction=None if point is None else point.load_fraction,
        message=failure.stable_message,
        details=dict(failure.details),
    )


__all__ = [
    "HprMapGenerationError",
    "HprSimulatorFailure",
    "diagnostic_from_failure",
    "sanitize_details",
]
