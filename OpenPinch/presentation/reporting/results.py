"""Serialize family-adapted analysis results at the presentation boundary."""

from __future__ import annotations

from typing import Any

from ...domain.value import Value
from .adapters import adapt_analysis_result


def target_to_result(target, isTotal: bool = False):
    """Return the family-owned report contract for one analysis result."""
    return adapt_analysis_result(target, is_total=isTotal)


def serialize_target(target, isTotal: bool = False) -> dict[str, Any]:
    return _serialise_report_data(
        target_to_result(target, isTotal=isTotal).model_dump(mode="python")
    )


def _serialise_report_data(value: Any) -> Any:
    if isinstance(value, Value):
        return value.to_dict()
    if isinstance(value, dict):
        return {key: _serialise_report_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialise_report_data(item) for item in value]
    if hasattr(value, "model_dump"):
        return _serialise_report_data(value.model_dump(mode="python"))
    return value


__all__ = ["serialize_target", "target_to_result"]
