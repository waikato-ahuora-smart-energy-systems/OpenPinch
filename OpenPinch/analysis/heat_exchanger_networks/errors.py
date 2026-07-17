"""Shared exceptions for HEN synthesis services."""

from __future__ import annotations


class WorkflowContractError(RuntimeError):
    """Raised when task fan-out would violate the synthesis workflow contract."""


__all__ = ["WorkflowContractError"]
