"""Shared typing helpers for ProblemTable update keyword arguments."""

from __future__ import annotations

from typing import TypeAlias, TypedDict

import numpy as np

from .enums import ProblemTableLabel

ProblemTableColumnKey: TypeAlias = str | ProblemTableLabel
ProblemTableColumnUpdates: TypeAlias = dict[ProblemTableColumnKey, np.ndarray]


class ProblemTableUpdateKwargs(TypedDict):
    """Keyword arguments accepted by ``ProblemTable.update``."""

    T_col: np.ndarray
    updates: ProblemTableColumnUpdates


__all__ = [
    "ProblemTableColumnKey",
    "ProblemTableColumnUpdates",
    "ProblemTableUpdateKwargs",
]
