"""Shared typing helpers for ProblemTable update keyword arguments."""

from typing import TypeAlias, TypedDict

import numpy as np

from .enums import ProblemTableLabel

ProblemTableColumnKey: TypeAlias = str | ProblemTableLabel
ProblemTableColumnUpdates: TypeAlias = dict[ProblemTableColumnKey, np.ndarray]


class ProblemTableUpdateKwargs(TypedDict):
    """Keyword arguments accepted by ``ProblemTable.update``."""

    T_col: np.ndarray
    updates: ProblemTableColumnUpdates
