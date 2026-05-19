"""Shared typing helpers for ProblemTable update keyword arguments."""

from typing import TypeAlias, TypedDict

import numpy as np

ProblemTableColumnUpdates: TypeAlias = dict[str, np.ndarray]


class ProblemTableUpdateKwargs(TypedDict):
    """Keyword arguments accepted by ``ProblemTable.update``."""

    T_col: np.ndarray
    updates: ProblemTableColumnUpdates
