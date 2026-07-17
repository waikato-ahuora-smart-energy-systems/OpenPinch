"""Energy-transfer payload cleaning and serialization."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...domain.configuration import tol
from ...domain.enums import GraphType


def _save_graph_data(diagram: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {GraphType.ETD.value: diagram}


def _as_float_array(values) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _decimal_places() -> int:
    return int(-np.log10(tol))


def _clean_array(values: np.ndarray) -> np.ndarray:
    return np.where(np.abs(values) <= tol, 0.0, values)


def _clean_optional(value: float | None) -> float | None:
    return None if value is None else _clean_value(value)


def _clean_value(value: float) -> float:
    value = float(value)
    return 0.0 if abs(value) <= tol else value
