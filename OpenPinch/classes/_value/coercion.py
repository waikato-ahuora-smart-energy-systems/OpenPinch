"""Low-level input-shape predicates used by the parent Value model."""

from __future__ import annotations

from typing import Any

import numpy as np


def is_value_with_unit(data: Any) -> bool:
    """Return whether an object exposes value and unit attributes."""
    return hasattr(data, "value") and hasattr(data, "unit")


def is_bool_like(data: Any) -> bool:
    """Return whether input is a Python or NumPy boolean scalar."""
    return isinstance(data, (bool, np.bool_))
