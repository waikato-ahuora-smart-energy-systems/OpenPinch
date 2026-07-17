"""Callable protocol implemented by scalar optimisation backends."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class OptimisationBackend(Protocol):
    """Execute a bounded multistart minimisation."""

    def __call__(
        self,
        *,
        func,
        bounds,
        x0_ls=None,
        args=(),
        constraints=(),
        **options: Any,
    ) -> tuple[np.ndarray, np.ndarray]: ...


__all__ = ["OptimisationBackend"]
