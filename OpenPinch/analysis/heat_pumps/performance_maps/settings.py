"""Fixed TESPy convergence settings for schema 1.0 map generation."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, slots=True)
class TespyConvergenceSettings:
    """One immutable, versioned set of public TESPy solve arguments."""

    identifier: str
    solve_arguments: MappingProxyType[str, Any]

    def as_provenance(self) -> dict[str, object]:
        """Return a detached JSON-compatible representation."""
        return {
            "identifier": self.identifier,
            "solve_arguments": dict(self.solve_arguments),
        }


TESPY_CONVERGENCE_SETTINGS = TespyConvergenceSettings(
    identifier="openpinch-tespy-hpr-convergence-v1",
    solve_arguments=MappingProxyType(
        {
            "max_iter": 50,
            "min_iter": 4,
            "init_previous": False,
            "use_cuda": False,
            "print_results": False,
            "robust_relax": False,
            "oscillation_damping": False,
            "skip_postprocess": False,
        }
    ),
)

__all__ = ["TESPY_CONVERGENCE_SETTINGS", "TespyConvergenceSettings"]
