"""Shared discovery for the repository's standard end-to-end problem corpus."""

from __future__ import annotations

from pathlib import Path

from tests.support.paths import REPOSITORY_ROOT

STANDARD_PROBLEM_INPUTS = REPOSITORY_ROOT / "examples" / "stream_data"


def standard_problem_paths() -> tuple[Path, ...]:
    """Return every standard problem path in stable filename order."""
    return tuple(
        sorted(STANDARD_PROBLEM_INPUTS.glob("p_*.json"), key=lambda path: path.name)
    )


__all__ = ["STANDARD_PROBLEM_INPUTS", "standard_problem_paths"]
