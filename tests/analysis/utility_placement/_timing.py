"""Time budgets for the utility-placement performance tests."""

from __future__ import annotations

import os

# Shared CI runners are slower and noisier than a developer machine, and
# coverage slows them further, so CI gets a looser budget.
CI_BUDGET_FACTOR = 3.0


def time_budget(seconds: float) -> float:
    """Return the time budget in seconds for this machine."""
    if os.environ.get("CI"):
        return seconds * CI_BUDGET_FACTOR
    return seconds
