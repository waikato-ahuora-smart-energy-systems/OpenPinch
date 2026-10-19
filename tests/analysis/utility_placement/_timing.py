"""Time budgets for the utility-placement performance tests."""

from __future__ import annotations

import os
import sys

# Shared CI runners are slower and noisier than a developer machine, and
# coverage slows them further, so CI gets a looser budget.
CI_BUDGET_FACTOR = 3.0
# Pyodide (WebAssembly) runs these about ten times slower than native CPython
# (a 16 ms native p95 took 219 ms on a busy machine), and its empty os.environ
# hides CI, so this factor also covers the shared runner.
WASM_BUDGET_FACTOR = 10.0


def time_budget(seconds: float) -> float:
    """Return the time budget in seconds for this machine."""
    if sys.platform == "emscripten":
        return seconds * WASM_BUDGET_FACTOR
    if os.environ.get("CI"):
        return seconds * CI_BUDGET_FACTOR
    return seconds
