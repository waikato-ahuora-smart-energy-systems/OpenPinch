"""Time budgets scale with the machine the performance tests run on."""

import sys

from tests.analysis.utility_placement._timing import (
    CI_BUDGET_FACTOR,
    WASM_BUDGET_FACTOR,
    time_budget,
)


def test_developer_machine_gets_the_stated_budget(monkeypatch):
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setattr(sys, "platform", "darwin")
    assert time_budget(0.1) == 0.1


def test_ci_runners_get_the_ci_factor(monkeypatch):
    monkeypatch.setenv("CI", "true")
    monkeypatch.setattr(sys, "platform", "linux")
    assert time_budget(0.1) == 0.1 * CI_BUDGET_FACTOR


def test_webassembly_gets_the_wasm_factor_without_ci_in_its_environment(
    monkeypatch,
):
    # Pyodide starts with an empty os.environ, so CI is never visible there.
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setattr(sys, "platform", "emscripten")
    assert time_budget(0.1) == 0.1 * WASM_BUDGET_FACTOR
