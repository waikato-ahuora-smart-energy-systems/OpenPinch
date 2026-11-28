"""A heat pump step that cannot be solved is skipped with a warning.

The second search refuses its first design when the existing network's
coolers and heaters cannot serve the machine's fixed duties at their
approaches. That is a heat pump the retrofit cannot use, not a failed
retrofit: the result is the search without the heat pump, and a warning
says the step was left out and why.
"""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from OpenPinch.analysis.heat_exchanger_networks.retrofit.search import (
    RetrofitSettings,
)
from OpenPinch.application import heat_pump_retrofit
from tests.application.heat_pump_stream_cases import step_of_no_size


def test_a_step_whose_first_design_does_not_solve_is_left_out(monkeypatch):
    base = SimpleNamespace(frontier=(SimpleNamespace(network=object()),))
    calls = []

    def search(problem, existing, settings, **arguments):
        calls.append(arguments)
        if "heat_pump" in arguments:
            raise ValueError("the heat pump 'HP' added as streams does not solve")
        return base

    case = SimpleNamespace(
        _build_execution_master_zone=lambda: object(),
        to_problem_json=lambda: {"streams": []},
    )
    monkeypatch.setattr(heat_pump_retrofit, "run_retrofit_search", search)
    monkeypatch.setattr(
        heat_pump_retrofit,
        "heat_pump_step",
        lambda *args, **kwargs: (step_of_no_size(), object()),
    )
    monkeypatch.setattr(heat_pump_retrofit, "heat_pump_case", lambda *a, **k: case)
    monkeypatch.setattr(heat_pump_retrofit, "snapshot_problem", lambda problem: problem)
    monkeypatch.setattr(
        heat_pump_retrofit,
        "temporary_zone_configuration",
        lambda root, configuration: nullcontext(),
    )
    with pytest.warns(UserWarning, match="heat pump step.*left out.*does not solve"):
        result = heat_pump_retrofit.retrofit_with_heat_pump(
            object(), object(), RetrofitSettings(heat_pump_cycle="carnot")
        )

    assert result is base
    assert len(calls) == 2
