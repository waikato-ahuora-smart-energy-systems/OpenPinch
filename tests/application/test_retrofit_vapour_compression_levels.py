"""The vapour compression retrofit targets 2 condensers and 1 evaporator.

Unless the problem's options or the retrofit's name the levels. On a
low-temperature retrofit at the retrofit's default search, the zone's
default 3 condensers and 2 evaporators found no machine at any size in
85 s; 2 and 1 found an 858 kW machine netting 181,000 $/y in 79 s, and
1 and 1 a 379 kW machine netting 90,000 $/y in 97 s.
"""

from __future__ import annotations

from tests.analysis.heat_exchanger_networks.heat_pump_levels_case import (
    EXISTING,
    levels_problem,
)


def _levels_seen(monkeypatch, problem, **arguments):
    from OpenPinch.application import heat_pump_retrofit

    seen = []

    def streams(case, existing, settings, **_):
        seen.append(
            (case.master_zone.config.hpr.n_cond, case.master_zone.config.hpr.n_evap)
        )
        return object()

    monkeypatch.setattr(heat_pump_retrofit, "retrofit_with_heat_pump", streams)
    problem.design.retrofit(EXISTING, heat_pump_cycle="vapour_compression", **arguments)
    return seen


def test_vapour_compression_retrofit_defaults_to_two_condensers_one_evaporator(
    monkeypatch,
):
    assert _levels_seen(monkeypatch, levels_problem()) == [(2, 1)]


def test_levels_in_the_problem_options_are_kept(monkeypatch):
    problem = levels_problem()
    problem.update_options({"HPR_N_COND": 3})

    assert _levels_seen(monkeypatch, problem) == [(3, 1)]


def test_levels_in_the_retrofit_options_are_kept(monkeypatch):
    seen = _levels_seen(monkeypatch, levels_problem(), options={"HPR_N_EVAP": 2})

    assert seen == [(2, 2)]
