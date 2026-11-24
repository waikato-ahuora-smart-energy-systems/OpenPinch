"""A heat pump designed once over each period's source and sink profiles.

``compute_shared_profile_heat_pump_target`` is the profile target of
several periods: one machine whose design serves every period's profiles,
each period's load its own sink's, weighted by the period's share of the
year.
"""

from __future__ import annotations

import numpy as np
import pytest

from OpenPinch.analysis.heat_pumps import service
from OpenPinch.domain.configuration import Configuration

T_VALS = np.array([150.0, 100.0, 50.0])
PROFILES = {
    "base": (T_VALS, np.array([0.0, 0.0, -80.0]), np.array([120.0, 0.0, 0.0])),
    "peak": (T_VALS, np.array([0.0, 0.0, -60.0]), np.array([150.0, 0.0, 0.0])),
}


def test_each_period_is_a_case_at_its_own_sinks_load(monkeypatch):
    captured = {}

    def solve(**kwargs):
        captured.update(kwargs)
        return "target"

    monkeypatch.setattr(service, "get_multiperiod_hpr_targets", solve)

    result = service.compute_shared_profile_heat_pump_target(
        profiles=PROFILES,
        period_ids={"base": 0, "peak": 1},
        weights={"base": 0.25, "peak": 0.75},
        config=Configuration(),
    )

    assert result == "target"
    cases = captured["period_cases"]
    assert [case.period_id for case in cases] == ["base", "peak"]
    assert [case.weight for case in cases] == [0.25, 0.75]
    assert [case.solver_case.args.Q_hpr_target for case in cases] == pytest.approx(
        [120.0, 150.0]
    )


def test_a_period_without_load_runs_no_machine_and_keeps_its_share(monkeypatch):
    # As for a zone's shared design: a period with no sink is no case, and
    # its weight still counts in the weighted operation.
    captured = {}

    def solve(**kwargs):
        captured.update(kwargs)
        return "target"

    monkeypatch.setattr(service, "get_multiperiod_hpr_targets", solve)
    idle = (T_VALS, np.array([0.0, 0.0, -60.0]), np.zeros(3))

    service.compute_shared_profile_heat_pump_target(
        profiles={"base": idle, "peak": PROFILES["peak"]},
        period_ids={"base": 0, "peak": 1},
        weights={"base": 0.25, "peak": 0.75},
        config=Configuration(),
    )

    assert [case.period_id for case in captured["period_cases"]] == ["peak"]
    assert captured["selected_period_id"] == "peak"
    assert captured["idle_period_weight"] == pytest.approx(0.25)


def test_no_load_in_any_period_is_no_target(monkeypatch):
    # As a single period's profile target: no load, no machine.
    solved = []
    monkeypatch.setattr(
        service, "get_multiperiod_hpr_targets", lambda **kwargs: solved.append(kwargs)
    )
    idle = (T_VALS, np.array([0.0, 0.0, -60.0]), np.zeros(3))

    result = service.compute_shared_profile_heat_pump_target(
        profiles={"base": idle, "peak": idle},
        period_ids={"base": 0, "peak": 1},
        weights={"base": 0.5, "peak": 0.5},
        config=Configuration(),
    )

    assert result is None
    assert solved == []
