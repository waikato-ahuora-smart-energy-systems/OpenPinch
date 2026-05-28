"""Regression tests for indirect integration entry helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from OpenPinch.classes.problem_table import ProblemTable
from OpenPinch.classes.stream import Stream
from OpenPinch.classes.stream_collection import StreamCollection
from OpenPinch.classes.zone import Zone
from OpenPinch.lib.config import Configuration
from OpenPinch.lib.enums import TT, ZT
from OpenPinch.lib.enums import ProblemTableLabel as PT
from OpenPinch.services.indirect_heat_integration import (
    indirect_integration_entry as indirect,
)


def test_compute_indirect_integration_targets_auto_aligns_utility_profile_grids(
    monkeypatch,
):
    zone = Zone(name="Plant", type=ZT.S.value, zone_config=Configuration())
    zone.targets[TT.TZ.value] = SimpleNamespace(
        hot_utilities=StreamCollection(),
        cold_utilities=StreamCollection(),
        heat_recovery_target=25.0,
        hot_utility_target=10.0,
        heat_recovery_limit=50.0,
    )
    zone.net_hot_streams.add(
        Stream(name="NetHot", t_supply=300.0, t_target=100.0, heat_flow=20.0)
    )

    site_pt = ProblemTable(
        {
            PT.T: [300.0, 100.0],
            PT.H_HOT: [30.0, 10.0],
            PT.H_COLD: [5.0, 0.0],
        }
    )
    utility_pt = ProblemTable(
        {
            PT.T: [300.0, 200.0, 100.0],
            PT.H_HOT: [40.0, 25.0, 10.0],
            PT.H_COLD: [8.0, 4.0, 1.0],
        }
    )
    expected_h_net_ut = utility_pt[PT.H_HOT] - utility_pt[PT.H_COLD]
    expected_h_net_ut = expected_h_net_ut - expected_h_net_ut.min()
    expected_h_cold_ut = utility_pt[PT.H_COLD] - utility_pt[PT.H_COLD].max()

    calls = {"count": 0}

    def fake_get_process_heat_cascade(*args, **kwargs):
        calls["count"] += 1
        return site_pt.copy if calls["count"] == 1 else utility_pt.copy

    monkeypatch.setattr(
        indirect, "get_process_heat_cascade", fake_get_process_heat_cascade
    )
    monkeypatch.setattr(
        indirect,
        "_match_utility_gen_and_use_at_same_level",
        lambda hot_utilities, cold_utilities: (hot_utilities, cold_utilities),
    )
    monkeypatch.setattr(indirect, "_save_graph_data", lambda pt: {})
    monkeypatch.setattr(
        indirect,
        "_compute_utility_cost",
        lambda hot_utilities, cold_utilities: 0.0,
    )

    target = indirect.compute_indirect_integration_targets(zone)

    assert calls["count"] == 2
    assert target.pt[PT.T].tolist() == [300.0, 200.0, 100.0]
    assert np.allclose(target.pt[PT.H_NET_UT], expected_h_net_ut)
    assert np.allclose(target.pt[PT.H_HOT_UT], utility_pt[PT.H_HOT])
    assert np.allclose(target.pt[PT.H_COLD_UT], expected_h_cold_ut)
