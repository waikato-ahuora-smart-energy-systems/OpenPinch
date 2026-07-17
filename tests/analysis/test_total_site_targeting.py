"""Regression tests for indirect integration entry helpers."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

import OpenPinch.analysis.targeting.total_site as indirect
from OpenPinch.domain.configuration import Configuration
from OpenPinch.domain.enums import GT, TT, ZT
from OpenPinch.domain.enums import ProblemTableLabel as PT
from OpenPinch.domain.problem_table import ProblemTable
from OpenPinch.domain.stream import Stream
from OpenPinch.domain.stream_collection import StreamCollection
from OpenPinch.domain.zone import Zone
from tests.support.paths import FIXTURES_ROOT

FIXTURE_PATH = FIXTURES_ROOT / "indirect_integration_cases.json"


def _fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _utility_stream(spec: dict) -> Stream:
    return Stream(
        name=spec["name"],
        t_supply=spec["t_supply"],
        t_target=spec["t_target"],
        heat_flow=spec["heat_flow"],
        price=spec["price"],
        is_process_stream=False,
    )


def _utility_collection(specs: list[dict]) -> StreamCollection:
    collection = StreamCollection()
    for spec in specs:
        collection.add(_utility_stream(spec))
    return collection


def _target_utilities(fixture: dict, target_spec: dict) -> tuple[StreamCollection, ...]:
    hot_spec = {
        **fixture["base_utilities"]["hot"][0],
        "heat_flow": target_spec["hot_utility_heat_flow"],
    }
    cold_spec = {
        **fixture["base_utilities"]["cold"][0],
        "heat_flow": target_spec["cold_utility_heat_flow"],
    }
    return _utility_collection([hot_spec]), _utility_collection([cold_spec])


def test_compute_indirect_integration_targets_auto_aligns_utility_profile_grids(
    monkeypatch,
):
    zone = Zone(name="Plant", type=ZT.S.value, config=Configuration())
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

    calls = {"count": 0, "period_indices": []}

    def fake_get_process_heat_cascade(
        *, hot_streams, cold_streams, is_shifted, period_idx
    ):
        calls["count"] += 1
        calls["period_indices"].append(period_idx)
        return site_pt.copy if calls["count"] == 1 else utility_pt.copy

    monkeypatch.setattr(
        indirect, "get_process_heat_cascade", fake_get_process_heat_cascade
    )
    monkeypatch.setattr(
        indirect,
        "_match_utility_gen_and_use_at_same_level",
        lambda hot_utilities, cold_utilities, period_idx=0: (
            hot_utilities,
            cold_utilities,
        ),
    )
    monkeypatch.setattr(indirect, "_save_graph_data", lambda pt: {})
    monkeypatch.setattr(
        indirect,
        "_compute_utility_cost",
        lambda utilities, idx=None: 0.0,
    )

    target = indirect.compute_indirect_integration_targets(zone)

    assert calls["count"] == 2
    assert calls["period_indices"] == [0, 0]
    assert target.pt[PT.T].tolist() == [300.0, 200.0, 100.0]
    assert np.allclose(target.pt[PT.H_NET_UT], expected_h_net_ut)
    assert np.allclose(target.pt[PT.H_HOT_UT], utility_pt[PT.H_HOT])
    assert np.allclose(target.pt[PT.H_COLD_UT], expected_h_cold_ut)


def test_compute_total_subzone_utility_targets_sums_static_fixture_targets():
    fixture = _fixture()
    zone = Zone(name="Site", type=ZT.S.value, config=Configuration())
    zone.hot_utilities = _utility_collection(fixture["base_utilities"]["hot"])
    zone.cold_utilities = _utility_collection(fixture["base_utilities"]["cold"])
    zone.targets[TT.DI.value] = SimpleNamespace(heat_recovery_limit=100.0)

    for target_spec in fixture["subzone_targets"]:
        subzone = Zone(name=target_spec["name"], parent_zone=zone)
        hot_utilities, cold_utilities = _target_utilities(fixture, target_spec)
        subzone.targets[TT.DI.value] = SimpleNamespace(
            hot_utility_target=target_spec["hot_utility_target"],
            cold_utility_target=target_spec["cold_utility_target"],
            heat_recovery_target=target_spec["heat_recovery_target"],
            utility_cost=target_spec["utility_cost"],
            hot_utilities=hot_utilities,
            cold_utilities=cold_utilities,
        )
        zone.add_zone(subzone)

    target = indirect.compute_total_subzone_utility_targets(zone)

    assert target.hot_utility_target == pytest.approx(20.0)
    assert target.cold_utility_target == pytest.approx(10.0)
    assert target.heat_recovery_target == pytest.approx(35.0)
    assert target.degree_of_int == pytest.approx(0.35)
    assert target.utility_cost == pytest.approx(3.0)
    assert float(target.hot_utilities[0].heat_flow[0]) == pytest.approx(20.0)
    assert float(target.cold_utilities[0].heat_flow[0]) == pytest.approx(10.0)


def test_compute_total_subzone_utility_targets_handles_zero_recovery_limit():
    fixture = _fixture()
    zone = Zone(name="Site", type=ZT.S.value, config=Configuration())
    zone.hot_utilities = _utility_collection(fixture["base_utilities"]["hot"])
    zone.cold_utilities = _utility_collection(fixture["base_utilities"]["cold"])
    zone.targets[TT.DI.value] = SimpleNamespace(heat_recovery_limit=0.0)

    target = indirect.compute_total_subzone_utility_targets(zone)

    assert target.degree_of_int == pytest.approx(1.0)
    assert target.hot_utility_target == pytest.approx(0.0)
    assert target.cold_utility_target == pytest.approx(0.0)


def test_compute_indirect_integration_targets_returns_none_without_net_streams():
    fixture = _fixture()
    zone = Zone(name="Site", type=ZT.S.value, config=Configuration())
    zone.targets[TT.TZ.value] = SimpleNamespace(
        hot_utilities=_utility_collection(fixture["base_utilities"]["hot"]),
        cold_utilities=_utility_collection(fixture["base_utilities"]["cold"]),
        heat_recovery_target=0.0,
        hot_utility_target=0.0,
        heat_recovery_limit=0.0,
    )

    assert indirect.compute_indirect_integration_targets(zone) is None


def test_match_utility_generation_and_use_balances_same_temperature_level():
    fixture = _fixture()["matching_utilities"]
    hot_utilities = _utility_collection([fixture["hot"]])
    cold_utilities = _utility_collection([fixture["cold"], fixture["unmatched_cold"]])

    matched_hot, matched_cold = indirect._match_utility_gen_and_use_at_same_level(
        hot_utilities=hot_utilities,
        cold_utilities=cold_utilities,
        period_idx=0,
    )

    assert float(matched_hot[0].heat_flow[0]) == pytest.approx(2.0)
    assert float(matched_cold[0].heat_flow[0]) == pytest.approx(0.0)
    assert float(matched_cold[1].heat_flow[0]) == pytest.approx(2.0)


def test_indirect_helpers_compute_cost_shift_profiles_and_graph_slices():
    fixture = _fixture()["matching_utilities"]
    utilities = _utility_collection([fixture["hot"], fixture["cold"]])

    expected_cost = sum(utility.utility_cost[0] for utility in utilities)
    assert indirect._compute_utility_cost(utilities, idx=0) == pytest.approx(
        expected_cost
    )

    shifted = indirect._shift_site_process_profiles(
        T_col=np.array([300.0, 200.0]),
        H_hot=np.array([40.0, 10.0]),
        H_cold=np.array([12.0, 2.0]),
    )
    assert shifted["updates"][PT.H_HOT].tolist() == [0.0, -30.0]
    assert shifted["updates"][PT.H_COLD].tolist() == [10.0, 0.0]

    pt = ProblemTable(
        {
            PT.T: [100.12345, 80.98765],
            PT.H_HOT: [10.0, 2.0],
            PT.H_COLD: [6.0, 1.0],
            PT.H_HOT_UT: [3.0, 0.0],
            PT.H_COLD_UT: [0.0, -1.0],
            PT.H_NET_UT: [0.0, 2.0],
            PT.H_NET_HP: [1.0, 4.0],
        }
    )

    graphs = indirect._save_graph_data(pt)

    assert sorted(graphs) == sorted([GT.SUGCC.value, GT.TSP.value])
    assert graphs[GT.TSP.value][PT.T].tolist() == [100.1234, 80.9876]
    assert list(graphs[GT.SUGCC.value].columns) == [
        PT.T.value,
        PT.H_NET_UT.value,
        PT.H_NET_HP.value,
    ]
