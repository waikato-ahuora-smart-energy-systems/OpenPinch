"""Regression tests for indirect integration entry helpers."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

import OpenPinch.analysis.targeting.direct as direct
import OpenPinch.analysis.targeting.indirect as indirect
from OpenPinch.domain.configuration import Configuration
from OpenPinch.domain.enums import GraphType, TargetType, ZoneType
from OpenPinch.domain.enums import ProblemTableLabel as ProblemTableLabel
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
        supply_temperature=spec["t_supply"],
        target_temperature=spec["t_target"],
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


@st.composite
def _direct_target_profile_specs(draw):
    profile_count = draw(st.integers(min_value=1, max_value=5))
    duties = st.floats(
        min_value=1.0,
        max_value=10000.0,
        allow_nan=False,
        allow_infinity=False,
    )
    return [(draw(duties), draw(duties)) for _index in range(profile_count)]


@given(_direct_target_profile_specs())
def test_reconstructed_subzone_direct_profiles_conserve_utility_duties(
    profile_specs,
):
    zone = Zone(name="Site", type=ZoneType.S.value, config=Configuration())
    expected_hot_utility = 0.0
    expected_cold_utility = 0.0

    for index, (hot_utility_duty, cold_utility_duty) in enumerate(profile_specs):
        subzone = Zone(name=f"Process {index}", parent_zone=zone)
        hot_utilities = StreamCollection(
            [
                Stream(
                    name="HU",
                    supply_temperature=250.0,
                    target_temperature=300.0,
                    heat_flow=hot_utility_duty,
                    is_process_stream=False,
                )
            ]
        )
        cold_utilities = StreamCollection(
            [
                Stream(
                    name="CU",
                    supply_temperature=50.0,
                    target_temperature=20.0,
                    heat_flow=cold_utility_duty,
                    is_process_stream=False,
                )
            ]
        )
        subzone.targets[TargetType.DI.value] = SimpleNamespace(
            pt=ProblemTable(
                {
                    ProblemTableLabel.T: [300.0, 150.0, 20.0],
                    ProblemTableLabel.H_NET_A: [
                        hot_utility_duty,
                        0.0,
                        cold_utility_duty,
                    ],
                }
            ),
            hot_utilities=hot_utilities,
            cold_utilities=cold_utilities,
            period_idx=0,
        )
        subzone.net_hot_streams.add(
            Stream("Poison hot", 200.0, 100.0, heat_flow=999999.0)
        )
        subzone.net_cold_streams.add(
            Stream("Poison cold", 80.0, 180.0, heat_flow=888888.0)
        )
        zone.add_zone(subzone)
        expected_hot_utility += hot_utility_duty
        expected_cold_utility += cold_utility_duty

    net_hot_streams, net_cold_streams = indirect._reconstruct_subzone_direct_profiles(
        zone
    )

    assert net_hot_streams is zone.subzone_net_hot_streams
    assert net_cold_streams is zone.subzone_net_cold_streams
    assert sum(float(stream.heat_flow[0]) for stream in net_cold_streams) == (
        pytest.approx(expected_hot_utility)
    )
    assert sum(float(stream.heat_flow[0]) for stream in net_hot_streams) == (
        pytest.approx(expected_cold_utility)
    )
    assert all(float(stream.heat_flow[0]) < 999999.0 for stream in net_hot_streams)
    assert all(float(stream.heat_flow[0]) < 888888.0 for stream in net_cold_streams)
    assert all(
        float(stream.heat_flow[0]) == 999999.0
        for subzone in zone.subzones.values()
        for stream in subzone.net_hot_streams
    )
    assert all(
        float(stream.heat_flow[0]) == 888888.0
        for subzone in zone.subzones.values()
        for stream in subzone.net_cold_streams
    )


def test_reconstructed_profiles_retain_sub_four_decimal_cascade_precision():
    zone = Zone(name="Site", type=ZoneType.S.value, config=Configuration())
    subzone = Zone(name="Fine Process", parent_zone=zone)
    temperatures = [200.00004, 200.00001, 100.00003]
    enthalpies = [10.00006, 0.0, 5.00004]
    pt = ProblemTable(
        {
            ProblemTableLabel.T: temperatures,
            ProblemTableLabel.H_NET_A: enthalpies,
        }
    )
    pt_real = ProblemTable(
        {
            ProblemTableLabel.T: temperatures,
            ProblemTableLabel.H_NET: enthalpies,
        }
    )
    hot_utilities = StreamCollection(
        [
            Stream(
                name="HU",
                supply_temperature=300.0,
                target_temperature=250.0,
                heat_flow=10.00006,
                is_process_stream=False,
            )
        ]
    )
    cold_utilities = StreamCollection(
        [
            Stream(
                name="CU",
                supply_temperature=20.0,
                target_temperature=50.0,
                heat_flow=5.00004,
                is_process_stream=False,
            )
        ]
    )
    direct._save_graph_data(pt, pt_real)
    subzone.targets[TargetType.DI.value] = SimpleNamespace(
        pt=pt,
        hot_utilities=hot_utilities,
        cold_utilities=cold_utilities,
        period_idx=0,
    )
    zone.add_zone(subzone)

    net_hot_streams, net_cold_streams = indirect._reconstruct_subzone_direct_profiles(
        zone
    )

    assert len(net_hot_streams) == len(net_cold_streams) == 1
    assert float(net_cold_streams[0].heat_flow[0]) == pytest.approx(10.00006)
    assert float(net_cold_streams[0].supply_temperature[0]) == pytest.approx(200.00001)
    assert float(net_cold_streams[0].target_temperature[0]) == pytest.approx(200.00004)
    assert float(net_hot_streams[0].heat_flow[0]) == pytest.approx(5.00004)
    assert float(net_hot_streams[0].supply_temperature[0]) == pytest.approx(200.00001)
    assert float(net_hot_streams[0].target_temperature[0]) == pytest.approx(100.00003)


def test_compute_indirect_integration_targets_auto_aligns_utility_profile_grids(
    monkeypatch,
):
    zone = Zone(name="Plant", type=ZoneType.S.value, config=Configuration())
    zone.targets[TargetType.SA.value] = SimpleNamespace(
        hot_utilities=StreamCollection(),
        cold_utilities=StreamCollection(),
        heat_recovery_target=25.0,
        hot_utility_target=10.0,
        heat_recovery_limit=50.0,
    )
    zone.net_hot_streams.add(
        Stream(
            name="NetHot",
            supply_temperature=300.0,
            target_temperature=100.0,
            heat_flow=20.0,
        )
    )

    site_pt = ProblemTable(
        {
            ProblemTableLabel.T: [300.0, 100.0],
            ProblemTableLabel.H_HOT: [30.0, 10.0],
            ProblemTableLabel.H_COLD: [5.0, 0.0],
        }
    )
    utility_pt = ProblemTable(
        {
            ProblemTableLabel.T: [300.0, 200.0, 100.0],
            ProblemTableLabel.H_HOT: [40.0, 25.0, 10.0],
            ProblemTableLabel.H_COLD: [8.0, 4.0, 1.0],
        }
    )
    expected_h_net_ut = (
        utility_pt[ProblemTableLabel.H_HOT] - utility_pt[ProblemTableLabel.H_COLD]
    )
    expected_h_net_ut = expected_h_net_ut - expected_h_net_ut.min()
    expected_h_cold_ut = (
        utility_pt[ProblemTableLabel.H_COLD]
        - utility_pt[ProblemTableLabel.H_COLD].max()
    )

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
        "_target_site_utilities",
        lambda hot_utilities, cold_utilities, idx=None: (
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
    assert target.pt[ProblemTableLabel.T].tolist() == [300.0, 200.0, 100.0]
    assert np.allclose(target.pt[ProblemTableLabel.H_NET_UT], expected_h_net_ut)
    assert np.allclose(
        target.pt[ProblemTableLabel.H_HOT_UT], utility_pt[ProblemTableLabel.H_HOT]
    )
    assert np.allclose(target.pt[ProblemTableLabel.H_COLD_UT], expected_h_cold_ut)


def test_compute_subzone_aggregate_target_sums_static_fixture_targets():
    fixture = _fixture()
    zone = Zone(name="Site", type=ZoneType.S.value, config=Configuration())
    zone.hot_utilities = _utility_collection(fixture["base_utilities"]["hot"])
    zone.cold_utilities = _utility_collection(fixture["base_utilities"]["cold"])
    zone.targets[TargetType.DI.value] = SimpleNamespace(heat_recovery_limit=100.0)

    for target_spec in fixture["subzone_targets"]:
        subzone = Zone(name=target_spec["name"], parent_zone=zone)
        hot_utilities, cold_utilities = _target_utilities(fixture, target_spec)
        subzone.targets[TargetType.DI.value] = SimpleNamespace(
            hot_utility_target=target_spec["hot_utility_target"],
            cold_utility_target=target_spec["cold_utility_target"],
            heat_recovery_target=target_spec["heat_recovery_target"],
            utility_cost=target_spec["utility_cost"],
            hot_utilities=hot_utilities,
            cold_utilities=cold_utilities,
        )
        zone.add_zone(subzone)

    target = indirect.compute_subzone_aggregate_target(zone)

    assert target.hot_utility_target == pytest.approx(20.0)
    assert target.cold_utility_target == pytest.approx(10.0)
    assert target.heat_recovery_target == pytest.approx(35.0)
    assert target.degree_of_int == pytest.approx(0.35)
    assert target.utility_cost == pytest.approx(3.0)
    assert float(target.hot_utilities[0].heat_flow[0]) == pytest.approx(20.0)
    assert float(target.cold_utilities[0].heat_flow[0]) == pytest.approx(10.0)


def test_compute_subzone_aggregate_target_handles_zero_recovery_limit():
    fixture = _fixture()
    zone = Zone(name="Site", type=ZoneType.S.value, config=Configuration())
    zone.hot_utilities = _utility_collection(fixture["base_utilities"]["hot"])
    zone.cold_utilities = _utility_collection(fixture["base_utilities"]["cold"])
    zone.targets[TargetType.DI.value] = SimpleNamespace(heat_recovery_limit=0.0)

    target = indirect.compute_subzone_aggregate_target(zone)

    assert target.degree_of_int == pytest.approx(1.0)
    assert target.hot_utility_target == pytest.approx(0.0)
    assert target.cold_utility_target == pytest.approx(0.0)


def test_compute_indirect_integration_targets_returns_none_without_net_streams():
    fixture = _fixture()
    zone = Zone(name="Site", type=ZoneType.S.value, config=Configuration())
    zone.targets[TargetType.SA.value] = SimpleNamespace(
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
    assert shifted["updates"][ProblemTableLabel.H_HOT].tolist() == [0.0, -30.0]
    assert shifted["updates"][ProblemTableLabel.H_COLD].tolist() == [10.0, 0.0]

    pt = ProblemTable(
        {
            ProblemTableLabel.T: [100.12345, 80.98765],
            ProblemTableLabel.H_HOT: [10.0, 2.0],
            ProblemTableLabel.H_COLD: [6.0, 1.0],
            ProblemTableLabel.H_HOT_UT: [3.0, 0.0],
            ProblemTableLabel.H_COLD_UT: [0.0, -1.0],
            ProblemTableLabel.H_NET_UT: [0.0, 2.0],
            ProblemTableLabel.H_NET_HP: [1.0, 4.0],
        }
    )

    graphs = indirect._save_graph_data(pt)

    assert sorted(graphs) == sorted([GraphType.SUGCC.value, GraphType.TSP.value])
    assert graphs[GraphType.TSP.value][ProblemTableLabel.T].tolist() == [
        100.1234,
        80.9876,
    ]
    assert list(graphs[GraphType.SUGCC.value].columns) == [
        ProblemTableLabel.T.value,
        ProblemTableLabel.H_NET_UT.value,
        ProblemTableLabel.H_NET_HP.value,
    ]


def _cascaded_site_problem():
    """Zone A generates 400 kW LPS; zone B uses 300 kW HW below it.

    The site cascade lets the LPS surplus supply the hot water, so the site
    needs no hot utility and rejects 100 kW.
    """
    from OpenPinch import PinchProblem

    def utility(name, kind, t_supply, t_target):
        return {
            "name": name,
            "type": kind,
            "t_supply": t_supply,
            "t_target": t_target,
            "dt_cont": 0.0,
            "price": 10.0,
        }

    def stream(zone, name, t_supply, t_target, heat_flow):
        return {
            "zone": zone,
            "name": name,
            "t_supply": t_supply,
            "t_target": t_target,
            "heat_flow": heat_flow,
            "dt_cont": 5.0,
        }

    return PinchProblem(
        {
            "streams": [
                stream("A", "h1", 200.0, 160.0, 400.0),
                stream("B", "c1", 40.0, 60.0, 300.0),
            ],
            "utilities": [
                utility("HPS", "Hot", 250.0, 249.0),
                utility("LPS", "Both", 150.0, 149.0),
                utility("HW", "Both", 90.0, 70.0),
                utility("CW", "Cold", 20.0, 25.0),
            ],
        },
        project_name="Site",
    )


def test_total_site_utility_lists_and_cost_follow_the_site_cascade():
    problem = _cascaded_site_problem()
    problem.target.all_heat_integration()
    target = problem.master_zone.targets[TargetType.II.value]

    hot_total = sum(float(u.heat_flow[0]) for u in target.hot_utilities)
    cold_total = sum(float(u.heat_flow[0]) for u in target.cold_utilities)

    assert target.hot_utility_target == pytest.approx(0.0, abs=1e-6)
    assert target.cold_utility_target == pytest.approx(100.0, rel=1e-9)
    assert hot_total == pytest.approx(target.hot_utility_target, abs=1e-6)
    assert cold_total == pytest.approx(target.cold_utility_target, rel=1e-9)
    # 100 kW at 10 $/MWh.
    assert target.utility_cost == pytest.approx(1.0, rel=1e-9)


@pytest.mark.parametrize("case", ["zonal_site", "Four-stream-Yee-and-Grossmann-1990-1"])
def test_sample_total_site_utility_lists_sum_to_the_site_targets(case):
    from OpenPinch import PinchProblem

    problem = PinchProblem(
        f"OpenPinch/tutorials/sample_cases/{case}.json", project_name="Site"
    )
    problem.target.all_heat_integration()
    target = problem.master_zone.targets[TargetType.II.value]

    hot = [float(u.heat_flow[0]) for u in target.hot_utilities]
    cold = [float(u.heat_flow[0]) for u in target.cold_utilities]
    scale = max(target.hot_utility_target, target.cold_utility_target)

    assert sum(hot) == pytest.approx(target.hot_utility_target, abs=1e-9 * scale)
    assert sum(cold) == pytest.approx(target.cold_utility_target, abs=1e-9 * scale)
    if target.hot_utility_target == pytest.approx(0.0, abs=1e-9 * scale):
        assert all(duty == pytest.approx(0.0, abs=1e-9 * scale) for duty in hot)
