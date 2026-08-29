"""Regression tests for direct integration entry analysis routines."""

from types import SimpleNamespace

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

import OpenPinch.analysis.targeting.direct as direct
from OpenPinch.analysis.targeting.direct import (
    _add_net_segment_period,
    _create_net_hot_and_cold_stream_collections_for_site_analysis,
    _find_next_available_utility,
    _initialise_utility_index,
    build_area_cost_target_data,
    should_update_balanced_composite_curves,
)
from OpenPinch.analysis.targeting.utilities import _assign_utility
from OpenPinch.domain.configuration import tol
from OpenPinch.domain.enums import GraphType, ProblemTableLabel, ZoneType
from OpenPinch.domain.problem_table import ProblemTable
from OpenPinch.domain.stream import Stream
from OpenPinch.domain.stream_collection import StreamCollection


def _balanced_problem_table():
    return ProblemTable(
        {
            ProblemTableLabel.T: [300.0, 200.0],
            ProblemTableLabel.H_HOT_BAL: [100.0, 0.0],
            ProblemTableLabel.H_COLD_BAL: [100.0, 0.0],
            ProblemTableLabel.R_HOT_BAL: [0.1, 0.1],
            ProblemTableLabel.R_COLD_BAL: [0.2, 0.2],
        }
    )


def test_balanced_curve_predicate_and_area_cost_payload(monkeypatch):
    direct_config = SimpleNamespace(balanced_cc_enabled=False)
    assert should_update_balanced_composite_curves(direct_config, False) is False
    assert should_update_balanced_composite_curves(direct_config, True) is True

    zone = SimpleNamespace(
        config=SimpleNamespace(),
        hot_streams=StreamCollection(),
        cold_streams=StreamCollection(),
        hot_utilities=StreamCollection(),
        cold_utilities=StreamCollection(),
    )
    monkeypatch.setattr(direct, "get_min_number_hx", lambda **_kwargs: 3)
    monkeypatch.setattr(direct, "get_area_targets", lambda **_kwargs: 120.0)
    monkeypatch.setattr(
        direct,
        "get_capital_cost_targets",
        lambda **_kwargs: ("capital", "annual"),
    )

    payload = build_area_cost_target_data(
        _balanced_problem_table(),
        _balanced_problem_table(),
        zone,
        idx=None,
        enabled=True,
    )

    assert payload == {
        "area": 120.0,
        "num_units": 3,
        "capital_cost": "capital",
        "total_cost": "annual",
    }

    assert (
        build_area_cost_target_data(
            _balanced_problem_table(),
            _balanced_problem_table(),
            zone,
            idx=None,
            enabled=False,
        )
        == {}
    )


def test_compute_direct_integration_targets_uses_empty_area_payload_when_disabled(
    monkeypatch,
):
    table = ProblemTable(
        {
            ProblemTableLabel.T: [300.0, 200.0],
            ProblemTableLabel.H_NET: [100.0, 0.0],
            ProblemTableLabel.H_NET_A: [100.0, 0.0],
            ProblemTableLabel.H_HOT: [100.0, 0.0],
            ProblemTableLabel.H_COLD: [0.0, 100.0],
            ProblemTableLabel.H_HOT_UT: [0.0, 0.0],
            ProblemTableLabel.H_COLD_UT: [0.0, 0.0],
        }
    )
    zone = SimpleNamespace(
        name="Area",
        address="Area",
        type=ZoneType.P.value,
        parent_zone=None,
        period_ids=None,
        all_streams=StreamCollection(),
        hot_streams=StreamCollection(),
        cold_streams=StreamCollection(),
        hot_utilities=StreamCollection(),
        cold_utilities=StreamCollection(),
        config=SimpleNamespace(
            direct=SimpleNamespace(
                vertical_gcc_enabled=False,
                assisted_ht_enabled=False,
                assisted_ht_dt=0.0,
                balanced_cc_enabled=False,
            ),
            targeting=SimpleNamespace(area_cost_enabled=False),
        ),
    )

    monkeypatch.setattr(direct, "get_process_heat_cascade", lambda **_kwargs: table)
    monkeypatch.setattr(direct, "get_heat_recovery_target_from_pt", lambda _pt: 0.0)
    monkeypatch.setattr(direct, "get_additional_GCCs", lambda pt, **_kwargs: pt)
    monkeypatch.setattr(direct, "get_utility_targets", lambda **_kwargs: None)
    monkeypatch.setattr(
        direct,
        "_create_net_hot_and_cold_stream_collections_for_site_analysis",
        lambda **_kwargs: (StreamCollection(), StreamCollection()),
    )
    monkeypatch.setattr(
        direct,
        "set_zonal_targets",
        lambda **_kwargs: {
            "hot_utility_target": 0.0,
            "cold_utility_target": 0.0,
            "heat_recovery_target": 0.0,
        },
    )
    monkeypatch.setattr(direct, "_save_graph_data", lambda *_args: {})
    monkeypatch.setattr(
        direct.DirectIntegrationTarget,
        "model_validate",
        staticmethod(lambda data: data),
    )

    result = direct.compute_direct_integration_targets(zone)

    assert "area" not in result
    assert result["zone_name"] == "Area"


def test_compute_direct_integration_targets_uses_unshifted_real_table_for_limit(
    monkeypatch,
):
    shifted = ProblemTable(
        {
            ProblemTableLabel.T: [300.0, 200.0],
            ProblemTableLabel.H_NET: [20.0, 50.0],
            ProblemTableLabel.H_NET_A: [20.0, 50.0],
            ProblemTableLabel.H_HOT: [100.0, 0.0],
            ProblemTableLabel.H_COLD: [0.0, 50.0],
            ProblemTableLabel.H_HOT_UT: [0.0, 0.0],
            ProblemTableLabel.H_COLD_UT: [0.0, 0.0],
        }
    )
    real_limit = ProblemTable(
        {
            ProblemTableLabel.T: [300.0, 200.0],
            ProblemTableLabel.H_NET: [0.0, 0.0],
            ProblemTableLabel.H_NET_A: [0.0, 0.0],
            ProblemTableLabel.H_HOT: [100.0, 0.0],
            ProblemTableLabel.H_COLD: [0.0, 100.0],
            ProblemTableLabel.H_HOT_UT: [0.0, 0.0],
            ProblemTableLabel.H_COLD_UT: [0.0, 0.0],
        }
    )
    real_aligned = ProblemTable(
        {
            ProblemTableLabel.T: [300.0, 200.0],
            ProblemTableLabel.H_NET: [50.0, 50.0],
            ProblemTableLabel.H_NET_A: [50.0, 50.0],
            ProblemTableLabel.H_HOT: [100.0, 0.0],
            ProblemTableLabel.H_COLD: [50.0, 150.0],
            ProblemTableLabel.H_HOT_UT: [0.0, 0.0],
            ProblemTableLabel.H_COLD_UT: [0.0, 0.0],
        }
    )

    def fake_cascade(**kwargs):
        if kwargs["is_shifted"]:
            return shifted
        if "known_heat_recovery" in kwargs:
            return real_aligned
        return real_limit

    zone = SimpleNamespace(
        name="Area",
        address="Area",
        type=ZoneType.P.value,
        parent_zone=None,
        period_ids=None,
        all_streams=StreamCollection(),
        hot_streams=StreamCollection(),
        cold_streams=StreamCollection(),
        hot_utilities=StreamCollection(),
        cold_utilities=StreamCollection(),
        config=SimpleNamespace(
            direct=SimpleNamespace(
                vertical_gcc_enabled=False,
                assisted_ht_enabled=False,
                assisted_ht_dt=0.0,
                balanced_cc_enabled=False,
            ),
            targeting=SimpleNamespace(area_cost_enabled=False),
        ),
    )
    monkeypatch.setattr(direct, "get_process_heat_cascade", fake_cascade)
    monkeypatch.setattr(direct, "get_additional_GCCs", lambda pt, **_kwargs: pt)
    monkeypatch.setattr(direct, "get_utility_targets", lambda **_kwargs: None)
    monkeypatch.setattr(
        direct,
        "_create_net_hot_and_cold_stream_collections_for_site_analysis",
        lambda **_kwargs: (StreamCollection(), StreamCollection()),
    )
    monkeypatch.setattr(direct, "_save_graph_data", lambda *_args: {})
    monkeypatch.setattr(
        direct.DirectIntegrationTarget,
        "model_validate",
        staticmethod(lambda data: data),
    )

    result = direct.compute_direct_integration_targets(zone)

    assert result["heat_recovery_target"] == pytest.approx(50.0)
    assert result["heat_recovery_limit"] == pytest.approx(100.0)
    assert result["degree_of_int"] == pytest.approx(0.5)
    assert result["pt_real"] is real_aligned


def test_save_graph_data_rounds_graph_copies_without_mutating_problem_tables():
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
    pt_before = pt.data.copy()
    pt_real_before = pt_real.data.copy()

    graphs = direct._save_graph_data(pt, pt_real)

    np.testing.assert_array_equal(pt.data, pt_before)
    np.testing.assert_array_equal(pt_real.data, pt_real_before)
    assert graphs[GraphType.GCC.value][ProblemTableLabel.T].tolist() == [
        200.0,
        200.0,
        100.0,
    ]
    assert graphs[GraphType.GCC.value][ProblemTableLabel.H_NET_A].tolist() == [
        10.0001,
        0.0,
        5.0,
    ]


def test_initialise_utility_index_returns_first_available():
    u1 = Stream("U1", 200, 250, heat_flow=0.0)
    u2 = Stream("U2", 200, 250, heat_flow=75.0)
    idx = _initialise_utility_index([u1, u2], [u1.heat_flow, u2.heat_flow])
    assert idx == 1


def test_utility_targeting_respects_maximum_heat_flow_before_fallback() -> None:
    capped_hot = Stream(
        "Capped steam",
        250.0,
        200.0,
        heat_flow=0.0,
        maximum_heat_flow=30.0,
        is_process_stream=False,
    )
    fallback_hot = Stream(
        "HU",
        300.0,
        299.99,
        heat_flow=0.0,
        is_process_stream=False,
    )
    hot = _assign_utility(
        T_vals=np.asarray([250.0, 150.0, 50.0]),
        H_vals=np.asarray([100.0, 50.0, 0.0]),
        u_ls=StreamCollection([fallback_hot, capped_hot]),
        pinch_row=2,
        is_hot_ut=True,
        is_real_temperatures=False,
        idx=None,
    )

    assert hot.get_stream_by_name("Capped steam").heat_flow.value == pytest.approx(
        30.0
    )
    assert hot.get_stream_by_name("HU").heat_flow.value == pytest.approx(70.0)


@given(cap=st.floats(min_value=0.0, max_value=100.0))
def test_utility_targeting_capacity_property(cap) -> None:
    named = Stream(
        "Named",
        300.0,
        299.99,
        heat_flow=0.0,
        maximum_heat_flow=cap,
        is_process_stream=False,
    )
    fallback = Stream(
        "HU",
        310.0,
        309.99,
        heat_flow=0.0,
        is_process_stream=False,
    )
    result = _assign_utility(
        T_vals=np.asarray([250.0, 150.0, 50.0]),
        H_vals=np.asarray([100.0, 50.0, 0.0]),
        u_ls=StreamCollection([fallback, named]),
        pinch_row=2,
        is_hot_ut=True,
        is_real_temperatures=False,
        idx=None,
    )

    named_duty = float(result.get_stream_by_name("Named").heat_flow.value)
    fallback_duty = float(result.get_stream_by_name("HU").heat_flow.value)
    assert named_duty <= cap + 1e-9
    assert named_duty + fallback_duty == pytest.approx(100.0)


def test_add_net_segment_period_consumes_residuals():
    u1 = Stream("U1", 200, 250, heat_flow=120.0)
    u2 = Stream("U2", 200, 250, heat_flow=80.0)
    residuals = [u1.heat_flow[0], u2.heat_flow[0]]
    net_streams = StreamCollection()
    next_idx, next_k = _add_net_segment_period(
        400,
        300,
        0,
        150,
        [u1, u2],
        residuals,
        net_streams,
        k=1,
    )
    assert next_idx == 1
    assert next_k == 2
    assert abs(residuals[0]) < tol
    assert abs(residuals[1] - 50) < tol
    assert len(net_streams) == 2
    assert net_streams[0].name == "Segment 1"
    assert net_streams[1].name == "Segment 1-1"


def test_net_utility_helpers_cover_empty_invalid_and_zero_residual_paths():
    assert _find_next_available_utility(0, StreamCollection(), []) == -1
    assert _add_net_segment_period(
        400.0,
        300.0,
        -1,
        100.0,
        StreamCollection(),
        [],
        StreamCollection(),
        k=1,
    ) == (-1, 1)

    utility = Stream("U1", 200, 250, heat_flow=50.0)
    with pytest.raises(ValueError, match="Infeasible temperature interval"):
        _create_net_hot_and_cold_stream_collections_for_site_analysis(
            T_vals=np.array([300.0, 300.0]),
            H_vals=np.array([0.0, 50.0]),
            hot_utilities=StreamCollection([utility]),
            cold_utilities=StreamCollection(),
        )


def test_add_net_segment_period_skips_depleted_current_utility():
    u1 = Stream("U1", 200, 250, heat_flow=0.0)
    u2 = Stream("U2", 200, 250, heat_flow=80.0)
    residuals = [0.0, 80.0]
    net_streams = StreamCollection()

    next_idx, next_k = _add_net_segment_period(
        400.0,
        300.0,
        0,
        40.0,
        [u1, u2],
        residuals,
        net_streams,
        k=1,
    )

    assert next_idx == 1
    assert next_k == 2
    assert residuals == [0.0, 40.0]
    assert len(net_streams) == 1
