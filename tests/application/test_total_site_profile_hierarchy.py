"""Regression coverage for hierarchy-isolated Total Site process profiles."""

from __future__ import annotations

import numpy as np
import pytest

from OpenPinch import PinchProblem
from OpenPinch.domain.enums import GraphType, ProblemTableLabel, TargetType
from OpenPinch.domain.stream import Stream
from OpenPinch.domain.stream_collection import StreamCollection

ABS_DUTY_TOL = 0.2
REL_DUTY_TOL = 1e-6


def _curve_span(target, column: ProblemTableLabel) -> float:
    profile = target.graphs[GraphType.TSP.value][column]
    return float(np.max(profile) - np.min(profile))


def _stream_duty(streams: StreamCollection) -> float:
    return float(sum(float(stream.heat_flow[0]) for stream in streams))


def _assert_total_site_profile_duties(problem: PinchProblem) -> None:
    zone = problem.master_zone
    total_process = zone.targets[TargetType.TZ.value]
    total_site = zone.targets[TargetType.TS.value]

    assert _curve_span(total_site, ProblemTableLabel.H_COLD) == pytest.approx(
        total_process.hot_utility_target,
        rel=REL_DUTY_TOL,
        abs=ABS_DUTY_TOL,
    )
    assert _curve_span(total_site, ProblemTableLabel.H_HOT) == pytest.approx(
        total_process.cold_utility_target,
        rel=REL_DUTY_TOL,
        abs=ABS_DUTY_TOL,
    )
    assert _stream_duty(zone.subzone_net_cold_streams) == pytest.approx(
        total_process.hot_utility_target,
        rel=REL_DUTY_TOL,
        abs=ABS_DUTY_TOL,
    )
    assert _stream_duty(zone.subzone_net_hot_streams) == pytest.approx(
        total_process.cold_utility_target,
        rel=REL_DUTY_TOL,
        abs=ABS_DUTY_TOL,
    )


def test_pulp_mill_total_site_profiles_use_immediate_subzone_direct_targets():
    problem = PinchProblem("pulp_mill.json", project_name="Site")

    problem.target.all_heat_integration()

    total_site = problem.master_zone.targets[TargetType.TS.value]
    assert _curve_span(total_site, ProblemTableLabel.H_COLD) == pytest.approx(
        212431.388,
        rel=REL_DUTY_TOL,
        abs=ABS_DUTY_TOL,
    )
    assert _curve_span(total_site, ProblemTableLabel.H_HOT) == pytest.approx(
        115316.151,
        rel=REL_DUTY_TOL,
        abs=ABS_DUTY_TOL,
    )
    assert total_site.hot_utility_target == pytest.approx(180094.613)
    assert total_site.cold_utility_target == pytest.approx(82979.376)
    _assert_total_site_profile_duties(problem)

    for process_zone in problem.master_zone.subzones.values():
        direct = process_zone.targets[TargetType.DI.value]
        assert _stream_duty(process_zone.net_cold_streams) == pytest.approx(
            direct.hot_utility_target,
            rel=REL_DUTY_TOL,
            abs=ABS_DUTY_TOL,
        )
        assert _stream_duty(process_zone.net_hot_streams) == pytest.approx(
            direct.cold_utility_target,
            rel=REL_DUTY_TOL,
            abs=ABS_DUTY_TOL,
        )
        expected_child_hot_utility = sum(
            child.targets[TargetType.DI.value].hot_utility_target
            for child in process_zone.subzones.values()
        )
        expected_child_cold_utility = sum(
            child.targets[TargetType.DI.value].cold_utility_target
            for child in process_zone.subzones.values()
        )
        assert _stream_duty(process_zone.subzone_net_cold_streams) == pytest.approx(
            expected_child_hot_utility,
            rel=REL_DUTY_TOL,
            abs=ABS_DUTY_TOL,
        )
        assert _stream_duty(process_zone.subzone_net_hot_streams) == pytest.approx(
            expected_child_cold_utility,
            rel=REL_DUTY_TOL,
            abs=ABS_DUTY_TOL,
        )


def test_total_site_profiles_ignore_poisoned_child_zone_net_stream_state():
    problem = PinchProblem("pulp_mill.json", project_name="Site")
    problem.target.all_heat_integration()

    for process_zone in problem.master_zone.subzones.values():
        process_zone.net_hot_streams = StreamCollection(
            [
                Stream(
                    name="Poisoned hot profile",
                    supply_temperature=180.0,
                    target_temperature=80.0,
                    heat_flow=999999.0,
                )
            ]
        )
        process_zone.net_cold_streams = StreamCollection(
            [
                Stream(
                    name="Poisoned cold profile",
                    supply_temperature=60.0,
                    target_temperature=160.0,
                    heat_flow=888888.0,
                )
            ]
        )

    problem.target.total_site_heat_integration()

    _assert_total_site_profile_duties(problem)


def test_total_site_profile_targeting_is_idempotent_after_all_heat_integration():
    problem = PinchProblem("pulp_mill.json", project_name="Site")
    problem.target.all_heat_integration()
    original = problem.master_zone.targets[TargetType.TS.value]
    original_spans = (
        _curve_span(original, ProblemTableLabel.H_COLD),
        _curve_span(original, ProblemTableLabel.H_HOT),
    )

    repeated = problem.target.total_site_heat_integration()

    assert (
        _curve_span(repeated, ProblemTableLabel.H_COLD),
        _curve_span(repeated, ProblemTableLabel.H_HOT),
    ) == pytest.approx(original_spans, rel=REL_DUTY_TOL, abs=ABS_DUTY_TOL)


def test_pulp_mill_sugcc_preserves_hps_connector_and_lps_ledge():
    problem = PinchProblem("pulp_mill.json", project_name="Site")
    problem.target.all_heat_integration()

    graph = problem.plot.site_utility_grand_composite_curve(return_graph_data=True)
    segments = [segment["data_points"] for segment in graph["segments"]]

    assert any(
        len(points) == 2
        and points[0]["x"] == pytest.approx(27253.71, abs=ABS_DUTY_TOL)
        and points[1]["x"] == pytest.approx(27253.71, abs=ABS_DUTY_TOL)
        and points[0]["y"] == pytest.approx(279.0, abs=0.01)
        and points[1]["y"] == pytest.approx(138.5254, abs=0.01)
        for points in segments
    )
    assert any(
        points[0]["x"] == pytest.approx(27253.71, abs=ABS_DUTY_TOL)
        and points[0]["y"] == pytest.approx(138.5254, abs=0.01)
        and points[-1]["x"] == pytest.approx(11259.17, abs=ABS_DUTY_TOL)
        and points[-1]["y"] == pytest.approx(138.4254, abs=0.01)
        for points in segments
    )
