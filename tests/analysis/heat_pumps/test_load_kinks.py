"""Load kinks of heat pump sink and refrigeration source net load profiles."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from OpenPinch.analysis.heat_pumps.common.load_kinks import find_hpr_load_kinks
from OpenPinch.contracts.hpr_load_kinks import HprLoadKinkKind

# Problem tables run from the hottest row down.
_T = np.array([100.0, 90.0, 80.0, 60.0, 40.0, 30.0])
_SINK = np.array([100.0, 60.0, 60.0, 40.0, 0.0, 0.0])


def _kinds(profile):
    return [(v.load_fraction, v.kind) for v in profile.vertices]


def test_sink_profile_walks_up_from_the_pinch_and_reports_every_vertex() -> None:
    profile = find_hpr_load_kinks(_T, _SINK, np.zeros_like(_T))

    assert profile.maximum_duty == pytest.approx(100.0, abs=1e-12)
    assert _kinds(profile) == [
        (pytest.approx(0.4, abs=1e-12), HprLoadKinkKind.STEEPENS),
        (pytest.approx(0.6, abs=1e-12), HprLoadKinkKind.STEP),
    ]
    steepens, step = profile.vertices
    assert steepens.duty == pytest.approx(40.0, abs=1e-12)
    assert steepens.temperature == pytest.approx(60.0, abs=1e-12)
    assert steepens.slope_before == pytest.approx(50.0, abs=1e-9)
    assert steepens.slope_after == pytest.approx(100.0, abs=1e-9)
    assert step.temperature == pytest.approx(80.0, abs=1e-12)
    assert step.slope_after is None
    assert profile.logical_load_fractions == pytest.approx((0.4, 0.6, 1.0), abs=1e-12)


def test_steepening_factor_sets_the_kink_threshold() -> None:
    profile = find_hpr_load_kinks(_T, _SINK, np.zeros_like(_T), steepening=2.5)

    assert profile.vertices[0].kind is HprLoadKinkKind.MINOR
    assert profile.logical_load_fractions == pytest.approx((0.6, 1.0), abs=1e-12)


def test_flattening_and_isothermal_vertices_are_classified() -> None:
    temperatures = np.array([120.0, 100.0, 100.0, 50.0])
    sink = np.array([100.0, 90.0, 50.0, 0.0])

    profile = find_hpr_load_kinks(temperatures, sink, np.zeros(4))

    assert _kinds(profile) == [
        (pytest.approx(0.5, abs=1e-12), HprLoadKinkKind.FLATTENS),
        (pytest.approx(0.9, abs=1e-12), HprLoadKinkKind.STEEPENS),
    ]
    assert profile.vertices[1].slope_before == pytest.approx(0.0, abs=1e-12)
    assert profile.logical_load_fractions == pytest.approx((0.9, 1.0), abs=1e-12)


def test_refrigeration_walks_the_source_profile_down_from_the_pinch() -> None:
    temperatures = np.array([100.0, 80.0, 60.0, 50.0, 40.0, 0.0])
    source = np.array([0.0, 0.0, -30.0, -30.0, -40.0, -100.0])

    profile = find_hpr_load_kinks(
        temperatures, np.zeros(6), source, is_refrigeration=True
    )

    assert profile.maximum_duty == pytest.approx(100.0, abs=1e-12)
    assert _kinds(profile) == [
        (pytest.approx(0.3, abs=1e-12), HprLoadKinkKind.STEP),
        (pytest.approx(0.4, abs=1e-12), HprLoadKinkKind.MINOR),
    ]
    assert profile.vertices[0].temperature == pytest.approx(60.0, abs=1e-12)
    assert profile.vertices[0].slope_before == pytest.approx(200.0 / 3.0, rel=1e-12)
    assert profile.logical_load_fractions == pytest.approx((0.3, 1.0), abs=1e-12)


def test_repeated_and_collinear_rows_do_not_create_kinks() -> None:
    temperatures = np.array([100.0, 100.0, 75.0, 75.0, 50.0, 50.0])
    sink = np.array([100.0, 100.0, 50.0, 50.0, 0.0, 0.0])

    profile = find_hpr_load_kinks(temperatures, sink, np.zeros(6))

    assert _kinds(profile) == [(pytest.approx(0.5, abs=1e-12), HprLoadKinkKind.MINOR)]
    assert profile.logical_load_fractions == (1.0,)


def test_empty_profile_and_invalid_steepening_raise() -> None:
    with pytest.raises(ValueError, match="no heat pump sink"):
        find_hpr_load_kinks(_T, np.zeros_like(_T), np.zeros_like(_T))
    with pytest.raises(ValueError, match="no refrigeration source"):
        find_hpr_load_kinks(_T, _SINK, np.zeros_like(_T), is_refrigeration=True)
    with pytest.raises(ValueError, match="greater than 1"):
        find_hpr_load_kinks(_T, _SINK, np.zeros_like(_T), steepening=1.0)


@st.composite
def _sink_profiles(draw):
    count = draw(st.integers(min_value=2, max_value=8))
    rises = draw(
        st.lists(
            st.floats(min_value=0.0, max_value=50.0), min_size=count, max_size=count
        )
    )
    duties = draw(
        st.lists(
            st.floats(min_value=0.0, max_value=500.0), min_size=count, max_size=count
        )
    )
    if sum(duties) <= 1.0:
        duties[0] = 100.0
    temperatures = 20.0 + np.concatenate(([0.0], np.cumsum(rises)))
    sink = np.concatenate(([0.0], np.cumsum(duties)))
    return temperatures[::-1], sink[::-1]


@given(_sink_profiles(), st.floats(min_value=1.1, max_value=5.0))
def test_vertices_are_ordered_and_logical_loads_follow_the_rule(
    case, steepening
) -> None:
    temperatures, sink = case

    profile = find_hpr_load_kinks(
        temperatures, sink, np.zeros_like(sink), steepening=steepening
    )
    scaled = find_hpr_load_kinks(
        temperatures, 3.0 * sink, np.zeros_like(sink), steepening=steepening
    )

    fractions = [v.load_fraction for v in profile.vertices]
    assert all(0.0 < f < 1.0 for f in fractions)
    assert fractions == sorted(set(fractions))
    assert profile.logical_load_fractions[-1] == 1.0
    for vertex in profile.vertices:
        if vertex.kind is HprLoadKinkKind.STEP:
            assert vertex.slope_after is None
        elif vertex.kind is HprLoadKinkKind.STEEPENS:
            assert vertex.slope_after >= steepening * vertex.slope_before
    np.testing.assert_allclose(
        [v.load_fraction for v in scaled.vertices], fractions, rtol=1e-9, atol=1e-12
    )
    assert [v.kind for v in scaled.vertices] == [v.kind for v in profile.vertices]
