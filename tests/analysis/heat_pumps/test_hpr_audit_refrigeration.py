"""Regression tests for the refrigeration redesign (audit area 3c)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from OpenPinch.analysis.heat_pumps.common.shared import refrigeration_allowance
from OpenPinch.domain.stream import Stream
from OpenPinch.domain.stream_collection import StreamCollection


def _args(streams, **overrides) -> SimpleNamespace:
    values = {
        "is_heat_pumping": False,
        "untargeted_cooling_streams": streams,
        "refrigeration_allowance": None,
        "T_env": 15.0,
        "dt_env_cont": 10.0,
        "dtcont_hp": 0.0,
        "T_cooling_water": 20.0,
        "dt_cooling_water": 5.0,
        "period_idx": 0,
    }
    return SimpleNamespace(**(values | overrides))


def _cooling(name, supply, target, duty) -> StreamCollection:
    streams = StreamCollection()
    streams.add(
        Stream(name, supply_temperature=supply, target_temperature=target, heat_flow=duty)
    )
    return streams


def test_warm_untargeted_cooling_needs_no_refrigeration_allowance():
    # 80 -> 40 degC is all above the 25 degC cooling-water level.
    args = _args(_cooling("Warm", 80.0, 40.0, 100.0))

    assert refrigeration_allowance(args) == pytest.approx(0.0)


def test_sub_ambient_untargeted_cooling_is_allowed_for():
    # 12 -> 2 degC is below both cooling water and air (25 degC here): only
    # default refrigeration can serve it, so it is not the design's shortfall.
    args = _args(_cooling("Cold rest", 12.0, 2.0, 40.0))

    allowance = refrigeration_allowance(args)

    assert allowance == pytest.approx(40.0)
    assert args.refrigeration_allowance == pytest.approx(40.0)  # cached


def test_heat_pumping_has_no_refrigeration_allowance():
    args = _args(_cooling("Cold rest", 12.0, 2.0, 40.0), is_heat_pumping=True)

    assert refrigeration_allowance(args) == 0.0
