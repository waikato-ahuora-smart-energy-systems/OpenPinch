"""Regression tests for the refrigeration redesign (audit area 3c)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from OpenPinch.analysis.heat_pumps.common.preprocessing import (
    construct_HPRTargetInputs,
)
from OpenPinch.analysis.heat_pumps.common.shared import refrigeration_allowance
from OpenPinch.domain.configuration import Configuration
from OpenPinch.domain.stream import Stream
from OpenPinch.domain.stream_collection import StreamCollection


def _args(streams, Q_hpr_target, **overrides) -> SimpleNamespace:
    values = {
        "is_heat_pumping": False,
        "bckgrd_hot_streams": streams,
        "Q_hpr_target": Q_hpr_target,
        "T_hot": None,
        "refrigeration_allowance": None,
        "T_env": 15.0,
        "dt_env_cont": 10.0,
        "dtcont_hp": 0.0,
        "T_cooling_water": 20.0,
        "dt_cooling_water": 5.0,
        "period_idx": 0,
    }
    return SimpleNamespace(**(values | overrides))


def _cooling(*segments) -> StreamCollection:
    streams = StreamCollection()
    for name, supply, target, duty in segments:
        streams.add(
            Stream(
                name,
                supply_temperature=supply,
                target_temperature=target,
                heat_flow=duty,
            )
        )
    return streams


def test_warm_untargeted_cooling_needs_no_refrigeration_allowance():
    # The targeted coldest 40 kW is 12 -> 2 degC; the rest, 80 -> 40 degC, is
    # all above the 25 degC cooling-water level.
    cooling = _cooling(("Warm", 80.0, 40.0, 100.0), ("Cold", 12.0, 2.0, 40.0))
    args = _args(cooling, 40.0)

    assert refrigeration_allowance(args) == pytest.approx(0.0)


def test_sub_ambient_untargeted_cooling_is_allowed_for():
    # The targeted coldest 50 kW is 2 -> -8 degC; the rest, 12 -> 2 degC, is
    # below both cooling water and air (25 degC here): only default
    # refrigeration can serve it, so it is not the design's shortfall.
    cooling = _cooling(("Cold rest", 12.0, 2.0, 40.0), ("Coldest", 2.0, -8.0, 50.0))
    args = _args(cooling, 50.0)

    allowance = refrigeration_allowance(args)

    assert allowance == pytest.approx(40.0)
    assert args.refrigeration_allowance == pytest.approx(40.0)  # cached


def test_allowance_is_exact_when_the_split_falls_off_the_temperature_grid():
    # 300 kW released between 0 and -0.1 degC (3000 kW/K, like a utility
    # level). The coldest 112.3456789 kW ends at -0.0625514... degC, which the
    # problem table rounds; the allowance must still be the exact rest.
    Q_target = 112.3456789
    args = construct_HPRTargetInputs(
        Q_target,
        np.array([10.0, 0.0, -0.1]),
        np.array([0.0, 0.0, -300.0]),
        np.zeros(3),
        is_heat_pumping=False,
        config=Configuration(),
    )

    assert refrigeration_allowance(args) == pytest.approx(300.0 - Q_target, rel=1e-12)


def test_heat_pumping_has_no_refrigeration_allowance():
    args = _args(_cooling(("Cold rest", 12.0, 2.0, 40.0)), 40.0, is_heat_pumping=True)

    assert refrigeration_allowance(args) == 0.0
