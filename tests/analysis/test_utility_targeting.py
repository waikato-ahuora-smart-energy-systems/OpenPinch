"""Regression tests for utility targeting analysis routines."""

import json
import os

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from OpenPinch.analysis.numerics import *
from OpenPinch.analysis.targeting.grand_composite import (
    get_seperated_gcc_heat_load_profiles,
)
from OpenPinch.analysis.targeting.utilities import (
    _assign_utility,
    _calculate_assigned_utility_duties,
    _maximise_utility_duty,
    target_utilities_for_load_profiles,
)
from OpenPinch.application._problem.input.construction import prepare_problem
from OpenPinch.contracts.input import TargetInput
from OpenPinch.contracts.output import TargetOutput
from OpenPinch.domain._value.resolution import get_scalar_value
from OpenPinch.domain.enums import (
    ProblemTableLabel,
    StreamLoc,
)
from OpenPinch.domain.problem_table import ProblemTable
from OpenPinch.domain.stream import Stream
from OpenPinch.domain.stream_collection import StreamCollection
from tests.support.paths import FIXTURES_ROOT

UTILITY_FIXTURE_ROOT = FIXTURES_ROOT / "utility_targeting"

"""Tests for target_utilities_for_load_profiles."""


def test_target_utilities_for_load_profiles_rejects_missing_required_utilities():
    with pytest.raises(ValueError, match="No hot utilities provided"):
        target_utilities_for_load_profiles(
            hot_utilities=StreamCollection(),
            cold_utilities=StreamCollection(),
            T_vals=np.array([200.0, 100.0]),
            H_net_cold=np.array([10.0, 0.0]),
            H_net_hot=np.array([0.0, 0.0]),
            pinch_idx=(0, 1),
        )

    with pytest.raises(ValueError, match="No cold utilities provided"):
        target_utilities_for_load_profiles(
            hot_utilities=StreamCollection(),
            cold_utilities=StreamCollection(),
            T_vals=np.array([200.0, 100.0]),
            H_net_cold=np.array([0.0, 0.0]),
            H_net_hot=np.array([0.0, 10.0]),
            pinch_idx=(0, 1),
        )


def test_assign_utility_rejects_non_vector_heat_segment():
    with pytest.raises(ValueError, match="Error in utility targeting"):
        _assign_utility(
            T_vals=np.array([200.0, 100.0]),
            H_vals=np.array([[0.0, 10.0], [20.0, 30.0]]),
            u_ls=StreamCollection(),
            pinch_row=1,
            is_hot_ut=True,
            is_real_temperatures=False,
            idx=None,
        )


def test_maximise_utility_duty_returns_zero_for_single_point_segment():
    assert (
        _maximise_utility_duty(
            T_segment=np.array([200.0]),
            H_segment=np.array([0.0]),
            Ts=210.0,
            Tt=190.0,
            is_hot_ut=True,
            Q_assigned=0.0,
        )
        == 0.0
    )


def test_sensible_hot_utility_duty_is_limited_by_interior_gcc_breakpoint():
    """Regression for the Process crossing observed in tutorial notebook 19."""
    temperatures = np.asarray([173.0, 168.0, 158.0, 97.0, 69.57, 62.96, 33.0])
    process_gcc = np.asarray([139.22, 135.2, 124.17, 38.52, 22.01, 18.03, 0.0])
    supply = 174.01
    target = 62.9556

    duty = _maximise_utility_duty(
        T_segment=temperatures,
        H_segment=process_gcc,
        Ts=supply,
        Tt=target,
        is_hot_ut=True,
        Q_assigned=0.0,
    )

    utility_profile = duty * np.clip(
        (temperatures - target) / (supply - target), 0.0, 1.0
    )
    assert duty == pytest.approx(125.65401322978228)
    assert np.all(utility_profile <= process_gcc + 1e-9)


@given(
    is_hot=st.booleans(),
    total=st.floats(min_value=10.0, max_value=200.0),
    interior_fraction=st.floats(min_value=0.05, max_value=0.4),
)
def test_sensible_utility_profile_never_crosses_gcc_breakpoints(
    is_hot: bool,
    total: float,
    interior_fraction: float,
) -> None:
    temperatures = np.asarray([100.0, 50.0, 0.0])
    interior = total * interior_fraction
    process_gcc = np.asarray(
        [total, interior, 0.0] if is_hot else [0.0, interior, total]
    )
    supply, target = (110.0, 0.0) if is_hot else (-10.0, 100.0)
    utility = Stream(
        "Sensible utility",
        supply,
        target,
        heat_flow=0.0,
        is_process_stream=False,
    )
    pure_duties = _calculate_assigned_utility_duties(
        T_vals=temperatures,
        H_vals=process_gcc,
        u_ls=StreamCollection([utility]),
        pinch_row=2 if is_hot else 0,
        is_hot_ut=is_hot,
        is_real_temperatures=True,
        idx=None,
    )
    assert float(utility.heat_flow.value) == 0.0

    targeted = _assign_utility(
        T_vals=temperatures,
        H_vals=process_gcc,
        u_ls=StreamCollection([utility]),
        pinch_row=2 if is_hot else 0,
        is_hot_ut=is_hot,
        is_real_temperatures=True,
        idx=None,
    )
    duty = float(targeted.get_stream_by_name("Sensible utility").heat_flow.value)
    assert pure_duties == pytest.approx((duty,), abs=1e-12)
    fractions = (
        np.clip((temperatures - target) / (supply - target), 0.0, 1.0)
        if is_hot
        else np.clip((target - temperatures) / (target - supply), 0.0, 1.0)
    )

    assert np.all(duty * fractions <= process_gcc + 1e-9)


def get_test_filenames():
    """Return test filenames used by this test module."""
    test_data_dir = UTILITY_FIXTURE_ROOT
    return [
        filename
        for filename in os.listdir(test_data_dir)
        if filename.startswith("p_") and filename.endswith(".json")
    ]


@pytest.mark.parametrize("filename", get_test_filenames())
def test_target_utility(filename):
    # Set the file path to the directory of this script
    p_file_path = UTILITY_FIXTURE_ROOT / filename
    r_file_path = UTILITY_FIXTURE_ROOT / f"r{filename[1:]}"
    with open(p_file_path) as json_data:
        input_data = json.load(json_data)
    data = TargetInput.model_validate(input_data)
    plant_profiles = input_data["plant_profile_data"]

    with open(r_file_path) as json_data:
        wkb_res = json.load(json_data)
    wkb_res = TargetOutput.model_validate(wkb_res)

    site = prepare_problem(
        streams=data.streams, utilities=data.utilities, options=data.options
    )

    for plant in plant_profiles:
        plant_name = plant["name"]
        plant_data = plant["data"]
        z = site.get_subzone(plant_name)
        pt = ProblemTable({ProblemTableLabel.T: plant_data["T"]})
        pt[ProblemTableLabel.H_NET] = plant_data["H_net"]
        pt.update(
            **get_seperated_gcc_heat_load_profiles(
                T_col=pt[ProblemTableLabel.T],
                H_net=pt[ProblemTableLabel.H_NET],
            )
        )
        z.hot_utilities, z.cold_utilities = target_utilities_for_load_profiles(
            hot_utilities=z.hot_utilities,
            cold_utilities=z.cold_utilities,
            T_vals=pt[ProblemTableLabel.T],
            H_net_cold=pt[ProblemTableLabel.H_NET_COLD],
            H_net_hot=pt[ProblemTableLabel.H_NET_HOT],
            pinch_idx=pt.pinch_idx(ProblemTableLabel.H_NET),
            is_real_temperatures=False,
        )

        t = None
        i = 0
        for t in wkb_res.targets:
            if f"Site/{plant_name}" == t.scope:
                break
            i += 1
        assert i < len(wkb_res.targets)

        for u in t.hot_utilities:
            s = z.hot_utilities[".".join([StreamLoc.HotU.value, u.name])]
            assert s is not None
            h_u = get_scalar_value(u.heat_flow)
            h_s = get_scalar_value(s.heat_flow)
            scalar = 1e-6 + max(h_s, h_u)
            assert abs(h_u - h_s) < 0.001 * scalar

        for u in t.cold_utilities:
            s = z.cold_utilities[".".join([StreamLoc.ColdU.value, u.name])]
            assert s is not None
            h_u = get_scalar_value(u.heat_flow)
            h_s = get_scalar_value(s.heat_flow)
            scalar = 1e-6 + max(h_s, h_u)
            assert abs(h_u - h_s) < 0.001 * scalar

        assert len(z.hot_utilities) == len(t.hot_utilities)
        assert len(z.cold_utilities) == len(t.cold_utilities)
    pass
