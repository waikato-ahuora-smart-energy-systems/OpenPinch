"""Regression tests for utility targeting analysis routines."""

import json
import os

import numpy as np
import pytest

from OpenPinch.analysis.numerics import *
from OpenPinch.analysis.targeting.grand_composite import (
    get_seperated_gcc_heat_load_profiles,
)
from OpenPinch.analysis.targeting.utilities import (
    _assign_utility,
    _maximise_utility_duty,
    target_utilities_for_load_profiles,
)
from OpenPinch.application._problem.input.construction import prepare_problem
from OpenPinch.contracts.input import TargetInput
from OpenPinch.contracts.output import TargetOutput
from OpenPinch.domain._value.resolution import get_scalar_value
from OpenPinch.domain.enums import (
    PT,
    TT,
    StreamLoc,
)
from OpenPinch.domain.problem_table import ProblemTable
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
        pt = ProblemTable({PT.T: plant_data["T"]})
        pt[PT.H_NET] = plant_data["H_net"]
        pt.update(
            **get_seperated_gcc_heat_load_profiles(
                T_col=pt[PT.T],
                H_net=pt[PT.H_NET],
            )
        )
        z.hot_utilities, z.cold_utilities = target_utilities_for_load_profiles(
            hot_utilities=z.hot_utilities,
            cold_utilities=z.cold_utilities,
            T_vals=pt[PT.T],
            H_net_cold=pt[PT.H_NET_COLD],
            H_net_hot=pt[PT.H_NET_HOT],
            pinch_idx=pt.pinch_idx(PT.H_NET),
            is_real_temperatures=False,
        )

        t = None
        i = 0
        for t in wkb_res.targets:
            if plant_name == t.name.replace("/" + TT.DI.value, ""):
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
