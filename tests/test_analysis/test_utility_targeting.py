"""Regression tests for utility targeting analysis routines."""

import json
import os

import pytest

from OpenPinch.classes import *
from OpenPinch.lib import *
from OpenPinch.services.common.gcc_manipulation import (
    get_seperated_gcc_heat_load_profiles,
)
from OpenPinch.services.common.utility_targeting import _target_utility
from OpenPinch.services.input_data_processing.data_preparation import prepare_problem
from OpenPinch.utils.miscellaneous import *

"""Tests for _target_utility"""


def get_test_filenames():
    """Return test filenames used by this test module."""
    test_data_dir = os.path.dirname(__file__) + "/test_utility_targeting_data"
    return [
        filename
        for filename in os.listdir(test_data_dir)
        if filename.startswith("p_") and filename.endswith(".json")
    ]


@pytest.mark.parametrize("filename", get_test_filenames())
def test_target_utility(filename):
    # Set the file path to the directory of this script
    filepath = os.path.dirname(__file__) + "/test_utility_targeting_data"
    p_file_path = filepath + "/" + filename
    r_file_path = filepath + "/r" + filename[1:]
    with open(p_file_path) as json_data:
        input_data = json.load(json_data)
    data = GetInputOutputData.model_validate(input_data)

    with open(r_file_path) as json_data:
        wkb_res = json.load(json_data)
    wkb_res = TargetOutput.model_validate(wkb_res)

    site = prepare_problem(
        streams=data.streams, utilities=data.utilities, options=data.options
    )

    for plant in data.plant_profile_data:
        z = site.get_subzone(plant.name)
        pt = ProblemTable({PT.T: plant.data.T})
        pt[PT.H_NET] = plant.data.H_net
        pt.update(
            **get_seperated_gcc_heat_load_profiles(
                T_col=pt[PT.T],
                H_net=pt[PT.H_NET],
            )
        )
        z.hot_utilities, z.cold_utilities = _target_utility(
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
            if plant.name == t.name.replace("/" + TT.DI.value, ""):
                break
            i += 1
        assert i < len(wkb_res.targets)

        for u in t.hot_utilities:
            s = z.hot_utilities[".".join([StreamLoc.HotU.value, u.name])]
            assert s is not None
            h_u = get_value(u.heat_flow)
            h_s = get_value(s.heat_flow)
            scalar = 1e-6 + max(h_s, h_u)
            assert abs(h_u - h_s) < 0.001 * scalar

        for u in t.cold_utilities:
            s = z.cold_utilities[".".join([StreamLoc.ColdU.value, u.name])]
            assert s is not None
            h_u = get_value(u.heat_flow)
            h_s = get_value(s.heat_flow)
            scalar = 1e-6 + max(h_s, h_u)
            assert abs(h_u - h_s) < 0.001 * scalar

        assert len(z.hot_utilities) == len(t.hot_utilities)
        assert len(z.cold_utilities) == len(t.cold_utilities)
    pass
