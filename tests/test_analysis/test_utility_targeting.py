import json
import os

import pytest

from OpenPinch.analysis.data_preparation import prepare_problem
from OpenPinch.utils.miscellaneous import *
from OpenPinch.classes import *
from OpenPinch.lib import *

"""Tests for _target_utility"""
from OpenPinch.analysis.utility_targeting import _target_utility


def get_test_filenames():
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
    p_file_path = filepath + "/p_" + filename[2:]
    r_file_path = filepath + "/r_" + filename[2:]
    with open(p_file_path) as json_data:
        input_data = json.load(json_data)

    data = GetInputOutputData.model_validate(input_data)
    site = prepare_problem(
        streams=data.streams, utilities=data.utilities, options=data.options
    )
    for plant in data.plant_profile_data:
        z: Zone = site.get_subzone(plant.name)
        # z: Zone = site.subzones[plant.name]
        GHLP_P = ProblemTable({PT.T.value: plant.data.T})
        GHLP_P.col[PT.H_COLD_NET.value] = plant.data.H_cold_net
        GHLP_P.col[PT.H_HOT_NET.value] = plant.data.H_hot_net

        z.hot_utilities = _target_utility(
            z.hot_utilities, GHLP_P, PT.T.value, PT.H_COLD_NET.value
        )
        z.cold_utilities = _target_utility(
            z.cold_utilities, GHLP_P, PT.T.value, PT.H_HOT_NET.value
        )

    with open(r_file_path) as json_data:
        wkb_res = json.load(json_data)
    wkb_res = TargetOutput.model_validate(wkb_res)

    for z0 in wkb_res.targets:
        if "Direct Integration" in z0.name:
            z = site.get_subzone(z0.name.replace("/Direct Integration", ""))
            if 0:
                print("\n\nName:", z.name)

                for i in range(len(z.hot_utilities)):
                    print(
                        z.hot_utilities[i].name + ":",
                        round(get_value(z0.hot_utilities[i].heat_flow), 2),
                        z0.hot_utilities[i].name + ":",
                        round(get_value(z0.hot_utilities[i].heat_flow), 2),
                        sep="\t",
                    )
                for i in range(len(z.cold_utilities)):
                    print(
                        z.cold_utilities[i].name + ":",
                        round(get_value(z0.cold_utilities[i].heat_flow), 2),
                        z0.cold_utilities[i].name + ":",
                        round(get_value(z0.cold_utilities[i].heat_flow), 2),
                        sep="\t",
                    )

                print(
                    "dQh:",
                    round(get_value(z0.Qh), 2)
                    - round(sum([get_value(u.heat_flow) for u in z.hot_utilities]), 2),
                    "dQh:",
                    round(get_value(z0.Qh), 2)
                    - round(sum([get_value(u.heat_flow) for u in z0.hot_utilities]), 2),
                    sep="\t",
                )
                print(
                    "dQc:",
                    round(get_value(z0.Qc), 2)
                    - round(sum([get_value(u.heat_flow) for u in z.cold_utilities]), 2),
                    "dQc:",
                    round(get_value(z0.Qc), 2)
                    - round(
                        sum([get_value(u.heat_flow) for u in z0.cold_utilities]), 2
                    ),
                    sep="\t",
                )
            else:
                for i in range(len(z.hot_utilities)):
                    assert (
                        abs(
                            get_value(z.hot_utilities[i].heat_flow)
                            - get_value(z0.hot_utilities[i].heat_flow)
                        )
                        < 0.001
                    )
                for i in range(len(z.cold_utilities)):
                    assert (
                        abs(
                            get_value(z.cold_utilities[i].heat_flow)
                            - get_value(z0.cold_utilities[i].heat_flow)
                        )
                        < 0.001
                    )
    pass
