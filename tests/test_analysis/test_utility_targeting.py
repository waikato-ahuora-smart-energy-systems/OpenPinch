import pytest
import os, json
import pandas as pd
from OpenPinch.lib import * 
from OpenPinch.analysis.support_methods import *
from OpenPinch.classes import Zone
from OpenPinch.analysis.data_preparation import prepare_problem_struture


"""Tests for target_utility"""
from OpenPinch.analysis.utility_targeting import target_utility

def get_test_filenames():
    test_data_dir = os.path.dirname(__file__) + '/test_utility_targeting_data'
    return [
        filename
        for filename in os.listdir(test_data_dir)
        if filename.startswith("p_") and filename.endswith(".json")
    ]

@pytest.mark.parametrize("filename", get_test_filenames())
def test_target_utility(filename):
    # Set the file path to the directory of this script
    filepath = os.path.dirname(__file__) + '/test_utility_targeting_data'
    p_file_path = filepath + '/p_' + filename[2:]
    r_file_path = filepath + '/r_' + filename[2:]
    with open(p_file_path) as json_data:
        input_data = json.load(json_data)
    
    data = GetInputOutputData.model_validate(input_data)
    site = prepare_problem_struture(streams=data.streams, utilities=data.utilities, options=data.options)
    for plant in data.plant_profile_data:
        z: Zone = site.get_subzone(plant.name)
        # z: Zone = site.subzones[plant.name]
        GHLP_P = ProblemTable({PT.T.value: plant.data.T})
        GHLP_P.col[PT.H_COLD_NET.value] = plant.data.H_cold_net
        GHLP_P.col[PT.H_HOT_NET.value] = plant.data.H_hot_net

        z.hot_utilities = target_utility(z.hot_utilities, GHLP_P, PT.T.value, PT.H_COLD_NET.value)
        z.cold_utilities = target_utility(z.cold_utilities, GHLP_P, PT.T.value, PT.H_HOT_NET.value)

    with open(r_file_path) as json_data:
        wkb_res = json.load(json_data)
    wkb_res = TargetOutput.model_validate(wkb_res)
    
    for z0 in wkb_res.targets:
        if "Direct Integration" in z0.name:
            z = site.get_subzone(z0.name.replace("/Direct Integration", ''))
            if 0:
                print('\n\nName:', z.name)
                
                for i in range(len(z.hot_utilities)):
                    print(z.hot_utilities[i].name + ':', round(get_value(z0.hot_utilities[i].heat_flow), 2), z0.hot_utilities[i].name + ':', round(get_value(z0.hot_utilities[i].heat_flow), 2), sep='\t')
                for i in range(len(z.cold_utilities)):
                    print(z.cold_utilities[i].name + ':', round(get_value(z0.cold_utilities[i].heat_flow), 2), z0.cold_utilities[i].name + ':', round(get_value(z0.cold_utilities[i].heat_flow), 2), sep='\t')

                print('dQh:', round(get_value(z0.Qh), 2) - round(sum([get_value(u.heat_flow) for u in z.hot_utilities]), 2), 'dQh:', round(get_value(z0.Qh), 2) - round(sum([get_value(u.heat_flow) for u in z0.hot_utilities]), 2), sep='\t')
                print('dQc:', round(get_value(z0.Qc), 2) - round(sum([get_value(u.heat_flow) for u in z.cold_utilities]), 2), 'dQc:', round(get_value(z0.Qc), 2) - round(sum([get_value(u.heat_flow) for u in z0.cold_utilities]), 2), sep='\t')
            else:
                for i in range(len(z.hot_utilities)):
                    assert(abs(get_value(z.hot_utilities[i].heat_flow) - get_value(z0.hot_utilities[i].heat_flow)) < 0.001)
                for i in range(len(z.cold_utilities)):
                    assert(abs(get_value(z.cold_utilities[i].heat_flow) - get_value(z0.cold_utilities[i].heat_flow)) < 0.001)
    pass

"""Tests for _calc_GCC_without_pockets."""
from OpenPinch.analysis.utility_targeting import _calc_GCC_without_pockets

def make_df(t_vals, h_net_vals) -> ProblemTable:
    return ProblemTable({
        PT.T.value: pd.Series(t_vals, dtype="float64"),
        PT.H_NET.value: pd.Series(h_net_vals, dtype="float64"),
    })

@pytest.mark.parametrize("t_vals, h_net_vals, expected_middle, expect_flat_pocket", [
    # Pocket should be flattened
    ([300, 250, 200, 150, 100], [500, 600, 550, 0, 100], 195.45454545454544, True),
    # Increasing = no flatten
    ([300, 250, 200, 150, 100], [100, 200, 300, 400, 500], 300, False),
    # Decreasing = no flatten
    ([300, 250, 200, 150, 100], [500, 400, 300, 200, 100], 300, False),
    # Flat already
    ([300, 250, 200, 150, 100], [100, 100, 100, 100, 100], 100, False),
    # Zeros
    ([300, 250, 200, 150, 100], [0, 0, 0, 0, 0], 0, False),
])
def test_calc_gcc_without_pockets(t_vals, h_net_vals, expected_middle, expect_flat_pocket):
    pt = make_df(t_vals, h_net_vals)
    pt_updated = _calc_GCC_without_pockets(pt)

    # Structure
    assert isinstance(pt_updated, ProblemTable)
    assert PT.H_NET_NP.value in pt_updated.columns
    assert PT.T.value in pt_updated.columns
    assert len(pt_updated) >= len(t_vals)

    # Pocket flattening check
    if expect_flat_pocket:
        # Get pinch points
        from OpenPinch.analysis.support_methods import get_pinch_loc
        h_idx, c_idx, valid = get_pinch_loc(pt_updated)

        if valid and h_idx + 1 < c_idx:
            values = pt_updated.loc[h_idx + 1 : c_idx - 1, PT.H_NET_NP.value]
            first_val = values.iloc[0]
            assert all(abs(val - first_val) < 1e-6 for val in values), (
                f"Pocket not flattened: found values {list(values)} between pinches."
            )


from OpenPinch.analysis.utility_targeting import _calc_GCC_with_vertical_heat_transfer

def test_calc_gcc_with_vertical_heat_transfer_clears_top_and_bottom():
    pt = ProblemTable({
        PT.T.value: [400, 300, 200, 100],
        PT.H_COLD.value: [500, 300, 100, 50],
        PT.H_HOT.value: [250, 150, 100, 0],
        PT.H_NET.value: [250, 150, 0, 50]
    })

    result = _calc_GCC_with_vertical_heat_transfer(pt.copy)
    
    assert (PT.H_NET_V.value in result.columns)
    assert result.loc[0, PT.H_NET_V.value] == 250  # Top value nonzero
    assert result.loc[-1, PT.H_NET_V.value] == 50  # Bottom-out should zero


from OpenPinch.analysis.utility_targeting import _calc_GCC_actual

def test_calc_gcc_actual_combines_columns():
    pt = ProblemTable({
        PT.H_NET_NP.value: [100, 200, 300],
        PT.H_NET_V.value: [300, 200, 100]
    })

    result = _calc_GCC_actual(pt.copy)

    expected = [100, 200, 300]  # Because f_horizontal = 1.0
    assert result[PT.H_NET_A.value].to_list() == expected


from OpenPinch.analysis.utility_targeting import _calc_seperated_heat_load_profiles

def test_calc_separated_heat_profiles_categorises_correctly():
    pt = ProblemTable({
        PT.T.value: [400, 300, 200],
        PT.H_NET_A.value: [300, 0, 100],
        PT.RCP_UT_NET.value: [5, 10, 15]
    })

    result = _calc_seperated_heat_load_profiles(pt.copy)

    assert PT.H_HOT_NET.value in result.columns
    assert PT.H_COLD_NET.value in result.columns
    assert result.loc[-1, PT.H_HOT_NET.value] == -100
    assert result.loc[0, PT.H_COLD_NET.value] == 300

