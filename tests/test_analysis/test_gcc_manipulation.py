import os

import pandas as pd, numpy as np
import pytest

from OpenPinch.utils.miscellaneous import *
from OpenPinch.classes import *
from OpenPinch.lib import *

"""Tests for gcc manipulation"""
from OpenPinch.analysis.gcc_manipulation import *


def get_test_filenames():
    test_data_dir = os.path.dirname(__file__) + "/test_utility_targeting_data"
    return [
        filename
        for filename in os.listdir(test_data_dir)
        if filename.startswith("p_") and filename.endswith(".json")
    ]


"""Tests for get_GCC_without_pockets."""

def make_df(t_vals, h_net_vals) -> ProblemTable:
    return ProblemTable(
        {
            PT.T.value: pd.Series(t_vals, dtype="float64"),
            PT.H_NET.value: pd.Series(h_net_vals, dtype="float64"),
        }
    )


@pytest.mark.parametrize(
    "t_vals, h_net_vals, expected_middle, expect_flat_pocket",
    [
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
    ],
)
def test_get_gcc_without_pockets(
    t_vals, h_net_vals, expected_middle, expect_flat_pocket
):
    pt = make_df(t_vals, h_net_vals)
    result = get_GCC_without_pockets(pt)

    # Structure
    assert isinstance(result, ProblemTable)
    assert PT.H_NET_NP.value in result.columns
    assert PT.T.value in result.columns
    assert len(pt) >= len(t_vals)

    # Pocket flattening check
    if expect_flat_pocket:
        # Get pinch points
        h_idx, c_idx, valid = pt.pinch_idx()

        if valid and h_idx + 1 < c_idx:
            values = pt.loc[h_idx + 1 : c_idx - 1, PT.H_NET_NP.value]
            first_val = values.iloc[0]
            assert all(
                abs(val - first_val) < 1e-6 for val in values
            ), f"Pocket not flattened: found values {list(values)} between pinches."


def test_get_gcc_with_vertical_heat_transfer_clears_top_and_bottom():
    pt = ProblemTable(
        {
            PT.T.value: [400, 300, 200, 100],
            PT.H_COLD.value: [500, 300, 100, 50],
            PT.H_HOT.value: [250, 150, 100, 0],
            PT.H_NET.value: [250, 150, 0, 50],
        }
    )

    pt_copy = pt.copy
    result = get_GCC_with_vertical_heat_transfer(
        pt_copy.col[PT.H_COLD.value],
        pt_copy.col[PT.H_HOT.value],
        pt_copy.col[PT.H_NET.value],
    )

    assert PT.H_NET_V.value in result
    assert result[PT.H_NET_V.value][0] == 250  # Top value nonzero
    assert result[PT.H_NET_V.value][-1] == 50  # Bottom-out should zero


def test_get_GCC_needing_utility_combines_columns():
    pt = ProblemTable(
        {PT.H_NET_NP.value: [100, 200, 300]}
    )

    result = get_GCC_needing_utility(pt.col[PT.H_NET_NP.value])

    expected = [100, 200, 300] 
    assert (result[PT.H_NET_A.value] == expected).all()


def test_get_separated_heat_profiles_categorises_correctly():
    pt = ProblemTable(
        {
            PT.T.value: [400, 300, 200],
            PT.H_NET_A.value: [300, 0, 100],
            PT.RCP_UT_NET.value: [5, 10, 15],
        }
    )

    result_cols = get_seperated_gcc_heat_load_profiles(pt.copy)

    assert PT.H_HOT_NET.value in result_cols
    assert PT.H_COLD_NET.value in result_cols
    assert result_cols[PT.H_HOT_NET.value][-1] == -100
    assert result_cols[PT.H_COLD_NET.value][0] == 300
