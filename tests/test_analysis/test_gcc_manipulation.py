"""Regression tests for gcc manipulation analysis routines."""

import os

import numpy as np
import pandas as pd
import pytest

from OpenPinch.classes import *
from OpenPinch.lib import *
from OpenPinch.services.common.gcc_manipulation import *
from OpenPinch.utils.miscellaneous import *

"""Tests for gcc manipulation"""


def get_test_filenames():
    """Return test filenames used by this test module."""
    test_data_dir = os.path.dirname(__file__) + "/test_utility_targeting_data"
    return [
        filename
        for filename in os.listdir(test_data_dir)
        if filename.startswith("p_") and filename.endswith(".json")
    ]


"""Tests for get_GCC_without_pockets."""


def make_df(t_vals, h_net_vals) -> ProblemTable:
    """Build df data used by this test module."""
    return ProblemTable(
        {
            PT.T: pd.Series(t_vals, dtype="float64"),
            PT.H_NET: pd.Series(h_net_vals, dtype="float64"),
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
            values = pt.loc[h_idx + 1 : c_idx - 1, PT.H_NET_NP]
            first_val = values.iloc[0]
            assert all(abs(val - first_val) < 1e-6 for val in values), (
                f"Pocket not flattened: found values {list(values)} between pinches."
            )


def test_get_gcc_with_vertical_heat_transfer_clears_top_and_bottom():
    pt = ProblemTable(
        {
            PT.T: [400, 300, 200, 100],
            PT.H_COLD: [500, 300, 100, 50],
            PT.H_HOT: [250, 150, 100, 0],
            PT.H_NET: [250, 150, 0, 50],
        }
    )

    pt_copy = pt.copy
    result = get_GCC_with_vertical_heat_transfer(
        T_col=pt_copy[PT.T],
        h_cold=pt_copy[PT.H_COLD],
        h_hot=pt_copy[PT.H_HOT],
        h_net=pt_copy[PT.H_NET],
    )

    assert np.allclose(result["T_col"], pt_copy[PT.T])
    assert PT.H_NET_V in result["updates"]
    assert result["updates"][PT.H_NET_V][0] == 250  # Top value nonzero
    assert result["updates"][PT.H_NET_V][-1] == 50  # Bottom-out should zero


def test_get_gcc_with_partial_pockets_cuts_pocket_and_inserts_breakpoints():
    pt = ProblemTable(
        {
            PT.T: [200.0, 150.0, 100.0, 50.0, 0.0],
            PT.H_NET: [0.0, 50.0, 100.0, 50.0, 0.0],
            PT.H_NET_NP: [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )

    result = get_GCC_with_partial_pockets(
        T_col=pt[PT.T],
        h_net=pt[PT.H_NET],
        h_net_np=pt[PT.H_NET_NP],
        dt_cut=60.0,
    )

    assert np.allclose(result["T_col"], [200.0, 150.0, 130.0, 100.0, 70.0, 50.0, 0.0])
    assert np.allclose(
        result["updates"][PT.H_NET_PK],
        [0.0, 0.0, 0.0, 30.0, 0.0, 0.0, 0.0],
    )
    assert np.allclose(
        result["updates"][PT.H_NET_AI],
        [0.0, 50.0, 70.0, 70.0, 70.0, 50.0, 0.0],
    )


def test_get_additional_gccs_uses_assisted_profile_as_actual_when_enabled():
    pt = ProblemTable(
        {
            PT.T: [200.0, 150.0, 100.0, 50.0, 0.0],
            PT.H_NET: [0.0, 50.0, 100.0, 50.0, 0.0],
        }
    )

    result = get_additional_GCCs(
        pt,
        do_assisted_ht_calc=True,
        assisted_ht_dt_cut=60.0,
    )

    assert np.allclose(result[PT.T], [200.0, 150.0, 130.0, 100.0, 70.0, 50.0, 0.0])
    assert np.allclose(result[PT.H_NET_NP], np.zeros(7))
    assert np.allclose(result[PT.H_NET_A], result[PT.H_NET_AI])
    assert np.allclose(
        result[PT.H_NET_A],
        [0.0, 50.0, 70.0, 70.0, 70.0, 50.0, 0.0],
    )


def test_get_GCC_needing_utility_combines_columns():
    pt = ProblemTable({PT.H_NET_NP: [100, 200, 300]})

    result = get_GCC_needing_utility(
        T_col=np.array([300.0, 200.0, 100.0]),
        h_net=pt[PT.H_NET_NP],
    )

    expected = [100, 200, 300]
    assert (result["updates"][PT.H_NET_A] == expected).all()


def test_get_gcc_without_pockets_updates_pinch_after_insertions():
    pt = ProblemTable(
        {
            PT.T: [
                259.0,
                258.9,
                245.0,
                235.0,
                210.0,
                195.0,
                191.666667,
                185.0,
                145.0,
                99.0,
                85.0,
                75.0,
                35.0,
                25.0,
                11.1,
                11.0,
            ],
            PT.H_NET: [
                750.0,
                750.0,
                750.0,
                900.0,
                525.0,
                300.0,
                333.33333,
                400.0,
                0.0,
                920.0,
                1200.0,
                1400.0,
                1200.0,
                1000.0,
                1000.0,
                1000.0,
            ],
        }
    )

    result = get_GCC_without_pockets(pt)

    assert any(abs(t - 225.0) < 1e-6 for t in result[PT.T])
    assert any(abs(t - 175.0) < 1e-6 for t in result[PT.T])


def test_get_separated_heat_profiles_categorises_correctly():
    pt = ProblemTable(
        {
            PT.T: [400, 300, 200],
            PT.H_NET_A: [300, 0, 100],
            PT.RCP_UT_NET: [5, 10, 15],
        }
    )

    result_cols = get_seperated_gcc_heat_load_profiles(
        T_col=pt[PT.T],
        H_net=pt[PT.H_NET_A],
    )

    assert np.allclose(result_cols["T_col"], pt[PT.T])
    assert PT.H_NET_HOT in result_cols["updates"]
    assert PT.H_NET_COLD in result_cols["updates"]
    assert result_cols["updates"][PT.H_NET_HOT][-1] == -100
    assert result_cols["updates"][PT.H_NET_COLD][0] == 300
