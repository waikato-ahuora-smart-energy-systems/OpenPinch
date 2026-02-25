import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import uuid

from OpenPinch.classes import *
from OpenPinch.lib import *


# Core helper function
def make_stream(name, t_supply, t_target, dt, cp=0, htc=0):
    return Stream(
        name,
        t_supply,
        t_target,
        dt,
        abs(cp * (t_supply - t_target)),
        htc if htc > 0 else 1,
    )


def _make_problem_table_for_interval_tests():
    data = {
        PT.T.value: [300.0, 200.0, 100.0],
        PT.DELTA_T.value: [0.0, 100.0, 100.0],
        PT.CP_HOT.value: [0.0, 1.0, 0.5],
        PT.DELTA_H_HOT.value: [0.0, 100.0, 50.0],
        PT.CP_COLD.value: [0.0, 0.5, 1.0],
        PT.DELTA_H_COLD.value: [0.0, 50.0, 100.0],
        PT.CP_NET.value: [0.0, -0.5, 0.5],
        PT.DELTA_H_NET.value: [0.0, -50.0, 50.0],
        PT.H_HOT.value: [200.0, 100.0, 50.0],
        PT.H_COLD.value: [20.0, 10.0, 5.0],
        PT.H_NET.value: [50.0, 0.0, 50.0],
        PT.H_NET_NP.value: [50.0, 0.0, 50.0],
        PT.H_NET_A.value: [50.0, 0.0, 50.0],
        PT.H_NET_V.value: [50.0, 0.0, 50.0],
    }
    return ProblemTable(data)


"""Tests for the _sum_mcp_between_temperature_boundaries function."""
from OpenPinch.analysis.problem_table_analysis import (
    _sum_mcp_between_temperature_boundaries,
)


def test_no_overlap_streams_are_skipped():
    T = [300, 200, 100]
    hot = [make_stream("H1", 400, 500, 2)]
    cold = [make_stream("C1", 10, 80, 1.5)]

    CP_hot, _ = _sum_mcp_between_temperature_boundaries(
        T, hot, is_shifted=False
    )
    CP_cold, _ = _sum_mcp_between_temperature_boundaries(
        T, cold, is_shifted=False
    )

    assert all(m == 0 for m in CP_hot)
    assert all(m == 0 for m in CP_cold)

def test_export_writes_to_results_dir_and_uses_filename_stem():
    table = ProblemTable({PT.T.value: [100], PT.H_NET.value: [50]})
    unique_name = f"problem_table_test_{uuid.uuid4().hex}"
    expected_dir = Path(__file__).resolve().parents[2] / "results"
    output_path = table.export(unique_name)

    try:
        assert output_path.parent == expected_dir
        assert output_path.name == f"{unique_name}.xlsx"
        assert output_path.is_file()
    finally:
        if output_path.exists():
            output_path.unlink()


def test_stream_with_zero_cp_does_not_affect_result():
    T = [300, 200, 100]
    hot = [make_stream("H1", 150, 250, 0)]
    cold = [make_stream("C1", 100, 200, 0)]

    CP_hot, rCP_hot = _sum_mcp_between_temperature_boundaries(
        T, hot, is_shifted=False
    )
    CP_cold, rCP_cold = _sum_mcp_between_temperature_boundaries(
        T, cold, is_shifted=False
    )

    assert CP_hot == [0, 0, 0]
    assert CP_cold == [0, 0, 0]
    assert rCP_hot == [0, 0, 0]
    assert rCP_cold == [0, 0, 0]


def test_stream_on_boundary_is_included():
    T = [300, 200, 100]
    hot = [make_stream("H1", 300, 200, 0, 1)]
    cold = [make_stream("C1", 100, 200, 0, 1)]

    CP_hot, _ = _sum_mcp_between_temperature_boundaries(
        T, hot, is_shifted=False
    )
    CP_cold, _ = _sum_mcp_between_temperature_boundaries(
        T, cold, is_shifted=False
    )

    assert CP_hot == [0, 1, 0]
    assert CP_cold == [0, 0, 1]


def test_stream_spanning_multiple_intervals():
    T = [300, 250, 200, 150]
    hot = [make_stream("H1", 160, 290, 0, 2)]
    cold = []

    CP_hot, _ = _sum_mcp_between_temperature_boundaries(
        T, hot, is_shifted=False
    )
    CP_cold, _ = _sum_mcp_between_temperature_boundaries(
        T, cold, is_shifted=False
    )

    assert CP_hot == [0, 2, 2, 2]
    assert CP_cold == [0, 0, 0, 0]


def test_empty_stream_lists_returns_zero():
    T = [300, 200, 100]
    hot, cold = [], []

    CP_hot, _ = _sum_mcp_between_temperature_boundaries(
        T, hot, is_shifted=False
    )
    CP_cold, _ = _sum_mcp_between_temperature_boundaries(
        T, cold, is_shifted=False
    )

    assert CP_hot == [0, 0, 0]
    assert CP_cold == [0, 0, 0]


"""Tests for problem_table_algorithm function."""
from OpenPinch.analysis.problem_table_analysis import problem_table_algorithm


def make_simple_problem_table():
    data = {
        PT.T.value: [400, 300, 200],
        PT.CP_HOT.value: [0, 2.0, 1.0],
        PT.CP_COLD.value: [0, 1.0, 2.0],
        PT.DELTA_T.value: [0, 0, 0],
        PT.CP_NET.value: [0, 0, 0],
        PT.DELTA_H_HOT.value: [0, 0, 0],
        PT.DELTA_H_COLD.value: [0, 0, 0],
        PT.DELTA_H_NET.value: [0, 0, 0],
        PT.H_HOT.value: [0, 0, 0],
        PT.H_COLD.value: [0, 0, 0],
        PT.H_NET.value: [0, 0, 0],
    }
    return ProblemTable(data)


def test_calc_problem_table_cascade_correct():
    pt_real = make_simple_problem_table()
    result = problem_table_algorithm(pt_real.copy)

    # Check composite curves calculated
    assert result.loc[0, PT.H_NET.value] == 0  # GCC starts at 0
    assert round(result.col[PT.H_NET.value].min(), 7) == 0  # GCC min shift


def test_delta_t_computation():
    pt_real = make_simple_problem_table()
    result = problem_table_algorithm(pt_real.copy)
    assert result.col[PT.DELTA_T.value].tolist() == [0, 100, 100]


def test_net_mcp_and_delta_h():
    pt_real = make_simple_problem_table()
    result = problem_table_algorithm(pt_real.copy)
    expected_CP_NET = [0, 0, 0]
    expected_delta_h_net = [0, 0, 0]
    assert result.col[PT.CP_NET.value].tolist() == expected_CP_NET
    assert result.col[PT.DELTA_H_NET.value].tolist() == expected_delta_h_net


def test_shifting_behavior():
    pt_real = make_simple_problem_table()
    result = problem_table_algorithm(pt_real.copy)
    assert result.col[PT.H_NET.value].min() == 0
    assert (
        abs(result.loc[-1, PT.H_COLD.value] - result.loc[-1, PT.H_HOT.value]) < 1e-6
    )  # Should be nearly equal after shift


"""Tests for ProblemTable pinch-related helpers."""


def test_insert_temperature_interval_basic():
    pt_ls = [
        [250, 200, 100],
        [0, 50, 100],
        [0, 2.0, 1.0],
        [0, 2.0, 1.0],
        [200.0, 100.0, 0.0],
        [0, 3.0, 2.0],
        [0, 3.0, 2.0],
        [350.0, 200.0, 0],
        [0, -1.0, -1.0],
        [0, -50.0, -100.0],
        [150.0, 100.0, 0.0],
    ]

    expected_ls = [
        [250, 225.0, 200, 100],
        [0, 25.0, 25.0, 100],
        [0, 2.0, 2.0, 1.0],
        [0, 2.0, 2.0, 1.0],
        [200.0, 150.0, 100.0, 0.0],
        [0, 3.0, 3.0, 2.0],
        [0, 3.0, 3.0, 2.0],
        [350.0, 275.0, 200.0, 0],
        [0, -1.0, -1.0, -1.0],
        [0, -25.0, -25.0, -100.0],
        [150.0, 125.0, 100.0, 0.0],
    ]

    pt_real = ProblemTable(pt_ls)
    expected = ProblemTable(expected_ls)

    inserted = pt_real.insert_temperature_interval([225.0])
    assert inserted == 1

    result = pt_real
    for row_index in range(expected.shape[0]):
        for col in [PT.T.value, PT.H_COLD.value, PT.H_HOT.value, PT.H_NET.value]:
            expected_val = expected.loc[row_index, col]
            result_val = result.loc[row_index, col]

            if pd.isna(expected_val) and pd.isna(result_val):
                continue

            if isinstance(expected_val, float) and isinstance(result_val, float):
                assert result_val == pytest.approx(expected_val, abs=1e-6)
            else:
                assert result_val == expected_val


def test_insert_temperature_interval_gcc_basic():
    pt_ls = [[250, 200, 100], [150.0, 100.0, 0.0]]
    expected_ls = [[250, 225.0, 200, 100], [150.0, 125.0, 100.0, 0.0]]

    pt_real = ProblemTable(pt_ls)
    expected = ProblemTable(expected_ls)

    inserted = pt_real.insert_temperature_interval([225.0])
    assert inserted == 1

    result = pt_real
    for row_index in range(expected.shape[0]):
        for col in [PT.T.value, PT.H_NET.value]:
            expected_val = expected.loc[row_index, col]
            result_val = result.loc[row_index, col]

            if pd.isna(expected_val) and pd.isna(result_val):
                continue

            if isinstance(expected_val, float) and isinstance(result_val, float):
                assert result_val == pytest.approx(expected_val, abs=1e-6)
            else:
                assert result_val == expected_val


def test_insert_temperature_interval_adds_top_interval_with_zero_heat():
    pt = _make_problem_table_for_interval_tests()
    labels = [
        PT.H_HOT.value,
        PT.H_COLD.value,
        PT.H_NET.value,
        PT.H_NET_NP.value,
        PT.H_NET_A.value,
        PT.H_NET_V.value,
    ]
    original_first = {label: pt.loc[0, label] for label in labels}

    inserted = pt.insert_temperature_interval(350.0)

    assert inserted == 1
    assert pt.shape[0] == 4
    assert pt.loc[0, PT.T.value] == pytest.approx(350.0)
    assert pt.loc[0, PT.DELTA_T.value] == pytest.approx(
        pt.loc[0, PT.T.value] - pt.loc[1, PT.T.value]
    )
    assert pt.loc[1, PT.DELTA_T.value] == pytest.approx(
        pt.loc[0, PT.T.value] - pt.loc[1, PT.T.value]
    )

    for label in (PT.CP_HOT.value, PT.CP_COLD.value, PT.CP_NET.value):
        assert pt.loc[0, label] == pytest.approx(0.0)

    for label in (PT.DELTA_H_HOT.value, PT.DELTA_H_COLD.value, PT.DELTA_H_NET.value):
        assert pt.loc[0, label] == pytest.approx(0.0)

    for label in labels:
        expected = original_first[label]
        actual = pt.loc[0, label]
        if np.isnan(expected):
            assert np.isnan(actual)
        else:
            assert actual == pytest.approx(expected)


def test_insert_temperature_interval_appends_bottom_interval_with_zero_heat():
    pt = _make_problem_table_for_interval_tests()
    labels = [
        PT.H_HOT.value,
        PT.H_COLD.value,
        PT.H_NET.value,
        PT.H_NET_NP.value,
        PT.H_NET_A.value,
        PT.H_NET_V.value,
    ]
    last_idx_before = pt.shape[0] - 1
    last_temperature = pt.loc[last_idx_before, PT.T.value]
    original_last = {label: pt.loc[last_idx_before, label] for label in labels}

    inserted = pt.insert_temperature_interval(50.0)

    assert inserted == 1
    assert pt.shape[0] == 4

    last_idx = pt.shape[0] - 1
    assert pt.loc[last_idx, PT.T.value] == pytest.approx(50.0)
    assert pt.loc[last_idx, PT.DELTA_T.value] == pytest.approx(
        pt.loc[last_idx - 1, PT.T.value] - pt.loc[last_idx, PT.T.value]
    )
    assert pt.loc[last_idx, PT.DELTA_T.value] == pytest.approx(
        last_temperature - pt.loc[last_idx, PT.T.value]
    )

    for label in (PT.CP_HOT.value, PT.CP_COLD.value, PT.CP_NET.value):
        assert pt.loc[last_idx, label] == pytest.approx(0.0)

    for label in (PT.DELTA_H_HOT.value, PT.DELTA_H_COLD.value, PT.DELTA_H_NET.value):
        assert pt.loc[last_idx, label] == pytest.approx(0.0)

    for label in labels:
        expected = original_last[label]
        actual = pt.loc[last_idx, label]
        if np.isnan(expected):
            assert np.isnan(actual)
        else:
            assert actual == pytest.approx(expected)


def test_insert_temperature_interval_vectorises_across_top_middle_and_bottom():
    pt = _make_problem_table_for_interval_tests()

    inserted = pt.insert_temperature_interval([350.0, 250.0, 50.0, 250.0])

    assert inserted == 3
    assert pt.shape[0] == 6
    assert pt.col[PT.T.value].tolist() == [350.0, 300.0, 250.0, 200.0, 100.0, 50.0]

    zero_columns = (
        PT.CP_HOT.value,
        PT.CP_COLD.value,
        PT.CP_NET.value,
        PT.DELTA_H_HOT.value,
        PT.DELTA_H_COLD.value,
        PT.DELTA_H_NET.value,
    )
    for idx in (0, -1):
        for label in zero_columns:
            assert pt.loc[idx, label] == pytest.approx(0.0)

    mid_idx = int(np.where(np.isclose(pt.col[PT.T.value], 250.0))[0][0])
    assert pt.loc[mid_idx, PT.CP_HOT.value] == pytest.approx(pt.loc[mid_idx + 1, PT.CP_HOT.value])
    assert pt.loc[mid_idx, PT.CP_COLD.value] == pytest.approx(pt.loc[mid_idx + 1, PT.CP_COLD.value])
    assert pt.loc[mid_idx, PT.DELTA_T.value] == pytest.approx(
        pt.loc[mid_idx - 1, PT.T.value] - pt.loc[mid_idx, PT.T.value]
    )
    assert pt.loc[mid_idx, PT.DELTA_H_HOT.value] == pytest.approx(
        pt.loc[mid_idx, PT.DELTA_T.value] * pt.loc[mid_idx, PT.CP_HOT.value]
    )
    assert pt.loc[mid_idx, PT.H_HOT.value] == pytest.approx(150.0)
    assert pt.loc[mid_idx, PT.H_COLD.value] == pytest.approx(15.0)
    assert pt.loc[mid_idx, PT.H_NET.value] == pytest.approx(25.0)


@pytest.mark.parametrize(
    "input_vals, expected",
    [
        ([0.0, 10.0, 20.0], (0, 0, True)),
        ([10.0, 5.0, 0.0], (2, 2, True)),
        ([10.0, 1e-7, -1e-7, 15.0], (1, 2, True)),
        ([10.0, 5.0, 1.0], (2, 0, False)),
        ([0.0, 0.0, 0.0], (2, 0, False)),
        ([1e-9, 2e-9, -1e-9, 5e-7], (3, 0, False)),
    ],
)
def test_get_pinch_loc(input_vals, expected):
    table = ProblemTable({PT.H_NET.value: input_vals})
    assert table.pinch_idx() == expected


@pytest.mark.parametrize(
    "case, h_vals, t_vals, expected",
    [
        ("standard_case", [100, 0.0, 100], [300, 250, 200], (250, 250)),
        ("pinch_at_bottom", [100, 50, 0.0], [300, 250, 200], (200, 200)),
        ("pinch_at_top", [0.0, 50, 100], [300, 250, 200], (300, 300)),
        ("no_pinch", [100, 100, 100], [300, 250, 200], (None, None)),
        ("hot_below_cold", [100, 0.0, 0.0], [300, 250, 200], (250, 250)),
    ],
)
def test_get_pinch_temperatures(case, h_vals, t_vals, expected):
    table = ProblemTable({PT.T.value: t_vals, PT.H_NET.value: h_vals})
    assert table.pinch_temperatures() == expected, case


def test_shift_heat_cascade_with_enum_col():
    table = ProblemTable({PT.H_NET.value: [0, 100, 200], PT.H_HOT.value: [0, 50, 150]})
    shifted = table.shift_heat_cascade(10.0, PT.H_NET.value)

    assert shifted[PT.H_NET.value].to_list() == [10.0, 110.0, 210.0]
    assert shifted[PT.H_HOT.value].to_list() == [0, 50, 150]


def test_shift_heat_cascade_with_str_col():
    table = ProblemTable({PT.H_NET.value: [0, 100, 200], PT.H_HOT.value: [0, 50, 150]})
    shifted = table.shift_heat_cascade(-25.0, PT.H_NET.value)

    assert shifted[PT.H_NET.value].to_list() == [-25.0, 75.0, 175.0]
    assert shifted[PT.H_HOT.value].to_list() == [0, 50, 150]


"""Test cases for the _insert_temperature_interval_into_pt_at_constant_h function."""
from OpenPinch.analysis.problem_table_analysis import (
    _insert_temperature_interval_into_pt_at_constant_h,
)


def test_insert_constant_h_projection_hcc_to_ccc():
    pt_ls = [
        [
            450,
            303.34,
            245.0,
            240.0,
            239.99,
            235.0,
            195.0,
            191.67,
            185.0,
            145.0,
            99.0,
            75.0,
            35.0,
            30.01,
            30.0,
            25.0,
            15,
        ],
        [
            0,
            146.66,
            58.34,
            5.0,
            0.01,
            4.99,
            40.0,
            3.33,
            6.67,
            40.0,
            46.0,
            24.0,
            40.0,
            4.99,
            0.01,
            5.0,
            10.0,
        ],
        [
            0,
            0,
            0,
            15.0,
            15.0,
            15.0,
            15.0,
            40.0,
            40.0,
            40.0,
            40.0,
            40.0,
            15.0,
            0,
            0,
            0,
            0,
        ],
        [0, 0, 0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0, 0, 0, 0],
        [
            6150.0,
            6150.0,
            6150.0,
            6075.0,
            6074.85,
            6000.0,
            5400.0,
            5266.67,
            5000.0,
            3400.0,
            1560.0,
            600.0,
            0,
            0,
            0,
            0,
            0,
        ],
        [
            0,
            0,
            0,
            0,
            0,
            0,
            30.0,
            30.0,
            30.0,
            50.0,
            20.0,
            20.0,
            20.0,
            20.0,
            20.0,
            20.0,
            0,
        ],
        [0, 0, 0, 0, 0, 0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0],
        [
            6900.0,
            6900.0,
            6900.0,
            6900.0,
            6900.0,
            6900.0,
            5700.0,
            5600.0,
            5400.0,
            3400.0,
            2480.0,
            2000.0,
            1200.0,
            1100.2,
            1100.0,
            1000.0,
            1000.0,
        ],
        [
            0,
            0,
            0,
            15.0,
            15.0,
            15.0,
            -15.0,
            10.0,
            10.0,
            -10.0,
            20.0,
            20.0,
            -5.0,
            -20.0,
            -20.0,
            -20.0,
            0,
        ],
        [
            0,
            0.0,
            0.0,
            75.0,
            0.15,
            74.85,
            -600.0,
            33.33,
            66.67,
            -400.0,
            920.0,
            480.0,
            -200.0,
            -99.8,
            -0.2,
            -100.0,
            0.0,
        ],
        [
            750.0,
            750.0,
            750.0,
            825.0,
            825.15,
            900.0,
            300.0,
            333.33,
            400.0,
            0,
            920.0,
            1400.0,
            1200.0,
            1100.2,
            1100.0,
            1000.0,
            1000.0,
        ],
    ]
    expected_ls = [
        [
            450,
            303.34,
            245.0,
            240.0,
            239.99,
            235.0,
            210.0,
            195.0,
            191.67,
            185.0,
            145.0,
            99.0,
            85.0,
            75.0,
            35.0,
            30.01,
            30.0,
            25.0,
            15,
        ],
        [
            0,
            146.66,
            58.34,
            5.0,
            0.01,
            4.99,
            25.0,
            15.0,
            0.83,
            6.67,
            40.0,
            23.0,
            14.0,
            10.0,
            40.0,
            4.99,
            0.01,
            5.0,
            10.0,
        ],
        [
            0,
            0,
            0,
            15.0,
            15.0,
            15.0,
            15.0,
            15.0,
            40.0,
            40.0,
            40.0,
            40.0,
            40.0,
            40.0,
            15.0,
            0,
            0,
            0,
            0,
        ],
        [
            0,
            0,
            0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0,
            0,
            0,
            0,
        ],
        [
            6150.0,
            6150.0,
            6150.0,
            6075.0,
            6074.85,
            6000.0,
            5625.0,
            5400.0,
            5266.67,
            5000.0,
            3400.0,
            1560.0,
            1000.0,
            600.0,
            0,
            0,
            0,
            0,
            0,
        ],
        [
            0,
            0,
            0,
            0,
            0,
            0,
            30.0,
            30.0,
            30.0,
            50.0,
            20.0,
            20.0,
            20.0,
            20.0,
            20.0,
            20.0,
            20.0,
            20.0,
            0,
        ],
        [
            0,
            0,
            0,
            0,
            0,
            0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0,
        ],
        [
            6900.0,
            6900.0,
            6900.0,
            6900.0,
            6900.0,
            6900.0,
            6150.0,
            5700.0,
            5600.0,
            5400.0,
            3400.0,
            2480.0,
            2200.0,
            2000.0,
            1200.0,
            1100.2,
            1100.0,
            1000.0,
            1000.0,
        ],
        [
            0,
            0,
            0,
            15.0,
            15.0,
            15.0,
            -15.0,
            -15.0,
            10.0,
            10.0,
            -10.0,
            20.0,
            20.0,
            20.0,
            -5.0,
            -20.0,
            -20.0,
            -20.0,
            0,
        ],
        [
            0,
            0.0,
            0.0,
            75.0,
            0.15,
            74.85,
            -375.0,
            -225.0,
            8.33,
            66.67,
            -400.0,
            460.0,
            280.0,
            200.0,
            -200.0,
            -99.8,
            -0.2,
            -100.0,
            0.0,
        ],
        [
            750.0,
            750.0,
            750.0,
            825.0,
            825.15,
            900.0,
            525.0,
            300.0,
            333.33,
            400.0,
            0,
            920.0,
            1200.0,
            1400.0,
            1200.0,
            1100.2,
            1100.0,
            1000.0,
            1000.0,
        ],
    ]
    pt_real = ProblemTable(pt_ls)
    expected = ProblemTable(expected_ls)

    result = _insert_temperature_interval_into_pt_at_constant_h(pt_real)

    for row_index in range(expected.data.shape[0]):
        for col in [PT.T.value, PT.H_COLD.value, PT.H_HOT.value, PT.H_NET.value]:
            expected_val = expected.loc[row_index, col]
            result_val = result.loc[row_index, col]

            if pd.isna(expected_val) and pd.isna(result_val):
                continue

            assert expected_val == result_val or ( # 210, 6150, 85, 1000
                isinstance(expected_val, float)
                and isinstance(result_val, float)
                and abs(expected_val - result_val) < 1e-2
            ), f"Mismatch at row {row_index}, col {col}: expected {expected_val}, got {result_val}"


"""Tests for set_zonal_targets function"""
from OpenPinch.analysis.problem_table_analysis import set_zonal_targets

# --- Fixtures & Helpers ---


@pytest.fixture
def dummy_zone():
    return Zone(name="Z_Test")


@pytest.fixture
def dummy_problem_table():
    # Basic dummy PT with 3 intervals
    return ProblemTable(
        {
            PT.T.value: [400, 300, 200],
            PT.H_HOT.value: [1000, 600, 0],
            PT.H_COLD.value: [0, 400, 800],
            PT.H_NET.value: [200, 300, 400],
        }
    )


@pytest.fixture
def dummy_problem_table_star():
    # Star PT should be shifted relative to original
    return ProblemTable(
        {
            PT.T.value: [400, 300, 200],
            PT.H_HOT.value: [950, 550, 0],
            PT.H_COLD.value: [0, 350, 750],
            PT.H_NET.value: [150, 300, 450],
        }
    )


# --- Core Functionality ---


def test_zonal_targets_computed_correctly(
    dummy_zone, dummy_problem_table, dummy_problem_table_star
):
    z = set_zonal_targets(dummy_problem_table_star, dummy_problem_table)

    assert z["hot_utility_target"] == 150
    assert z["cold_utility_target"] == 450
    assert z["heat_recovery_target"] == pytest.approx(500)
    assert z["heat_recovery_limit"] == pytest.approx(600)
    assert z["degree_of_int"] == pytest.approx(5 / 6)


# --- Edge Cases ---


def test_zero_heat_recovery_limit_sets_degree_to_one(dummy_zone):
    pt_real = ProblemTable(
        {
            PT.T.value: [400, 300],
            PT.H_HOT.value: [100, 100],  # No change → ΔH = 0
            PT.H_NET.value: [0, 100],  # Forces negative limit
        }
    )
    pt = ProblemTable(
        {PT.T.value: [400, 300], PT.H_HOT.value: [200, 100], PT.H_NET.value: [50, 100]}
    )
    z = set_zonal_targets(pt, pt_real)

    assert z["heat_recovery_limit"] == 0
    assert z["degree_of_int"] == 1.0


def test_negative_heat_recovery_target():
    pt_real = ProblemTable(
        {PT.T.value: [400, 300], PT.H_HOT.value: [500, 300], PT.H_NET.value: [0, 200]}
    )
    pt = ProblemTable(
        {
            PT.T.value: [400, 300],
            PT.H_HOT.value: [450, 250],
            PT.H_NET.value: [600, 700],  # H_net ends higher than H_HOT start
        }
    )

    z = set_zonal_targets(pt, pt_real)

    assert z["heat_recovery_target"] < 0
    assert isinstance(z["degree_of_int"], float)


def test_single_row_problem_table():
    pt_real = ProblemTable(
        {
            PT.T.value: [400],
            PT.H_HOT.value: [100],
            PT.H_NET.value: [50],
        }
    )
    pt = ProblemTable(
        {
            PT.T.value: [400],
            PT.H_HOT.value: [90],
            PT.H_NET.value: [40],
        }
    )

    z = set_zonal_targets(pt, pt_real)

    assert z["hot_utility_target"] == 40
    assert z["cold_utility_target"] == 40
    assert z["heat_recovery_target"] == 50
    assert isinstance(z["degree_of_int"], float)


"""Test get_process_heat_cascade"""
from OpenPinch.analysis.problem_table_analysis import get_process_heat_cascade, get_heat_recovery_target_from_pt, set_zonal_targets


def test_problem_table_algorithm_executes():
    z = Zone(name="P")
    z.config = Configuration()
    z.hot_streams.add(Stream("stream A", t_supply=400, t_target=200, heat_flow=1.0))
    z.cold_streams.add(Stream("stream B", t_supply=200, t_target=300, heat_flow=2.0))
    z.pt = ProblemTable()
    z.pt_real = ProblemTable()

    pt = get_process_heat_cascade(
        hot_streams=z.hot_streams, 
        cold_streams=z.cold_streams, 
        all_streams=z.all_streams, 
        zone_config=z.config,
        is_shifted=True,
    )
    pt_real = get_process_heat_cascade(
        hot_streams=z.hot_streams, 
        cold_streams=z.cold_streams, 
        all_streams=z.all_streams, 
        zone_config=z.config,
        is_shifted=False,
        known_heat_recovery=get_heat_recovery_target_from_pt(pt)
    )
    target_values = set_zonal_targets(
        pt=pt,
        pt_real=pt_real,
    )
    assert target_values["hot_utility_target"] == 1.0
    assert target_values["cold_utility_target"] == 0.0
    assert target_values["heat_recovery_target"] == 1.0


"""Test _shift_pt_to_set_heat_recovery"""
from OpenPinch.analysis.problem_table_analysis import _shift_pt_to_set_heat_recovery


def test_correct_pt_composite_curves_shifts_columns():
    pt = ProblemTable({PT.H_COLD.value: [10, 20], PT.H_NET.value: [5, 15]})
    corrected: ProblemTable = _shift_pt_to_set_heat_recovery(
        pt.copy, heat_recovery_target=30, current_heat_recovery=50
    )

    assert (corrected.col[PT.H_COLD.value] == [30, 40]).all()
    assert (corrected.col[PT.H_NET.value] == [25, 35]).all()


"""Tests for the create_problem_table_with_t_int function."""
from OpenPinch.analysis.problem_table_analysis import create_problem_table_with_t_int


def make_stream(name, t_supply, t_target, dt, cp=0, htc=0):
    return Stream(
        name=name,
        t_supply=t_supply,
        t_target=t_target,
        dt_cont=dt,
        heat_flow=abs(cp * (t_supply - t_target)),
        htc=htc if htc > 0 else 1,
    )


def test_returns_correct_intervals_basic():
    hot = [make_stream("H1", 400, 300, 10)]
    cold = [make_stream("C1", 100, 200, 10)]
    T_star: ProblemTable
    T: ProblemTable
    T_star = create_problem_table_with_t_int(hot + cold, True)
    T = create_problem_table_with_t_int(hot + cold, False)

    assert PT.T.value in T_star.columns
    assert PT.T.value in T.columns
    assert T_star[PT.T.value].to_list() == [390, 290, 210, 110]
    assert T[PT.T.value].to_list() == [400, 300, 200, 100]


def test_includes_utilities():
    hu = [make_stream("HU", 500, 550, -10)]
    cu = [make_stream("CU", 30, 70, 10)]
    T_star: ProblemTable
    T: ProblemTable
    T_star = create_problem_table_with_t_int(hu + cu, True)
    T = create_problem_table_with_t_int(hu + cu, False)

    assert T_star[PT.T.value].to_list() == [540, 490, 80, 40]
    assert T[PT.T.value].to_list() == [550, 500, 70, 30]


def test_includes_turbine_config():
    zone_config = Configuration()
    zone_config.DO_TURBINE_WORK = True
    zone_config.T_TURBINE_BOX = 200
    zone_config.P_TURBINE_BOX = 5  # pressure in bar
    zone_config.DO_EXERGY_TARGETING = True

    T_star: ProblemTable
    T_star = create_problem_table_with_t_int(zone_config=zone_config)

    assert zone_config.T_TURBINE_BOX in T_star[PT.T.value].to_list()
    assert zone_config.T_ENV in T_star[PT.T.value].to_list()


def test_empty_inputs():
    T_star: ProblemTable
    T: ProblemTable
    T_star = create_problem_table_with_t_int()
    T = create_problem_table_with_t_int()
    assert T_star.data.size == 0
    assert T.data.size == 0
