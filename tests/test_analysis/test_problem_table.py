import pandas as pd
import pytest

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


"""Tests for _problem_table_algorithm function."""
from OpenPinch.analysis.problem_table_analysis import _problem_table_algorithm


def make_simple_problem_table():
    data = {
        PT.T.value: [400, 300, 200],
        PT.CP_HOT.value: [0, 2.0, 1.0],
        PT.CP_COLD.value: [0, 1.0, 2.0],
        PT.DELTA_T.value: [0, 0, 0],
        PT.MCP_NET.value: [0, 0, 0],
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
    result = _problem_table_algorithm(pt_real.copy)

    # Check composite curves calculated
    assert result.loc[0, PT.H_NET.value] == 0  # GCC starts at 0
    assert round(result.col[PT.H_NET.value].min(), 7) == 0  # GCC min shift


def test_delta_t_computation():
    pt_real = make_simple_problem_table()
    result = _problem_table_algorithm(pt_real.copy)
    assert result.col[PT.DELTA_T.value].tolist() == [0, 100, 100]


def test_net_mcp_and_delta_h():
    pt_real = make_simple_problem_table()
    result = _problem_table_algorithm(pt_real.copy)
    expected_mcp_net = [0, -1.0, 1.0]
    expected_delta_h_net = [0, -100.0, 100.0]
    assert result.col[PT.MCP_NET.value].tolist() == expected_mcp_net
    assert result.col[PT.DELTA_H_NET.value].tolist() == expected_delta_h_net


def test_shifting_behavior():
    pt_real = make_simple_problem_table()
    result = _problem_table_algorithm(pt_real.copy)
    assert result.col[PT.H_NET.value].min() == 0
    assert (
        abs(result.loc[-1, PT.H_COLD.value] - result.loc[-1, PT.H_HOT.value]) < 1e-6
    )  # Should be nearly equal after shift


"""Tests for ProblemTable pinch-related helpers."""


def test_insert_temperature_interval_basic():
    pt_ls = [
        [250, 200, 100],
        [None, 50, 100],
        [None, 2.0, 1.0],
        [None, 2.0, 1.0],
        [200.0, 100.0, 0.0],
        [None, 3.0, 2.0],
        [None, 3.0, 2.0],
        [350.0, 200.0, 0],
        [None, -1.0, -1.0],
        [None, -50.0, -100.0],
        [150.0, 100.0, 0.0],
    ]

    expected_ls = [
        [250, 225.0, 200, 100],
        [None, 25.0, 25.0, 100],
        [None, 2.0, 2.0, 1.0],
        [None, 2.0, 2.0, 1.0],
        [200.0, 150.0, 100.0, 0.0],
        [None, 3.0, 3.0, 2.0],
        [None, 3.0, 3.0, 2.0],
        [350.0, 275.0, 200.0, 0],
        [None, -1.0, -1.0, -1.0],
        [None, -25.0, -25.0, -100.0],
        [150.0, 125.0, 100.0, 0.0],
    ]

    pt_real = ProblemTable(pt_ls)
    expected = ProblemTable(expected_ls)

    result, inserted = pt_real.insert_temperature_interval([225.0])
    assert inserted == 1

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

    result, inserted = pt_real.insert_temperature_interval([225.0])
    assert inserted == 1

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
    assert table.get_pinch_loc() == expected


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
    assert table.get_pinch_temperatures() == expected, case


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


"""Tests for _set_zonal_targets function"""
from OpenPinch.analysis.problem_table_analysis import _set_zonal_targets

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
    z = _set_zonal_targets(dummy_problem_table_star, dummy_problem_table)

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
    z = _set_zonal_targets(pt, pt_real)

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

    z = _set_zonal_targets(pt, pt_real)

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

    z = _set_zonal_targets(pt, pt_real)

    assert z["hot_utility_target"] == 40
    assert z["cold_utility_target"] == 40
    assert z["heat_recovery_target"] == 50
    assert isinstance(z["degree_of_int"], float)


"""Test get_process_heat_cascade"""
from OpenPinch.analysis.problem_table_analysis import get_process_heat_cascade


def test_problem_table_algorithm_executes():
    z = Zone(name="Z")
    z.config = Configuration()
    z.hot_streams.add(Stream("stream A", t_supply=400, t_target=200, heat_flow=1.0))
    z.cold_streams.add(Stream("stream B", t_supply=200, t_target=300, heat_flow=2.0))
    z.pt = ProblemTable()
    z.pt_real = ProblemTable()

    _, _, target_values = get_process_heat_cascade(
        z.hot_streams, z.cold_streams, z.all_streams, z.config
    )
    assert target_values["hot_utility_target"] == 1.0
    assert target_values["cold_utility_target"] == 0.0
    assert target_values["heat_recovery_target"] == 1.0


"""Test _add_temperature_intervals_at_constant_h"""
from OpenPinch.analysis.problem_table_analysis import (
    _add_temperature_intervals_at_constant_h,
)


def test_add_temperature_intervals_skips_when_target_zero(monkeypatch):
    dummy_pt = ProblemTable(
        {PT.T.value: [400, 300], PT.H_HOT.value: [0, 0], PT.H_COLD.value: [0, 0]}
    )
    monkeypatch.setattr(
        "OpenPinch.analysis.problem_table_analysis._insert_temperature_interval_into_pt_at_constant_h",
        lambda *args: args[1],
    )
    result_pt = dummy_pt.copy
    _add_temperature_intervals_at_constant_h(result_pt, 0.0)
    assert result_pt == dummy_pt


def test_add_temperature_intervals_calls_insert(monkeypatch):
    calls = []

    def fake_insert(*args):
        calls.append(args)
        return args[0]  # Return the full DataFrame, not a column name

    monkeypatch.setattr(
        "OpenPinch.analysis.problem_table_analysis._insert_temperature_interval_into_pt_at_constant_h",
        fake_insert,
    )

    dummy_pt = ProblemTable(
        {PT.T.value: [400, 300], PT.H_HOT.value: [0, 0], PT.H_COLD.value: [0, 0]}
    )
    dummy_pt = dummy_pt.copy
    _add_temperature_intervals_at_constant_h(dummy_pt, 50.0)

    assert len(calls) == 1


"""Test _shift_pt_real_composite_curves"""
from OpenPinch.analysis.problem_table_analysis import _shift_pt_real_composite_curves


def test_correct_pt_composite_curves_shifts_columns():
    pt = ProblemTable({PT.H_COLD.value: [10, 20], PT.H_NET.value: [5, 15]})
    corrected: ProblemTable = _shift_pt_real_composite_curves(
        pt.copy, heat_recovery_target=30, heat_recovery_limit=50
    )

    assert (corrected.col[PT.H_COLD.value] == [30, 40]).all()
    assert (corrected.col[PT.H_NET.value] == [25, 35]).all()


"""Tests for the get_temperature_intervals function."""
from OpenPinch.analysis.problem_table_analysis import _get_temperature_intervals


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
    T_star, T = _get_temperature_intervals(hot + cold)

    assert PT.T.value in T_star.columns
    assert PT.T.value in T.columns
    assert T_star[PT.T.value].to_list() == [390, 290, 210, 110]
    assert T[PT.T.value].to_list() == [400, 300, 200, 100]


def test_includes_utilities():
    hu = [make_stream("HU", 500, 550, -10)]
    cu = [make_stream("CU", 30, 70, 10)]
    T_star: ProblemTable
    T: ProblemTable
    T_star, T = _get_temperature_intervals(hu + cu)

    assert T_star[PT.T.value].to_list() == [540, 490, 80, 40]
    assert T[PT.T.value].to_list() == [550, 500, 70, 30]


def test_includes_turbine_config():
    config = Configuration()
    config.DO_TURBINE_WORK = True
    config.T_TURBINE_BOX = 200
    config.P_TURBINE_BOX = 5  # pressure in bar
    config.DO_EXERGY_TARGETING = True

    T_star, _ = _get_temperature_intervals(config=config)

    assert config.T_TURBINE_BOX in T_star[PT.T.value].to_list()
    assert config.T_ENV in T_star[PT.T.value].to_list()


def test_empty_inputs():
    T_star, T = _get_temperature_intervals()
    assert T_star.data.size == 0
    assert T.data.size == 0
