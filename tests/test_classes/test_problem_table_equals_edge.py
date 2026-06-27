"""Targeted edge-path tests for ProblemTable equality internals."""

import numpy as np
import pytest

from OpenPinch.classes.problem_table import ProblemTable
from OpenPinch.lib.enums import ProblemTableLabel as PT


class _FakeArray:
    def __init__(self, arr: np.ndarray, fake_dtype):
        self._arr = arr
        self.dtype = fake_dtype
        self.shape = arr.shape
        self.size = arr.size

    def __getitem__(self, key):
        return self._arr[key]


def test_equals_uses_numeric_allclose_fast_path():
    left = ProblemTable({"a": [1.0, 2.0], "b": [3.0, 4.0]}, add_default_labels=False)
    right = ProblemTable({"a": [1.0, 2.0], "b": [3.0, 4.0]}, add_default_labels=False)
    assert left._equals(right) is True


def test_equals_columnwise_allclose_branch_with_fake_object_dtype():
    left = ProblemTable({"a": [1.0, 2.0], "b": [3.0, 4.0]}, add_default_labels=False)
    right = ProblemTable({"a": [1.0, 20.0], "b": [3.0, 4.0]}, add_default_labels=False)

    left.data = _FakeArray(np.asarray([[1.0, 3.0], [2.0, 4.0]], dtype=float), object)
    right.data = _FakeArray(np.asarray([[1.0, 3.0], [20.0, 4.0]], dtype=float), object)

    assert left._equals(right) is False


def test_equals_object_nan_and_numeric_mismatch_branch():
    left = ProblemTable({"a": [0.0, 0.0, 0.0]}, add_default_labels=False)
    right = ProblemTable({"a": [0.0, 0.0, 0.0]}, add_default_labels=False)
    left.data = np.array([[np.nan], [1.0], ["x"]], dtype=object)
    right.data = np.array([[np.nan], [2.0], ["x"]], dtype=object)

    assert left._equals(right) is False


# ===== Merged from test_problem_table_extra.py =====
"""Additional edge-branch coverage for ProblemTable."""


def _default_table() -> ProblemTable:
    return ProblemTable(
        {
            PT.T: [200.0, 100.0],
            PT.DELTA_T: [0.0, 100.0],
            PT.CP_HOT: [1.0, 1.0],
            PT.CP_COLD: [1.0, 1.0],
            PT.CP_NET: [0.0, 0.0],
            PT.H_NET: [10.0, 0.0],
        }
    )


def test_index_views_and_iloc_paths():
    pt = _default_table()
    _ = pt.icol[0]
    pt.icol[0] = [300.0, 250.0]
    assert pt.icol[0][0] == pytest.approx(300.0)

    _ = pt.iloc[(0, 0)]
    pt.iloc[(1, 0)] = 150.0
    assert pt.iloc[(1, 0)] == pytest.approx(150.0)
    assert pt.iloc[(0, PT.T)] == pytest.approx(300.0)
    pt.iloc[(1, PT.T)] = 125.0
    assert pt.iloc[(1, PT.T)] == pytest.approx(125.0)


def test_uninitialised_column_setters_and_len_zero():
    pt = ProblemTable()
    assert len(pt) == 0

    pt[PT.T] = [100.0, 90.0]
    pt.cols[PT.T] = [110.0, 95.0]
    pt[PT.H_NET] = [5.0, 4.0]
    assert pt[PT.T].tolist() == [110.0, 95.0]
    assert pt[PT.H_NET].tolist() == [5.0, 4.0]


def test_cols_view_get_and_set_on_initialised_table():
    pt = _default_table()
    assert pt.col[PT.T][0] == pytest.approx(200.0)
    pt.col[PT.T] = [210.0, 110.0]
    assert pt.col[PT.T].tolist() == [210.0, 110.0]

    values = pt.cols[[PT.T, PT.H_NET]]
    assert values.shape == (2, 2)
    mixed_values = pt.cols[[PT.T, PT.H_NET.value]]
    assert mixed_values.shape == (2, 2)
    pt.cols[PT.H_NET] = [5.0, 4.0]
    assert pt[PT.H_NET].tolist() == [5.0, 4.0]


def test_problem_table_getitem_setitem_and_slice_enum_paths():
    pt = _default_table()

    assert np.array_equal(pt[PT.T], pt[PT.T])

    pt[PT.H_NET] = [8.0, 7.0]
    assert pt[PT.H_NET].tolist() == [8.0, 7.0]

    single = pt.slice(PT.T)
    assert isinstance(single, ProblemTable)
    assert single.to_list() == [200.0, 100.0]

    subset = pt.slice([PT.T, PT.H_NET])
    assert isinstance(subset, ProblemTable)
    assert subset.columns == [PT.T.value, PT.H_NET.value]
    assert subset.to_list() == [[200.0, 100.0], [8.0, 7.0]]

    mixed_subset = pt.slice([PT.T, PT.H_NET.value])
    assert mixed_subset.columns == [PT.T.value, PT.H_NET.value]

    with pytest.raises(TypeError, match="pt\\.slice"):
        _ = pt[[PT.T, PT.H_NET]]
    with pytest.raises(TypeError, match="single-column assignment"):
        pt[[PT.T, PT.H_NET]] = [[1.0, 2.0], [3.0, 4.0]]


def test_loc_accepts_enum_column_keys():
    pt = _default_table()

    assert pt.loc[0, PT.T] == pytest.approx(200.0)
    pt.loc[1, PT.H_NET] = 3.5
    assert pt.loc[1, PT.H_NET] == pytest.approx(3.5)


def test_equals_object_and_shape_branches():
    pt = _default_table()
    assert pt._equals("not-a-table") is False

    pt_other_cols = ProblemTable({"x": [1.0], "y": [2.0]}, add_default_labels=False)
    assert pt._equals(pt_other_cols) is False

    pt_none_a = ProblemTable()
    pt_none_b = ProblemTable()
    assert pt_none_a._equals(pt_none_b) is True
    assert pt._equals(pt_none_a) is False

    pt_short = ProblemTable({PT.T: [200.0], PT.H_NET: [0.0]})
    assert pt._equals(pt_short) is False

    pt_empty_a = ProblemTable({PT.T: []})
    pt_empty_b = ProblemTable({PT.T: []})
    assert pt_empty_a._equals(pt_empty_b) is True


def test_equals_object_dtype_cast_nan_and_mismatch_paths():
    cols = ["a", "b", "c", "d"]
    left = ProblemTable({c: [0.0] for c in cols}, add_default_labels=False)
    right = ProblemTable({c: [0.0] for c in cols}, add_default_labels=False)

    left.data = np.array([["1.0", np.nan, 2.0, "x"]], dtype=object)
    right.data = np.array([[1.0, np.nan, 2.0 + 1e-8, "x"]], dtype=object)
    assert left._equals(right) is True
    assert (left == right) is True
    assert (left != right) is False

    bad_numeric = ProblemTable({c: [0.0] for c in cols}, add_default_labels=False)
    bad_numeric.data = np.array([[1.0, np.nan, 9.0, "x"]], dtype=object)
    assert left._equals(bad_numeric) is False

    bad_text = ProblemTable({c: [0.0] for c in cols}, add_default_labels=False)
    bad_text.data = np.array([["1.0", np.nan, 2.0, "y"]], dtype=object)
    assert left._equals(bad_text) is False


def test_equals_columnwise_numeric_continue_path():
    cols = ["a", "b"]
    left = ProblemTable({c: [0.0, 0.0] for c in cols}, add_default_labels=False)
    right = ProblemTable({c: [0.0, 0.0] for c in cols}, add_default_labels=False)

    class _FakeArray:
        def __init__(self, arr, fake_dtype):
            self._arr = arr
            self.dtype = fake_dtype
            self.shape = arr.shape
            self.size = arr.size

        def __getitem__(self, key):
            return self._arr[key]

    left.data = _FakeArray(np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float), object)
    right.data = _FakeArray(np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float), object)
    assert left._equals(right) is True


def test_equals_fallback_numeric_pair_continue_path():
    cols = ["a"]
    left = ProblemTable({c: [0.0, 0.0] for c in cols}, add_default_labels=False)
    right = ProblemTable({c: [0.0, 0.0] for c in cols}, add_default_labels=False)

    # force object-column fallback with mixed numeric/text values
    left.data = np.array([[1.0], ["x"]], dtype=object)
    right.data = np.array([[1.0 + 1e-9], ["x"]], dtype=object)

    assert left._equals(right) is True


def test_to_list_and_shift_heat_cascade_branches():
    pt = _default_table()
    _ = pt.to_list(col=PT.T)
    pt.round(0)
    assert pt[PT.T][0] == pytest.approx(200.0)
    shifted_enum = pt.shift_heat_cascade(5.0, PT.H_NET)
    assert shifted_enum[PT.H_NET][0] == pytest.approx(15.0)

    idx = pt.col_index[PT.H_NET.value]
    shifted_idx = pt.shift_heat_cascade(-2.0, idx)
    assert shifted_idx[PT.H_NET][0] == pytest.approx(13.0)
    assert pt.pinch_idx(idx) == pt.pinch_idx(PT.H_NET)


def test_problem_table_column_label_validation_edges():
    table = ProblemTable(
        {"empty": np.array([np.nan]), "filled": np.array([1.0])},
        add_default_labels=False,
    )

    assert table.columns == ["empty", "filled"]
    assert np.isnan(table["empty"]).all()
    with pytest.raises(TypeError, match="Column labels"):
        ProblemTable._validate_column_name(object())
    with pytest.raises(TypeError, match="sequences"):
        ProblemTable._validate_column_names(object())
    with pytest.raises(ValueError, match="Duplicate column label"):
        ProblemTable._validate_column_mapping({PT.T: [1.0], PT.T.value: [2.0]})


def test_insert_temperature_interval_helper_guard_paths():
    pt_none = ProblemTable()
    assert pt_none.insert_temperature_interval([120.0]) == 0

    pt = _default_table()
    assert pt._Ts_needing_insertion(np.array([], dtype=float)).size == 0
    deduped = pt._dedupe_monotonic(np.array([5.0, 5.0, 4.0], dtype=float))
    assert deduped.tolist() == [5.0, 4.0]

    same_data, inserted = pt._apply_interval_map({}, np.array([]), np.array([]))
    assert inserted == 0
    assert np.allclose(same_data, pt.data, equal_nan=True)

    rows, top_adj, bot_adj = pt._build_mid_block(pt.data[0], pt.data[1], np.array([]))
    assert rows.size == 0
    assert top_adj.shape == pt.data[0].shape
    assert bot_adj.shape == pt.data[1].shape

    new_data = np.zeros((2, pt.data.shape[1]))
    row_meta = [{"type": "top"}, {"type": "top"}]
    pt._rebuild_edge_block(new_data, row_meta, positions=[0], is_top=True)

    pt._insert_mid_block(new_data, positions=[], upper_pos=0, lower_pos=1)
    pt._insert_mid_block(new_data, positions=[0], upper_pos=-1, lower_pos=-1)


def test_temperature_block_and_interpolation_edge_paths():
    mini = ProblemTable(
        {
            PT.T: [100.0],
            PT.DELTA_T: [0.0],
        },
        add_default_labels=False,
    )
    empty_block, _ = mini._build_top_or_bottom_block(
        row_neighbor=np.array([100.0, 0.0], dtype=float),
        T_vals=np.array([]),
        is_top_block=True,
    )
    assert empty_block.shape == (0, 2)

    full = _default_table()
    row_neighbor = full.data[0].copy()
    block, _ = full._build_top_or_bottom_block(
        row_neighbor=row_neighbor,
        T_vals=np.array([220.0, 240.0, 260.0], dtype=float),
        is_top_block=True,
    )
    assert block.shape[0] == 3

    rows0, (t_idx, delta_idx) = full._initialise_insert_rows(
        full.data[0], full.data[1], np.array([])
    )
    assert rows0.shape[0] == 0
    assert isinstance(t_idx, int) and isinstance(delta_idx, int)

    full._interpolate_heat_columns(
        rows=np.empty((0, full.data.shape[1])),
        row_top=full.data[0],
        row_bot=full.data[1],
        t_idx=full.col_index[PT.T.value],
    )

    same_top = full.data[0].copy()
    same_bot = full.data[0].copy()
    rows_same = np.array([same_top.copy()])
    full._interpolate_heat_columns(
        rows=rows_same,
        row_top=same_top,
        row_bot=same_bot,
        t_idx=full.col_index[PT.T.value],
    )

    top = full.data[0].copy()
    bot = full.data[1].copy()
    top[full.col_index[PT.H_HOT.value]] = np.nan
    bot[full.col_index[PT.H_HOT.value]] = 11.0
    rows = np.array([top.copy()])
    rows[0, full.col_index[PT.T.value]] = 150.0
    full._interpolate_heat_columns(
        rows=rows,
        row_top=top,
        row_bot=bot,
        t_idx=full.col_index[PT.T.value],
    )
    assert rows[0, full.col_index[PT.H_HOT.value]] == pytest.approx(11.0)

    adjusted = full._adjust_bottom_row(
        row_bot=bot,
        rows=np.empty((0, full.data.shape[1])),
        t_idx=full.col_index[PT.T.value],
        delta_idx=full.col_index[PT.DELTA_T.value],
    )
    assert np.allclose(adjusted, bot, equal_nan=True)


def test_row_mutation_update_sort_and_export_errors():
    pt = _default_table()
    n_before = len(pt)
    pt.insert({PT.T: 180.0, PT.H_NET: 5.0}, 1)
    assert len(pt) == n_before + 1

    pt.update_row(1, {PT.T: 175.0, "unknown": 1.0})
    assert pt.loc[1, PT.T] == pytest.approx(175.0)

    assert pt.update({}) is pt

    with pytest.raises(ValueError, match="uninitialised"):
        ProblemTable().update(
            {PT.H_NET: np.array([1.0])},
            T_col=np.array([1.0]),
        )

    pt.delete_row(1)
    assert len(pt) == n_before

    with pytest.raises(KeyError, match="not found"):
        pt.sort_by_column("not_a_column")

    pt.sort_by_column(PT.T, ascending=False)
    assert pt[PT.T][0] >= pt[PT.T][-1]

    with pytest.raises(ValueError, match="uninitialised"):
        ProblemTable().export("x")


def test_update_requires_T_col_for_non_empty_updates():
    pt = _default_table()

    with pytest.raises(TypeError, match="T_col"):
        pt.update({PT.H_NET: np.array([1.0, 2.0])})
    with pytest.raises(TypeError, match="updates"):
        pt.update(
            ["not", "a", "dict"],
            T_col=np.array([200.0, 100.0]),
        )


@pytest.mark.parametrize(
    ("T_col", "exc_type"),
    [
        ([200.0, 100.0], TypeError),
        (np.array([[200.0, 100.0]]), ValueError),
    ],
)
def test_update_rejects_invalid_T_col(T_col, exc_type):
    pt = _default_table()

    with pytest.raises(exc_type, match="1D numpy.ndarray"):
        pt.update(
            {PT.H_NET: np.array([1.0, 2.0])},
            T_col=T_col,
        )


def test_update_rejects_non_numeric_T_col_values():
    pt = _default_table()

    with pytest.raises(ValueError, match="numeric temperature"):
        pt.update(
            {PT.H_NET: np.array([1.0, 2.0])},
            T_col=np.array(["hot", "cold"], dtype=object),
        )


@pytest.mark.parametrize(
    ("values", "exc_type"),
    [
        ([1.0, 2.0], TypeError),
        (np.array([[1.0, 2.0]]), ValueError),
    ],
)
def test_update_rejects_non_vector_numpy_column_values(values, exc_type):
    pt = _default_table()

    with pytest.raises(exc_type, match="1D numpy.ndarray"):
        pt.update(
            {PT.H_NET: values},
            T_col=np.array([200.0, 100.0]),
        )


def test_update_rejects_unknown_or_temperature_columns():
    pt = _default_table()

    with pytest.raises(KeyError, match="not found"):
        pt.update(
            {"not_a_column": np.array([1.0, 2.0])},
            T_col=np.array([200.0, 100.0]),
        )

    with pytest.raises(ValueError, match="temperature column"):
        pt.update(
            {PT.T: np.array([200.0, 100.0])},
            T_col=np.array([200.0, 100.0]),
        )


def test_update_rejects_length_mismatch_against_T_col():
    pt = _default_table()

    with pytest.raises(ValueError, match="T_col"):
        pt.update(
            {PT.H_NET: np.array([1.0])},
            T_col=np.array([200.0, 100.0]),
        )


def test_update_assigns_directly_when_temperature_grids_match():
    pt = _default_table()

    out = pt.update(
        {PT.H_NET: np.array([3.0, 4.0])},
        T_col=np.array([200.0 + 1e-7, 100.0 - 1e-7]),
    )

    assert out is pt
    assert pt[PT.H_NET].tolist() == [3.0, 4.0]
