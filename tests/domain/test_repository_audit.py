"""Problem-table boundary regressions from the repository audit."""

from copy import deepcopy

import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from OpenPinch.domain.enums import ProblemTableLabel
from OpenPinch.domain.problem_table import ProblemTable


@pytest.mark.parametrize("values", [[], [np.nan], [np.nan, np.nan]])
def test_slicing_missing_columns_preserves_rows(values):
    table = ProblemTable({ProblemTableLabel.T: values})
    selected = table._slice_columns([ProblemTableLabel.H_NET])
    assert selected.shape == (len(values), 1)
    assert np.isnan(selected[ProblemTableLabel.H_NET]).all()


@pytest.mark.parametrize("source", [{}, []])
def test_empty_input_has_a_two_dimensional_buffer(source):
    table = ProblemTable(source)
    assert table.shape == (0, len(ProblemTableLabel))


def test_empty_custom_table_has_a_two_dimensional_buffer():
    assert ProblemTable({}, add_default_labels=False).shape == (0, 0)


def test_padding_columns_does_not_mutate_input():
    columns = [[200, 100]]
    before = deepcopy(columns)
    ProblemTable(columns)
    assert columns == before


@pytest.mark.parametrize("source_kind", ["mapping", "columns"])
@example(integer=0)
@settings(max_examples=30, deadline=None)
@given(integer=st.integers(min_value=-1000, max_value=1000))
def test_integer_input_preserves_fractional_updates_and_insertions(
    source_kind, integer
):
    columns = {label: [integer] for label in ProblemTableLabel}
    table = (
        ProblemTable(columns, add_default_labels=False)
        if source_kind == "mapping"
        else ProblemTable(list(columns.values()))
    )
    updated = integer + 0.25
    inserted = integer + 0.75
    table.update_row(0, {ProblemTableLabel.T: updated})
    table.insert({ProblemTableLabel.T: inserted}, index=1)
    np.testing.assert_array_equal(table[ProblemTableLabel.T], [updated, inserted])
    assert np.isnan(table[ProblemTableLabel.H_NET][1])
