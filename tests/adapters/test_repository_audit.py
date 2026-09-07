"""Adversarial tabular input cases from the repository audit."""

from io import StringIO

import pandas as pd
import pytest

from OpenPinch.adapters.io.csv import get_problem_from_csv
from OpenPinch.adapters.io.tabular import problem_records_from_frame
from tests.adapters.test_csv import _stream_rows, _utility_rows


@pytest.mark.parametrize("field,column", [("price", 5), ("t_supply", 2)])
def test_csv_rejects_non_numeric_unit_bearing_utility_values(field, column):
    rows = _utility_rows()
    rows[2][column] = "not a number"
    streams = StringIO(pd.DataFrame(_stream_rows()).to_csv(header=False, index=False))
    utilities = StringIO(pd.DataFrame(rows).to_csv(header=False, index=False))
    with pytest.raises(ValueError, match=field):
        get_problem_from_csv(streams, utilities)


@pytest.mark.parametrize("value", [None, "", "  ", float("nan")])
def test_tabular_numeric_blanks_remain_missing(value):
    result = problem_records_from_frame(
        pd.DataFrame({"price": [value]}), {"price": "$/MWh"}
    )
    assert result[0]["price"] == {"value": None, "unit": "$/MWh"}
