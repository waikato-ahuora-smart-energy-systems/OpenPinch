"""Additional edge-branch tests for input validation helpers."""

import pandas as pd

from OpenPinch.utils.input_validation import validate_stream_data, validate_utility_data


def test_validate_stream_data_stateful_isna_hits_normalize_label_nan_path(monkeypatch):
    import OpenPinch.utils.input_validation as iv

    real_isna = pd.isna
    calls = {"sentinel": 0}

    def fake_isna(value):
        if value == "SENTINEL":
            calls["sentinel"] += 1
            return calls["sentinel"] >= 3
        return real_isna(value)

    monkeypatch.setattr(iv.pd, "isna", fake_isna)
    out = validate_stream_data([{"zone": "SENTINEL", "name": "H1"}])
    assert out[0]["zone"] == "SENTINEL"


def test_validate_utility_data_empty_dataframe_returns_dataframe():
    df = pd.DataFrame(columns=["name", "type"])
    out = validate_utility_data(df)
    assert isinstance(out, pd.DataFrame)
    assert out.empty


def test_validate_utility_data_list_skips_non_dict_records():
    out = validate_utility_data(
        [
            "not-a-dict",
            {"name": None, "type": "Hot"},
            {"name": "HP Steam", "type": "Hot"},
        ]
    )
    assert out == [{"name": "HP Steam", "type": "Hot"}]
