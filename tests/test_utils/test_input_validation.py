"""Regression tests for input validation utility helpers."""

import pandas as pd
from OpenPinch.utils.input_validation import (
    validate_stream_data,
    validate_utility_data,
)
from OpenPinch.utils.input_validation import validate_stream_data, validate_utility_data
import OpenPinch.utils.input_validation as iv


def test_validate_stream_data_none_returns_empty_list():
    assert validate_stream_data(None) == []


def test_validate_stream_data_empty_dataframe_returns_empty_dataframe():
    df = pd.DataFrame(columns=["zone", "name"])
    out = validate_stream_data(df)
    assert isinstance(out, pd.DataFrame)
    assert out.empty


def test_validate_stream_data_list_normalises_defaults_and_inherits():
    streams = [
        {"zone": None, "name": None, "t_supply": 150.0},
        {"zone": "", "name": "", "t_supply": 140.0},
        {"zone": "2", "name": "3", "t_supply": 130.0},
        {"zone": "Area.1", "name": "Stream.1", "t_supply": 120.0},
        {"zone": None, "name": None, "t_supply": 110.0},
    ]

    out = validate_stream_data(streams)

    assert [s["zone"] for s in out] == [
        "Process Zone",
        "Process Zone",
        "Z2",
        "Area-1",
        "Area-1",
    ]
    assert [s["name"] for s in out] == ["S1", "S2", "S3", "Stream-1", "S4"]


def test_validate_stream_data_list_skips_non_dict_and_all_empty_rows():
    streams = [
        "not-a-dict",
        {},
        {"zone": None, "name": None},
        {"zone": "Zone A", "name": "H1"},
    ]

    out = validate_stream_data(streams)

    assert len(out) == 1
    assert out[0]["zone"] == "Zone A"
    assert out[0]["name"] == "H1"


def test_validate_stream_data_list_default_name_avoids_existing_name_collisions():
    streams = [
        {"zone": "Zone A", "name": "S1"},
        {"zone": None, "name": None, "t_supply": 120.0},
        {"zone": None, "name": None, "t_supply": 110.0},
    ]

    out = validate_stream_data(streams)

    assert [s["name"] for s in out] == ["S1", "S2", "S3"]


def test_validate_stream_data_dataframe_normalises_data():
    df = pd.DataFrame(
        [
            {"zone": "Zone.1", "name": "1", "t_supply": 100.0},
            {"zone": None, "name": None, "t_supply": 90.0},
            {"zone": "", "name": "", "t_supply": 80.0},
        ]
    )

    out = validate_stream_data(df)

    assert isinstance(out, pd.DataFrame)
    assert out["zone"].tolist() == ["Zone-1", "Zone-1", "Zone-1"]
    assert out["name"].tolist() == ["S1", "S2", "S3"]


def test_validate_utilities_data_none_returns_empty_list():
    assert validate_utility_data(None) == []


def test_validate_utilities_data_list_filters_missing_names():
    utilities = [
        {"name": None, "type": "Hot"},
        {"name": "", "type": "Cold"},
        {"name": "   ", "type": "Cold"},
        {"name": "HP Steam", "type": "Hot"},
    ]

    out = validate_utility_data(utilities)

    assert len(out) == 1
    assert out[0]["name"] == "HP Steam"


def test_validate_utilities_data_dataframe_filters_missing_names():
    df = pd.DataFrame(
        [
            {"name": None, "type": "Hot"},
            {"name": "", "type": "Cold"},
            {"name": "HP Steam", "type": "Hot"},
            {"name": "CW", "type": "Cold"},
        ]
    )

    out = validate_utility_data(df)

    assert isinstance(out, pd.DataFrame)
    assert out["name"].tolist() == ["HP Steam", "CW"]


# ===== Merged from test_input_validation_extra.py =====
"""Additional edge-branch tests for input validation helpers."""


def test_validate_stream_data_stateful_isna_hits_normalize_label_nan_path(monkeypatch):

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
