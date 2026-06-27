"""Regression tests for capital cost and area targeting analysis routines."""

import json
from pathlib import Path

import numpy as np
import pytest

import OpenPinch.services.common.capital_cost_and_area_targeting as area_targeting
from OpenPinch.classes import Stream, StreamCollection
from OpenPinch.lib.config import Configuration
from OpenPinch.lib.enums import PT
from OpenPinch.services.common.capital_cost_and_area_targeting import (
    _count_crossing,
    _count_crossing_ranges,
    _count_utility_range_container,
    _count_utility_range_containers,
    _map_interval_resistances_to_tdf,
    get_area_targets,
    get_balanced_CC,
    get_capital_cost_targets,
    get_min_number_hx,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "capital_cost_area_cases.json"
)


def _capital_area_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _make_stream(
    name: str,
    t_supply: float,
    t_target: float,
    *,
    heat_flow: float = 100.0,
    dt_cont: float = 0.0,
    htc: float = 10.0,
    is_process_stream: bool = True,
) -> Stream:
    """Helper to create a configured stream for the unit tests."""
    return Stream(
        name=name,
        t_supply=t_supply,
        t_target=t_target,
        heat_flow=heat_flow,
        dt_cont=dt_cont,
        htc=htc,
        is_process_stream=is_process_stream,
    )


def test_get_capital_cost_targets_uses_configured_costing_coefficients():
    fixture = _capital_area_fixture()["costing"]
    config = Configuration(
        options={
            "COSTING_HX_UNIT_COST": fixture["hx_unit_cost"],
            "COSTING_HX_AREA_COEFF": fixture["hx_area_coeff"],
            "COSTING_HX_AREA_EXP": fixture["hx_area_exp"],
            "COSTING_DISCOUNT_RATE": fixture["discount_rate"],
            "COSTING_SERVICE_LIFE": fixture["service_life"],
        }
    )

    capital_cost, annual_capital_cost = get_capital_cost_targets(
        fixture["area"],
        fixture["num_units"],
        config,
    )

    expected_capital = (
        fixture["num_units"] * fixture["hx_unit_cost"]
        + fixture["num_units"]
        * fixture["hx_area_coeff"]
        * (fixture["area"] / fixture["num_units"]) ** fixture["hx_area_exp"]
    )
    recovery_factor = (
        fixture["discount_rate"]
        * (1 + fixture["discount_rate"]) ** fixture["service_life"]
        / ((1 + fixture["discount_rate"]) ** fixture["service_life"] - 1)
    )
    assert capital_cost.value == pytest.approx(expected_capital)
    assert capital_cost.unit == "$"
    assert annual_capital_cost.value == pytest.approx(
        expected_capital * recovery_factor
    )
    assert annual_capital_cost.unit == "$/y"


def test_get_balanced_composite_curve_uses_static_resistance_fixture():
    fixture = _capital_area_fixture()["balanced_composite"]

    result = get_balanced_CC(
        np.array(fixture["T_col"]),
        np.array(fixture["H_hot"]),
        np.array(fixture["H_cold"]),
        np.array(fixture["H_hot_ut"]),
        np.array(fixture["H_cold_ut"]),
        dT_vals=np.array(fixture["dT_vals"]),
        RCP_hot=np.array(fixture["RCP_hot"]),
        RCP_cold=np.array(fixture["RCP_cold"]),
        RCP_hot_ut=np.array(fixture["RCP_hot_ut"]),
        RCP_cold_ut=np.array(fixture["RCP_cold_ut"]),
    )

    updates = result["updates"]
    np.testing.assert_allclose(result["T_col"], np.array(fixture["T_col"]))
    np.testing.assert_allclose(updates[PT.H_HOT_BAL], np.array([100.0, 80.0, 0.0]))
    np.testing.assert_allclose(updates[PT.H_COLD_BAL], np.array([100.0, 60.0, 0.0]))
    np.testing.assert_allclose(updates[PT.RCP_HOT_BAL], np.array([1.5, 2.5, 4.5]))
    np.testing.assert_allclose(updates[PT.RCP_COLD_BAL], np.array([3.0, 4.0, 5.0]))
    np.testing.assert_allclose(
        updates[PT.R_HOT_BAL],
        np.array(fixture["expected_R_hot_bal"]),
    )
    np.testing.assert_allclose(
        updates[PT.R_COLD_BAL],
        np.array(fixture["expected_R_cold_bal"]),
    )


def test_get_balanced_composite_curve_without_resistance_data_returns_heat_updates():
    fixture = _capital_area_fixture()["balanced_composite"]

    result = get_balanced_CC(
        np.array(fixture["T_col"]),
        np.array(fixture["H_hot"]),
        np.array(fixture["H_cold"]),
        np.array(fixture["H_hot_ut"]),
        np.array(fixture["H_cold_ut"]),
    )

    assert set(result["updates"]) == {PT.H_HOT_BAL, PT.H_COLD_BAL}
    np.testing.assert_allclose(result["updates"][PT.H_HOT_BAL], [100.0, 80.0, 0.0])
    np.testing.assert_allclose(result["updates"][PT.H_COLD_BAL], [100.0, 60.0, 0.0])


def test_get_area_targets_returns_expected_total_area_from_static_fixture():
    fixture = _capital_area_fixture()["area_target"]

    area = get_area_targets(
        np.array(fixture["T_vals"]),
        np.array(fixture["H_hot_bal"]),
        np.array(fixture["H_cold_bal"]),
        np.array(fixture["R_hot_bal"]),
        np.array(fixture["R_cold_bal"]),
    )

    assert area == pytest.approx(fixture["expected_area"])


def test_get_area_targets_rejects_internal_shape_mismatch(monkeypatch):
    monkeypatch.setattr(
        area_targeting,
        "get_temperature_driving_forces",
        lambda *_args, **_kwargs: {
            "delta_T1": np.array([30.0, 20.0]),
            "delta_T2": np.array([20.0, 10.0]),
            "dh_vals": np.array([100.0, 80.0]),
            "t_h1": np.array([390.0, 340.0]),
            "t_h2": np.array([360.0, 310.0]),
            "t_c1": np.array([370.0, 345.0]),
            "t_c2": np.array([355.0, 305.0]),
        },
    )
    monkeypatch.setattr(
        area_targeting,
        "_map_interval_resistances_to_tdf",
        lambda *_args, **_kwargs: np.array([0.4]),
    )

    with pytest.raises(ValueError, match="arrays are unequal"):
        get_area_targets(
            np.array([400.0, 350.0, 300.0]),
            np.array([200.0, 100.0, 0.0]),
            np.array([200.0, 100.0, 0.0]),
            np.array([0.0, 0.2, 0.4]),
            np.array([0.0, 0.5, 0.7]),
        )


def test_get_area_targets_requires_balanced_curves():
    T_vals = np.array([300.0, 200.0])
    H_hot_bal = np.array([200.0, 0.0])
    H_cold_bal = np.array([150.0, 0.0])
    R_hot_bal = np.zeros_like(T_vals)
    R_cold_bal = np.zeros_like(T_vals)

    with pytest.raises(
        ValueError, match="requires the inputted composite curves to be balanced"
    ):
        get_area_targets(
            T_vals,
            H_hot_bal,
            H_cold_bal,
            R_hot_bal,
            R_cold_bal,
        )


def test_get_min_number_hx_counts_crossings_and_utilities():
    T_vals = np.array([400.0, 350.0, 300.0, 250.0])
    H_hot_bal = np.array([300.0, 200.0, 100.0, 0.0])
    H_cold_bal = np.array([300.0, 250.0, 100.0, 0.0])

    hot_streams = StreamCollection()
    hot_streams.add(_make_stream("H1", 420.0, 290.0))

    cold_streams = StreamCollection()
    cold_streams.add(_make_stream("C1", 280.0, 360.0))

    hot_utilities = StreamCollection()
    hot_utilities.add(
        _make_stream(
            "HU1",
            390.0,
            310.0,
            heat_flow=50.0,
            is_process_stream=False,
        )
    )

    cold_utilities = StreamCollection()
    cold_utilities.add(
        _make_stream(
            "CU1",
            320.0,
            360.0,
            heat_flow=40.0,
            is_process_stream=False,
        )
    )

    num_hx = get_min_number_hx(
        T_vals,
        H_hot_bal,
        H_cold_bal,
        hot_streams,
        cold_streams,
        hot_utilities,
        cold_utilities,
    )

    assert num_hx == 3


def test_get_min_number_hx_returns_zero_without_pinch_intervals():
    T_vals = np.array([300.0, 250.0, 200.0])
    H_hot_bal = np.array([100.0, 50.0, 0.0])
    H_cold_bal = np.array([100.0, 50.0, 0.0])

    empty_streams = StreamCollection()

    assert (
        get_min_number_hx(
            T_vals,
            H_hot_bal,
            H_cold_bal,
            empty_streams,
            empty_streams,
            empty_streams,
            empty_streams,
        )
        == 0
    )


def test_get_min_number_hx_returns_zero_when_no_balanced_positions_exist():
    empty_streams = StreamCollection()

    assert (
        get_min_number_hx(
            np.array([300.0, 250.0]),
            np.array([100.0, 0.0]),
            np.array([50.0, 25.0]),
            empty_streams,
            empty_streams,
            empty_streams,
            empty_streams,
        )
        == 0
    )


def test_area_targeting_private_interval_helpers_cover_empty_and_scalar_paths():
    streams = StreamCollection()
    streams.add(_make_stream("H1", 420.0, 290.0))
    utilities = StreamCollection()
    utilities.add(
        _make_stream(
            "HU1",
            390.0,
            310.0,
            heat_flow=50.0,
            is_process_stream=False,
        )
    )

    assert _count_crossing(300.0, 400.0, streams) == 1
    assert _count_utility_range_container(300.0, 400.0, utilities) == 1
    assert (
        _count_crossing_ranges(
            np.array([], dtype=float),
            np.array([], dtype=float),
            streams,
        )
        == 0
    )
    assert (
        _count_utility_range_containers(
            np.array([], dtype=float),
            np.array([], dtype=float),
            utilities,
        )
        == 0
    )


def test_map_interval_resistances_to_tdf_aligns_hot_and_cold_intervals():
    resistance = _map_interval_resistances_to_tdf(
        np.array([400.0, 350.0, 300.0]),
        np.array([0.0, 0.2, 0.4]),
        np.array([0.0, 0.5, 0.7]),
        np.array([390.0, 340.0]),
        np.array([360.0, 310.0]),
        np.array([370.0, 345.0]),
        np.array([355.0, 305.0]),
    )

    np.testing.assert_allclose(resistance, np.array([0.7, 1.1]))
