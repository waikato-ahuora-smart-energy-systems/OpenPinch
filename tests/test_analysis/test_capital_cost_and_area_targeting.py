import numpy as np
import pytest

from OpenPinch.analysis.capital_cost_and_area_targeting import (
    get_area_targets,
    get_min_number_hx,
)
from OpenPinch.classes import Stream, StreamCollection


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

# TODO: Check the test
# def test_get_area_targets_returns_expected_total_area(monkeypatch):
#     T_vals = np.array([300.0, 250.0, 200.0])
#     H_hot_bal = np.array([300.0, 150.0, 0.0])
#     H_cold_bal = np.array([300.0, 200.0, 0.0])
#     R_hot_bal = np.array([0.0, 0.2, 0.3])
#     R_cold_bal = np.array([0.0, 0.4, 0.1])

#     tdf_payload = {
#         "delta_T1": np.array([30.0, 20.0]),
#         "delta_T2": np.array([20.0, 10.0]),
#         "dh_vals": np.array([100.0, 80.0]),
#         "t_h1": np.array([295.0, 245.0]),
#         "t_h2": np.array([255.0, 205.0]),
#         "t_c1": np.array([265.0, 225.0]),
#         "t_c2": np.array([235.0, 195.0]),
#     }
#     monkeypatch.setattr(
#         "OpenPinch.analysis.capital_cost_and_area_targeting.get_temperature_driving_forces",
#         lambda *args, **kwargs: tdf_payload,
#     )

#     area = get_area_targets(
#         T_vals,
#         H_hot_bal,
#         H_cold_bal,
#         R_hot_bal,
#         R_cold_bal,
#     )

#     lmtd = (tdf_payload["delta_T1"] - tdf_payload["delta_T2"]) / np.log(
#         tdf_payload["delta_T1"] / tdf_payload["delta_T2"]
#     )
#     resistance_totals = np.array([0.6, 0.4])
#     overall_u = 1.0 / resistance_totals
#     expected_area = np.sum(tdf_payload["dh_vals"] / (overall_u * lmtd))

#     assert area == pytest.approx(expected_area, rel=1e-9)


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
