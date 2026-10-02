"""Regression tests for the core-targeting audit fixes (area 2)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from OpenPinch import PinchProblem
from OpenPinch.analysis.targeting.utilities import _warn_on_unmet_utility_demand
from OpenPinch.domain.stream import Stream
from OpenPinch.domain.stream_collection import StreamCollection


def _single_zone_payload(**utility_overrides) -> dict:
    steam = {"name": "Steam", "type": "Hot", "t_supply": 200.0, "price": 30.0}
    steam |= utility_overrides
    return {
        "streams": [
            {
                "zone": "Site",
                "name": "C1",
                "t_supply": 50.0,
                "t_target": 150.0,
                "heat_flow": 200.0,
                "dt_cont": 5.0,
            }
        ],
        "utilities": [
            steam,
            {"name": "CW", "type": "Cold", "t_supply": 10.0, "price": 5.0},
        ],
    }


# 2.2 Indirect targeting on a zone without subzones


def test_indirect_targeting_on_a_single_zone_points_to_direct_targeting():
    problem = PinchProblem(_single_zone_payload(), project_name="Site")

    with pytest.raises(ValueError, match="has no subzones"):
        problem.target.indirect_heat_integration()


def test_indirect_targeting_on_a_site_with_subzones_still_runs():
    problem = PinchProblem("zonal_site.json", project_name="Site")

    target = problem.target.indirect_heat_integration()

    assert target.hot_utility_target >= 0.0


# 2.3 Unmet utility demand


def test_capped_utility_shortfall_goes_to_the_default_hot_utility():
    problem = PinchProblem(
        _single_zone_payload(maximum_heat_flow=50.0), project_name="Site"
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        target = problem.target.direct_heat_integration()

    by_name = {u.name: float(u.heat_flow.value) for u in target.hot_utilities}
    assert by_name["Steam"] == pytest.approx(50.0)
    assert by_name["HU"] == pytest.approx(150.0)
    assert sum(by_name.values()) == pytest.approx(target.hot_utility_target)


def test_unmet_utility_demand_is_reported():
    hot = StreamCollection()
    hot.add(
        Stream(
            "Steam",
            supply_temperature=200.0,
            target_temperature=199.9,
            heat_flow=50.0,
            is_process_stream=False,
        )
    )

    with pytest.warns(UserWarning, match="150 is unmet"):
        _warn_on_unmet_utility_demand(
            hot_required=200.0,
            cold_required=0.0,
            hot_utilities=hot,
            cold_utilities=StreamCollection(),
            idx=0,
        )


# 2.4 Near-isothermal streams


def test_near_isothermal_stream_keeps_its_duty():
    problem = PinchProblem(
        {
            "streams": [
                {
                    "zone": "Site",
                    "name": "Condensing vapour",
                    "t_supply": 100.0,
                    "t_target": 100.0 - 5e-6,
                    "heat_flow": 50.0,
                    "dt_cont": 5.0,
                }
            ],
            "utilities": [],
        },
        project_name="Site",
    )

    stream = problem.hot_streams[0]
    span = float(stream.supply_temperature.value - stream.target_temperature.value)
    assert span == pytest.approx(0.01)
    assert float(stream.heat_flow.value) == pytest.approx(50.0)
    target = problem.target.direct_heat_integration()
    assert target.cold_utility_target == pytest.approx(50.0)


# 2.5 Zero approach and 2.6 negative dt_cont in area targeting


def _two_stream_payload(*, dt_cont=None, options=None) -> dict:
    # Equal CPs overlapping from 100 to 150 degC: at zero approach the curves
    # touch over that whole range. Targets: 50 kW hot and 50 kW cold utility.
    streams = [
        {
            "zone": "Site",
            "name": "H1",
            "t_supply": 150.0,
            "t_target": 50.0,
            "heat_flow": 100.0,
            "htc": 1.0,
        },
        {
            "zone": "Site",
            "name": "C1",
            "t_supply": 100.0,
            "t_target": 200.0,
            "heat_flow": 100.0,
            "htc": 1.0,
        },
    ]
    if dt_cont is not None:
        for stream in streams:
            stream["dt_cont"] = dt_cont
    return {"streams": streams, "utilities": [], "options": options or {}}


def test_zero_approach_gives_infinite_area_but_keeps_energy_targets():
    problem = PinchProblem(
        _two_stream_payload(options={"THERMAL_DT_CONT": 0.0}), project_name="Site"
    )

    with pytest.warns(UserWarning, match="area and capital cost targets are infinite"):
        target = problem.target.heat_exchanger_area_and_cost()

    assert target.hot_utility_target == pytest.approx(50.0)
    assert target.area == float("inf")


def test_negative_dt_cont_runs_through_direct_and_area_targeting():
    problem = PinchProblem(_two_stream_payload(dt_cont=-2.0), project_name="Site")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        direct = problem.target.direct_heat_integration()
        area = problem.target.heat_exchanger_area_and_cost()

    assert direct.hot_utility_target >= 0.0
    assert area.area == float("inf")


# 2.7 Exergy on real temperatures


def test_exergetic_gcc_breaks_at_the_real_ambient_temperature():
    from OpenPinch.analysis.exergy.service import build_exergy_gcc_curve
    from OpenPinch.domain.enums import ProblemTableLabel

    # A hot-dominated GCC interval from 30 to 0 degC on shifted temperatures
    # is 35 to 5 degC on real temperatures with a 5 K contribution.
    curve = build_exergy_gcc_curve(
        temperatures=[30.0, 0.0],
        heat_loads=[0.0, 30.0],
        t_env=15.0,
        dt_cont_shift=5.0,
    )
    temperatures = curve[ProblemTableLabel.T.value]

    assert 0.0 in temperatures  # the exergetic temperature is 0 at ambient
    assert len(temperatures) == 3


def test_exergy_nlp_curves_move_hot_branches_to_real_temperatures():
    from OpenPinch.analysis.exergy.service import build_exergy_nlp_curves

    # A hot branch from 12 to 2 degC shifted is 17 to 7 degC real: part of it
    # is above the 15 degC ambient, so it is partly an exergy source.
    kwargs = {
        "temperatures": [12.0, 2.0],
        "branches": [("hot", [0.0, 10.0])],
        "t_env": 15.0,
    }

    unshifted = build_exergy_nlp_curves(**kwargs)
    shifted = build_exergy_nlp_curves(**kwargs, dt_cont_shift=5.0)

    assert unshifted["source_total"] == pytest.approx(0.0)
    assert shifted["source_total"] > 0.0


# 2.9 Utility priority by shifted level


def test_hot_utilities_are_used_in_shifted_temperature_order():
    # Real order: LP (180) is colder than HP (200). Shifted order: HP drops by
    # its 30 K contribution to 170, below LP at 179, so HP is used first.
    payload = _single_zone_payload()
    payload["utilities"] = [
        {"name": "HP", "type": "Hot", "t_supply": 200.0, "dt_cont": 30.0},
        {"name": "LP", "type": "Hot", "t_supply": 180.0, "dt_cont": 1.0},
        {"name": "CW", "type": "Cold", "t_supply": 10.0},
    ]
    problem = PinchProblem(payload, project_name="Site")

    target = problem.target.direct_heat_integration()

    by_name = {u.name: float(u.heat_flow.value) for u in target.hot_utilities}
    assert by_name["HP"] == pytest.approx(target.hot_utility_target)
    assert by_name["LP"] == pytest.approx(0.0)


# 2.10 Small robustness items


def test_composite_curve_with_a_small_duty_is_kept():
    from OpenPinch.analysis.graphs.composite import clean_composite_curve_ends

    temperatures, duties = clean_composite_curve_ends(
        [100.0, 90.0, 80.0], [0.0, 0.0005, 0.001]
    )

    assert len(duties) == 3


def test_secondary_pinch_is_found_at_megawatt_scale():
    from OpenPinch.domain.enums import ProblemTableLabel
    from OpenPinch.domain.problem_table import ProblemTable

    table = ProblemTable(
        {
            ProblemTableLabel.T: [300.0, 200.0, 100.0],
            ProblemTableLabel.H_NET: [5.0e6, 1.0e-4, 2.0e6],
        }
    )

    hot_row, cold_row, valid = table.pinch_idx(ProblemTableLabel.H_NET)

    assert (hot_row, cold_row, valid) == (1, 1, True)


def test_single_row_table_gives_no_net_streams():
    from OpenPinch.analysis.targeting.direct import (
        _create_net_hot_and_cold_stream_collections_for_site_analysis,
    )

    hot_utilities = StreamCollection()
    hot_utilities.add(
        Stream(
            "HU",
            supply_temperature=200.0,
            target_temperature=199.0,
            heat_flow=10.0,
            is_process_stream=False,
        )
    )

    net_hot, net_cold = _create_net_hot_and_cold_stream_collections_for_site_analysis(
        T_vals=np.asarray([100.0]),
        H_vals=np.asarray([0.0]),
        hot_utilities=hot_utilities,
        cold_utilities=StreamCollection(),
        idx=0,
    )

    assert len(net_hot) == 0 and len(net_cold) == 0
