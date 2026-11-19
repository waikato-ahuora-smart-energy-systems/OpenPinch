"""A site's utility levels, read from its problem for the site balance.

A hot and a cold utility of one name with matching temperatures are one
main (a ``Both`` utility); anything else stays a level of its own. Each
level keeps its temperatures and prices per period, and a hot utility's
``maximum_heat_flow`` caps what the site buys into its level.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import OpenPinch
from OpenPinch.analysis.heat_exchanger_networks.site.levels import (
    site_utility_system,
    utility_capacities,
)
from OpenPinch.analysis.heat_exchanger_networks.solver.arrays import (
    problem_to_solver_arrays,
)
from OpenPinch.analysis.heat_exchanger_networks.solver.utility_slices import (
    with_utility_slices,
)
from OpenPinch.application.problem import PinchProblem

ZONAL_SITE = (
    Path(OpenPinch.__file__).parent / "tutorials" / "sample_cases" / "zonal_site.json"
)


def _quantity(value, unit):
    return {"value": value, "unit": unit}


def _utility(name, kind, supply, target, price, cap=None):
    return {
        "name": name,
        "type": kind,
        "t_supply": _quantity(supply, "degC"),
        "t_target": _quantity(target, "degC"),
        "dt_cont": _quantity(2.5, "delta_degC"),
        "price": _quantity(price, "$/MWh"),
        "htc": _quantity(1.0, "kW/m^2/delta_degC"),
        "heat_flow": None,
        "maximum_heat_flow": None if cap is None else _quantity(cap, "kW"),
    }


def _stream(name, supply, target, heat_flow):
    return {
        "zone": "Site/A",
        "name": name,
        "t_supply": _quantity(supply, "degC"),
        "t_target": _quantity(target, "degC"),
        "heat_flow": _quantity(heat_flow, "kW"),
        "dt_cont": _quantity(5.0, "delta_degC"),
        "htc": _quantity(1.0, "kW/m^2/delta_degC"),
    }


def _problem(utilities) -> PinchProblem:
    return PinchProblem(
        source={
            "streams": [
                _stream("H1", 200.0, 60.0, 140.0),
                _stream("C1", 50.0, 190.0, 140.0),
            ],
            "utilities": utilities,
            "zone_tree": {
                "name": "Site",
                "type": "Site",
                "children": [{"name": "A", "type": "Process Zone"}],
            },
        }
    )


def test_zonal_site_has_a_boiler_four_mains_and_cooling_water():
    """Every price is 30 $/MWh: 249 $/kW/y over 8300 h."""
    arrays = problem_to_solver_arrays(PinchProblem(source=str(ZONAL_SITE)), 10.0)

    levels = site_utility_system(arrays).levels

    kinds = {level.name: level.kind for level in levels.levels}
    assert kinds == {
        "VHPS": "hot",
        "HPS": "both",
        "MPS": "both",
        "LPS": "both",
        "HW": "both",
        "CW": "cold",
    }
    hw = levels.level("HW")
    assert hw.t_high == pytest.approx((90.0,), abs=1e-9)
    assert hw.t_low == pytest.approx((70.0,), abs=1e-9)
    assert hw.import_price == pytest.approx((249.0,), rel=1e-12)
    assert hw.dump_price == pytest.approx((249.0,), rel=1e-12)
    assert levels.period_weights == pytest.approx((1.0,), rel=1e-12)


def test_same_named_hot_and_cold_utilities_at_other_temperatures_stay_apart():
    """A hot utility "W" at 150 C and a cold utility "W" at 20 C are two
    levels; heat raised into the cold one cannot meet use of the hot one."""
    problem = _problem(
        [
            _utility("W", "Hot", 150.0, 149.9, 40.0),
            _utility("W", "Cold", 20.0, 25.0, 5.0),
        ]
    )

    system = site_utility_system(problem_to_solver_arrays(problem, 10.0))

    kinds = {level.name: level.kind for level in system.levels.levels}
    assert kinds == {"W (hot)": "hot", "W (cold)": "cold"}


def test_a_hot_utilitys_maximum_heat_flow_caps_its_imports():
    problem = _problem(
        [
            _utility("LPS", "Both", 150.0, 149.9, 30.0, cap=12.5),
            _utility("CW", "Cold", 20.0, 25.0, 5.0),
        ]
    )
    arrays = problem_to_solver_arrays(problem, 10.0)

    system = site_utility_system(
        arrays, capacities=utility_capacities(problem.master_zone)
    )

    assert system.levels.level("LPS").import_capacity == pytest.approx((12.5,))
    assert system.levels.level("CW").import_capacity is None


def test_each_utility_of_sliced_arrays_belongs_to_its_parents_level():
    """Hot water sliced at a stream's supply is still the HW level."""
    arrays = problem_to_solver_arrays(PinchProblem(source=str(ZONAL_SITE)), 10.0)
    system = site_utility_system(arrays)
    sliced = with_utility_slices(arrays)
    names = [level.name for level in system.levels.levels]

    hot, cold = system.members(sliced)

    for identity, level in zip(
        sliced.utility_identities["hot_utilities"], hot, strict=True
    ):
        parent = identity.split("@")[0].removeprefix("Hot Utility.")
        assert names[level] == parent
    assert len(cold) == len(sliced.utility_identities["cold_utilities"])
