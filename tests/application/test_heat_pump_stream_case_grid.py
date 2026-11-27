"""A heat pump stream beside a process temperature keeps its duty.

A converged heat pump target puts a level at a kink of the composite
curves, to the search's tolerance: a condenser 1.4e-6 K under the
temperature the cold stream ends at, on the levels case at a multiplier
of 1.5. The problem table merges temperatures within 1e-6 K, so the
condenser's 0.01 K stream was stretched onto the process temperature and
gained 1e-4 of its duty (0.027 of 270 kW). The case then had more
condenser heat than the target placed and no network could be designed
for it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from OpenPinch import PinchProblem
from OpenPinch.application.case_derivation import _case_with_heat_pump_streams
from OpenPinch.application.heat_pump_design import _on_process_temperatures, _scaled
from OpenPinch.domain.stream import Stream
from OpenPinch.domain.stream_collection import StreamCollection
from tests.analysis.heat_exchanger_networks.heat_pump_levels_case import (
    levels_problem,
)

# The levels case's Carnot target at a multiplier of 1.5, as a search gave
# it: the cold stream ends at 120 degC, 127.5 degC shifted, and the
# condenser sits its 1.5 K above that, less the search's tolerance.
CONDENSER = (128.99999863591916, 269.99992107601383)
EVAPORATOR = (38.33619490856593, 170.65522036573628)


def _level(name, supply, target, duty):
    return Stream(
        name=name,
        supply_temperature=supply,
        target_temperature=target,
        heat_flow=duty,
        delta_t_contribution=0.0,
        heat_transfer_coefficient=1.0,
        is_process_stream=False,
    )


def _target(root, *, condenser, evaporator):
    """A Carnot target of one condenser and one evaporator, each given as
    its temperature (degC) and duty (kW), as the stream writer reads it."""
    (t_cond, q_cond), (t_evap, q_evap) = condenser, evaporator
    return SimpleNamespace(
        provenance=SimpleNamespace(zone_address=root.address),
        config=root.config,
        hpr_shared_design=False,
        hpr_details=SimpleNamespace(amb_streams=StreamCollection(), period_outputs={}),
        hpr_hot_streams={"HP_H1": _level("HP_H1", t_cond, t_cond - 0.01, q_cond)},
        hpr_cold_streams={"HP_C1": _level("HP_C1", t_evap - 0.01, t_evap, q_evap)},
    )


def _case_at_the_design_multiplier(multiplier):
    problem = levels_problem()
    root = problem._require_prepared_root_zone()
    target = _target(root, condenser=CONDENSER, evaporator=EVAPORATOR)
    approach = float(root.config.hpr.dt_cont)
    case = _case_with_heat_pump_streams(problem, target, dt_cont=approach, machine="HP")
    # As the design case does.
    case = _on_process_temperatures(
        case, _scaled(problem, multiplier), approach=approach * multiplier
    )
    return _scaled(case, multiplier)


def test_a_condenser_just_under_a_process_temperature_is_moved_onto_it():
    # The basic pinch case at a multiplier of 2.0: F3 ends at 230 degC,
    # 240 degC shifted, and the condenser's own 2 K above that is 242 degC.
    # The search stopped 1e-5 K under it, which left the top 1e-5 K of F3
    # to the hot utility: a second pinch at the condenser and no network
    # (no design from 1e-5 to 1e-4 K under; twenty networks on it).
    from tests.application.heat_pump_stream_cases import basic_source

    problem = PinchProblem(source=basic_source(), project_name="basic")
    root = problem._require_prepared_root_zone()
    target = _target(
        root,
        condenser=(241.99999007149836, 1149.9998460724764),
        evaporator=(113.46405008375231, 690.7189940361919),
    )
    case = _case_with_heat_pump_streams(problem, target, dt_cont=1.0, machine="HP")

    moved = _on_process_temperatures(case, _scaled(problem, 2.0), approach=2.0)

    records = {
        record["name"]: record
        for record in moved.to_problem_json()["streams"]
        if record.get("heat_pump")
    }
    assert records["HP_H1"]["t_supply"]["value"] == 242.0
    # The evaporator is beside no process temperature and stays where the
    # target put it, to the last digit: a level between two others sits at
    # its approach limit, and a design of three condensers that closed
    # with its levels as given did not with them rounded to 1e-5 K.
    assert records["HP_C1"]["t_target"]["value"] == 113.46405008375231
    direct = _scaled(moved, 2.0).target.direct_heat_integration()
    assert (direct.hot_pinch, direct.cold_pinch) == (150.0, 150.0)


def test_a_condenser_a_hair_under_a_process_temperature_keeps_its_duty():
    direct = _case_at_the_design_multiplier(1.5).target.direct_heat_integration()

    # Two 600 kW process streams balance, so the case's utilities differ
    # by the heat pump's work: the condenser's duty less the evaporator's.
    surplus = float(direct.cold_utility_target) - float(direct.hot_utility_target)
    assert surplus == pytest.approx(CONDENSER[1] - EVAPORATOR[1], abs=1e-6)


def _quantity(value, unit="degC"):
    key = "values" if isinstance(value, list) else "value"
    return {key: value, "unit": unit}


def _record(name, supply, target, duty=600.0, *, contribution=5.0, mark=None):
    return {
        "zone": "Site/P",
        "name": name,
        "t_supply": _quantity(supply),
        "t_target": _quantity(target),
        "heat_flow": _quantity(duty, "kW"),
        "dt_cont": _quantity(contribution, "delta_degC"),
        "htc": _quantity(1.0, "kW/m^2/delta_degC"),
        "heat_pump": mark,
    }


def _plant(*streams, periods=None):
    """H1 from 90 to 30 degC and the cold streams given, with steam and
    cooling water; ``periods`` names the periods of a problem with several."""
    utilities = [
        {
            "name": name,
            "type": kind,
            "t_supply": _quantity(supply),
            "t_target": _quantity(target),
            "dt_cont": _quantity(2.5, "delta_degC"),
            "price": _quantity(price, "$/MWh"),
            "htc": _quantity(1.0, "kW/m^2/delta_degC"),
            "heat_flow": None,
        }
        for name, kind, supply, target, price in (
            ("HPS", "Hot", 250.0, 249.9, 40.0),
            ("CW", "Cold", 10.0, 15.0, 5.0),
        )
    ]
    options = {"THERMAL_DT_CONT": 5.0}
    if periods:
        options |= {
            "PROBLEM_PERIOD_IDS": list(periods),
            "PROBLEM_PERIOD_WEIGHTS": [1.0 / len(periods)] * len(periods),
        }
    return PinchProblem(
        source={
            "streams": [_record("H1", 90.0, 30.0), *streams],
            "utilities": utilities,
            "options": options,
            "zone_tree": {
                "name": "Site",
                "type": "Site",
                "children": [{"name": "P", "type": "Process Zone"}],
            },
        }
    )


_CONDENSER = {"machine": "HP", "role": "condenser", "placement": "process"}


def test_a_stream_already_on_a_process_temperature_is_not_moved_off_it():
    # At a multiplier of 1.5 C1 ends at 127.5 degC shifted and C2 at
    # 127.49005. The condenser starts on the first, its own 1.5 K above,
    # and ends 5e-5 K from the second: the end must not pull the stream
    # off the temperature its start is on.
    cold = (_record("C1", 20.0, 120.0), _record("C2", 20.0, 119.99005))
    condenser = _record(
        "HP_H1", 129.0, 128.99, 270.0, contribution=1.0, mark=_CONDENSER
    )
    case = _plant(*cold, condenser)

    moved = _on_process_temperatures(case, _scaled(_plant(*cold), 1.5), approach=1.5)

    (record,) = [r for r in moved.to_problem_json()["streams"] if r["name"] == "HP_H1"]
    assert record["t_supply"]["value"] == 129.0


def test_each_period_is_moved_onto_its_own_process_temperatures():
    # C1 ends at 120 degC in the first period and 122.5 in the second:
    # 127.5 and 130 degC shifted at a multiplier of 1.5. A condenser 1e-5 K
    # under 129 degC is beside a process temperature in the first period
    # only; in the second it is a level of its own and stays as given.
    cold = _record("C1", 20.0, [120.0, 122.5])
    condenser = _record(
        "HP_H1",
        [128.99999, 128.99999],
        [128.98999, 128.98999],
        [270.0, 270.0],
        contribution=1.0,
        mark=_CONDENSER,
    )
    periods = ("first", "second")
    case = _plant(cold, condenser, periods=periods)
    process = _scaled(_plant(cold, periods=periods), 1.5)

    moved = _on_process_temperatures(case, process, approach=1.5)

    (record,) = [r for r in moved.to_problem_json()["streams"] if r["name"] == "HP_H1"]
    assert record["t_supply"]["values"] == pytest.approx([129.0, 128.99999], abs=1e-9)
