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
