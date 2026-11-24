"""The exchanger cap is on condensers and evaporators, by their own keys.

``HENS_HEAT_PUMP_STREAM_EXCHANGERS`` limits the exchangers on a condenser
or an evaporator. The air an evaporator draws on is a stream of the case
too, and one air stream can feed several evaporators: it is not capped.
"""

from __future__ import annotations

from OpenPinch.application.heat_pump_design import capped_heat_pump_streams
from tests.application.heat_pump_stream_cases import full_load_target
from tests.application.test_heat_pump_stream_case_air_source import _short_of_heat


def test_the_capped_streams_are_the_condenser_and_the_evaporator_by_key():
    problem = _short_of_heat()
    case = problem.with_heat_pump_streams(full_load_target(problem))

    # The case also holds the air stream, ``Plant.AIR_H1``.
    assert capped_heat_pump_streams(case) == {"Plant.HP_H1", "Plant.HP_C1"}
