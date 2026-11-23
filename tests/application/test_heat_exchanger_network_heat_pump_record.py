"""``record=`` is refused with ``heat_pump=``.

A ``HeatExchangerNetworkRecord`` collects one synthesis run, and a design
with a heat pump is several: one without it and one per multiplier with
it. Filling the record from the first alone would show a caller polling
it a ranking that leaves every heat pump network out.
"""

from __future__ import annotations

import pytest

from OpenPinch import PinchProblem
from OpenPinch.analysis.heat_exchanger_networks.results.record import (
    HeatExchangerNetworkRecord,
)
from tests.application.heat_pump_stream_cases import basic_source


def test_a_record_cannot_follow_a_design_with_a_heat_pump():
    problem = PinchProblem(source=basic_source(), project_name="basic")

    with pytest.raises(ValueError, match="record= is not available with heat_pump="):
        problem.design.heat_exchanger_network(
            # A design that takes a second, should the refusal ever go.
            heat_pump={"search": {"restarts": 0, "evaluations": 1}},
            width=0,
            depth=0,
            max_parallel=1,
            record=HeatExchangerNetworkRecord(),
        )
