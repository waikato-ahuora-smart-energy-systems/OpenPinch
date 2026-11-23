"""A heat pump network is designed on a problem with several periods.

Crude preheat train, three periods, Carnot heat pump at full load solved
as one machine over all of them. The network is synthesised on the median
basis and fitted to every period. The fit closes each period's balance
with a heater or cooler at a stream end where it needs one, but never on
a heat pump stream: the condenser and the evaporator are served in full
by process streams in every period.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import OpenPinch
from OpenPinch import PinchProblem
from OpenPinch.analysis.heat_exchanger_networks.solver import backend
from OpenPinch.analysis.heat_exchanger_networks.solver.dependencies import (
    MissingSynthesisDependencyError,
    MissingSynthesisSolverError,
)
from OpenPinch.domain.enums import HeatExchangerKind

MULTIPERIOD = (
    Path(OpenPinch.__file__).parent
    / "tutorials/sample_cases/crude_preheat_train_multiperiod.json"
)


def _highs_available() -> bool:
    try:
        backend.require_solver_backend("highs")
    except (MissingSynthesisDependencyError, MissingSynthesisSolverError):
        return False
    return True


pytestmark = [
    pytest.mark.synthesis,
    pytest.mark.skipif(not _highs_available(), reason="needs highspy"),
]


def _served(network, stream: str) -> list[float]:
    """The heat (kW) recovery exchangers move on ``stream`` in each period."""

    on_stream = [
        exchanger
        for exchanger in network.exchangers
        if stream
        in (
            exchanger.source_stream.rsplit(".", 1)[-1],
            exchanger.sink_stream.rsplit(".", 1)[-1],
        )
    ]
    assert on_stream
    assert all(e.kind is HeatExchangerKind.RECOVERY for e in on_stream)
    return [
        sum(e.period_states[n].duty for e in on_stream)
        for n in range(len(on_stream[0].period_states))
    ]


def test_the_fit_serves_each_heat_pump_stream_by_process_heat_in_every_period():
    problem = PinchProblem(
        source=json.loads(MULTIPERIOD.read_text(encoding="utf-8")),
        project_name="periods",
    )

    design = problem.design.heat_exchanger_network(
        heat_pump={
            "load": {"fraction": 1.0},
            "levels": {"condensers": 1, "evaporators": 1},
            "search": {"restarts": 0, "evaluations": 1},
        },
        heat_pump_cost=200.0,
        width=0,
        depth=0,
        max_parallel=1,
    )

    with_pump = [o.network for o in design.top(1000) if o.network.heat_pumps]
    assert with_pump
    for network in with_pump:
        machine = network.heat_pumps[0]
        condenser = _served(network, "HP_H1")
        evaporator = _served(network, "HP_C1")
        # Turndown, base, peak: the largest is the machine's capacity.
        assert len(condenser) == 3
        assert max(condenser) == pytest.approx(machine.condenser_duty, rel=1e-4)
        assert max(evaporator) == pytest.approx(machine.evaporator_duty, rel=1e-4)
        assert condenser[0] < condenser[1] < condenser[2]
