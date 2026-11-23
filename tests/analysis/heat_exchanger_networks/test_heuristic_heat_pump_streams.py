"""The heuristic designs a case that holds a heat pump's two streams.

A heat pump target adds a condensing hot stream above the pinch and an
evaporating cold stream below it, each sized to its pocket of the grand
composite curve. The cascade closes at both levels, so each pinch side has
a second pinch, and a design that ticks streams off from the process pinch
alone runs out of temperature there. Split at every pinch, the heuristic
returns networks at the heat pump's target.

Left out: half load at multiplier 0.75. There the condenser level is the
hotter of two exact pinches and pinch design splits at it; the side model
below it fails for the MILP and the heuristic alike.
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
from OpenPinch.domain.enums import HeatExchangerNetworkDesignMethod

BASIC_PINCH = (
    Path(OpenPinch.__file__).parent / "tutorials/sample_cases/basic_pinch.json"
)
# The heat pump streams' contribution at the design's multiplier (K).
HEAT_PUMP_DT_CONT = 1.0
# ``target.carnot_heat_pump(load_fraction=load, condensers=1, evaporators=1)``
# on the basic pinch case with every contribution scaled by the multiplier
# (17 s a call, so recorded): (load, multiplier) to the condenser's and the
# evaporator's (supply degC, target degC, kW) and the hot utility left (kW).
HEAT_PUMP_TARGETS = {
    (1.0, 0.75): (
        (222.249999, 222.239999, 649.992137),
        (121.247311, 121.257311, 429.85378),
        0.065,
    ),
    (1.0, 1.0): (
        (225.999999, 225.989999, 749.997934),
        (119.660928, 119.670928, 486.58145),
        0.075,
    ),
    (1.0, 1.5): (
        (233.499999, 233.489999, 949.987553),
        (116.777314, 116.787314, 594.253721),
        0.095,
    ),
    (0.5, 1.0): (
        (200.999999, 200.989999, 374.998459),
        (130.116466, 130.126466, 277.470686),
        375.037,
    ),
    (0.5, 1.5): (
        (201.833332, 201.823332, 474.998928),
        (129.050572, 129.060572, 348.785881),
        475.048,
    ),
}

pytestmark = pytest.mark.synthesis


def _highs_available() -> bool:
    try:
        backend.require_solver_backend("highs")
    except (MissingSynthesisDependencyError, MissingSynthesisSolverError):
        return False
    return True


def _case_with_heat_pump(load: float, multiplier: float) -> dict:
    """The basic pinch case with the two streams of its heat pump target."""

    source = json.loads(BASIC_PINCH.read_text(encoding="utf-8"))
    condenser, evaporator, _hot_utility = HEAT_PUMP_TARGETS[(load, multiplier)]
    for name, (supply, target, duty) in (("HP_H1", condenser), ("HP_C1", evaporator)):
        source["streams"].append(
            {
                "zone": "Plant",
                "name": name,
                "t_supply": {"value": supply, "unit": "degC"},
                "t_target": {"value": target, "unit": "degC"},
                "heat_flow": {"value": duty, "unit": "kW"},
                # The design scales it back to ``HEAT_PUMP_DT_CONT``.
                "dt_cont": {"value": HEAT_PUMP_DT_CONT / multiplier, "unit": "degC"},
                "htc": {"value": 1.0, "unit": "kW/m^2/degC"},
            }
        )
    return source


@pytest.mark.skipif("not _highs_available()", reason="needs highspy")
@pytest.mark.parametrize(("load", "multiplier"), sorted(HEAT_PUMP_TARGETS))
def test_the_heuristic_realises_a_network_with_the_heat_pump_streams(load, multiplier):
    design = PinchProblem(
        source=_case_with_heat_pump(load, multiplier), project_name="heat-pump"
    ).design.heat_exchanger_network(
        width=0,
        depth=0,
        dt_cont_multipliers=[multiplier],
        max_parallel=1,
        options={"HENS_PDM_MILP": False},
    )

    best = design.top(1)[0]
    assert best.method is HeatExchangerNetworkDesignMethod.HeuristicPinchDesign
    assert best.network.summary_metrics["hot_utility_load"] == pytest.approx(
        HEAT_PUMP_TARGETS[(load, multiplier)][2], abs=0.5
    )
