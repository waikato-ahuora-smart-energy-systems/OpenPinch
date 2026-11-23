"""Pinch design splits a case with heat pump streams at the process pinch.

A condenser sized to its pocket makes its level an exact pinch, and it is
hotter than the process pinch, so it is the one targeting reports as the
hot pinch. Split there, the side below the condenser holds the stretch
between the two pinches, which a heat pump target leaves short of heat by
its rounding (8e-5 kW here), and a side below a pinch takes no heater: it
has no solution. The process pinch, the pinch of the case without the
heat pump's streams, puts that stretch on the side with the heater.
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

BASIC_PINCH = (
    Path(OpenPinch.__file__).parent / "tutorials/sample_cases/basic_pinch.json"
)
MULTIPLIER = 0.75
# ``target.carnot_heat_pump(load_fraction=0.5, condensers=1, evaporators=1)``
# on the basic pinch case with every contribution scaled by 0.75:
# (name, role, supply degC, target degC, kW). The target leaves 325.0 kW of
# hot utility.
HEAT_PUMP = [
    ("HP_H1", "condenser", 200.58332400254318, 200.5733240025432, 324.9998592575617),
    ("HP_C1", "evaporator", 130.6690551729923, 130.6790551729923, 241.4188934426253),
]

pytestmark = pytest.mark.synthesis


def _highs_available() -> bool:
    try:
        backend.require_solver_backend("highs")
    except (MissingSynthesisDependencyError, MissingSynthesisSolverError):
        return False
    return True


def _half_load_case() -> dict:
    source = json.loads(BASIC_PINCH.read_text(encoding="utf-8"))
    for name, role, supply, target, duty in HEAT_PUMP:
        source["streams"].append(
            {
                "zone": "Plant",
                "name": name,
                "t_supply": {"value": supply, "unit": "degC"},
                "t_target": {"value": target, "unit": "degC"},
                "heat_flow": {"value": duty, "unit": "kW"},
                # The design at 0.75 scales it to HPR_DT_CONT, 1 K.
                "dt_cont": {"value": 1.0 / MULTIPLIER, "unit": "delta_degC"},
                "heat_pump": {"machine": "HP", "role": role},
            }
        )
    return source


@pytest.mark.skipif("not _highs_available()", reason="needs highspy")
def test_the_milp_designs_a_heat_pump_case_whose_condenser_is_the_hot_pinch():
    design = PinchProblem(
        source=_half_load_case(), project_name="half"
    ).design.heat_exchanger_network(
        width=0,
        depth=0,
        dt_cont_multipliers=[MULTIPLIER],
        max_parallel=1,
        options={"HENS_PDM_MILP": True, "HENS_HEURISTIC_PDM_NETWORKS": 0},
    )

    best = design.top(1)[0].network
    assert best.summary_metrics["hot_utility_load"] == pytest.approx(325.0, abs=0.5)
