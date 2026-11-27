"""A segmented stream's branches report their split fractions.

In an isothermal-mixing stage each branch of a stream spans the stream's
whole temperature change there, so its share of the flow is its share of the
stream's duty in the stage. That share was taken as the duty over the
stream's heat capacity times its temperature change; a segmented stream has
no single heat capacity, and its split fractions came back ``None`` (review
finding 37).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from OpenPinch import PinchProblem
from OpenPinch.analysis.heat_exchanger_networks.extraction.service import (
    extract_heat_exchanger_network,
)
from OpenPinch.analysis.heat_exchanger_networks.solver.arrays import (
    problem_to_solver_arrays,
)


def _problem() -> PinchProblem:
    """H1 in two segments heats C1 and C2 in one stage."""
    return PinchProblem(
        {
            "streams": [
                {
                    "zone": "Site",
                    "name": "H1",
                    "segments": [
                        {"t_supply": 200.0, "t_target": 150.0, "heat_flow": 50.0},
                        {"t_supply": 150.0, "t_target": 100.0, "heat_flow": 50.0},
                    ],
                },
                {
                    "zone": "Site",
                    "name": "C1",
                    "t_supply": 50.0,
                    "t_target": 110.0,
                    "heat_flow": 60.0,
                },
                {
                    "zone": "Site",
                    "name": "C2",
                    "t_supply": 50.0,
                    "t_target": 90.0,
                    "heat_flow": 40.0,
                },
            ],
            "utilities": [
                {"name": "HU", "type": "Hot", "t_supply": 250.0, "t_target": 249.99},
                {"name": "CU", "type": "Cold", "t_supply": 20.0, "t_target": 20.01},
            ],
        }
    )


def test_a_segmented_streams_branches_split_its_flow_by_duty():
    arrays = problem_to_solver_arrays(_problem(), 10.0)
    # The isothermal model's solution: H1 gives 60 kW to C1 and 40 to C2.
    solved = SimpleNamespace(
        S=1,
        Q_r=[[[60.0], [40.0]]],
        Q_r_by_period=[[[[60.0], [40.0]]]],
        z=[[[1.0], [1.0]]],
        z_allowed=[[[1], [1]]],
        Q_h=[0.0, 0.0],
        Q_c=[0.0],
        z_hu=[0.0, 0.0],
        z_cu=[0.0],
        T_h=[[473.15, 373.15]],
        T_c=[[383.15, 323.15], [363.15, 323.15]],
        T_h_by_period=[[[473.15, 373.15]]],
        T_c_by_period=[[[383.15, 323.15], [363.15, 323.15]]],
        area_r=[[[999.0], [999.0]]],
        # The model holds a segmented stream's heat capacity as 0.
        f_h=[0.0],
        f_c=[1.0, 1.0],
    )

    network = extract_heat_exchanger_network(solved, arrays, run_id="split")

    shares = {
        e.sink_stream: e.period_states[0].source_split_fraction
        for e in network.exchangers
    }
    assert shares == pytest.approx({"Site.C1": 0.6, "Site.C2": 0.4})
