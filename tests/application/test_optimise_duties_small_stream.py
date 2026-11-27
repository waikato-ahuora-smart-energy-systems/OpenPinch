"""A small stream's exchangers are kept beside a large stream's.

The fixed-structure model took an exchanger with a duty under 1e-4 of the
largest stream load as idle. Beside a 50,000 kW stream that is 5 kW, so a
3 kW stream's heater and cooler were removed and ``optimise_duties``
failed: "every process stream needs at least one exchanger in a fixed
structure; no exchanger serves: P.Csmall".
"""

from __future__ import annotations

import pytest

from OpenPinch import PinchProblem


def _problem(big: float, small: float) -> PinchProblem:
    def stream(name, supply, target, duty):
        return {
            "zone": "P",
            "name": name,
            "t_supply": supply,
            "t_target": target,
            "heat_flow": duty,
            "htc": 1.0,
            "dt_cont": 5.0,
        }

    return PinchProblem(
        {
            "streams": [
                stream("H1", 200.0, 80.0, big),
                stream("C1", 40.0, 150.0, 0.9 * big),
                stream("Hsmall", 120.0, 60.0, small),
                stream("Csmall", 30.0, 90.0, small),
            ],
            "utilities": [
                {
                    "name": "HU",
                    "type": "Hot",
                    "t_supply": 250.0,
                    "t_target": 249.9,
                    "htc": 1.0,
                    "price": 30.0,
                },
                {
                    "name": "CU",
                    "type": "Cold",
                    "t_supply": 15.0,
                    "t_target": 20.0,
                    "htc": 1.0,
                    "price": 5.0,
                },
            ],
        },
        project_name="tolerance",
    )


@pytest.mark.synthesis
@pytest.mark.solver
def test_a_small_streams_exchangers_are_kept_beside_a_large_stream():
    view = _problem(50000.0, 3.0).design.optimise_duties(
        {
            "recovery": [("H1", "C1", 1)],
            "heaters": ["C1", "Csmall"],
            "coolers": ["H1", "Hsmall"],
        },
        objective="utility",
    )

    duties = {
        exchanger.exchanger_id: exchanger.period_states[0].duty
        for exchanger in view.selected_network.exchangers
    }
    assert duties["hot-utility:HU->Csmall"] == pytest.approx(3.0, abs=1e-3)
    assert duties["cold-utility:Hsmall->CU"] == pytest.approx(3.0, abs=1e-3)
