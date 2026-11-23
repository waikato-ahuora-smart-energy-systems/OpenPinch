"""A process stream whose profile kinks inside a match.

The basic pinch case with one more hot stream ``V`` shaped like a water
vapour compression condenser: desuperheating from 224.75 to 196.3 degC
(5.7 kW) and on to 174.76 degC (4.67 kW), then condensing between 174.76
and 174.75 degC (177.1 kW). Matched whole against a cold stream that it
heats from near its plateau, both match ends keep their approach while,
at the kink after 10.4 kW, the condensing stream sits below the cold
stream: a crossing inside the match that a check of the ends misses.
"""

from __future__ import annotations

from OpenPinch import PinchProblem
from tests.application.heat_pump_stream_cases import basic_source


def kink_problem(options: dict | None = None) -> PinchProblem:
    source = basic_source()
    source["streams"].append(
        {
            "zone": "Plant",
            "name": "V",
            "dt_cont": {"value": 1.0, "unit": "delta_degC"},
            "segments": [
                {"t_supply": 224.75, "t_target": 196.3, "heat_flow": 5.7},
                {"t_supply": 196.3, "t_target": 174.76, "heat_flow": 4.67},
                {"t_supply": 174.76, "t_target": 174.75, "heat_flow": 177.1},
            ],
        }
    )
    source["options"] = {**source.get("options", {}), **(options or {})}
    return PinchProblem(source=source, project_name="kink")
