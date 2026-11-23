"""A site of two processes linked by two steam levels.

Process A rejects 1000 kW between 170 and 150 degC, which can raise LPS
(140 degC) and nothing hotter. Process B needs 800 kW between 150 and
170 degC, which takes MPS (180 degC). LPS cannot be let down to MPS, so
the site imports 800 kW of MPS and dumps 1000 kW of LPS. Read as one
process the two streams would recover 400 kW from each other, leaving
400 kW of heating and 600 kW of cooling.
"""

from __future__ import annotations

from OpenPinch import PinchProblem

SITE_HOT_UTILITY = 800.0
SITE_COLD_UTILITY = 1000.0


def _value(value: float, unit: str = "degC") -> dict:
    return {"value": value, "unit": unit}


def _stream(zone: str, name: str, supply: float, target: float, duty: float) -> dict:
    return {
        "zone": zone,
        "name": name,
        "t_supply": _value(supply),
        "t_target": _value(target),
        "heat_flow": _value(duty, "kW"),
        "dt_cont": _value(5.0),
        "htc": _value(1.0, "kW/m^2/degC"),
    }


def _utility(name: str, kind: str, supply: float, target: float, price: float) -> dict:
    return {
        "name": name,
        "type": kind,
        "t_supply": _value(supply),
        "t_target": _value(target),
        "dt_cont": _value(0.0),
        "price": _value(price, "$/MWh"),
        "htc": _value(1.0, "kW/m^2/degC"),
        "heat_flow": None,
    }


def two_level_site_source(options: dict | None = None) -> dict:
    return {
        "streams": [
            _stream("A", "A hot", 170.0, 150.0, 1000.0),
            _stream("B", "B cold", 150.0, 170.0, 800.0),
        ],
        "utilities": [
            _utility("MPS", "Both", 180.0, 179.9, 40.0),
            _utility("LPS", "Both", 140.0, 139.9, 20.0),
            _utility("CW", "Cold", 20.0, 25.0, 2.0),
        ],
        "options": dict(options or {}),
    }


def two_level_site(options: dict | None = None) -> PinchProblem:
    return PinchProblem(
        source=two_level_site_source(options), project_name="two level site"
    )


def utility_heat_pump_target(problem: PinchProblem, **arguments):
    """The site's utility heat pump at full load, with one evaluation.

    One evaluation of the search takes its starting point as it is, which
    is enough for what the target is sized on; the levels it lands at are
    not an optimum.
    """
    return problem.target.heat_pump_and_refrigeration(
        "carnot",
        placement="utility",
        load={"fraction": 1.0},
        levels={"condensers": 1, "evaporators": 1},
        search={"restarts": 0, "evaluations": 1},
        **arguments,
    )
