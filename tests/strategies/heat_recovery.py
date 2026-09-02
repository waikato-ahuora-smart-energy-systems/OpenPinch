"""Reusable process-stream strategies for inverse heat-recovery properties."""

from __future__ import annotations

from hypothesis import strategies as st


def _value(draw, *, minimum: int, maximum: int) -> float:
    return float(draw(st.integers(min_value=minimum, max_value=maximum)))


@st.composite
def heat_recovery_problem_payloads(draw):
    """Generate bounded scalar sensible and segmented process stream sets."""
    hot_supply = _value(draw, minimum=180, maximum=300)
    hot_span = _value(draw, minimum=50, maximum=120)
    cold_supply = _value(draw, minimum=0, maximum=80)
    cold_span = _value(draw, minimum=70, maximum=150)
    hot_duty = _value(draw, minimum=50, maximum=500)
    cold_duty = _value(draw, minimum=50, maximum=500)
    hot_target = hot_supply - hot_span
    cold_target = cold_supply + cold_span

    def stream(
        *,
        name: str,
        supply: float,
        target: float,
        duty: float,
        segmented: bool,
    ) -> dict:
        base = {
            "zone": "Site/Process",
            "name": name,
            "dt_cont": _value(draw, minimum=0, maximum=30),
            "htc": 1.0,
            "active": True,
        }
        if not segmented:
            return base | {
                "t_supply": supply,
                "t_target": target,
                "heat_flow": duty,
            }
        midpoint = (supply + target) / 2.0
        return base | {
            "segments": [
                {
                    "name": "S1",
                    "t_supply": supply,
                    "t_target": midpoint,
                    "heat_flow": duty * 0.4,
                },
                {
                    "name": "S2",
                    "t_supply": midpoint,
                    "t_target": target,
                    "heat_flow": duty * 0.6,
                },
            ]
        }

    streams = [
        stream(
            name="Hot active",
            supply=hot_supply,
            target=hot_target,
            duty=hot_duty,
            segmented=draw(st.booleans()),
        ),
        stream(
            name="Cold active",
            supply=cold_supply,
            target=cold_target,
            duty=cold_duty,
            segmented=draw(st.booleans()),
        ),
        {
            "zone": "Site/Process",
            "name": "Hot inactive",
            "t_supply": hot_supply + 20.0,
            "t_target": hot_target,
            "heat_flow": hot_duty,
            "dt_cont": 29.0,
            "htc": 1.0,
            "active": False,
        },
    ]
    return {
        "streams": streams,
        "utilities": [],
        "zone_tree": {
            "name": "Site",
            "type": "Site",
            "children": [{"name": "Process", "type": "Process Zone"}],
        },
        "options": {},
    }


@st.composite
def multiperiod_heat_recovery_problem_payloads(draw):
    """Generate bounded aligned multiperiod sensible stream sets."""
    period_count = draw(st.integers(min_value=2, max_value=3))
    period_ids = ["0", *[f"period-{index}" for index in range(1, period_count)]]

    def values(minimum: int, maximum: int) -> list[float]:
        return [
            float(value)
            for value in draw(
                st.lists(
                    st.integers(min_value=minimum, max_value=maximum),
                    min_size=period_count,
                    max_size=period_count,
                )
            )
        ]

    hot_supply = values(180, 300)
    hot_span = values(50, 120)
    cold_supply = values(0, 80)
    cold_span = values(70, 150)
    hot_duty = values(50, 500)
    cold_duty = values(50, 500)

    def period_value(data: list[float], unit: str) -> dict:
        return {"values": data, "unit": unit}

    return {
        "streams": [
            {
                "zone": "Site/Process",
                "name": "Hot active",
                "t_supply": period_value(hot_supply, "degC"),
                "t_target": period_value(
                    [supply - span for supply, span in zip(hot_supply, hot_span)],
                    "degC",
                ),
                "heat_flow": period_value(hot_duty, "kW"),
                "dt_cont": 11.0,
                "htc": 1.0,
                "active": True,
            },
            {
                "zone": "Site/Process",
                "name": "Cold active",
                "t_supply": period_value(cold_supply, "degC"),
                "t_target": period_value(
                    [supply + span for supply, span in zip(cold_supply, cold_span)],
                    "degC",
                ),
                "heat_flow": period_value(cold_duty, "kW"),
                "dt_cont": 9.0,
                "htc": 1.0,
                "active": True,
            },
            {
                "zone": "Site/Process",
                "name": "Cold inactive",
                "t_supply": period_value(cold_supply, "degC"),
                "t_target": period_value(
                    [value + 10.0 for value in cold_supply],
                    "degC",
                ),
                "heat_flow": period_value([50.0] * period_count, "kW"),
                "dt_cont": 30.0,
                "htc": 1.0,
                "active": False,
            },
        ],
        "utilities": [],
        "zone_tree": {
            "name": "Site",
            "type": "Site",
            "children": [{"name": "Process", "type": "Process Zone"}],
        },
        "options": {"PROBLEM_PERIOD_IDS": period_ids},
    }


__all__ = [
    "heat_recovery_problem_payloads",
    "multiperiod_heat_recovery_problem_payloads",
]
