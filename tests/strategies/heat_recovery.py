"""Reusable process-stream strategies for inverse heat-recovery properties."""

from __future__ import annotations

from hypothesis import strategies as st


def _value(draw, *, minimum: int, maximum: int) -> float:
    return float(draw(st.integers(min_value=minimum, max_value=maximum)))


def _duty(draw) -> float:
    return float(
        draw(
            st.one_of(
                st.integers(min_value=1, max_value=500),
                st.sampled_from([1e-12, 5e-7, 1e-6, 1e-3, 0.5]),
            )
        )
    )


@st.composite
def heat_recovery_problem_payloads(draw):
    """Generate bounded scalar sensible and segmented process stream sets."""
    hot_supply = _value(draw, minimum=180, maximum=300)
    hot_span = _value(draw, minimum=50, maximum=120)
    cold_supply = _value(draw, minimum=0, maximum=80)
    cold_span = _value(draw, minimum=70, maximum=150)
    hot_duty = _duty(draw)
    cold_duty = _duty(draw)
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
        # Segment validation requires every segment duty to exceed 1e-6. Keep
        # micro-duty examples sensible so the generated input remains valid.
        if not segmented or duty <= 2.5e-6:
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
    for index in range(draw(st.integers(min_value=0, max_value=2))):
        supply = _value(draw, minimum=160, maximum=320)
        span = _value(draw, minimum=20, maximum=140)
        streams.append(
            stream(
                name=f"Hot active {index + 2}",
                supply=supply,
                target=supply - span,
                duty=_duty(draw),
                segmented=draw(st.booleans()),
            )
        )
    for index in range(draw(st.integers(min_value=0, max_value=2))):
        supply = _value(draw, minimum=-20, maximum=120)
        span = _value(draw, minimum=20, maximum=170)
        streams.append(
            stream(
                name=f"Cold active {index + 2}",
                supply=supply,
                target=supply + span,
                duty=_duty(draw),
                segmented=draw(st.booleans()),
            )
        )
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
    period_ids = draw(
        st.sampled_from(
            [
                ["0", "peak"],
                ["value", "unit"],
                ["base", "peak", "night"],
            ]
        )
    )
    period_count = len(period_ids)

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


@st.composite
def threshold_problem_payloads(draw):
    """Generate balanced two-stream threshold problems with a positive plateau."""
    cold_supply = _value(draw, minimum=20, maximum=80)
    temperature_span = _value(draw, minimum=50, maximum=100)
    threshold_approach = _value(draw, minimum=10, maximum=80)
    heat_recovery = _value(draw, minimum=50, maximum=500)
    hot_target = cold_supply + threshold_approach
    hot_supply = hot_target + temperature_span
    cold_target = cold_supply + temperature_span
    return {
        "streams": [
            {
                "zone": "Site/Process",
                "name": "Hot threshold stream",
                "t_supply": hot_supply,
                "t_target": hot_target,
                "heat_flow": heat_recovery,
                "dt_cont": _value(draw, minimum=0, maximum=30),
                "htc": 1.0,
            },
            {
                "zone": "Site/Process",
                "name": "Cold threshold stream",
                "t_supply": cold_supply,
                "t_target": cold_target,
                "heat_flow": heat_recovery,
                "dt_cont": _value(draw, minimum=0, maximum=30),
                "htc": 1.0,
            },
        ],
        "utilities": [],
        "zone_tree": {
            "name": "Site",
            "type": "Site",
            "children": [{"name": "Process", "type": "Process Zone"}],
        },
        "options": {},
    }


__all__ = [
    "heat_recovery_problem_payloads",
    "multiperiod_heat_recovery_problem_payloads",
    "threshold_problem_payloads",
]
