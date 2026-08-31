"""Domain-valid strategies for stateful ``PinchProblem`` properties."""

from __future__ import annotations

from hypothesis import strategies as st


@st.composite
def nested_zone_multiplier_cases(draw):
    """Generate a nested zone tree, valid descendant path, and multiplier."""
    depth = draw(st.integers(min_value=2, max_value=5))
    zone_names = [f"Area{index}" for index in range(1, depth + 1)]
    selected_depth = draw(st.integers(min_value=1, max_value=depth))
    multiplier = draw(
        st.floats(
            min_value=0.0,
            max_value=20.0,
            allow_nan=False,
            allow_infinity=False,
        )
    )

    child = None
    for name in reversed(zone_names):
        node = {"name": name, "type": "Process Zone"}
        if child is not None:
            node["children"] = [child]
        child = node

    leaf_path = "/".join(("Site", *zone_names))
    selected_path = "/".join(("Site", *zone_names[:selected_depth]))
    payload = {
        "streams": [
            {
                "zone": leaf_path,
                "name": "Generated cold stream",
                "t_supply": 20.0,
                "t_target": 100.0,
                "heat_flow": 80.0,
                "dt_cont": 10.0,
            }
        ],
        "utilities": [],
        "zone_tree": {
            "name": "Site",
            "type": "Site",
            "children": [child],
        },
        "options": {},
    }
    return payload, selected_path, multiplier


def _transactional_payload(
    *,
    suffix: str,
    hot_supply: float,
    hot_target: float,
    cold_supply: float,
    cold_target: float,
    heat_flow: float,
) -> dict[str, object]:
    zone_path = f"Stateful/Plant{suffix}"
    return {
        "streams": [
            {
                "zone": zone_path,
                "name": f"Hot {suffix}",
                "t_supply": hot_supply,
                "t_target": hot_target,
                "heat_flow": heat_flow,
                "dt_cont": 5.0,
            },
            {
                "zone": zone_path,
                "name": f"Cold {suffix}",
                "t_supply": cold_supply,
                "t_target": cold_target,
                "heat_flow": heat_flow * 0.8,
                "dt_cont": 5.0,
            },
        ],
        "utilities": [
            {
                "name": f"Hot utility {suffix}",
                "type": "Hot",
                "t_supply": hot_supply + 50.0,
                "price": 1.0,
            },
            {
                "name": f"Cold utility {suffix}",
                "type": "Cold",
                "t_supply": cold_supply - 20.0,
                "price": 1.0,
            },
        ],
        "zone_tree": {
            "name": "Stateful",
            "type": "Site",
            "children": [{"name": f"Plant{suffix}", "type": "Process Zone"}],
        },
        "options": {},
    }


@st.composite
def transactional_problem_scenarios(draw):
    """Generate two valid loads and a shrinkable state-machine command sequence."""
    hot_supply = float(draw(st.integers(min_value=150, max_value=240)))
    hot_span = float(draw(st.integers(min_value=40, max_value=100)))
    cold_supply = float(draw(st.integers(min_value=10, max_value=60)))
    cold_span = float(draw(st.integers(min_value=60, max_value=120)))
    heat_flow = float(draw(st.integers(min_value=50, max_value=500)))
    offset = float(draw(st.integers(min_value=1, max_value=20)))

    primary = _transactional_payload(
        suffix="A",
        hot_supply=hot_supply,
        hot_target=hot_supply - hot_span,
        cold_supply=cold_supply,
        cold_target=cold_supply + cold_span,
        heat_flow=heat_flow,
    )
    secondary = _transactional_payload(
        suffix="B",
        hot_supply=hot_supply + offset,
        hot_target=hot_supply - hot_span + offset,
        cold_supply=cold_supply + offset,
        cold_target=cold_supply + cold_span + offset,
        heat_flow=heat_flow + offset,
    )
    commands = draw(
        st.lists(
            st.sampled_from(
                ("load_primary", "load_secondary", "target", "fail_replacement")
            ),
            min_size=0,
            max_size=8,
        )
    )
    return (primary, secondary), commands


__all__ = ["nested_zone_multiplier_cases", "transactional_problem_scenarios"]
