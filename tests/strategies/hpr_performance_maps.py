"""Domain-specific Hypothesis strategies for HPR performance maps."""

from __future__ import annotations

from hypothesis import strategies as st

_IDENTIFIERS = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789-_",
    min_size=1,
    max_size=24,
)
_FINITE_JSON_NUMBERS = st.floats(
    min_value=-1_000_000.0,
    max_value=1_000_000.0,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)
_JSON_SCALARS = st.one_of(
    st.none(),
    st.booleans(),
    _FINITE_JSON_NUMBERS,
    st.text(max_size=30),
)
json_values = st.recursive(
    _JSON_SCALARS,
    lambda children: st.one_of(
        st.lists(children, max_size=4),
        st.dictionaries(_IDENTIFIERS, children, max_size=4),
    ),
    max_leaves=12,
)
provenance_objects = st.dictionaries(
    _IDENTIFIERS,
    json_values,
    min_size=1,
    max_size=5,
)


@st.composite
def hpr_performance_map_payloads(draw):
    """Generate one physically valid alpha-schema HPR map payload."""
    mode = draw(st.sampled_from(("heat_pump", "refrigeration")))
    map_id = draw(_IDENTIFIERS)
    reference_capacity = float(draw(st.integers(min_value=10, max_value=10_000)))
    source_temperature = float(draw(st.integers(min_value=-40, max_value=80)))
    lift = float(draw(st.integers(min_value=1, max_value=120)))
    sink_temperature = source_temperature + lift
    load_fractions = draw(
        st.lists(
            st.sampled_from((0.25, 0.5, 0.75, 1.0)),
            min_size=1,
            max_size=4,
            unique=True,
        ).map(sorted)
    )
    curve_id = f"curve-{source_temperature:g}-{sink_temperature:g}"
    points = []
    for index, load_fraction in enumerate(load_fractions):
        useful_duty = reference_capacity * load_fraction
        electric_power = useful_duty / 4.0
        if mode == "heat_pump":
            q_source = useful_duty - electric_power
            q_sink = useful_duty
        else:
            q_source = useful_duty
            q_sink = useful_duty + electric_power
        points.append(
            {
                "name": f"point-{index}-{load_fraction:g}",
                "curve_id": curve_id,
                "source_temperature": source_temperature,
                "sink_temperature": sink_temperature,
                "load_fraction": load_fraction,
                "q_source": q_source,
                "q_sink": q_sink,
                "electric_power": electric_power,
                "cop": 4.0,
            }
        )
    return {
        "schema_version": "1.0",
        "map_id": map_id,
        "mode": mode,
        "units": {
            "source_temperature": "degC",
            "sink_temperature": "degC",
            "q_source": "kW",
            "q_sink": "kW",
            "electric_power": "kW",
        },
        "reference_capacity": reference_capacity,
        "reference_capacity_basis": ("q_sink" if mode == "heat_pump" else "q_source"),
        "interpolation_topology": "ordered_part_load_curve",
        "thermodynamic_backend": draw(st.sampled_from(("coolprop", "tespy"))),
        "model_id": draw(_IDENTIFIERS),
        "provenance": draw(provenance_objects),
        "points": points,
        "cop_convention": "heating" if mode == "heat_pump" else "cooling",
        "energy_balance_tolerance": 1e-6,
        "temperature_match_tolerance": 1e-6,
    }


@st.composite
def hpr_request_coordinate_sets(draw):
    """Generate unsorted unique coordinate sets for canonical requests."""
    sources = draw(
        st.lists(
            st.integers(min_value=-40, max_value=80),
            min_size=1,
            max_size=5,
            unique=True,
        )
    )
    sinks = draw(
        st.lists(
            st.integers(min_value=-20, max_value=160),
            min_size=1,
            max_size=5,
            unique=True,
        )
    )
    loads = draw(
        st.lists(
            st.sampled_from((0.1, 0.25, 0.5, 0.75, 1.0)),
            min_size=1,
            max_size=5,
            unique=True,
        )
    )
    return tuple(map(float, sources)), tuple(map(float, sinks)), tuple(loads)


__all__ = [
    "hpr_performance_map_payloads",
    "hpr_request_coordinate_sets",
    "json_values",
    "provenance_objects",
]
