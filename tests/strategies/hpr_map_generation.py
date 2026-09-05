"""Reusable Hypothesis strategies for HPR map generation."""

from __future__ import annotations

from hypothesis import strategies as st

from OpenPinch.analysis.heat_pumps.performance_maps.models import (
    HprPointSimulation,
    HprTargetMapBasis,
)
from OpenPinch.contracts.hpr_performance_map import HprPerformanceMapRequest

_IDENTIFIERS = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789-_",
    min_size=1,
    max_size=16,
)


@st.composite
def hpr_target_map_bases(draw, *, backend: str = "coolprop"):
    """Generate supported single-stage target bases using a stable fluid."""
    mode = draw(st.sampled_from(("heat_pump", "refrigeration")))
    evaporating_temperature = float(draw(st.integers(min_value=-30, max_value=20)))
    lift = float(
        draw(
            st.integers(
                min_value=20,
                max_value=min(70, int(90 - evaporating_temperature)),
            )
        )
    )
    source_approach = float(draw(st.integers(min_value=0, max_value=5)))
    sink_approach = float(draw(st.integers(min_value=0, max_value=5)))
    return HprTargetMapBasis(
        target_id=draw(_IDENTIFIERS),
        mode=mode,
        simulation_backend=backend,
        cycle_id="single_stage_vapour_compression",
        model_id="openpinch-vapour-compression-v1",
        refrigerant_spec="R134a",
        nominal_evaporating_temperature=evaporating_temperature,
        nominal_condensing_temperature=evaporating_temperature + lift,
        nominal_useful_duty=float(draw(st.integers(min_value=10, max_value=10_000))),
        source_approach_temperature=source_approach,
        sink_approach_temperature=sink_approach,
        compressor_isentropic_efficiency=draw(
            st.floats(
                min_value=0.5,
                max_value=0.9,
                allow_nan=False,
                allow_infinity=False,
            )
        ),
        superheat=float(draw(st.integers(min_value=0, max_value=10))),
        subcooling=float(draw(st.integers(min_value=0, max_value=10))),
        internal_hx_gas_temperature_change=float(
            draw(st.integers(min_value=0, max_value=10))
        ),
        source_provenance={"strategy": "hpr_target_map_bases"},
    )


@st.composite
def hpr_map_requests(draw):
    """Generate small canonicalizable active HPR map grids."""
    sources = draw(
        st.lists(
            st.integers(min_value=-25, max_value=25),
            min_size=1,
            max_size=3,
            unique=True,
        )
    )
    minimum_sink = max(sources) + 15
    sinks = draw(
        st.lists(
            st.integers(min_value=minimum_sink, max_value=minimum_sink + 80),
            min_size=1,
            max_size=3,
            unique=True,
        )
    )
    loads = draw(
        st.lists(
            st.sampled_from((0.1, 0.25, 0.5, 0.75, 1.0)),
            min_size=1,
            max_size=4,
            unique=True,
        )
    )
    capacity = draw(st.one_of(st.none(), st.integers(min_value=10, max_value=5_000)))
    return HprPerformanceMapRequest(
        map_id=draw(_IDENTIFIERS),
        source_temperatures=sources,
        sink_temperatures=sinks,
        load_fractions=loads,
        reference_capacity=capacity,
    )


@st.composite
def explicit_molar_mixture_text(draw):
    """Generate parseable explicit mixtures with positive finite totals."""
    components = draw(
        st.lists(
            st.sampled_from(("R32", "R125", "R134a", "R143a")),
            min_size=1,
            max_size=4,
            unique=True,
        )
    )
    fractions = draw(
        st.lists(
            st.integers(min_value=0, max_value=100),
            min_size=len(components),
            max_size=len(components),
        ).filter(lambda values: sum(values) > 0)
    )
    body = "&".join(
        f"{component}[{fraction}]"
        for component, fraction in zip(components, fractions, strict=True)
    )
    return f"HEOS::{body}", tuple(components), tuple(map(float, fractions))


@st.composite
def balanced_point_simulations(draw, *, mode: str):
    """Generate normalized simulations with exact positive energy closure."""
    useful_duty = float(draw(st.integers(min_value=10, max_value=10_000)))
    compressor_power = useful_duty / float(draw(st.integers(min_value=2, max_value=8)))
    if mode == "heat_pump":
        q_source = useful_duty - compressor_power
        q_sink = useful_duty
    else:
        q_source = useful_duty
        q_sink = useful_duty + compressor_power
    return HprPointSimulation(
        q_source=q_source,
        q_sink=q_sink,
        compressor_power=compressor_power,
        converged=True,
        engine_details={"strategy": "balanced"},
    )


__all__ = [
    "balanced_point_simulations",
    "explicit_molar_mixture_text",
    "hpr_map_requests",
    "hpr_target_map_bases",
]
