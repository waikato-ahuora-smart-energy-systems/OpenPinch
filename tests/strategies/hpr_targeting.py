"""Reusable strategies for HPR targeting backend integration."""

from __future__ import annotations

from hypothesis import strategies as st

from OpenPinch.contracts.hpr import HprTargetSimulationRecord

BACKENDS = st.sampled_from(("coolprop", "tespy"))
INVALID_BACKENDS = st.text(min_size=1).filter(
    lambda value: value.strip().lower() not in {"coolprop", "tespy"}
)
MODES = st.sampled_from(("heat_pump", "refrigeration"))
PURE_FLUIDS = st.sampled_from(("R134a", "R32", "ammonia", "water"))
REGISTERED_BLENDS = st.sampled_from(("R407C", "R410A", "R404A"))
EXPLICIT_MOLAR_MIXTURES = st.sampled_from(
    (
        "HEOS::R32[0.5]&R125[0.5]",
        "HEOS::R32[0.3]&R125[0.4]&R143a[0.3]",
        "HEOS::R32[0.2]&R125[0.3]&R134a[0.1]&R143a[0.4]",
    )
)
WORKING_FLUIDS = st.one_of(PURE_FLUIDS, REGISTERED_BLENDS, EXPLICIT_MOLAR_MIXTURES)
FINITE_TEMPERATURES = st.floats(
    min_value=-80.0,
    max_value=180.0,
    allow_nan=False,
    allow_infinity=False,
)
POSITIVE_DUTIES = st.floats(
    min_value=1e-3,
    max_value=1e6,
    allow_nan=False,
    allow_infinity=False,
)


@st.composite
def hpr_target_values(draw):
    """Generate finite, physically signed targeting values."""
    useful_duty = draw(POSITIVE_DUTIES)
    cop = draw(
        st.floats(
            min_value=1.001,
            max_value=20.0,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    work = useful_duty / cop
    return {
        "useful_duty": useful_duty,
        "work": work,
        "cop": cop,
        "source_duty": useful_duty - work,
    }


@st.composite
def hpr_thermal_profiles(draw):
    """Generate detached hot-side and cold-side temperature-duty profiles."""
    values = draw(hpr_target_values())
    source_inlet = draw(FINITE_TEMPERATURES)
    sink_inlet = draw(FINITE_TEMPERATURES)
    source_drop = draw(
        st.floats(min_value=0.1, max_value=40.0, allow_nan=False, allow_infinity=False)
    )
    sink_rise = draw(
        st.floats(min_value=0.1, max_value=40.0, allow_nan=False, allow_infinity=False)
    )
    return (
        {
            "side": "source",
            "inlet_temperature": source_inlet,
            "outlet_temperature": source_inlet - source_drop,
            "duty": values["source_duty"],
        },
        {
            "side": "sink",
            "inlet_temperature": sink_inlet,
            "outlet_temperature": sink_inlet + sink_rise,
            "duty": values["useful_duty"],
        },
    )


@st.composite
def hpr_target_simulation_records(draw):
    """Generate valid detached single-stage simulation records."""
    evaporating = draw(FINITE_TEMPERATURES.filter(lambda value: value < 140.0))
    condensing = draw(
        st.floats(
            min_value=evaporating + 1.0,
            max_value=min(220.0, evaporating + 100.0),
            allow_nan=False,
            allow_infinity=False,
        )
    )
    backend = draw(BACKENDS)
    return HprTargetSimulationRecord(
        simulation_backend=backend,
        mode=draw(MODES),
        cycle_id="single_stage_vapour_compression",
        model_id="openpinch-vapour-compression-v1",
        refrigerant_spec=draw(WORKING_FLUIDS),
        nominal_evaporating_temperature=evaporating,
        nominal_condensing_temperature=condensing,
        nominal_useful_duty=draw(POSITIVE_DUTIES),
        source_approach_temperature=draw(
            st.floats(
                min_value=0.0, max_value=20.0, allow_nan=False, allow_infinity=False
            )
        ),
        sink_approach_temperature=draw(
            st.floats(
                min_value=0.0, max_value=20.0, allow_nan=False, allow_infinity=False
            )
        ),
        compressor_isentropic_efficiency=draw(
            st.floats(
                min_value=0.05, max_value=1.0, allow_nan=False, allow_infinity=False
            )
        ),
        superheat=draw(
            st.floats(
                min_value=0.0, max_value=30.0, allow_nan=False, allow_infinity=False
            )
        ),
        subcooling=draw(
            st.floats(
                min_value=0.0, max_value=30.0, allow_nan=False, allow_infinity=False
            )
        ),
        internal_hx_gas_temperature_change=draw(
            st.floats(
                min_value=0.0, max_value=30.0, allow_nan=False, allow_infinity=False
            )
        ),
        evaporator_count=1,
        condenser_count=1,
        period_id=draw(st.one_of(st.none(), st.text(min_size=1, max_size=12))),
        engine_version="8.0.0" if backend == "coolprop" else "0.11.2",
        power_boundary="compressor_only",
        assumptions={"generator": "hypothesis", "components": ["compressor", "valve"]},
    )


INVALID_RECORD_OVERRIDES = st.sampled_from(
    (
        {"simulation_backend": "other"},
        {"model_id": ""},
        {"refrigerant_spec": ""},
        {"nominal_useful_duty": 0.0},
        {"source_approach_temperature": -1.0},
        {"sink_approach_temperature": -1.0},
        {"compressor_isentropic_efficiency": 0.0},
        {"compressor_isentropic_efficiency": 1.01},
        {"evaporator_count": 0},
        {"condenser_count": 0},
        {"assumptions": {}},
    )
)


__all__ = [
    "BACKENDS",
    "EXPLICIT_MOLAR_MIXTURES",
    "FINITE_TEMPERATURES",
    "INVALID_BACKENDS",
    "INVALID_RECORD_OVERRIDES",
    "MODES",
    "POSITIVE_DUTIES",
    "PURE_FLUIDS",
    "REGISTERED_BLENDS",
    "WORKING_FLUIDS",
    "hpr_target_simulation_records",
    "hpr_target_values",
    "hpr_thermal_profiles",
]
