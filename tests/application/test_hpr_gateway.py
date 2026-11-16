"""target.heat_pump_and_refrigeration: one gateway for every registered cycle.

The seven named heat pump and refrigeration methods are wrappers over it.
Golden tests pin the targeting call each named method makes (surface, target,
configuration, runtime options and service): it is what the target depends
on, and it must not change.
"""

from __future__ import annotations

import pytest

from OpenPinch import PinchProblem
from OpenPinch.application._problem.accessors import target as target_module
from OpenPinch.contracts.hpr import (
    CarnotCycleOptions,
    HprLevels,
    HprLoad,
    HprSearchLimits,
    HprTargetRequest,
    MvrCycleOptions,
    VapourCompressionCycleOptions,
)

LEGACY_CALLS = {
    "carnot_heat_pump": dict(
        is_utility_heat_pump=True,
        is_cascade_cycle=False,
        load_fraction=0.5,
        condensers=2,
        evaporators=1,
        compressor_efficiency=0.7,
        expander_efficiency=0.8,
        minimum_approach_temperature=3.0,
        maximum_restarts=4,
        maximum_iterations=50,
        maximum_evaluations=1000,
    ),
    "carnot_refrigeration": dict(load_duty=120.0, condensers=1, evaporators=2),
    "vapour_compression_heat_pump": dict(
        is_cascade_cycle=True,
        refrigerants=["Ammonia", "Water"],
        initialize_from_carnot=False,
        sort_refrigerants=True,
        allow_integrated_expander=False,
        simulation_backend="coolprop",
        condensers=2,
        evaporators=1,
        compressor_efficiency=0.75,
        minimum_approach_temperature=4.0,
    ),
    "vapour_compression_refrigeration": dict(
        is_utility_refrigeration=True,
        is_cascade_cycle=False,
        refrigerants=["Propane"],
        load_fraction=0.8,
    ),
    "mvr_heat_pump": dict(
        refrigerants=["Ammonia"],
        mvr_fluids=["Water"],
        mvr_compressor_efficiency=0.7,
        mvr_stages=2,
        motor_efficiency=0.95,
        condensers=1,
        evaporators=1,
        maximum_iterations=20,
    ),
}

GOLDEN = {
    "carnot_heat_pump": {
        "surface": "carnot_heat_pump",
        "target_id": "Indirect Heat Pump",
        "configuration": {
            "HPR_DT_CONT": 3.0,
            "HPR_ETA_COMP": 0.7,
            "HPR_ETA_EXP": 0.8,
            "HPR_LOAD_FRACTION": 0.5,
            "HPR_LOAD_MODE": "fraction",
            "HPR_MAX_MULTISTART": 4,
            "HPR_N_COND": 2,
            "HPR_N_EVAP": 1,
            "HPR_TYPE": "Parallel Carnot cycles",
        },
        "options": {"maximum_evaluations": 1000, "maximum_iterations": 50},
        "direct_service": None,
        "indirect_service": "indirect_heat_pump_service",
    },
    "carnot_refrigeration": {
        "surface": "carnot_refrigeration",
        "target_id": "Direct Refrigeration",
        "configuration": {
            "HPR_LOAD_DUTY": 120.0,
            "HPR_LOAD_MODE": "duty",
            "HPR_N_COND": 1,
            "HPR_N_EVAP": 2,
            "HPR_TYPE": "Cascade Carnot cycles",
        },
        "options": {},
        "direct_service": "direct_refrigeration_service",
        "indirect_service": None,
    },
    "vapour_compression_heat_pump": {
        "surface": "vapour_compression_heat_pump",
        "target_id": "Direct Heat Pump",
        "configuration": {
            "HPR_DT_CONT": 4.0,
            "HPR_ETA_COMP": 0.75,
            "HPR_INITIALISE_SIMULATED_CYCLE": False,
            "HPR_INTEGRATED_EXPANDER_ENABLED": False,
            "HPR_N_COND": 2,
            "HPR_N_EVAP": 1,
            "HPR_REFRIGERANTS": ["Ammonia", "Water"],
            "HPR_REFRIGERANT_SORT_ENABLED": True,
            "HPR_TYPE": "Cascade vapour compression cycles",
        },
        "options": {"simulation_backend": "coolprop"},
        "direct_service": "direct_heat_pump_service",
        "indirect_service": None,
    },
    "vapour_compression_refrigeration": {
        "surface": "vapour_compression_refrigeration",
        "target_id": "Indirect Refrigeration",
        "configuration": {
            "HPR_LOAD_FRACTION": 0.8,
            "HPR_LOAD_MODE": "fraction",
            "HPR_REFRIGERANTS": ["Propane"],
            "HPR_TYPE": "Parallel vapour compression cycles",
        },
        "options": {"simulation_backend": "coolprop"},
        "direct_service": None,
        "indirect_service": "indirect_refrigeration_service",
    },
    "mvr_heat_pump": {
        "surface": "mvr_heat_pump",
        "target_id": "Direct Heat Pump",
        "configuration": {
            "HPR_MVR_COUNT": 2,
            "HPR_MVR_ETA_COMP": 0.7,
            "HPR_MVR_ETA_MOTOR": 0.95,
            "HPR_MVR_FLUIDS": ["Water"],
            "HPR_N_COND": 1,
            "HPR_N_EVAP": 1,
            "HPR_REFRIGERANTS": ["Ammonia"],
            "HPR_TYPE": "Vapour compression with MVR cascade",
        },
        "options": {"maximum_evaluations": 1000000, "maximum_iterations": 20},
        "direct_service": "direct_heat_pump_service",
        "indirect_service": None,
    },
}


def _captured(monkeypatch, call):
    """The targeting call ``call(problem)`` makes, stopped before it runs."""

    seen = {}

    class Stop(Exception):
        pass

    def record(self, **kwargs):
        seen.update(kwargs)
        raise Stop

    monkeypatch.setattr(target_module._TargetAccessor, "_execute", record)
    with pytest.raises(Stop):
        call(PinchProblem("basic_pinch.json"))

    def name(service):
        return None if service is None else service.__name__

    return {
        "surface": seen["surface"],
        "target_id": seen["target_id"],
        "configuration": dict(seen["configuration"]),
        "options": dict(seen["options"]),
        "direct_service": name(seen["direct_service"]),
        "indirect_service": name(seen["indirect_service"]),
    }


@pytest.mark.parametrize("method", list(LEGACY_CALLS))
def test_each_named_method_makes_its_golden_targeting_call(monkeypatch, method):
    call = _captured(
        monkeypatch,
        lambda problem: getattr(problem.target, method)(**LEGACY_CALLS[method]),
    )

    assert call == GOLDEN[method]


GATEWAY_CALLS = {
    "carnot_heat_pump": dict(
        cycle="carnot",
        duty="heat_pump",
        topology="parallel",
        placement="utility",
        load=HprLoad(fraction=0.5),
        levels=HprLevels(condensers=2, evaporators=1, minimum_approach_temperature=3.0),
        cycle_options=CarnotCycleOptions(
            compressor_efficiency=0.7, expander_efficiency=0.8
        ),
        search=HprSearchLimits(restarts=4, iterations=50, evaluations=1000),
    ),
    "carnot_refrigeration": dict(
        cycle="carnot",
        duty="refrigeration",
        load=HprLoad(duty=120.0),
        levels=HprLevels(condensers=1, evaporators=2),
    ),
    "vapour_compression_heat_pump": dict(
        cycle="vapour_compression",
        levels={"condensers": 2, "evaporators": 1, "minimum_approach_temperature": 4.0},
        cycle_options={
            "refrigerants": ["Ammonia", "Water"],
            "initialize_from_carnot": False,
            "sort_refrigerants": True,
            "allow_integrated_expander": False,
            "compressor_efficiency": 0.75,
        },
    ),
    "vapour_compression_refrigeration": dict(
        cycle="vapour_compression",
        duty="refrigeration",
        topology="parallel",
        placement="utility",
        load={"fraction": 0.8},
        cycle_options=VapourCompressionCycleOptions(refrigerants=["Propane"]),
    ),
    "mvr_heat_pump": dict(
        cycle="mvr",
        levels=HprLevels(condensers=1, evaporators=1),
        cycle_options=MvrCycleOptions(
            refrigerants=["Ammonia"],
            mvr_fluids=["Water"],
            mvr_compressor_efficiency=0.7,
            mvr_stages=2,
            motor_efficiency=0.95,
        ),
        search={"iterations": 20},
    ),
}


@pytest.mark.parametrize("method", list(GATEWAY_CALLS))
def test_the_gateway_makes_the_named_methods_targeting_call(monkeypatch, method):
    call = _captured(
        monkeypatch,
        lambda problem: problem.target.heat_pump_and_refrigeration(
            **GATEWAY_CALLS[method]
        ),
    )

    assert call == GOLDEN[method]


@pytest.mark.parametrize("method", list(GATEWAY_CALLS))
def test_a_request_makes_the_same_call_as_its_arguments(monkeypatch, method):
    request = HprTargetRequest.model_validate(
        {
            key: value.model_dump() if hasattr(value, "model_dump") else value
            for key, value in GATEWAY_CALLS[method].items()
        }
    )

    call = _captured(
        monkeypatch,
        lambda problem: problem.target.heat_pump_and_refrigeration(request=request),
    )

    assert call == GOLDEN[method]


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        ({"cycle": "stirling"}, ValueError, "unknown cycle 'stirling'"),
        ({"cycle": "mvr", "duty": "refrigeration"}, ValueError, "heat_pump"),
        ({"cycle": "carnot", "topology": "single"}, ValueError, "cascade, parallel"),
        ({"cycle": "brayton"}, NotImplementedError, "not available"),
        ({"load": {"fraction": 0.5, "duty": 10.0}}, ValueError, "one of"),
        (
            {"cycle_options": {"compressor_efficiency": 1.5}},
            ValueError,
            "compressor_efficiency",
        ),
        (
            {"cycle": "carnot", "cycle_options": {"refrigerants": ["Water"]}},
            ValueError,
            "refrigerants",
        ),
    ],
)
def test_the_gateway_rejects_what_the_cycle_does_not_support(arguments, error, message):
    with pytest.raises(error, match=message):
        PinchProblem("basic_pinch.json").target.heat_pump_and_refrigeration(**arguments)


def test_a_carnot_target_is_the_same_through_the_gateway():
    levels = dict(condensers=1, evaporators=1)
    named = PinchProblem("basic_pinch.json").target.carnot_heat_pump(
        load_fraction=0.25, maximum_restarts=1, **levels
    )
    gateway = PinchProblem("basic_pinch.json").target.heat_pump_and_refrigeration(
        "carnot",
        load=HprLoad(fraction=0.25),
        levels=HprLevels(**levels),
        search=HprSearchLimits(restarts=1),
    )

    assert gateway.hpr_success and named.hpr_success
    for field in ("utility_tot", "w_net", "w_hpr", "heat_recovery"):
        assert getattr(gateway.hpr_details, field) == pytest.approx(
            getattr(named.hpr_details, field), rel=1e-12
        ), field


@pytest.mark.parametrize(
    ("arguments", "key", "value"),
    [
        ({"expander_efficiency": 0.0}, "HPR_ETA_EXP", 0.0),
        ({"maximum_restarts": 0}, "HPR_MAX_MULTISTART", 0),
    ],
)
def test_the_named_methods_still_accept_their_zero_values(
    monkeypatch, arguments, key, value
):
    # No expander recovery, and no restarts beyond the first run, are valid.
    call = _captured(
        monkeypatch, lambda problem: problem.target.carnot_heat_pump(**arguments)
    )

    assert call["configuration"][key] == value
