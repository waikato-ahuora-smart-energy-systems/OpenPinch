"""Compressor efficiency models through the public HPR target accessors."""

import json

import pytest

from OpenPinch import PinchProblem

pytest.importorskip("CoolProp")

# A short search on a small case: these tests check the plumbing, not the optimum.
_SHORT = dict(
    load_fraction=0.25,
    condensers=1,
    evaporators=1,
    maximum_restarts=1,
    maximum_iterations=2,
    maximum_evaluations=20,
    options={
        "COSTING_HPR_CAPITAL_RECOVERY_ENABLED": False,
        "HPR_MAX_PRESSURE_RATIO": 1e6,
        "HPR_SCREEN_MIN_SUCTION_PRESSURE": 0.0,
    },
)


def test_a_compressor_function_drives_a_vapour_compression_target():
    seen = []

    def flat_map(conditions):
        seen.append(conditions.pressure_ratio)
        return 0.65

    target = PinchProblem("basic_pinch.json").target.vapour_compression_heat_pump(
        compressor_model=flat_map, refrigerants=["Water"], **_SHORT
    )

    assert seen and all(ratio > 1.0 for ratio in seen)
    record = target.hpr_details.target_simulation_record
    assert record.compressor_isentropic_efficiency == pytest.approx(0.65)


def test_provenance_names_the_compressor_function():
    def flat_map(conditions):
        return 0.65

    problem = PinchProblem("basic_pinch.json")
    target = problem.target.vapour_compression_heat_pump(
        compressor_model=flat_map, refrigerants=["Water"], **_SHORT
    )

    settings = json.loads(target.provenance.effective_settings_json)
    assert settings["HPR_COMPRESSOR_MODEL"] == {"custom_compressor_model": "flat_map"}
    assert json.loads(problem.results.model_dump_json())


@pytest.mark.parametrize("cascade", [True, False])
def test_each_stage_records_the_efficiency_its_model_gave(cascade):
    options = {**_SHORT["options"], "HPR_COMPRESSOR_MODEL": "pareto_front"}
    target = PinchProblem("basic_pinch.json").target.vapour_compression_heat_pump(
        refrigerants=["Water"],
        is_cascade_cycle=cascade,
        **{**_SHORT, "condensers": 2, "maximum_evaluations": 30, "options": options},
    )

    record = target.hpr_details.target_simulation_record
    stages = [loop.stages[0] for loop in record.loops]
    etas = [stage.compressor_isentropic_efficiency for stage in stages]
    works = [stage.compressor_work for stage in stages]
    assert all(0.4 <= eta < 0.8695 for eta in etas)
    assert record.compressor_isentropic_efficiency == pytest.approx(
        sum(eta * work for eta, work in zip(etas, works)) / sum(works)
    )


def test_an_mvr_target_takes_models_for_both_compressors():
    def vc_map(conditions):
        return 0.66

    def fan_map(conditions):
        assert conditions.refrigerant == "Water"
        return 0.8

    target = PinchProblem("basic_pinch.json").target.mvr_heat_pump(
        compressor_model=vc_map,
        mvr_compressor_model=fan_map,
        **{
            **_SHORT,
            "maximum_iterations": 3,
            "maximum_evaluations": 60,
            "options": {**_SHORT["options"], "HPR_REFRIGERANTS": ["Water"]},
        },
    )

    record = target.hpr_details.target_simulation_record
    etas = {
        loop.loop_role: loop.stages[0].compressor_isentropic_efficiency
        for loop in record.loops
    }
    assert etas == {"vapour_compression": 0.66, "mvr": 0.8}


def test_a_compressor_function_drives_a_refrigeration_target():
    seen = []

    def flat_map(conditions):
        seen.append(conditions)
        return 0.62

    target = PinchProblem("basic_pinch.json").target.vapour_compression_refrigeration(
        compressor_model=flat_map, refrigerants=["Ammonia"], **_SHORT
    )

    assert seen
    record = target.hpr_details.target_simulation_record
    assert record.compressor_isentropic_efficiency == pytest.approx(0.62)


def test_a_performance_map_needs_a_constant_compressor_efficiency():
    # The map's off-design cycles hold the efficiency constant, which would
    # misrepresent a target whose efficiency follows a model.
    from OpenPinch.analysis.heat_pumps.performance_maps.target_basis import (
        HprPerformanceMapCompatibilityError,
        build_hpr_target_map_basis,
    )

    options = {**_SHORT["options"], "HPR_COMPRESSOR_MODEL": "pareto_front"}
    target = PinchProblem("basic_pinch.json").target.vapour_compression_heat_pump(
        refrigerants=["Water"], **{**_SHORT, "options": options}
    )

    with pytest.raises(HprPerformanceMapCompatibilityError) as caught:
        build_hpr_target_map_basis(target)
    assert caught.value.code == "compressor_model"
    assert "pareto_front" in str(caught.value)
