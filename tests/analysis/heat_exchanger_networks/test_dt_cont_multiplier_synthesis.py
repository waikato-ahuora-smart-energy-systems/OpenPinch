"""HEN synthesis is keyed on a dt_cont multiplier, scaled as targeting scales."""

from __future__ import annotations

import json

import pytest

from OpenPinch.analysis.heat_exchanger_networks.execution.settings import (
    workflow_settings_from_problem,
)
from OpenPinch.analysis.heat_exchanger_networks.solver.arrays import (
    nominal_approach_temperature,
    problem_to_solver_arrays,
)
from OpenPinch.analysis.heat_exchanger_networks.targeting.pinch_design_method import (
    build_pinch_design_method_tasks,
)
from OpenPinch.application.problem import PinchProblem
from OpenPinch.contracts.synthesis.method import (
    HeatExchangerNetworkSynthesisMethodInput,
)
from OpenPinch.domain.configuration import Configuration
from OpenPinch.domain.configuration_fields import CONFIG_FIELD_SPECS
from OpenPinch.domain.enums import HeatExchangerNetworkDesignMethod
from tests.support.paths import REPOSITORY_ROOT

BASIC_PINCH = REPOSITORY_ROOT / "OpenPinch/tutorials/sample_cases/basic_pinch.json"


def _basic_pinch(
    options: dict | None = None, *, contributions: list[float] | None = None
) -> PinchProblem:
    payload = json.loads(BASIC_PINCH.read_text(encoding="utf-8"))
    # Distinct contributions, so scaling would show.
    for index, stream in enumerate(payload["streams"]):
        value = 2.0 + index if contributions is None else contributions[index]
        stream["dt_cont"] = {"value": value, "unit": "degC"}
    payload["options"] = payload.get("options", {}) | (options or {})
    problem = PinchProblem(source=payload, project_name="basic")
    problem.target.all_heat_integration()
    return problem


def _arrays(problem: PinchProblem, multiplier: float):
    config = problem.master_zone.config
    return problem_to_solver_arrays(
        problem, nominal_approach_temperature(config, multiplier)
    )


def _contribution(arrays, stream_key: str) -> float:
    for side in ("hot", "cold"):
        axis = arrays.axis_maps[f"{side}_process_streams"]
        if stream_key in axis:
            return float(arrays.arrays[f"T_{side[0]}_cont_period"][0][axis[stream_key]])
    raise KeyError(stream_key)


def _streams(problem: PinchProblem) -> dict:
    # The live root zone: ``master_zone`` returns a copy.
    zone = problem._master_zone
    return {**dict(zone.hot_streams.items()), **dict(zone.cold_streams.items())}


def _stream_keys(problem: PinchProblem) -> list[str]:
    return list(_streams(problem))


def test_dt_cont_multipliers_are_unset_by_default():
    # Unset, the search width picks them.
    assert CONFIG_FIELD_SPECS["HENS_DT_CONT_MULTIPLIERS"].default is None
    assert Configuration().hens.dt_cont_multipliers is None


def test_approach_temperatures_option_is_retired():
    assert "HENS_APPROACH_TEMPERATURES" not in CONFIG_FIELD_SPECS
    with pytest.raises(ValueError, match="Unknown configuration option"):
        Configuration(options={"HENS_APPROACH_TEMPERATURES": [14.0]})


def test_task_contract_is_keyed_on_the_multiplier():
    fields = HeatExchangerNetworkSynthesisMethodInput.model_fields
    assert "dt_cont_multiplier" in fields
    assert "approach_temperature" not in fields

    def task(multiplier: float) -> HeatExchangerNetworkSynthesisMethodInput:
        return HeatExchangerNetworkSynthesisMethodInput(
            run_id="run",
            method=HeatExchangerNetworkDesignMethod.PinchDesign,
            dt_cont_multiplier=multiplier,
        )

    assert task(1.0).task_id != task(1.5).task_id
    with pytest.raises(ValueError, match="finite and positive"):
        task(0.0)


def test_nominal_approach_temperature_is_twice_the_scaled_reference():
    config = Configuration(options={"THERMAL_DT_CONT": 7.5})
    assert nominal_approach_temperature(config, 1.0) == pytest.approx(15.0)
    assert nominal_approach_temperature(config, 0.4) == pytest.approx(6.0)
    with pytest.raises(ValueError, match="THERMAL_DT_CONT"):
        nominal_approach_temperature(
            Configuration(options={"THERMAL_DT_CONT": 0.0}), 1.0
        )


def test_settings_resolve_the_multiplier_sweep():
    default = workflow_settings_from_problem(_basic_pinch())
    assert default.dt_cont_multipliers == (0.75, 1.0, 1.5)

    swept = workflow_settings_from_problem(
        _basic_pinch({"HENS_DT_CONT_MULTIPLIERS": [0.5, 1.5]})
    )
    assert swept.dt_cont_multipliers == (0.5, 1.5)


def test_tier_one_builds_one_pinch_design_task_per_multiplier():
    settings = workflow_settings_from_problem(
        _basic_pinch({"HENS_DT_CONT_MULTIPLIERS": [1.0, 1.5]})
    )

    tasks = build_pinch_design_method_tasks(settings)

    assert [task.dt_cont_multiplier for task in tasks] == [1.0, 1.5]


def test_multiplier_one_uses_each_contribution_as_entered():
    problem = _basic_pinch()

    arrays = _arrays(problem, 1.0)

    contributions = sorted(
        float(value)
        for key in ("T_h_cont_period", "T_c_cont_period")
        for value in arrays.arrays[key][0]
    )
    entered = sorted(
        2.0 + index
        for index in range(len(json.loads(BASIC_PINCH.read_text())["streams"]))
    )
    assert contributions == pytest.approx(entered)


def test_multiplier_scales_unlocked_streams_and_keeps_locked_ones():
    problem = _basic_pinch()
    keys = _stream_keys(problem)
    locked_key = keys[0]
    streams = _streams(problem)
    streams[locked_key].delta_t_contribution_multiplier_locked = True
    full = _arrays(problem, 1.0)

    half = _arrays(problem, 0.5)

    assert _contribution(half, locked_key) == pytest.approx(
        _contribution(full, locked_key)
    )
    for key in keys[1:]:
        assert _contribution(half, key) == pytest.approx(0.5 * _contribution(full, key))


@pytest.mark.parametrize("multiplier", [1.0, 2.0])
def test_zero_contribution_becomes_a_tenth_of_a_kelvin_times_the_multiplier(
    multiplier: float,
):
    problem = _basic_pinch(contributions=[0.0] * 7)
    keys = _stream_keys(problem)
    streams = _streams(problem)
    streams[keys[0]].delta_t_contribution_multiplier_locked = True

    arrays = _arrays(problem, multiplier)

    assert _contribution(arrays, keys[0]) == pytest.approx(0.1)
    for key in keys[1:]:
        assert _contribution(arrays, key) == pytest.approx(0.1 * multiplier)
