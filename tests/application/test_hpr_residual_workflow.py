"""Real HPR-to-utility workflow and generated immutable-state checks."""

import json
from copy import deepcopy

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from OpenPinch import PinchProblem
from OpenPinch.contracts.graphs import Graph
from OpenPinch.contracts.input import TargetInput
from OpenPinch.domain.hpr import HPRResidualProfile


@pytest.fixture(scope="module")
def solved_hpr():
    problem = PinchProblem("heat_pump_targeting.json", project_name="Heat Pump Study")
    options = {"COSTING_HPR_PRICE_RATIO_COLD_TO_ELE": 0.1}
    kwargs = dict(
        load_fraction=0.25,
        condensers=1,
        evaporators=1,
        maximum_restarts=1,
        options=options,
    )
    hp = problem.target.carnot_heat_pump(**kwargs)
    rf = problem.target.carnot_refrigeration(**kwargs)
    return problem, hp, rf


def test_modes_emit_separate_nonempty_finite_curves(solved_hpr):
    problem, hp, rf = solved_hpr
    assert hp.hpr_load.cycle_cooling != pytest.approx(rf.hpr_load.cycle_cooling)
    assert hp.hpr_load.achieved <= hp.hpr_load.selected + 1e-5
    assert hp.hpr_load.cycle_heating <= hp.hpr_load.selected + 1e-5
    for target, mode in ((hp, "heat_pump"), (rf, "refrigeration")):
        load = target.hpr_load
        assert load.cycle_heating - load.cycle_cooling == pytest.approx(
            load.work, abs=1e-5
        )
        for prefix in ("grand_composite_curve", "net_load_profiles"):
            method = getattr(problem.plot, f"{prefix}_with_{mode}")
            raw = method(target=target, return_graph_data=True)
            assert method(return_graph_data=True) == raw
            assert raw["segments"]
            assert target.name in raw["name"]
            assert all(
                np.isfinite([p["x"], p["y"]]).all()
                for s in raw["segments"]
                for p in s["data_points"]
            )
            assert len(method(target=target).data) > 0
    with pytest.raises(ValueError, match="mode"):
        problem.plot.grand_composite_curve_with_refrigeration(target=hp)


def test_residual_allocation_reconciles_and_placement_retains_basis(solved_hpr):
    problem, hp, _ = solved_hpr
    original = deepcopy(problem.to_problem_json())
    residual = problem.residual_utility(base_target=hp)
    assert residual.results is None
    assert residual.to_problem_json()["streams"] == []
    utility = residual.target.direct_heat_integration()
    assert utility.integration_type == "Residual"
    assert utility.hot_utility_target == pytest.approx(hp.hot_utility_target, abs=1e-4)
    assert utility.cold_utility_target == pytest.approx(
        hp.cold_utility_target, abs=1e-4
    )
    optimized = residual.target.utility_placement(
        isothermal=2,
        options={"iteration_limit": 2, "evaluation_limit": 40, "seed": 20260715},
    )
    allocated = optimized.target.direct_heat_integration()
    assert allocated.hot_utility_target == pytest.approx(utility.hot_utility_target)
    assert allocated.cold_utility_target == pytest.approx(utility.cold_utility_target)
    assert (
        optimized.to_problem_json()["residual_basis"]
        == residual.to_problem_json()["residual_basis"]
    )
    assert problem.to_problem_json() == original
    assert residual.plot.grand_composite_curve().data


def test_residual_rejects_foreign_modified_and_unsupported_analyses(solved_hpr):
    problem, hp, _ = solved_hpr
    foreign = PinchProblem("heat_pump_targeting.json")
    with pytest.raises(ValueError, match="foreign"):
        foreign.residual_utility(base_target=hp)
    changed = hp.model_copy(
        update={"hpr_load": hp.hpr_load.model_copy(update={"selected": 1.0})}
    )
    with pytest.raises(ValueError, match="modified"):
        problem.residual_utility(base_target=changed)
    residual = problem.residual_utility(base_target=hp)
    with pytest.raises(ValueError, match="physical streams"):
        residual.target.carnot_heat_pump()
    with pytest.raises(ValueError, match="shifts"):
        residual.set_dt_cont_multiplier(2.0)
    with pytest.raises(ValueError, match="period"):
        residual.target.direct_heat_integration(period_id="missing")
    mixed = residual.to_problem_json()
    mixed["streams"] = problem.to_problem_json()["streams"]
    with pytest.raises(ValueError, match="physical streams"):
        TargetInput.model_validate(mixed)
    changed_source = PinchProblem(problem.to_problem_json())
    changed_source.project_name = "Other"
    with pytest.raises(ValueError):
        changed_source.plot.grand_composite_curve_with_heat_pump(target=hp)


@settings(max_examples=15, deadline=None)
@given(
    st.lists(
        st.sampled_from(["allocate", "roundtrip", "price"]), min_size=0, max_size=6
    )
)
def test_frozen_residual_state_sequences(solved_hpr, commands):
    problem, hp, _ = solved_hpr
    original = deepcopy(problem.to_problem_json())
    residual = problem.residual_utility(base_target=hp)
    basis = deepcopy(residual.to_problem_json()["residual_basis"])
    for command in commands:
        if command == "allocate":
            result = residual.target.direct_heat_integration()
            assert result.hot_utility_target == pytest.approx(
                max(hp.hpr_residual.profile.heating)
            )
        else:
            inputs = json.loads(json.dumps(residual.to_problem_json()))
            if command == "price":
                inputs["utilities"][0]["price"] = 17.0
            residual = PinchProblem(inputs, project_name=residual.project_name)
        assert residual.to_problem_json()["residual_basis"] == basis
        assert problem.to_problem_json() == original


@settings(max_examples=15, deadline=None)
@given(
    st.sampled_from(["heat_pump", "refrigeration"]),
    st.sampled_from(["grand_composite_curve", "net_load_profiles"]),
)
def test_graph_transport_and_repeat_observation_properties(solved_hpr, mode, prefix):
    problem, hp, rf = solved_hpr
    target = hp if mode == "heat_pump" else rf
    method = getattr(problem.plot, f"{prefix}_with_{mode}")
    before = target.hpr_residual.model_dump_json()
    graph = Graph.model_validate(method(target=target, return_graph_data=True))
    assert Graph.model_validate_json(graph.model_dump_json()) == graph
    assert Graph.model_validate(method(target=target, return_graph_data=True)) == graph
    assert target.hpr_residual.model_dump_json() == before
    assert any(segment.series_id for segment in graph.segments)


@pytest.mark.parametrize(
    "field,value",
    [
        ("net", (0.0,)),
        ("temperatures", (10.0, 20.0)),
        ("heating", (-1.0, 0.0)),
        ("cooling", (0.0, float("nan"))),
    ],
)
def test_residual_profile_rejects_invalid_records(field, value):
    data = dict(
        temperatures=(20.0, 10.0),
        net=(0.0, 10.0),
        heating=(0.0, 0.0),
        cooling=(0.0, 10.0),
        temperature_basis="shifted",
    )
    data[field] = value
    with pytest.raises(ValueError):
        HPRResidualProfile.model_validate(data)


def test_zero_selected_service_returns_no_target():
    problem = PinchProblem("heat_pump_targeting.json")
    assert problem.target.carnot_heat_pump(load_fraction=0.0) is None
    with pytest.raises(ValueError, match="unique"):
        problem.plot.grand_composite_curve_with_heat_pump()


def test_no_input_hpr_plot_has_clear_error():
    with pytest.raises(ValueError, match="No solved"):
        PinchProblem().plot.grand_composite_curve_with_heat_pump()


def test_segmented_utility_shape_survives_detachment():
    source = PinchProblem("heat_pump_targeting.json").to_problem_json()
    source["utilities"][0] = {
        "name": "Segmented heating",
        "type": "Hot",
        "dt_cont": 1.0,
        "segments": [
            {"t_supply": 260.0, "t_target": 259.95, "heat_flow": 1.0},
            {"t_supply": 259.95, "t_target": 259.9, "heat_flow": 2.0},
        ],
        "maximum_heat_flow": 1000.0,
        "price": 10.0,
    }
    problem = PinchProblem(source)
    hp = problem.target.carnot_heat_pump(
        load_fraction=0.25,
        condensers=1,
        evaporators=1,
        maximum_restarts=1,
        options={"COSTING_HPR_PRICE_RATIO_COLD_TO_ELE": 0.1},
    )
    residual = problem.residual_utility(base_target=hp)
    segmented = next(u for u in residual.hot_utilities if u.has_segments)
    assert float(segmented.heat_flow[0]) == 0.0
    assert float(segmented.maximum_heat_flow[0]) == 1000.0
    target = residual.target.direct_heat_integration()
    allocated = next(u for u in target.hot_utilities if u.has_segments)
    duties = [float(s.heat_flow[0]) for s in allocated.segments]
    assert duties[1] == pytest.approx(2 * duties[0])
    assert target.hot_utility_target == pytest.approx(hp.hot_utility_target, abs=1e-4)


def test_indirect_refrigeration_ambient_balance_and_residual_placement():
    problem = PinchProblem("heat_pump_targeting.json")
    target = problem.target.carnot_refrigeration(
        is_utility_refrigeration=True,
        load_fraction=0.25,
        condensers=1,
        evaporators=1,
        maximum_restarts=1,
        options={"COSTING_HPR_PRICE_RATIO_COLD_TO_ELE": 0.1},
    )
    data = target.hpr_residual
    assert data.profile.temperature_basis == "real"
    assert target.hpr_load.ambient_hot > 1.0
    residual = problem.residual_utility(base_target=target)
    allocated = residual.target.direct_heat_integration()
    physical_surplus = np.ptp(data.physical_hot_composite) - np.ptp(
        data.physical_cold_composite
    )
    assert physical_surplus == pytest.approx(
        allocated.cold_utility_target - allocated.hot_utility_target, abs=1e-4
    )
    optimized = residual.target.utility_placement(
        isothermal=2,
        options={"iteration_limit": 2, "evaluation_limit": 40, "seed": 20260715},
    )
    assert (
        optimized.to_problem_json()["residual_basis"]
        == residual.to_problem_json()["residual_basis"]
    )
    final = optimized.target.direct_heat_integration()
    assert final.cold_utility_target == pytest.approx(allocated.cold_utility_target)
