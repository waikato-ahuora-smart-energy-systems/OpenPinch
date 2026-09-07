"""Adversarial regression cases found during the repeated HPR change audit."""

from copy import deepcopy

import pytest

from OpenPinch import PinchProblem
from tests.application.test_hpr_residual_workflow import solved_hpr as _solved_hpr

solved_hpr = _solved_hpr


@pytest.mark.parametrize(
    "field,value", [("heat_transfer_coefficient", 123), ("fluid_name", "Water")]
)
def test_modified_hpr_utility_metadata_is_rejected(solved_hpr, field, value):
    problem, hp, _ = solved_hpr
    modified = deepcopy(hp)
    setattr(modified.hot_utilities[0], field, value)
    with pytest.raises(ValueError, match="modified"):
        problem.residual_utility(base_target=modified)


def test_modified_hpr_period_index_is_rejected(solved_hpr):
    problem, hp, _ = solved_hpr
    modified = hp.model_copy(update={"period_idx": None})
    with pytest.raises(ValueError, match="modified"):
        problem.residual_utility(base_target=modified)


@pytest.mark.parametrize("ids", [["other"], ["0", "other"]])
def test_residual_rejects_conflicting_configured_periods(solved_hpr, ids):
    problem, hp, _ = solved_hpr
    payload = problem.residual_utility(base_target=hp).to_problem_json()
    payload["options"]["PROBLEM_PERIOD_IDS"] = ids
    payload["options"]["PROBLEM_PERIOD_WEIGHTS"] = [1] * len(ids)
    with pytest.raises(ValueError, match="period"):
        PinchProblem(payload)


def test_residual_json_without_options_uses_frozen_period(solved_hpr):
    problem, hp, _ = solved_hpr
    payload = problem.residual_utility(base_target=hp).to_problem_json()
    payload["residual_basis"]["data"]["period_id"] = "winter"
    payload["options"] = None
    residual = PinchProblem(payload)
    assert list(residual.period_ids) == ["winter"]
    assert residual.config["PROBLEM_PERIOD_IDS"] == ["winter"]
    assert residual.target.all_heat_integration().targets[0].period_id == "winter"


def test_zero_load_does_not_return_or_plot_previous_hpr(solved_hpr):
    from OpenPinch.application._problem.targeting.state import snapshot_problem

    source, hp, _ = solved_hpr
    problem = snapshot_problem(source)
    assert problem.target.carnot_heat_pump(load_fraction=0) is None
    with pytest.raises(ValueError, match="unique"):
        problem.plot.grand_composite_curve_with_heat_pump()
    assert problem.plot.grand_composite_curve_with_heat_pump(target=hp).data


def test_residual_roundtrip_retains_zone_and_weight(solved_hpr):
    problem, hp, _ = solved_hpr
    residual = problem.residual_utility(base_target=hp)
    payload = residual.to_problem_json()
    payload["options"]["PROBLEM_PERIOD_WEIGHTS"] = [2.0]
    restored = PinchProblem(payload)
    assert restored.master_zone.name == payload["zone_tree"]["name"]
    assert list(restored.master_zone.weights) == [2.0]
    assert restored.target.direct_heat_integration(zone=payload["zone_tree"]["name"])


@pytest.mark.parametrize("change", ["children", "type", "shift", "shared_hpr"])
def test_residual_rejects_ignored_topology_and_shared_hpr(solved_hpr, change):
    problem, hp, _ = solved_hpr
    payload = problem.residual_utility(base_target=hp).to_problem_json()
    if change == "children":
        payload["zone_tree"]["children"] = [{"name": "ignored", "type": "Process Zone"}]
    elif change == "type":
        payload["zone_tree"]["type"] = "Site"
    elif change == "shift":
        payload["zone_tree"]["dt_cont_multiplier"] = 2.0
    else:
        payload["options"]["HPR_MULTIPERIOD_OPTIMIZATION_ENABLED"] = True
    with pytest.raises(ValueError, match="Residual|residual"):
        PinchProblem(payload)


@pytest.mark.parametrize("period_id,limit", [("base", 1000.0), ("peak", None)])
def test_residual_detachment_preserves_period_capacity_and_fluid_metadata(
    period_id, limit
):
    source = PinchProblem("heat_pump_targeting.json").to_problem_json()
    source["options"].update(PROBLEM_PERIOD_IDS=["base", "peak"])
    utility = source["utilities"][0]
    utility.update(
        maximum_heat_flow={"period_ids": ["base"], "values": [1000], "unit": "kW"},
        fluid_name="Water",
        fluid_phase="Gas",
        p_supply={"value": 2, "unit": "bar"},
        p_target={"value": 1.5, "unit": "bar"},
        h_supply={"value": 2800, "unit": "kJ/kg"},
        h_target={"value": 500, "unit": "kJ/kg"},
    )
    problem = PinchProblem(source)
    hp = problem.target.carnot_heat_pump(
        period_id=period_id,
        load_fraction=0.25,
        condensers=1,
        evaporators=1,
        maximum_restarts=1,
        options={"COSTING_HPR_PRICE_RATIO_COLD_TO_ELE": 0.1},
    )
    residual = problem.residual_utility(base_target=hp)
    copied = residual.hot_utilities.get_stream_by_name(utility["name"])
    original = hp.hot_utilities.get_stream_by_name(utility["name"])
    if limit is None:
        assert copied.maximum_heat_flow is None
    else:
        assert float(copied.maximum_heat_flow[0]) == limit
    assert copied.fluid_name == "Water"
    assert copied.fluid_phase == original.fluid_phase
    for field in (
        "supply_pressure",
        "target_pressure",
        "supply_enthalpy",
        "target_enthalpy",
    ):
        assert float(getattr(copied, field)[0]) == pytest.approx(
            float(getattr(original, field)[problem.period_ids[period_id]])
        )
    assert (
        residual.target.direct_heat_integration().hot_utility_target
        == pytest.approx(hp.hot_utility_target)
    )


@pytest.mark.parametrize("capacity_ids", [["peak"], ["base"], ["peak", "base"]])
def test_utility_transfer_binds_implicit_units_without_collapsing_period_arrays(
    capacity_ids,
):
    import math

    from tests.application.test_hpr_period_batch_boundaries import _two_period_payload

    receiver = PinchProblem(_two_period_payload())
    source = _two_period_payload()
    source["options"].update(
        PROBLEM_PERIOD_IDS=["peak", "base"],
        PROBLEM_PERIOD_WEIGHTS=[3, 2],
        INPUT_UNIT_HEAT_FLOW="MW",
    )
    source["utilities"] = [
        dict(
            name="steam",
            type="Hot",
            t_supply=220,
            t_target=210,
            heat_flow={"values": [0.2, 0.1]},
            maximum_heat_flow={
                "values": [0.3] * len(capacity_ids),
                "period_ids": capacity_ids,
            },
        )
    ]
    donor = PinchProblem(source)
    before = donor.to_problem_json()
    derived = receiver.with_utilities_from(donor)
    utility = derived.hot_utilities.get_stream_by_name("steam")
    nominal = derived.to_problem_json()["utilities"][0]["heat_flow"]
    assert nominal["values"] == pytest.approx([100, 200])
    for period_id, idx in derived.period_ids.items():
        actual = float(utility.maximum_heat_flow[idx])
        assert (
            actual == pytest.approx(300)
            if period_id in capacity_ids
            else math.isnan(actual)
        )
    assert derived.results is None
    assert donor.to_problem_json() == before
    assert PinchProblem(derived.to_problem_json()).period_ids == derived.period_ids
