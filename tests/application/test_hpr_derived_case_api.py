"""Derived HPR API contracts and utility ownership regressions."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from OpenPinch import PinchProblem
from tests.application.test_hpr_residual_workflow import solved_hpr as _solved_hpr

solved_hpr = _solved_hpr


@pytest.mark.parametrize(
    "method", ["direct_heat_integration", "all_heat_integration", "utility_placement"]
)
def test_shortcuts_validate_before_mutation(solved_hpr, method):
    problem, hp, _ = solved_hpr
    before = problem.to_problem_json()
    results = problem.results.model_dump_json()
    call = getattr(problem.target, method)
    for target in ("DHP", problem.results, hp.model_copy(update={"hpr_load": None})):
        with pytest.raises(ValueError):
            call(base_target=target)
    with pytest.raises(ValueError, match="zone"):
        call(base_target=hp, zone="missing")
    if method != "utility_placement":
        with pytest.raises(ValueError, match="period"):
            call(base_target=hp, period_id="missing")
        with pytest.raises(ValueError, match="subzones"):
            call(base_target=hp, include_subzones=True)
    else:
        with pytest.raises(ValueError, match="period"):
            call(base_target=hp, period_ids=["missing"])
    assert problem.to_problem_json() == before
    assert problem.results.model_dump_json() == results


def test_allocation_shortcuts_equal_explicit_case(solved_hpr):
    problem, hp, _ = solved_hpr
    before = problem.results.model_dump_json()
    residual = problem.residual_utility(base_target=hp)
    explicit = residual.target.direct_heat_integration()
    direct = problem.target.direct_heat_integration(base_target=hp)
    all_results = problem.target.all_heat_integration(base_target=hp)
    assert direct.hot_utility_target == pytest.approx(explicit.hot_utility_target)
    assert direct.cold_utility_target == pytest.approx(explicit.cold_utility_target)
    assert all_results.targets
    assert all(row.integration_type == "Residual" for row in all_results.targets)
    assert direct.provenance.owner_id != problem._analysis_owner_id
    assert hp.provenance.identity in direct.provenance.prerequisite_ids
    assert hp.provenance.identity in all_results.targets[0].provenance.prerequisite_ids
    assert problem.results.model_dump_json() == before
    assert residual.target.all_heat_integration().targets


def test_placement_returns_solved_and_transfer_is_unsolved(solved_hpr):
    problem, hp, _ = solved_hpr
    optimized = problem.target.utility_placement(
        base_target=hp,
        isothermal=2,
        options={"iteration_limit": 2, "evaluation_limit": 40, "seed": 20260715},
    )
    assert optimized.results.targets
    assert optimized.period_results
    assert optimized.utility_placement_result is not None
    assert not optimized.summary_frame().empty
    assert optimized.plot.grand_composite_curve().data
    new = problem.with_utilities_from(optimized)
    assert new.results is None
    assert new.utility_placement_result is None
    assert new.to_problem_json()["residual_basis"] is None
    assert new.to_problem_json()["streams"] == problem.to_problem_json()["streams"]
    assert (
        new.to_problem_json()["utilities"] == optimized.to_problem_json()["utilities"]
    )
    assert new._analysis_owner_id != problem._analysis_owner_id
    assert PinchProblem(optimized.to_problem_json()).results is None


def test_all_periods_rejects_explicit_target(solved_hpr):
    problem, hp, _ = solved_hpr
    for method in (
        "all_heat_integration",
        "direct_heat_integration",
        "utility_placement",
        "exergy",
    ):
        with pytest.raises(ValueError, match="base_target"):
            getattr(problem.target.all_periods, method)(base_target=hp)


def test_utility_transfer_preserves_component_graph():
    problem = PinchProblem("process_mvr.json")
    original = problem.components.add_process_mvr(
        "Evaporator vapour", mvr_stage_t_lift=10.0
    )
    donor = PinchProblem("process_mvr.json")
    derived = problem.with_utilities_from(donor, project_name="Derived")
    copied = next(iter(derived.process_components.values()))
    assert copied is not original
    assert copied.problem is derived
    assert original.problem is problem
    assert copied.active == original.active
    assert derived.results is None
    assert derived.project_name == "Derived"
    for record in copied.stream_records:
        for membership in record.original_memberships:
            assert membership.zone in list(
                derived._walk_zone_tree(derived._master_zone)
            )
    a = problem.target.direct_heat_integration()
    b = derived.target.direct_heat_integration()
    assert b.process_component_work_target == pytest.approx(
        a.process_component_work_target
    )


def test_utility_transfer_reorders_periods_and_retains_segments():
    from tests.application.test_hpr_period_batch_boundaries import _two_period_payload

    payload = _two_period_payload()
    receiver = PinchProblem(payload)
    payload["options"]["PROBLEM_PERIOD_IDS"] = ["peak", "base"]
    payload["utilities"] = [
        {
            "name": "steam",
            "type": "Hot",
            "segments": [
                {
                    "t_supply": 220,
                    "t_target": 210,
                    "heat_flow": {"values": [3, 4], "unit": "kW"},
                },
                {
                    "t_supply": 210,
                    "t_target": 200,
                    "heat_flow": {"values": [6, 8], "unit": "kW"},
                },
            ],
            "maximum_heat_flow": {
                "values": [12, 10],
                "period_ids": ["peak", "base"],
                "unit": "kW",
            },
            "price": {"values": [25, 30], "unit": "$/MWh"},
            "active": False,
            "htc": 2,
        }
    ]
    donor = PinchProblem(payload)
    derived = receiver.with_utilities_from(donor)
    utility = derived.to_problem_json()["utilities"][0]
    assert utility["segments"][0]["heat_flow"]["values"] == [4, 3]
    assert utility["price"]["values"] == [30, 25]
    assert utility["maximum_heat_flow"]["period_ids"] == ["peak", "base"]
    assert not utility["active"]
    assert derived.to_problem_json()["streams"] == receiver.to_problem_json()["streams"]
    assert (
        receiver.with_utilities_from(derived).to_problem_json()
        == derived.to_problem_json()
    )
    assert (
        PinchProblem(derived.to_problem_json()).to_problem_json()
        == derived.to_problem_json()
    )
    with pytest.raises(ValueError, match="period"):
        receiver.with_utilities_from(PinchProblem("heat_pump_targeting.json"))
    with pytest.raises(TypeError, match="PinchProblem"):
        receiver.with_utilities_from({})


@pytest.mark.parametrize("zone", ["Site", "AreaA"])
def test_ordinary_placement_finalizes_exact_period_and_scope(zone):
    from tests.application.test_hpr_period_batch_boundaries import _two_period_payload

    problem = PinchProblem(_two_period_payload())
    optimized = problem.target.utility_placement(
        zone=zone,
        period_ids=["peak"],
        isothermal=2,
        options={"iteration_limit": 1, "evaluation_limit": 25},
    )
    assert tuple(optimized.period_results) == ("peak",)
    assert all(row.period_id == "peak" for row in optimized.results.targets)
    assert all(
        row.scope == ("Site" if zone == "Site" else "Site/AreaA")
        for row in optimized.results.targets
    )
    assert optimized.utility_placement_result is not None
    assert problem.results is None


def test_finalization_failure_is_atomic(solved_hpr, monkeypatch):
    from OpenPinch.application import utility_placement

    problem, hp, _ = solved_hpr
    before = problem.results.model_dump_json()

    def fail(*args):
        raise RuntimeError("final allocation failed")

    monkeypatch.setattr(utility_placement, "_finalize_placement_case", fail)
    with pytest.raises(RuntimeError, match="final allocation"):
        problem.target.utility_placement(
            base_target=hp,
            isothermal=2,
            options={"iteration_limit": 1, "evaluation_limit": 25},
        )
    assert problem.results.model_dump_json() == before


def test_existing_consumers_infer_scalar_selection():
    from tests.application.test_hpr_period_batch_boundaries import _two_period_payload

    problem = PinchProblem(_two_period_payload())
    target = problem.target.direct_heat_integration(zone="AreaA", period_id="peak")
    observed = problem.target.exergy(base_target=target)
    assert observed.period_id == "peak"
    assert observed.scope == "Site/AreaA"


@settings(max_examples=12, deadline=None)
@given(
    order=st.permutations(["base", "peak"]),
    duties=st.lists(
        st.floats(min_value=1, max_value=20, allow_nan=False), min_size=2, max_size=2
    ),
    active=st.booleans(),
    unit=st.sampled_from(["kW", "W"]),
    commands=st.lists(st.sampled_from(["copy", "roundtrip", "allocate"]), max_size=4),
)
def test_transfer_state_model(order, duties, active, unit, commands):
    from tests.application.test_hpr_period_batch_boundaries import _two_period_payload

    receiver = PinchProblem(_two_period_payload())
    payload = _two_period_payload()
    payload["options"]["PROBLEM_PERIOD_IDS"] = list(order)
    payload["utilities"] = [
        {
            "name": "steam",
            "type": "Hot",
            "t_supply": 220,
            "t_target": 210,
            "heat_flow": {"values": duties, "unit": unit},
            "maximum_heat_flow": {"values": [1000, 2000], "unit": "kW"},
            "active": active,
        }
    ]
    donor = PinchProblem(payload)
    original = receiver.to_problem_json()
    current = receiver.with_utilities_from(donor)
    expected = current.to_problem_json()
    assert expected["utilities"][0]["heat_flow"]["values"] == [
        duties[order.index(sid)] for sid in receiver.period_ids
    ]
    for command in commands:
        if command == "copy":
            current = receiver.with_utilities_from(current)
        elif command == "roundtrip":
            current = PinchProblem(current.to_problem_json())
        else:
            current.target.direct_heat_integration(zone="AreaA", period_id="base")
        assert current.to_problem_json() == expected
        assert receiver.to_problem_json() == original
    assert current._analysis_owner_id != receiver._analysis_owner_id


def test_transfer_binds_implicit_units():
    from tests.application.test_hpr_period_batch_boundaries import _two_period_payload

    payload = _two_period_payload()
    payload["utilities"] = [
        {"name": "steam", "type": "Hot", "t_supply": 493.15, "t_target": 483.15}
    ]
    payload["options"]["INPUT_UNIT_TEMPERATURE"] = "K"
    donor = PinchProblem(payload)
    receiver = PinchProblem(_two_period_payload())
    new = receiver.with_utilities_from(donor)
    assert float(new.hot_utilities[0].supply_temperature[0]) == pytest.approx(220)


def test_residual_shortcut_never_retargets_process_or_hpr(solved_hpr, monkeypatch):
    from OpenPinch.application._problem.accessors import target as accessor

    problem, hp, _ = solved_hpr

    def unexpected(*args, **kwargs):
        raise AssertionError("Unexpected thermal solver")

    for name in (
        "direct_heat_integration_service",
        "direct_heat_pump_service",
        "indirect_heat_pump_service",
    ):
        monkeypatch.setattr(accessor, name, unexpected)
    result = problem.target.all_heat_integration(base_target=hp)
    assert result.targets
    optimized = problem.target.utility_placement(
        base_target=hp,
        isothermal=2,
        options={"iteration_limit": 1, "evaluation_limit": 25},
    )
    monkeypatch.setattr(accessor._TargetAccessor, "direct_heat_integration", unexpected)
    monkeypatch.setattr(accessor._TargetAccessor, "utility_placement", unexpected)
    assert not optimized.summary_frame().empty
    assert optimized.plot.grand_composite_curve().data
    assert optimized.period_results


def test_batch_and_thermal_override_guards(solved_hpr):
    from OpenPinch import PinchWorkspace

    problem, hp, _ = solved_hpr
    with pytest.raises(ValueError, match="frozen"):
        problem.target.direct_heat_integration(
            base_target=hp, options={"THERMAL_DT_CONT": 5}
        )
    workspace = PinchWorkspace(source="heat_pump_targeting.json")
    with pytest.raises(ValueError, match="base_target"):
        workspace.cases(("baseline",)).target.all_heat_integration(base_target=hp)


def test_hpr_shortcut_infers_child_zone_and_nondefault_period():
    payload = PinchProblem("heat_pump_targeting.json").to_problem_json()
    payload["options"]["PROBLEM_PERIOD_IDS"] = ["base", "peak"]
    problem = PinchProblem(payload)
    hp = problem.target.carnot_heat_pump(
        zone="Plant",
        period_id="peak",
        load_fraction=0.25,
        condensers=1,
        evaporators=1,
        maximum_restarts=1,
        options={"COSTING_HPR_PRICE_RATIO_COLD_TO_ELE": 0.1},
    )
    inferred = problem.target.direct_heat_integration(base_target=hp)
    explicit = problem.target.direct_heat_integration(
        base_target=hp, zone="Plant", period_id="peak"
    )
    assert inferred.period_id == "peak"
    assert inferred.hot_utility_target == pytest.approx(explicit.hot_utility_target)
    assert inferred.provenance.prerequisite_ids == (hp.provenance.identity,)
    with pytest.raises(ValueError, match="zone"):
        problem.target.all_heat_integration(base_target=hp, zone="Site")
    with pytest.raises(ValueError, match="period"):
        problem.target.all_heat_integration(base_target=hp, period_id="base")
    problem.update_options({"ENV_TEMPERATURE": 30.0})
    with pytest.raises(ValueError, match="stale"):
        problem.target.direct_heat_integration(base_target=hp)


def test_existing_base_selection_honours_runtime_period_options():
    from tests.application.test_hpr_period_batch_boundaries import _two_period_payload

    problem = PinchProblem(_two_period_payload())
    target = problem.target.direct_heat_integration(zone="AreaA", period_id="base")
    for method in (problem.target.exergy, problem.target.energy_transfer):
        with pytest.raises(ValueError, match="period"):
            method(base_target=target, options={"period_id": "peak"})
    # Named selectors retain their established precedence over advanced options.
    observed = problem.target.exergy(
        base_target=target, period_id="base", options={"period_id": "peak"}
    )
    assert observed.period_id == "base"
