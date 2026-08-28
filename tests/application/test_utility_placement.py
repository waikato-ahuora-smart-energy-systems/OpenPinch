"""Public utility-placement application integration tests."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from OpenPinch import PinchProblem, PinchWorkspace
from OpenPinch.application import utility_placement as placement_application
from OpenPinch.application.utility_placement import build_problem_placement_context
from OpenPinch.contracts.utility_placement import (
    UtilityLevelKind,
    UtilityPlacementBaseTarget,
    UtilityPlacementRequest,
)
from OpenPinch.domain.enums import GraphType, ProblemTableLabel, ZoneType
from OpenPinch.domain.zone import Zone


@pytest.mark.parametrize(
    "scope", [UtilityPlacementBaseTarget.DIRECT, UtilityPlacementBaseTarget.TOTAL_SITE]
)
def test_application_context_uses_isolated_target_and_complete_coordinates(
    scope,
) -> None:
    problem = PinchWorkspace(source="chocolate_factory.json").use_case("baseline")
    request = UtilityPlacementRequest(
        isothermal_level_count=2,
        base_target=scope,
        period_ids=("0",),
    )
    before = problem.to_problem_json()
    legacy_results = problem.results

    blueprints, context = build_problem_placement_context(problem, request)

    assert context.scope is scope
    assert context.periods[0].period_id == "0"
    assert context.periods[0].snapshot.shifted_temperatures
    assert context.periods[0].snapshot.real_temperatures
    assert context.periods[0].snapshot.real_hot_composite
    assert context.periods[0].snapshot.real_cold_composite
    assert len(context.periods[0].snapshot.real_hot_composite) == len(
        context.periods[0].snapshot.real_temperatures
    )
    assert len(context.periods[0].snapshot.real_cold_composite) == len(
        context.periods[0].snapshot.real_temperatures
    )
    assert context.periods[0].snapshot.entropy_slices
    assert context.periods[0].residual_hot_duty >= 0.0
    assert context.periods[0].residual_cold_duty >= 0.0
    assert len(context.periods[0].coordinate_bounds) == len(blueprints.all)
    assert problem.to_problem_json() == before
    assert problem.results is legacy_results


def test_application_context_resolves_auto_and_rejects_unknown_period_first() -> None:
    problem = PinchWorkspace(source="chocolate_factory.json").use_case("baseline")
    blueprints, context = build_problem_placement_context(
        problem,
        UtilityPlacementRequest(isothermal_level_count=2),
    )
    assert blueprints.all
    assert context.scope is UtilityPlacementBaseTarget.DIRECT

    with pytest.raises(Exception, match="period"):
        build_problem_placement_context(
            problem,
            UtilityPlacementRequest(
                isothermal_level_count=2,
                period_ids=("missing",),
            ),
        )


def test_profile_extraction_recovers_from_non_finite_separated_columns(
    monkeypatch,
) -> None:
    table = {
        ProblemTableLabel.T: (100.0, 50.0),
        ProblemTableLabel.H_NET_COLD: (float("nan"), 0.0),
        ProblemTableLabel.H_NET_HOT: (0.0, 1.0),
        ProblemTableLabel.H_NET_A: (float("nan"), 2.0),
        ProblemTableLabel.H_NET: (3.0, 4.0),
    }

    def separated(*, T_col, H_net):
        assert T_col == table[ProblemTableLabel.T]
        assert H_net == table[ProblemTableLabel.H_NET]
        return {
            "updates": {
                ProblemTableLabel.H_NET_COLD: (-3.0, 0.0),
                ProblemTableLabel.H_NET_HOT: (0.0, 4.0),
            }
        }

    monkeypatch.setattr(
        placement_application,
        "get_seperated_gcc_heat_load_profiles",
        separated,
    )

    assert placement_application._load_profiles(table) == (
        (3.0, 0.0),
        (0.0, 4.0),
    )


def test_profile_calibration_handles_empty_and_unrepresentable_profiles() -> None:
    assert placement_application._calibrate_profile((), residual_duty=0.0) == ()

    with pytest.raises(Exception, match="cannot represent"):
        placement_application._calibrate_profile((), residual_duty=1.0)


def test_process_entropy_extraction_skips_inactive_and_zero_duty_streams() -> None:
    inactive = SimpleNamespace(is_active=False)
    zero_duty = SimpleNamespace(
        is_active=True,
        name="zero",
        segments=(),
        supply_temperature=100.0,
        target_temperature=50.0,
        heat_flow=0.0,
    )
    zone = SimpleNamespace(
        hot_streams=(inactive, zero_duty),
        cold_streams=(),
    )

    assert placement_application._process_entropy_slices(zone, 0) == ()


def test_process_entropy_extraction_rejects_missing_physical_input() -> None:
    incomplete = SimpleNamespace(
        is_active=True,
        name="incomplete",
        segments=(),
        supply_temperature=None,
        target_temperature=50.0,
        heat_flow=1.0,
    )
    zone = SimpleNamespace(hot_streams=(incomplete,), cold_streams=())

    with pytest.raises(Exception, match="requires temperatures and duty"):
        placement_application._process_entropy_slices(zone, 0)


def test_coordinate_bounds_include_sensible_temperature_span() -> None:
    request = UtilityPlacementRequest(
        isothermal_level_count=2,
        sensible_level_count=1,
    )
    blueprints = placement_application.prepare_template_blueprints(request)

    bounds = placement_application._coordinate_bounds(
        request,
        blueprints,
        (50.0, 100.0),
    )

    sensible_keys = {
        bound.coordinate.template_key
        for bound in bounds
        if bound.coordinate.field.value == "temperature_span"
    }
    assert sensible_keys == {
        blueprint.key
        for blueprint in blueprints.all
        if blueprint.kind is UtilityLevelKind.SENSIBLE
    }


def _tiny_options():
    return {
        "iteration_limit": 1,
        "evaluation_limit": 100,
        "candidate_limit": 2,
        "run_count": 1,
    }


def _problem_with_utilities(utilities) -> PinchProblem:
    source = PinchWorkspace(source="chocolate_factory.json").case(
        "baseline"
    ).to_problem_json()
    source["utilities"] = utilities
    return PinchProblem(source=source, project_name="Site")


def test_existing_utilities_infer_kinds_expand_both_and_pad_sides() -> None:
    problem = _problem_with_utilities(
        [
            {"name": "Steam A", "type": "Hot", "t_supply": 180.0},
            {"name": "Steam B", "type": "Hot", "t_supply": 140.0},
            {
                "name": "Hot water",
                "type": "Hot",
                "t_supply": 90.0,
                "t_target": 70.0,
            },
            {
                "name": "Cooling water",
                "type": "Cold",
                "t_supply": 20.0,
                "t_target": 30.0,
            },
            {
                "name": "Dual water",
                "type": "Both",
                "t_supply": 5.0,
                "t_target": 15.0,
            },
        ]
    )
    selected_zone = placement_application._resolve_placement_zone(problem, None)

    request = placement_application._build_problem_placement_request(
        problem,
        selected_zone=selected_zone,
        isothermal=None,
        sensible=None,
        period_ids=("0",),
        options=_tiny_options(),
    )

    assert request.isothermal_level_count == 2
    assert request.sensible_level_count == 2
    assert len(request.hot_templates) == 4
    assert len(request.cold_templates) == 4
    assert [template.kind for template in request.hot_templates].count(
        UtilityLevelKind.ISOTHERMAL
    ) == 2
    assert [template.kind for template in request.cold_templates].count(
        UtilityLevelKind.ISOTHERMAL
    ) == 2
    assert {template.name for template in request.hot_templates} >= {
        "Steam A",
        "Steam B",
        "Hot water",
        "Dual water (hot)",
    }
    assert {template.name for template in request.cold_templates} >= {
        "Cooling water",
        "Dual water (cold)",
    }
    assert any(
        template.name.startswith("inferred_cold_iso_")
        for template in request.cold_templates
    )


@settings(max_examples=6, deadline=None)
@given(order=st.permutations((0, 1, 2, 3, 4)))
def test_existing_utility_inference_is_declaration_order_invariant(order) -> None:
    utilities = [
        {"name": "Steam A", "type": "Hot", "t_supply": 180.0},
        {"name": "Steam B", "type": "Hot", "t_supply": 140.0},
        {
            "name": "Hot water",
            "type": "Hot",
            "t_supply": 90.0,
            "t_target": 70.0,
        },
        {
            "name": "Cooling water",
            "type": "Cold",
            "t_supply": 20.0,
            "t_target": 30.0,
        },
        {
            "name": "Dual water",
            "type": "Both",
            "t_supply": 5.0,
            "t_target": 15.0,
        },
    ]

    def signature(problem):
        selected_zone = placement_application._resolve_placement_zone(problem, None)
        request = placement_application._build_problem_placement_request(
            problem,
            selected_zone=selected_zone,
            isothermal=None,
            sensible=None,
            period_ids=("0",),
            options=_tiny_options(),
        )
        return tuple(
            (template.side, template.kind, template.name)
            for template in request.hot_templates + request.cold_templates
        )

    expected = signature(_problem_with_utilities(utilities))
    reordered = [utilities[index] for index in order]

    assert signature(_problem_with_utilities(reordered)) == expected


def test_profiled_existing_utility_is_sensible_and_both_direction_is_paired() -> None:
    problem = _problem_with_utilities(
        [
            {"name": "Hot phase 1", "type": "Hot", "t_supply": 180.0},
            {"name": "Hot phase 2", "type": "Hot", "t_supply": 140.0},
            {"name": "Cold phase 1", "type": "Cold", "t_supply": 5.0},
            {"name": "Cold phase 2", "type": "Cold", "t_supply": 10.0},
            {
                "name": "Profiled",
                "type": "Both",
                "profile": {
                    "points": [
                        {"temperature": 20.0, "cumulative_heat": 0.0},
                        {"temperature": 40.0, "cumulative_heat": 10.0},
                    ]
                },
            },
        ]
    )
    selected_zone = placement_application._resolve_placement_zone(problem, None)

    request = placement_application._build_problem_placement_request(
        problem,
        selected_zone=selected_zone,
        isothermal=None,
        sensible=None,
        period_ids=("0",),
        options=_tiny_options(),
    )

    hot = next(template for template in request.hot_templates if "Profiled" in template.name)
    cold = next(
        template for template in request.cold_templates if "Profiled" in template.name
    )
    assert hot.kind is UtilityLevelKind.SENSIBLE
    assert cold.kind is UtilityLevelKind.SENSIBLE
    assert hot.side.value == "hot"
    assert cold.side.value == "cold"


def test_explicit_counts_override_existing_utilities_and_require_isothermal() -> None:
    problem = _problem_with_utilities(
        [{"name": "Existing", "type": "Hot", "t_supply": 180.0}]
    )
    selected_zone = placement_application._resolve_placement_zone(problem, None)

    request = placement_application._build_problem_placement_request(
        problem,
        selected_zone=selected_zone,
        isothermal=2,
        sensible=2,
        period_ids=("0",),
        options=_tiny_options(),
    )

    assert request.isothermal_level_count == 2
    assert request.sensible_level_count == 2
    assert request.hot_templates is None
    assert request.cold_templates is None

    with pytest.raises(Exception, match="isothermal"):
        placement_application._build_problem_placement_request(
            problem,
            selected_zone=selected_zone,
            isothermal=None,
            sensible=2,
            period_ids=("0",),
            options=_tiny_options(),
        )


def test_omitted_counts_require_inferable_existing_utilities() -> None:
    problem = _problem_with_utilities([])
    selected_zone = placement_application._resolve_placement_zone(problem, None)

    with pytest.raises(Exception, match="existing utilities"):
        placement_application._build_problem_placement_request(
            problem,
            selected_zone=selected_zone,
            isothermal=None,
            sensible=None,
            period_ids=("0",),
            options=_tiny_options(),
        )


def test_public_signature_uses_optional_counts_and_zone_only() -> None:
    parameters = inspect.signature(PinchProblem().target.utility_placement).parameters

    assert "isothermal" in parameters
    assert "sensible" in parameters
    assert parameters["isothermal"].default is None
    assert parameters["sensible"].default is None
    assert parameters["zone"].default is None
    assert "isothermal_level_count" not in parameters
    assert "sensible_level_count" not in parameters
    assert "hot_templates" not in parameters
    assert "cold_templates" not in parameters
    assert "base_target" not in parameters


@pytest.mark.parametrize(
    ("zone_type", "expected"),
    [
        (ZoneType.P, UtilityPlacementBaseTarget.DIRECT),
        (ZoneType.O, UtilityPlacementBaseTarget.DIRECT),
        (ZoneType.S, UtilityPlacementBaseTarget.TOTAL_SITE),
        (ZoneType.C, UtilityPlacementBaseTarget.INDIRECT),
        (ZoneType.R, UtilityPlacementBaseTarget.INDIRECT),
    ],
)
def test_zone_type_determines_placement_target_profile(zone_type, expected) -> None:
    zone = Zone(name="Selected", type=zone_type.value)

    assert placement_application._scope_for_zone(zone) is expected


def test_zone_resolution_defaults_to_root_and_accepts_unique_name_path_and_object() -> (
    None
):
    root = Zone(name="Community", type=ZoneType.C.value)
    site = Zone(name="Site A", type=ZoneType.S.value, parent_zone=root)
    process = Zone(name="Process A", type=ZoneType.P.value, parent_zone=site)
    operation = Zone(name="Operation A", type=ZoneType.O.value, parent_zone=process)
    root.add_zone(site)
    site.add_zone(process)
    process.add_zone(operation)
    problem = SimpleNamespace(_build_execution_master_zone=lambda: root)

    assert placement_application._resolve_placement_zone(problem, None) is root
    assert placement_application._resolve_placement_zone(problem, "Process A") is process
    assert (
        placement_application._resolve_placement_zone(
            problem, "Community/Site A/Process A/Operation A"
        )
        is operation
    )
    assert placement_application._resolve_placement_zone(problem, site) is site


def test_zone_resolution_rejects_ambiguous_missing_foreign_and_utility_zones() -> None:
    root = Zone(name="Community", type=ZoneType.C.value)
    left = Zone(name="Left", type=ZoneType.S.value, parent_zone=root)
    right = Zone(name="Right", type=ZoneType.S.value, parent_zone=root)
    left_process = Zone(name="Shared", type=ZoneType.P.value, parent_zone=left)
    right_process = Zone(name="Shared", type=ZoneType.P.value, parent_zone=right)
    utility = Zone(name="Utilities", type=ZoneType.U.value, parent_zone=root)
    root.add_zone(left)
    root.add_zone(right)
    root.add_zone(utility)
    left.add_zone(left_process)
    right.add_zone(right_process)
    problem = SimpleNamespace(_build_execution_master_zone=lambda: root)

    with pytest.raises(Exception, match="ambiguous"):
        placement_application._resolve_placement_zone(problem, "Shared")
    with pytest.raises(Exception, match="not found"):
        placement_application._resolve_placement_zone(problem, "Missing")
    with pytest.raises(Exception, match="belong"):
        placement_application._resolve_placement_zone(
            problem,
            Zone(name="Shared", type=ZoneType.P.value),
        )
    with pytest.raises(Exception, match="Utility Zone"):
        placement_application._scope_for_zone(utility)


def test_public_accessor_returns_a_detached_normal_case_with_retained_evidence() -> (
    None
):
    problem = PinchWorkspace(source="chocolate_factory.json").use_case("baseline")
    before = problem.to_problem_json()
    legacy_results = problem.results

    optimized_case = problem.target.utility_placement(
        isothermal=2,
        period_ids=("0",),
        options=_tiny_options(),
    )
    result = optimized_case.utility_placement_result

    assert isinstance(optimized_case, PinchProblem)
    assert result is not None
    assert result.best.thermodynamic_total is not None
    assert result.best.period_results[0].thermodynamic.process_entropy.value != 0.0
    assert len(optimized_case.to_problem_json()["utilities"]) == 4
    assert problem.utility_placement_result is None
    assert problem.results is legacy_results
    assert problem.to_problem_json() == before
    assert result == result.model_validate_json(result.model_dump_json())

    repeated_case = problem.target.utility_placement(
        isothermal=2,
        period_ids=("0",),
        options=_tiny_options(),
    )
    assert repeated_case.to_problem_json() == optimized_case.to_problem_json()
    assert repeated_case.utility_placement_result == result

    with pytest.raises(Exception, match="period"):
        problem.target.utility_placement(
            isothermal=2,
            period_ids=("missing",),
            options=_tiny_options(),
        )
    assert problem.utility_placement_result is None
    assert problem.results is legacy_results
    assert problem.to_problem_json() == before

    all_periods_case = problem.target.all_periods.utility_placement(
        isothermal=2,
        options=_tiny_options(),
    )
    assert all_periods_case.utility_placement_result.period_ids == tuple(
        problem.period_ids
    )


def test_default_thermodynamic_solution_follows_the_residual_process_profile() -> None:
    problem = PinchWorkspace(source="chocolate_factory.json").use_case("baseline")
    optimized_case = problem.target.utility_placement(
        isothermal=2,
        sensible=2,
        period_ids=("0",),
        options=_tiny_options(),
    )
    result = optimized_case.utility_placement_result
    period = result.best.period_results[0]
    active_hot = [level for level in period.hot_levels if level.allocated_duty.value > 0]
    active_cold = [
        level for level in period.cold_levels if level.allocated_duty.value > 0
    ]

    assert active_hot
    assert active_cold
    assert min(level.supply_temperature.value for level in active_hot) < 200.0
    assert max(level.target_temperature.value for level in active_cold) > 0.0
    assert result.best.thermodynamic_total is not None
    assert result.best.thermodynamic_total.value >= 0.0
    assert result.best.period_results[0].thermodynamic is not None
    assert (
        result.best.period_results[0].thermodynamic.total_entropy_generation.value
        == pytest.approx(result.best.thermodynamic_total.value)
    )


def test_optimized_utilities_replace_a_new_case_for_standard_gcc_and_tsp() -> None:
    workspace = PinchWorkspace(source="chocolate_factory.json")
    problem = workspace.use_case("baseline")
    before = problem.to_problem_json()
    optimized_case = problem.target.utility_placement(
        isothermal=2,
        sensible=2,
        period_ids=("0",),
        options=_tiny_options(),
    )
    added_case = workspace.add(
        optimized_case,
        name="optimized_utilities",
        activate=False,
    )
    assert added_case.utility_placement_result == optimized_case.utility_placement_result

    added_case.target.direct_heat_integration(period_id="0")
    gcc = added_case.plot.grand_composite_curve(return_graph_data=True)
    added_case.target.total_site_heat_integration(period_id="0")
    tsp = added_case.plot.total_site_profiles(return_graph_data=True)

    assert gcc["type"] == GraphType.GCC.value
    assert any(
        segment["title"].startswith("Utility GCC") for segment in gcc["segments"]
    )
    assert tsp["type"] == GraphType.TSP.value
    assert {"Hot Utility", "Cold Utility"} <= {
        segment["title"] for segment in tsp["segments"]
    }
    assert len(added_case.to_problem_json()["utilities"]) == 8
    assert problem.to_problem_json() == before
    assert "optimized_utilities" in workspace.list_cases()
    assert not hasattr(problem.plot, "utility_placement")


@pytest.mark.parametrize(
    ("zone", "scope"),
    [
        ("Almond", UtilityPlacementBaseTarget.DIRECT),
        (None, UtilityPlacementBaseTarget.TOTAL_SITE),
    ],
)
def test_named_case_replacement_is_deterministic_and_isolated_for_every_scope(
    zone,
    scope,
) -> None:
    workspace = PinchWorkspace(source="chocolate_factory.json")
    problem = workspace.use_case("baseline")
    before = problem.to_problem_json()
    first_case = problem.target.utility_placement(
        isothermal=2,
        sensible=2,
        zone=zone,
        period_ids=("0",),
        options=_tiny_options(),
    )
    second_case = problem.target.utility_placement(
        isothermal=2,
        sensible=2,
        zone=zone,
        period_ids=("0",),
        options=_tiny_options(),
    )
    assert second_case.to_problem_json() == first_case.to_problem_json()

    optimized_case = workspace.add(
        first_case,
        name=f"optimized_{scope.value}",
        activate=False,
    )
    if scope is UtilityPlacementBaseTarget.DIRECT:
        optimized_case.target.direct_heat_integration(zone=zone, period_id="0")
        graph = optimized_case.plot.grand_composite_curve(
            zone_name=zone,
            return_graph_data=True,
        )
        expected_type = GraphType.GCC.value
    else:
        optimized_case.target.total_site_heat_integration(period_id="0")
        graph = optimized_case.plot.total_site_profiles(return_graph_data=True)
        expected_type = GraphType.TSP.value

    assert graph["type"] == expected_type
    assert problem.to_problem_json() == before
    assert workspace.active_case_name == "baseline"


def test_public_total_site_workflow_covers_the_total_site_residual() -> None:
    problem = PinchWorkspace(source="chocolate_factory.json").use_case("baseline")

    optimized_case = problem.target.utility_placement(
        isothermal=2,
        sensible=2,
        period_ids=("0",),
        options=_tiny_options(),
    )
    result = optimized_case.utility_placement_result

    assert result.scope is UtilityPlacementBaseTarget.TOTAL_SITE
    assert result.best.period_results[0].feasible
    assert result.best.period_results[0].hot_coverage_residual.value == pytest.approx(
        0.0, abs=1e-6
    )
    assert result.best.period_results[0].cold_coverage_residual.value == pytest.approx(
        0.0, abs=1e-6
    )


def test_public_community_workflow_uses_indirect_aggregate_profile() -> None:
    seed = PinchProblem("zonal_site.json", project_name="Scope")
    source = seed.to_problem_json()
    source["zone_tree"]["name"] = "Scope"
    source["zone_tree"]["type"] = ZoneType.C.value
    problem = PinchProblem(source, project_name="Scope")

    optimized_case = problem.target.utility_placement(
        isothermal=2,
        sensible=2,
        period_ids=("0",),
        options=_tiny_options(),
    )

    assert (
        optimized_case.utility_placement_result.scope
        is UtilityPlacementBaseTarget.INDIRECT
    )
    assert len(optimized_case.to_problem_json()["utilities"]) == 8


def test_public_validation_fails_before_analysis_and_preserves_previous_cache() -> None:
    problem = PinchWorkspace(source="chocolate_factory.json").use_case("baseline")
    assert problem.utility_placement_result is None
    with pytest.raises(Exception, match="at least 2"):
        problem.target.utility_placement(isothermal=1)
    assert problem.utility_placement_result is None


def test_input_changes_invalidate_cached_placement_result() -> None:
    problem = PinchWorkspace(source="chocolate_factory.json").use_case("baseline")
    optimized_case = problem.target.utility_placement(
        isothermal=2,
        period_ids=("0",),
        options=_tiny_options(),
    )
    assert optimized_case.utility_placement_result is not None

    optimized_case.update_options({"THERMAL_DT_CONT": 11.0})

    assert optimized_case.utility_placement_result is None


def test_obsolete_placement_observation_methods_are_absent() -> None:
    problem = PinchProblem()

    assert not hasattr(problem, "utility_placement_metrics")
    assert not hasattr(problem, "utility_placement_summary_frame")
    assert not hasattr(problem, "utility_placement_report")


def test_workspace_add_validates_case_and_name_and_can_activate() -> None:
    workspace = PinchWorkspace(source="chocolate_factory.json")
    optimized_case = workspace.case("baseline").target.utility_placement(
        isothermal=2,
        period_ids=("0",),
        options=_tiny_options(),
    )

    with pytest.raises(TypeError, match="PinchProblem"):
        workspace.add({}, name="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="case name"):
        workspace.add(optimized_case, name="../invalid")

    added = workspace.add(optimized_case, name="optimized", activate=True)

    assert added is workspace.case("optimized")
    assert workspace.active_case_name == "optimized"
    assert added.utility_placement_result == optimized_case.utility_placement_result
    with pytest.raises(ValueError, match="already exists"):
        workspace.add(optimized_case, name="optimized")
