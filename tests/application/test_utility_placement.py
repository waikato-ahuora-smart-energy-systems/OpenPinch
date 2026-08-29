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
from OpenPinch.domain._value.resolution import get_scalar_value
from OpenPinch.domain.enums import GraphType, ProblemTableLabel, ZoneType
from OpenPinch.domain.zone import Zone
from tests.strategies.utility_placement import residual_profile_envelopes


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


def test_coordinate_bounds_cover_the_process_temperature_envelope() -> None:
    request = UtilityPlacementRequest(
        isothermal_level_count=2,
        sensible_level_count=2,
    )
    blueprints = placement_application.prepare_template_blueprints(request)
    temperatures = (180.0, 100.0, 20.0)

    bounds = placement_application._coordinate_bounds(
        request,
        blueprints,
        temperatures,
    )

    supply_bounds = [
        bound
        for bound in bounds
        if bound.coordinate.field.value == "supply_temperature"
    ]
    for bound in supply_bounds:
        assert bound.bounds.lower <= max(temperatures)
        assert bound.bounds.upper >= min(temperatures)
    sensible_span_bounds = [
        bound.bounds
        for bound in bounds
        if bound.coordinate.field.value == "temperature_span"
    ]
    assert sensible_span_bounds
    assert all(
        bound.upper >= max(temperatures) - min(temperatures)
        for bound in sensible_span_bounds
    )


@settings(max_examples=50, deadline=None)
@given(envelope=residual_profile_envelopes())
def test_coordinate_bounds_cover_generated_residual_profile_support(envelope) -> None:
    temperatures, hot_profile, cold_profile = envelope
    request = UtilityPlacementRequest(
        isothermal_level_count=2,
        sensible_level_count=2,
    )
    blueprints = placement_application.prepare_template_blueprints(request)

    bounds = placement_application._coordinate_bounds(
        request,
        blueprints,
        temperatures,
        hot_profile=hot_profile,
        cold_profile=cold_profile,
    )

    def changing_support(profile):
        support = tuple(
            temperature
            for index, temperature in enumerate(temperatures)
            if (index > 0 and abs(profile[index] - profile[index - 1]) > 1e-12)
            or (
                index < len(profile) - 1
                and abs(profile[index] - profile[index + 1]) > 1e-12
            )
        )
        return support or temperatures

    supply_bounds = {
        (bound.coordinate.template_key.side, bound.coordinate.template_key.name): (
            bound.bounds.lower,
            bound.bounds.upper,
        )
        for bound in bounds
        if bound.coordinate.field.value == "supply_temperature"
    }
    for (side, _), (lower, upper) in supply_bounds.items():
        support = changing_support(hot_profile if side.value == "hot" else cold_profile)
        assert lower <= min(support)
        assert upper >= max(support)
    assert all(
        bound.bounds.upper >= max(temperatures) - min(temperatures)
        for bound in bounds
        if bound.coordinate.field.value == "temperature_span"
    )


def _tiny_options():
    return {
        "iteration_limit": 1,
        "evaluation_limit": 100,
        "candidate_limit": 2,
        "run_count": 1,
    }


def _temperature_overlap_ratio(background_segments, utility_segments) -> float:
    background = [
        point["y"]
        for segment in background_segments
        for point in segment["data_points"]
    ]
    utility = [
        point["y"] for segment in utility_segments for point in segment["data_points"]
    ]
    background_span = max(background) - min(background)
    overlap = min(max(background), max(utility)) - max(min(background), min(utility))
    return max(0.0, overlap) / background_span


def _allocated_duties(levels) -> dict[str, float]:
    return {
        level.template_key.name: level.allocated_duty.value for level in levels
    }


def _targeted_duties(utilities, *, period_idx: int | None = None) -> dict[str, float]:
    return {
        utility.name: float(
            get_scalar_value(utility.heat_flow, period_idx=period_idx)
        )
        for utility in utilities
    }


def _problem_with_utilities(utilities) -> PinchProblem:
    source = (
        PinchWorkspace(source="chocolate_factory.json")
        .case("baseline")
        .to_problem_json()
    )
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

    hot = next(
        template for template in request.hot_templates if "Profiled" in template.name
    )
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
    assert (
        placement_application._resolve_placement_zone(problem, "Process A") is process
    )
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
    active_hot = [
        level for level in period.hot_levels if level.allocated_duty.value > 0
    ]
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
    assert result.best.period_results[
        0
    ].thermodynamic.total_entropy_generation.value == pytest.approx(
        result.best.thermodynamic_total.value
    )


def test_total_site_four_level_request_keeps_inactive_sensible_candidates() -> None:
    problem = PinchWorkspace(
        source="chocolate_factory.json", project_name="Site"
    ).use_case("baseline")
    optimized_case = problem.target.utility_placement(
        isothermal=2,
        sensible=2,
        period_ids=("0",),
        options={
            **_tiny_options(),
            "minimum_sensible_span": {
                "value": 10.0,
                "unit": "delta_degC",
            },
        },
    )
    period = optimized_case.utility_placement_result.best.period_results[0]

    sensible_cold = [
        level
        for level in period.cold_levels
        if level.kind is UtilityLevelKind.SENSIBLE and not level.is_fallback
    ]
    assert len(sensible_cold) == 2
    assert all(level.temperature_span.value > 5.0 for level in sensible_cold)
    assert any(level.allocated_duty.value == 0.0 for level in sensible_cold)


@pytest.mark.parametrize("zone", ["Almond", None])
def test_optimizer_evidence_exactly_matches_ordinary_retargeted_utility_duties(
    zone,
) -> None:
    problem = PinchWorkspace(
        source="chocolate_factory.json",
        project_name="Site",
    ).use_case("baseline")
    maximum_duties = None
    if zone is not None:
        maximum_duties = {
            f"hot_{suffix}": 20.0
            for suffix in ("iso_1", "iso_2", "sensible_1", "sensible_2")
        }
    optimized_case = problem.target.utility_placement(
        isothermal=2,
        sensible=2,
        zone=zone,
        period_ids=("0",),
        maximum_duties=maximum_duties,
        options=_tiny_options(),
    )
    evidence = optimized_case.utility_placement_result.best.period_results[0]

    if zone is None:
        target = optimized_case.target.total_site_heat_integration(period_id="0")
        graph = optimized_case.plot.total_site_profiles(return_graph_data=True)
        assert graph["type"] == GraphType.TSP.value
    else:
        target = optimized_case.target.direct_heat_integration(
            zone=zone,
            period_id="0",
        )
        graph = optimized_case.plot.grand_composite_curve(
            zone_name=zone,
            return_graph_data=True,
        )
        assert graph["type"] == GraphType.GCC.value
        assert all(
            float(utility) <= float(process) + 1e-6
            for process, utility in zip(
                target.pt[ProblemTableLabel.H_NET_A],
                target.pt[ProblemTableLabel.H_NET_UT],
                strict=True,
            )
        )

    assert _allocated_duties(evidence.hot_levels) == pytest.approx(
        _targeted_duties(target.hot_utilities), abs=1e-6
    )
    assert _allocated_duties(evidence.cold_levels) == pytest.approx(
        _targeted_duties(target.cold_utilities), abs=1e-6
    )
    assert evidence.residual_hot_duty.value == pytest.approx(
        float(target.hot_utility_target), abs=1e-6
    )
    assert evidence.residual_cold_duty.value == pytest.approx(
        float(target.cold_utility_target), abs=1e-6
    )


def test_returned_case_retargets_exact_duties_in_every_selected_period() -> None:
    source = {
        "streams": [
            {
                "zone": "Site/AreaA",
                "name": "HotA",
                "t_supply": {"values": [200.0, 220.0], "unit": "degC"},
                "t_target": {"values": [80.0, 100.0], "unit": "degC"},
                "heat_flow": {"values": [120.0, 160.0], "unit": "kW"},
                "dt_cont": 10.0,
                "htc": 1.0,
            },
            {
                "zone": "Site/AreaA",
                "name": "ColdA",
                "t_supply": {"values": [20.0, 30.0], "unit": "degC"},
                "t_target": {"values": [160.0, 180.0], "unit": "degC"},
                "heat_flow": {"values": [180.0, 240.0], "unit": "kW"},
                "dt_cont": 10.0,
                "htc": 1.0,
            },
        ],
        "utilities": [],
        "zone_tree": {
            "name": "Site",
            "type": "Site",
            "children": [{"name": "AreaA", "type": "Process Zone"}],
        },
        "options": {"PROBLEM_PERIOD_IDS": ["base", "peak"]},
    }
    problem = PinchProblem(source, project_name="Site")
    optimized_case = problem.target.utility_placement(
        isothermal=2,
        zone="AreaA",
        maximum_duties={"hot_iso_1": 10.0, "hot_iso_2": 10.0},
        options=_tiny_options(),
    )
    evidence_by_period = {
        period.period_id: period
        for period in optimized_case.utility_placement_result.best.period_results
    }

    for period_id, evidence in evidence_by_period.items():
        period_idx = optimized_case.period_ids[period_id]
        target = optimized_case.target.direct_heat_integration(
            zone="AreaA",
            period_id=period_id,
        )
        assert _allocated_duties(evidence.hot_levels) == pytest.approx(
            _targeted_duties(target.hot_utilities, period_idx=period_idx), abs=1e-6
        )
        assert _allocated_duties(evidence.cold_levels) == pytest.approx(
            _targeted_duties(target.cold_utilities, period_idx=period_idx), abs=1e-6
        )


def test_capped_process_dispatch_uses_available_levels_before_fallback() -> None:
    problem = PinchWorkspace(
        source="chocolate_factory.json", project_name="Site"
    ).use_case("baseline")
    optimized_case = problem.target.utility_placement(
        isothermal=2,
        sensible=2,
        zone="Almond",
        period_ids=("0",),
        maximum_duties={
            f"hot_{suffix}": 20.0
            for suffix in ("iso_1", "iso_2", "sensible_1", "sensible_2")
        },
        options=_tiny_options(),
    )
    result = optimized_case.utility_placement_result
    period = result.best.period_results[0]
    hot_levels = period.hot_levels
    named_active = [
        level
        for level in hot_levels
        if not level.is_fallback and level.allocated_duty.value > 1e-9
    ]
    hot_fallback = sum(
        level.allocated_duty.value for level in hot_levels if level.is_fallback
    )
    cold_fallback = sum(
        level.allocated_duty.value
        for level in period.cold_levels
        if level.is_fallback
    )
    expected_penalty = (
        hot_fallback / period.residual_hot_duty.value
    ) ** 2 + (cold_fallback / period.residual_cold_duty.value) ** 2

    assert len(named_active) >= 3
    assert period.fallback_penalty.value == pytest.approx(expected_penalty)
    assert result.best.fallback_penalty.value == pytest.approx(
        period.weight * expected_penalty
    )


@pytest.mark.parametrize("zone", ["Almond", None])
def test_notebook_scopes_cover_residual_profile_temperature_support(zone) -> None:
    problem = PinchWorkspace(
        source="chocolate_factory.json",
        project_name="Site",
    ).use_case("baseline")
    optimized_case = problem.target.utility_placement(
        isothermal=2,
        sensible=2,
        zone=zone,
        period_ids=("0",),
        options=_tiny_options(),
    )
    result = optimized_case.utility_placement_result
    _, context = build_problem_placement_context(problem, result.request)
    source_period = context.periods[0]
    result_period = result.best.period_results[0]
    temperatures = source_period.snapshot.shifted_temperatures

    def changing_support(profile):
        return {
            temperature
            for index, temperature in enumerate(temperatures)
            if (index > 0 and abs(profile[index] - profile[index - 1]) > 1e-9)
            or (
                index < len(profile) - 1
                and abs(profile[index] - profile[index + 1]) > 1e-9
            )
        }

    hot_support = changing_support(source_period.snapshot.hot_load_profile)
    cold_support = changing_support(source_period.snapshot.cold_load_profile)
    hot_support_floor = min(hot_support)
    hot_support_ceiling = max(hot_support)
    cold_support_floor = min(cold_support)
    active_hot = [
        level for level in result_period.hot_levels if level.allocated_duty.value > 0
    ]
    active_cold = [
        level for level in result_period.cold_levels if level.allocated_duty.value > 0
    ]

    for hot, cold in zip(
        result_period.hot_levels, result_period.cold_levels, strict=True
    ):
        assert cold.kind is hot.kind
        assert cold.supply_temperature.value == pytest.approx(
            hot.target_temperature.value
        )
        assert cold.target_temperature.value == pytest.approx(
            hot.supply_temperature.value
        )
        assert cold.temperature_span == hot.temperature_span

    assert (
        min(level.target_temperature.value for level in active_hot)
        <= hot_support_floor + 35.0
    )
    assert (
        min(
            abs(level.supply_temperature.value - hot_support_ceiling)
            for level in active_hot
        )
        <= 5.0
    )
    assert (
        min(
            abs(level.supply_temperature.value - cold_support_floor)
            for level in active_cold
        )
        <= 5.0
    )

    if zone is not None:
        optimized_case.target.direct_heat_integration(zone=zone, period_id="0")
        graph = optimized_case.plot.grand_composite_curve(
            zone_name=zone,
            return_graph_data=True,
        )
        background_segments = [
            segment for segment in graph["segments"] if segment["title"] == "GCC 1"
        ]
        utility_segments = [
            segment
            for segment in graph["segments"]
            if segment["title"].startswith("Utility GCC")
        ]
        assert _temperature_overlap_ratio(background_segments, utility_segments) > 0.75
    else:
        optimized_case.target.total_site_heat_integration(period_id="0")
        graph = optimized_case.plot.total_site_profiles(return_graph_data=True)
        by_title = {segment["title"]: segment for segment in graph["segments"]}
        assert (
            _temperature_overlap_ratio(
                [by_title["Cold CC"]],
                [by_title["Hot Utility"]],
            )
            > 0.75
        )
        assert by_title["Cold Utility"]["data_points"]


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
    assert (
        added_case.utility_placement_result == optimized_case.utility_placement_result
    )

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


@pytest.mark.parametrize("zone_type", [ZoneType.C, ZoneType.R])
def test_public_aggregate_workflow_retargets_exact_indirect_duties(zone_type) -> None:
    seed = PinchProblem("zonal_site.json", project_name="Scope")
    source = seed.to_problem_json()
    source["zone_tree"]["name"] = "Scope"
    source["zone_tree"]["type"] = zone_type.value
    problem = PinchProblem(source, project_name="Scope")

    optimized_case = problem.target.utility_placement(
        isothermal=2,
        sensible=2,
        period_ids=("0",),
        options=_tiny_options(),
    )

    result = optimized_case.utility_placement_result
    assert result.scope is UtilityPlacementBaseTarget.INDIRECT
    assert len(optimized_case.to_problem_json()["utilities"]) == 8
    evidence = result.best.period_results[0]
    target = optimized_case.target.indirect_heat_integration(period_id="0")
    assert _allocated_duties(evidence.hot_levels) == pytest.approx(
        _targeted_duties(target.hot_utilities), abs=1e-6
    )
    assert _allocated_duties(evidence.cold_levels) == pytest.approx(
        _targeted_duties(target.cold_utilities), abs=1e-6
    )


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
