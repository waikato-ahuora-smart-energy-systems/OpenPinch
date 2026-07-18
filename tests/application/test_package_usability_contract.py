"""Closed manifests for the process-engineer-facing package vocabulary."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

import OpenPinch
from OpenPinch import PinchProblem, PinchWorkspace

EXPECTED_ROOT_EXPORTS = {"PinchProblem", "PinchWorkspace"}

EXPECTED_TARGET_METHODS = {
    "all_heat_integration",
    "brayton_heat_pump",
    "brayton_refrigeration",
    "carnot_heat_pump",
    "carnot_refrigeration",
    "cogeneration",
    "direct_heat_integration",
    "energy_transfer",
    "exergy",
    "heat_exchanger_area_and_cost",
    "indirect_heat_integration",
    "isentropic_cogeneration",
    "mvr_heat_pump",
    "sun_smith_cogeneration",
    "total_site_heat_integration",
    "vapour_compression_heat_pump",
    "vapour_compression_refrigeration",
    "varbanov_cogeneration",
}

FORBIDDEN_AMBIGUOUS_TARGET_METHODS = {
    "all",
    "area_cost",
    "configured_analyses",
    "direct",
    "indirect",
}

EXPECTED_COMPONENT_METHODS = {"add_process_mvr"}
EXPECTED_DESIGN_METHODS = {
    "enhanced_heat_exchanger_network",
    "heat_exchanger_network",
    "multiperiod_heat_exchanger_network",
    "network_evolution",
    "open_hens",
    "pinch_design",
    "thermal_derivative",
}
EXPECTED_WORKSPACE_METHODS = {
    "case",
    "cases",
    "compare_cases",
    "compare_to",
    "export_excel",
    "list_cases",
    "load",
    "load_bundle",
    "metrics",
    "report",
    "save_bundle",
    "scenario",
    "set_dt_cont_multiplier",
    "show_dashboard",
    "summary_frame",
    "to_problem_json",
    "update_options",
    "use_case",
    "validate",
    "validation_report",
}


def test_root_exports_are_exact_and_intentional():
    assert set(OpenPinch.__all__) == EXPECTED_ROOT_EXPORTS
    assert OpenPinch.PinchProblem.__name__ == "PinchProblem"
    assert OpenPinch.PinchWorkspace.__name__ == "PinchWorkspace"


def test_target_contract_manifest_is_closed_and_descriptive():
    assert EXPECTED_TARGET_METHODS.isdisjoint(FORBIDDEN_AMBIGUOUS_TARGET_METHODS)
    assert all(len(method_name) > 5 for method_name in EXPECTED_TARGET_METHODS)
    assert "carnot_heat_pump" in EXPECTED_TARGET_METHODS
    assert "heat_pump" not in EXPECTED_TARGET_METHODS


def test_live_target_accessor_matches_the_closed_manifest():
    target = PinchProblem().target
    public_methods = {
        name
        for name in dir(target)
        if not name.startswith("_") and callable(getattr(target, name))
    }

    assert public_methods == EXPECTED_TARGET_METHODS
    assert not callable(target)
    for retired_name in FORBIDDEN_AMBIGUOUS_TARGET_METHODS:
        assert not hasattr(target, retired_name)


def test_component_and_design_accessors_match_closed_manifests():
    problem = PinchProblem()
    components = {
        name
        for name in dir(problem.components)
        if not name.startswith("_") and callable(getattr(problem.components, name))
    }
    design = {
        name
        for name in dir(problem.design)
        if not name.startswith("_") and callable(getattr(problem.design, name))
    }

    assert components == EXPECTED_COMPONENT_METHODS
    assert design == EXPECTED_DESIGN_METHODS
    assert not hasattr(problem, "add_component")
    for retired_name in (
        "heat_exchanger_network_synthesis",
        "enhanced_synthesis_method",
        "open_hens_method",
        "pinch_design_method",
        "thermal_derivative_method",
        "network_evolution_method",
    ):
        assert not hasattr(problem.design, retired_name)


def test_workspace_class_matches_the_case_only_manifest():
    methods = {
        name
        for name, value in inspect.getmembers(PinchWorkspace, inspect.isroutine)
        if not name.startswith("_")
    }

    assert methods == EXPECTED_WORKSPACE_METHODS
    for retired_name in (
        "from_json",
        "copy_case",
        "get_case_input",
        "list_variants",
        "get_variant_input",
        "input_view",
        "validate_variant",
        "set_variant_input",
        "solve_variant",
        "compare_variants",
        "configuration_field_metadata",
    ):
        assert not hasattr(PinchWorkspace, retired_name)


def test_root_only_quickstart_import_compiles():
    compile(
        "from OpenPinch import PinchProblem, PinchWorkspace\n"
        "problem = PinchProblem()\n"
        "workspace = PinchWorkspace()\n",
        "<openpinch-quickstart>",
        "exec",
    )


def test_hpr_signatures_use_named_engineering_arguments():
    target = PinchProblem().target
    carnot = inspect.signature(target.carnot_heat_pump).parameters
    vapour_compression = inspect.signature(
        target.vapour_compression_heat_pump
    ).parameters
    mvr = inspect.signature(target.mvr_heat_pump).parameters

    assert {"is_utility_heat_pump", "is_cascade_cycle", "load_fraction"} <= set(carnot)
    assert {"refrigerants", "initialize_from_carnot"} <= set(vapour_compression)
    assert {"mvr_fluids", "mvr_stages", "motor_efficiency"} <= set(mvr)
    assert "placement" not in carnot
    assert "cycle" not in carnot
    assert "kwargs" not in vapour_compression

    process_mvr = inspect.signature(
        PinchProblem().components.add_process_mvr
    ).parameters
    assert {"compressor_efficiency", "motor_efficiency"} <= set(process_mvr)
    assert "eta_mvr_comp" not in process_mvr
    assert "eta_motor" not in process_mvr


def test_hpr_rejects_conflicting_named_load_forms_before_execution():
    with pytest.raises(
        ValueError,
        match="Supply only one of load_fraction, load_duty, or period_loads",
    ):
        PinchProblem().target.carnot_heat_pump(
            load_fraction=0.25,
            load_duty=1000.0,
        )


def test_hpr_invocation_options_are_ephemeral(monkeypatch):
    problem = PinchProblem("crude_preheat_train.json")
    stored_config = dict(problem.config)
    observed = {}

    def fake_execute(self, **_kwargs):
        observed["type"] = self.master_zone.config.hpr.type
        observed["load"] = self.master_zone.config.hpr.load_fraction
        observed["refrigerants"] = self.master_zone.config.hpr.refrigerants
        return SimpleNamespace(name="Site/Vapour Compression Heat Pump")

    monkeypatch.setattr(PinchProblem, "_execute_targeting", fake_execute)

    problem.target.vapour_compression_heat_pump(
        load_fraction=0.25,
        refrigerants=["water", "ammonia"],
    )

    assert observed == {
        "type": "Cascade vapour compression cycles",
        "load": 0.25,
        "refrigerants": ["water", "ammonia"],
    }
    assert "HPR_TYPE" not in problem.config
    assert dict(problem.config) == stored_config
