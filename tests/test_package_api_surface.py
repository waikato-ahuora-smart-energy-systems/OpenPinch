"""API surface tests for the simplified OpenPinch package barrels."""

from __future__ import annotations

import OpenPinch
import OpenPinch.classes
import OpenPinch.lib
import OpenPinch.services
import OpenPinch.utils

EXPECTED_EXPORTS = {
    "OpenPinch": {
        "Configuration",
        "GraphAvailability",
        "GraphType",
        "HPRcycle",
        "NotebookMetadata",
        "PinchProblem",
        "PinchWorkspace",
        "ProblemReport",
        "ReportMetric",
        "SampleCaseMetadata",
        "StreamSegmentSchema",
        "StreamSchema",
        "StreamType",
        "TargetInput",
        "TargetOutput",
        "TargetType",
        "TemperatureHeatPointSchema",
        "TemperatureHeatProfileSchema",
        "UtilitySchema",
        "ValidationIssue",
        "ValidationReport",
        "ZoneTreeSchema",
        "ZoneType",
        "config_options",
        "copy_notebook",
        "copy_sample_case",
        "get_piecewise_linearisation_for_streams",
        "list_notebooks",
        "list_sample_cases",
        "notebook_metadata",
        "pinch_analysis_service",
        "read_sample_case",
        "sample_case_metadata",
    },
    "OpenPinch.lib": {
        "BaseTargetModel",
        "DirectIntegrationTarget",
        "Configuration",
        "Graph",
        "GraphSet",
        "HeatExchangerNetworkLabel",
        "HeatExchangerNetworkSynthesisResult",
        "HeatExchangerNetworkSynthesisTask",
        "HeatUtility",
        "HeatPumpTargetOutputs",
        "StreamSchema",
        "TargetInput",
        "TargetOutput",
        "TargetResults",
        "UtilitySchema",
        "ValueWithUnit",
        "ZoneTreeSchema",
        "ZoneType",
    },
    "OpenPinch.classes": {
        "HeatExchanger",
        "HeatExchangerNetwork",
        "PinchProblem",
        "PinchWorkspace",
        "ProblemTable",
        "Stream",
        "StreamCollection",
        "Value",
        "Zone",
    },
    "OpenPinch.utils": {
        "export_target_summary_to_excel_with_units",
        "get_problem_from_csv",
        "get_problem_from_excel",
        "graph_simple_cc_plot",
        "timing_decorator",
        "validate_stream_data",
        "validate_utility_data",
    },
    "OpenPinch.services": {
        "area_cost_targeting_service",
        "data_preprocessing_service",
        "direct_heat_pump_service",
        "direct_heat_integration_service",
        "direct_refrigeration_service",
        "indirect_heat_integration_service",
        "indirect_heat_pump_service",
        "indirect_refrigeration_service",
        "get_area_targets",
        "get_capital_cost_targets",
        "get_output_graph_data",
        "get_utility_targets",
        "power_cogeneration_service",
    },
}


MODULES = {
    "OpenPinch": OpenPinch,
    "OpenPinch.lib": OpenPinch.lib,
    "OpenPinch.classes": OpenPinch.classes,
    "OpenPinch.utils": OpenPinch.utils,
    "OpenPinch.services": OpenPinch.services,
}


def _public_exports(module) -> set[str]:
    if hasattr(module, "__all__"):
        return set(module.__all__)
    return {name for name in dir(module) if not name.startswith("_")}


def test_public_namespace_exports_match_snapshot():
    for name, module in MODULES.items():
        actual = _public_exports(module)
        expected = EXPECTED_EXPORTS[name]
        if name == "OpenPinch":
            assert actual == expected
        else:
            assert expected <= actual
        for exported_name in expected:
            assert hasattr(module, exported_name)


def test_openhens_compatibility_surfaces_are_not_root_exported():
    forbidden_exports = {
        "OpenHENS",
        "CaseStudy",
        "SynthesisStudy",
        "run_synthesis_workflow",
        "HeatExchangerNetworkDesignSpace",
        "HeatExchangerNetworkMethodSequence",
        "HeatExchangerNetworkSolveSetup",
        "HeatExchangerNetworkOutputs",
    }

    for module in MODULES.values():
        assert forbidden_exports.isdisjoint(_public_exports(module))


def test_heat_exchanger_area_slice_value_model_is_not_barrel_exported():
    slice_type_names = {
        "HeatExchangerSegmentAreaContribution",
        "HeatExchangerAreaSlice",
    }

    assert slice_type_names.isdisjoint(_public_exports(OpenPinch))
    assert slice_type_names.isdisjoint(_public_exports(OpenPinch.classes))


def test_parent_owned_runtime_records_are_absent_from_public_modules():
    from OpenPinch.classes import heat_exchanger, stream

    private_records = {
        "StreamSegment",
        "HeatExchangerPeriodState",
        "HeatExchangerAreaSlice",
    }
    assert private_records.isdisjoint(_public_exports(OpenPinch))
    assert private_records.isdisjoint(_public_exports(OpenPinch.classes))
    assert not hasattr(stream, "StreamSegment")
    assert not hasattr(heat_exchanger, "HeatExchangerPeriodState")
    assert not hasattr(heat_exchanger, "HeatExchangerAreaSlice")
    assert hasattr(OpenPinch, "StreamSegmentSchema")
