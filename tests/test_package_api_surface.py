"""API surface tests for the simplified OpenPinch package barrels."""

from __future__ import annotations

import OpenPinch
import OpenPinch.analysis
import OpenPinch.classes
import OpenPinch.lib
import OpenPinch.utils


EXPECTED_EXPORTS = {
    "OpenPinch": {
        "Configuration",
        "HeatPumpIntegrationComparison",
        "HeatPumpIntegrationScenario",
        "PinchProblem",
        "StreamCollection",
        "StreamSchema",
        "StreamType",
        "TargetInput",
        "TargetOutput",
        "UtilitySchema",
        "VisualiseInput",
        "VisualiseOutput",
        "ZoneTreeSchema",
        "ZoneType",
        "extract_results",
        "get_piecewise_linearisation_for_streams",
        "get_targets",
        "get_visualise",
        "pinch_analysis_service",
    },
    "OpenPinch.lib": {
        "Configuration",
        "Graph",
        "GraphSet",
        "HeatUtility",
        "HeatPumpIntegrationComparison",
        "HeatPumpIntegrationScenario",
        "StreamCollection",
        "StreamSchema",
        "TargetInput",
        "TargetOutput",
        "TargetResults",
        "UtilitySchema",
        "ValueWithUnit",
        "VisualiseInput",
        "VisualiseOutput",
        "ZoneTreeSchema",
        "ZoneType",
    },
    "OpenPinch.classes": {
        "CascadeVapourCompressionCycle",
        "EnergyTarget",
        "MultiStageSteamTurbine",
        "ParallelVapourCompressionCycles",
        "PinchProblem",
        "ProblemTable",
        "SimpleBraytonHeatPumpCycle",
        "Stream",
        "StreamCollection",
        "Value",
        "VapourCompressionCycle",
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
    "OpenPinch.analysis": {
        "compute_direct_integration_targets",
        "compute_indirect_integration_targets",
        "get_area_targets",
        "get_capital_cost_targets",
        "get_output_graph_data",
        "get_utility_targets",
        "prepare_problem",
        "visualise_graphs",
    },
}


MODULES = {
    "OpenPinch": OpenPinch,
    "OpenPinch.lib": OpenPinch.lib,
    "OpenPinch.classes": OpenPinch.classes,
    "OpenPinch.utils": OpenPinch.utils,
    "OpenPinch.analysis": OpenPinch.analysis,
}


def _public_exports(module) -> set[str]:
    if hasattr(module, "__all__"):
        return set(module.__all__)
    return {name for name in dir(module) if not name.startswith("_")}


def test_public_namespace_exports_match_snapshot():
    for name, module in MODULES.items():
        actual = _public_exports(module)
        expected = EXPECTED_EXPORTS[name]
        assert expected <= actual
        for exported_name in expected:
            assert hasattr(module, exported_name)
