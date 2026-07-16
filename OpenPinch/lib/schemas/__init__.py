"""Concern-based schema package for OpenPinch."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .common import (
        HPRMetric,
        MaybeVU,
        PeriodValueWithUnit,
        ScalarOrVU,
        ValueWithUnit,
    )
    from .graphs import DataPoint, Graph, GraphSet, Segment
    from .hpr import (
        HeatPumpTargetInputs,
        HeatPumpTargetOutputs,
        HPRBackendResult,
        HPRParsedState,
        HPRThermoArtifacts,
    )
    from .io import (
        NonLinearStream,
        StreamSchema,
        StreamSegmentSchema,
        TargetInput,
        TargetOutput,
        TemperatureHeatPointSchema,
        TemperatureHeatProfileSchema,
        UtilitySchema,
        ZoneTreeSchema,
    )
    from .reporting import (
        GraphAvailability,
        HeatUtility,
        PinchTemp,
        ProblemReport,
        ReportMetric,
        TargetResults,
    )
    from .synthesis.common import (
        HeatExchangerNetworkSynthesisExportRecord,
        HeatExchangerNetworkSynthesisManifest,
        SynthesisDesignMethod,
        SynthesisMethod,
        SynthesisOutputFormat,
        SynthesisTaskStatus,
    )
    from .synthesis.method import (
        HeatExchangerNetworkSynthesisMethodInput,
        HeatExchangerNetworkSynthesisMethodOutput,
    )
    from .synthesis.result import HeatExchangerNetworkSynthesisResult
    from .synthesis.task import (
        HeatExchangerNetworkSynthesisTask,
        HeatExchangerNetworkSynthesisTaskOutcome,
    )
    from .synthesis.topology import (
        HeatExchangerNetworkTopologyRestriction,
    )
    from .targets import (
        AnyTargetModel,
        BaseTargetModel,
        DirectHeatPumpTarget,
        DirectIntegrationTarget,
        DirectRefrigerationTarget,
        EnergyTransferTarget,
        HeatPumpTargetBase,
        IndirectHeatPumpTarget,
        IndirectRefrigerationTarget,
        TotalProcessTarget,
        TotalSiteTarget,
        UtilitySummaryTarget,
    )
    from .turbine import TurbineSolveResult, TurbineStageResult
    from .workspace import (
        ConfigurationFieldMetadata,
        GraphCatalogEntry,
        GraphDataEntry,
        InputRecordView,
        PinchWorkspaceBundle,
        ProblemTableDiffView,
        ProblemTableView,
        ScenarioComparisonView,
        ScenarioVariantBundleEntry,
        ScenarioVariantView,
        ScenarioWorkflowConfig,
        SummaryCard,
        TableView,
        ValidationIssue,
        ValidationReport,
        VariantInputView,
        VariantMetricDelta,
        ZoneNodeView,
    )

__all__ = [
    "HPRMetric",
    "MaybeVU",
    "ScalarOrVU",
    "PeriodValueWithUnit",
    "ValueWithUnit",
    "DataPoint",
    "Graph",
    "GraphSet",
    "Segment",
    "GraphAvailability",
    "HeatPumpTargetInputs",
    "HeatPumpTargetOutputs",
    "HPRParsedState",
    "HPRThermoArtifacts",
    "HPRBackendResult",
    "NonLinearStream",
    "StreamSchema",
    "StreamSegmentSchema",
    "TargetInput",
    "TargetOutput",
    "UtilitySchema",
    "TemperatureHeatPointSchema",
    "TemperatureHeatProfileSchema",
    "ZoneTreeSchema",
    "HeatUtility",
    "ProblemReport",
    "ReportMetric",
    "TargetResults",
    "PinchTemp",
    "HeatExchangerNetworkSynthesisExportRecord",
    "HeatExchangerNetworkSynthesisManifest",
    "HeatExchangerNetworkSynthesisMethodInput",
    "HeatExchangerNetworkSynthesisMethodOutput",
    "HeatExchangerNetworkSynthesisResult",
    "HeatExchangerNetworkSynthesisTask",
    "HeatExchangerNetworkSynthesisTaskOutcome",
    "HeatExchangerNetworkTopologyRestriction",
    "SynthesisDesignMethod",
    "SynthesisMethod",
    "SynthesisOutputFormat",
    "SynthesisTaskStatus",
    "AnyTargetModel",
    "BaseTargetModel",
    "DirectHeatPumpTarget",
    "DirectIntegrationTarget",
    "DirectRefrigerationTarget",
    "EnergyTransferTarget",
    "HeatPumpTargetBase",
    "IndirectHeatPumpTarget",
    "IndirectRefrigerationTarget",
    "TotalProcessTarget",
    "TotalSiteTarget",
    "UtilitySummaryTarget",
    "TurbineSolveResult",
    "TurbineStageResult",
    "ConfigurationFieldMetadata",
    "GraphCatalogEntry",
    "GraphDataEntry",
    "InputRecordView",
    "ProblemTableDiffView",
    "ProblemTableView",
    "ScenarioComparisonView",
    "ScenarioVariantBundleEntry",
    "ScenarioVariantView",
    "ScenarioWorkflowConfig",
    "PinchWorkspaceBundle",
    "SummaryCard",
    "TableView",
    "ValidationIssue",
    "ValidationReport",
    "VariantMetricDelta",
    "VariantInputView",
    "ZoneNodeView",
]

_EXPORTS: dict[str, tuple[str, str | None]] = {
    "HPRMetric": ("OpenPinch.lib.schemas.common", "HPRMetric"),
    "MaybeVU": ("OpenPinch.lib.schemas.common", "MaybeVU"),
    "PeriodValueWithUnit": ("OpenPinch.lib.schemas.common", "PeriodValueWithUnit"),
    "ScalarOrVU": ("OpenPinch.lib.schemas.common", "ScalarOrVU"),
    "ValueWithUnit": ("OpenPinch.lib.schemas.common", "ValueWithUnit"),
    "DataPoint": ("OpenPinch.lib.schemas.graphs", "DataPoint"),
    "Graph": ("OpenPinch.lib.schemas.graphs", "Graph"),
    "GraphSet": ("OpenPinch.lib.schemas.graphs", "GraphSet"),
    "Segment": ("OpenPinch.lib.schemas.graphs", "Segment"),
    "HeatPumpTargetInputs": ("OpenPinch.lib.schemas.hpr", "HeatPumpTargetInputs"),
    "HeatPumpTargetOutputs": ("OpenPinch.lib.schemas.hpr", "HeatPumpTargetOutputs"),
    "HPRBackendResult": ("OpenPinch.lib.schemas.hpr", "HPRBackendResult"),
    "HPRParsedState": ("OpenPinch.lib.schemas.hpr", "HPRParsedState"),
    "HPRThermoArtifacts": ("OpenPinch.lib.schemas.hpr", "HPRThermoArtifacts"),
    "NonLinearStream": ("OpenPinch.lib.schemas.io", "NonLinearStream"),
    "StreamSchema": ("OpenPinch.lib.schemas.io", "StreamSchema"),
    "StreamSegmentSchema": ("OpenPinch.lib.schemas.io", "StreamSegmentSchema"),
    "TargetInput": ("OpenPinch.lib.schemas.io", "TargetInput"),
    "TargetOutput": ("OpenPinch.lib.schemas.io", "TargetOutput"),
    "TemperatureHeatPointSchema": (
        "OpenPinch.lib.schemas.io",
        "TemperatureHeatPointSchema",
    ),
    "TemperatureHeatProfileSchema": (
        "OpenPinch.lib.schemas.io",
        "TemperatureHeatProfileSchema",
    ),
    "UtilitySchema": ("OpenPinch.lib.schemas.io", "UtilitySchema"),
    "ZoneTreeSchema": ("OpenPinch.lib.schemas.io", "ZoneTreeSchema"),
    "GraphAvailability": ("OpenPinch.lib.schemas.reporting", "GraphAvailability"),
    "HeatUtility": ("OpenPinch.lib.schemas.reporting", "HeatUtility"),
    "PinchTemp": ("OpenPinch.lib.schemas.reporting", "PinchTemp"),
    "ProblemReport": ("OpenPinch.lib.schemas.reporting", "ProblemReport"),
    "ReportMetric": ("OpenPinch.lib.schemas.reporting", "ReportMetric"),
    "TargetResults": ("OpenPinch.lib.schemas.reporting", "TargetResults"),
    "HeatExchangerNetworkSynthesisExportRecord": (
        "OpenPinch.lib.schemas.synthesis.common",
        "HeatExchangerNetworkSynthesisExportRecord",
    ),
    "HeatExchangerNetworkSynthesisManifest": (
        "OpenPinch.lib.schemas.synthesis.common",
        "HeatExchangerNetworkSynthesisManifest",
    ),
    "HeatExchangerNetworkSynthesisMethodInput": (
        "OpenPinch.lib.schemas.synthesis.method",
        "HeatExchangerNetworkSynthesisMethodInput",
    ),
    "HeatExchangerNetworkSynthesisMethodOutput": (
        "OpenPinch.lib.schemas.synthesis.method",
        "HeatExchangerNetworkSynthesisMethodOutput",
    ),
    "HeatExchangerNetworkSynthesisResult": (
        "OpenPinch.lib.schemas.synthesis.result",
        "HeatExchangerNetworkSynthesisResult",
    ),
    "HeatExchangerNetworkSynthesisTask": (
        "OpenPinch.lib.schemas.synthesis.task",
        "HeatExchangerNetworkSynthesisTask",
    ),
    "HeatExchangerNetworkSynthesisTaskOutcome": (
        "OpenPinch.lib.schemas.synthesis.task",
        "HeatExchangerNetworkSynthesisTaskOutcome",
    ),
    "HeatExchangerNetworkTopologyRestriction": (
        "OpenPinch.lib.schemas.synthesis.topology",
        "HeatExchangerNetworkTopologyRestriction",
    ),
    "SynthesisDesignMethod": (
        "OpenPinch.lib.schemas.synthesis.common",
        "SynthesisDesignMethod",
    ),
    "SynthesisMethod": (
        "OpenPinch.lib.schemas.synthesis.common",
        "SynthesisMethod",
    ),
    "SynthesisOutputFormat": (
        "OpenPinch.lib.schemas.synthesis.common",
        "SynthesisOutputFormat",
    ),
    "SynthesisTaskStatus": (
        "OpenPinch.lib.schemas.synthesis.common",
        "SynthesisTaskStatus",
    ),
    "AnyTargetModel": ("OpenPinch.lib.schemas.targets", "AnyTargetModel"),
    "BaseTargetModel": ("OpenPinch.lib.schemas.targets", "BaseTargetModel"),
    "DirectHeatPumpTarget": ("OpenPinch.lib.schemas.targets", "DirectHeatPumpTarget"),
    "DirectIntegrationTarget": (
        "OpenPinch.lib.schemas.targets",
        "DirectIntegrationTarget",
    ),
    "DirectRefrigerationTarget": (
        "OpenPinch.lib.schemas.targets",
        "DirectRefrigerationTarget",
    ),
    "EnergyTransferTarget": ("OpenPinch.lib.schemas.targets", "EnergyTransferTarget"),
    "HeatPumpTargetBase": ("OpenPinch.lib.schemas.targets", "HeatPumpTargetBase"),
    "IndirectHeatPumpTarget": (
        "OpenPinch.lib.schemas.targets",
        "IndirectHeatPumpTarget",
    ),
    "IndirectRefrigerationTarget": (
        "OpenPinch.lib.schemas.targets",
        "IndirectRefrigerationTarget",
    ),
    "TotalProcessTarget": ("OpenPinch.lib.schemas.targets", "TotalProcessTarget"),
    "TotalSiteTarget": ("OpenPinch.lib.schemas.targets", "TotalSiteTarget"),
    "UtilitySummaryTarget": ("OpenPinch.lib.schemas.targets", "UtilitySummaryTarget"),
    "TurbineSolveResult": ("OpenPinch.lib.schemas.turbine", "TurbineSolveResult"),
    "TurbineStageResult": ("OpenPinch.lib.schemas.turbine", "TurbineStageResult"),
    "ConfigurationFieldMetadata": (
        "OpenPinch.lib.schemas.workspace",
        "ConfigurationFieldMetadata",
    ),
    "GraphCatalogEntry": ("OpenPinch.lib.schemas.workspace", "GraphCatalogEntry"),
    "GraphDataEntry": ("OpenPinch.lib.schemas.workspace", "GraphDataEntry"),
    "InputRecordView": ("OpenPinch.lib.schemas.workspace", "InputRecordView"),
    "PinchWorkspaceBundle": ("OpenPinch.lib.schemas.workspace", "PinchWorkspaceBundle"),
    "ProblemTableDiffView": ("OpenPinch.lib.schemas.workspace", "ProblemTableDiffView"),
    "ProblemTableView": ("OpenPinch.lib.schemas.workspace", "ProblemTableView"),
    "ScenarioComparisonView": (
        "OpenPinch.lib.schemas.workspace",
        "ScenarioComparisonView",
    ),
    "ScenarioVariantBundleEntry": (
        "OpenPinch.lib.schemas.workspace",
        "ScenarioVariantBundleEntry",
    ),
    "ScenarioVariantView": ("OpenPinch.lib.schemas.workspace", "ScenarioVariantView"),
    "ScenarioWorkflowConfig": (
        "OpenPinch.lib.schemas.workspace",
        "ScenarioWorkflowConfig",
    ),
    "SummaryCard": ("OpenPinch.lib.schemas.workspace", "SummaryCard"),
    "TableView": ("OpenPinch.lib.schemas.workspace", "TableView"),
    "ValidationIssue": ("OpenPinch.lib.schemas.workspace", "ValidationIssue"),
    "ValidationReport": ("OpenPinch.lib.schemas.workspace", "ValidationReport"),
    "VariantInputView": ("OpenPinch.lib.schemas.workspace", "VariantInputView"),
    "VariantMetricDelta": ("OpenPinch.lib.schemas.workspace", "VariantMetricDelta"),
    "ZoneNodeView": ("OpenPinch.lib.schemas.workspace", "ZoneNodeView"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name)
    value = module if attribute is None else getattr(module, attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | set(_EXPORTS))
