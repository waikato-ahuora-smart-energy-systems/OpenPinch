"""Schemas for frontend-oriented scenario workspace contracts."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ValidationIssue(BaseModel):
    """One structured validation issue for frontend display."""

    severity: str
    path: str
    message: str
    section: Optional[str] = None
    record_index: Optional[int] = None
    field: Optional[str] = None
    record_label: Optional[str] = None


class ValidationReport(BaseModel):
    """Structured validation report for one editable case input."""

    valid: bool
    issues: List[ValidationIssue] = Field(default_factory=list)


class TableView(BaseModel):
    """Generic tabular data for frontend data-grid rendering."""

    columns: List[str] = Field(default_factory=list)
    rows: List[Dict[str, Any]] = Field(default_factory=list)


class SummaryCard(BaseModel):
    """One summary-card metric tied to a specific target."""

    card_id: str
    target_id: str
    target_name: str
    label: str
    value: Any = None
    unit: Optional[str] = None


class GraphCatalogEntry(BaseModel):
    """Frontend-friendly graph catalog row with stable identifiers."""

    graph_id: str
    graph_set_id: str
    target_id: str
    target_name: str
    zone_name: Optional[str] = None
    zone_address: Optional[str] = None
    target_type: Optional[str] = None
    graph_type: Optional[str] = None
    graph_name: str
    index: int


class GraphDataEntry(BaseModel):
    """Serializable graph data attached to one graph identifier."""

    graph_id: str
    graph_set_id: str
    target_id: str
    target_name: str
    graph_type: Optional[str] = None
    graph_name: str
    graph_data: Dict[str, Any]


class ProblemTableView(BaseModel):
    """Serializable shifted or real Problem Table view for one target."""

    table_id: str
    target_id: str
    target_name: str
    table_kind: str
    table: TableView


class ZoneNodeView(BaseModel):
    """Flattened zone-tree node with a stable path identifier."""

    zone_id: str
    path: str
    name: str
    zone_type: Optional[str] = None
    parent_id: Optional[str] = None
    dt_cont_multiplier: Optional[float] = None


class InputRecordView(BaseModel):
    """Editable stream or utility record with a stable path identifier."""

    record_id: str
    path: str
    section: str
    index: int
    name: Optional[str] = None
    zone: Optional[str] = None
    data: Dict[str, Any]


class VariantInputView(BaseModel):
    """Frontend-friendly editable input view for one workspace variant."""

    variant_name: str
    zones: List[ZoneNodeView] = Field(default_factory=list)
    streams: List[InputRecordView] = Field(default_factory=list)
    utilities: List[InputRecordView] = Field(default_factory=list)
    options: Dict[str, Any] = Field(default_factory=dict)


class ConfigurationFieldMetadata(BaseModel):
    """Declarative metadata for one frontend-editable configuration field."""

    name: str
    label: str
    field_type: str
    group: str
    config_path: List[str] = Field(default_factory=list)
    support_level: str
    runtime_status: str = "supported"
    enum_choices: List[str] = Field(default_factory=list)
    numeric_min: Optional[float] = None
    numeric_max: Optional[float] = None
    multiple: bool = False


class ScenarioWorkflowConfig(BaseModel):
    """Workflow settings associated with solving one variant."""

    workflow: str = "target"
    workflow_options: Dict[str, Any] = Field(default_factory=dict)


class ScenarioVariantView(BaseModel):
    """Serializable solved-or-invalid variant result for frontend consumption."""

    variant_name: str
    period_id: Optional[str] = None
    workflow: str
    workflow_options: Dict[str, Any] = Field(default_factory=dict)
    status: str
    support_level: str
    validation: ValidationReport
    warnings: List[str] = Field(default_factory=list)
    error_message: Optional[str] = None
    error_category: Optional[str] = None
    summary_cards: List[SummaryCard] = Field(default_factory=list)
    summary_table: Optional[TableView] = None
    graph_catalog: List[GraphCatalogEntry] = Field(default_factory=list)
    graph_data_entries: List[GraphDataEntry] = Field(default_factory=list)
    problem_tables: List[ProblemTableView] = Field(default_factory=list)


class VariantMetricDelta(BaseModel):
    """One metric delta between a base variant and another variant."""

    base_variant: str
    variant_name: str
    target_id: str
    target_name: str
    metric: str
    base_value: Any = None
    variant_value: Any = None
    unit: Optional[str] = None
    delta: Optional[float] = None


class ProblemTableDiffView(BaseModel):
    """Deterministic structural and cell-level diff summary for one table."""

    base_variant: str
    variant_name: str
    target_id: str
    target_name: str
    table_kind: str
    base_rows: int
    variant_rows: int
    shared_columns: List[str] = Field(default_factory=list)
    changed_cells: Optional[int] = None
    shape_changed: bool = False


class ScenarioComparisonView(BaseModel):
    """Serializable comparison view across one or more solved variants."""

    base_variant: str
    variant_names: List[str] = Field(default_factory=list)
    metric_deltas: List[VariantMetricDelta] = Field(default_factory=list)
    graph_catalogs: Dict[str, List[GraphCatalogEntry]] = Field(default_factory=dict)
    problem_table_diffs: List[ProblemTableDiffView] = Field(default_factory=list)


class ScenarioVariantBundleEntry(BaseModel):
    """One persisted variant entry inside a scenario workspace bundle."""

    case_input: Dict[str, Any]
    workflow: ScenarioWorkflowConfig = Field(default_factory=ScenarioWorkflowConfig)
    cached_view: Optional[ScenarioVariantView] = None


class PinchWorkspaceBundle(BaseModel):
    """Portable persisted multi-case study bundle."""

    schema_version: str = "1"
    project_name: Optional[str] = None
    baseline_name: str = "baseline"
    variants: Dict[str, ScenarioVariantBundleEntry]


__all__ = [
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
