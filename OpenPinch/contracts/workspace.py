"""Schemas for validation reports and persisted workspaces."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


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


class WorkspaceCaseBundleEntry(BaseModel):
    """One persisted case input inside a workspace bundle."""

    case_input: Dict[str, Any]

    model_config = ConfigDict(extra="forbid")


class PinchWorkspaceBundle(BaseModel):
    """Portable persisted multi-case study bundle."""

    schema_version: Literal["3"] = "3"
    project_name: Optional[str] = None
    baseline_name: str = "baseline"
    cases: Dict[str, WorkspaceCaseBundleEntry]

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _reject_unsupported_schema(cls, value):
        if isinstance(value, dict):
            schema_version = value.get("schema_version", "3")
            if schema_version != "3":
                raise ValueError(
                    "Unsupported workspace schema_version "
                    f"{schema_version!r}; expected '3'."
                )
        return value


__all__ = [
    "ConfigurationFieldMetadata",
    "PinchWorkspaceBundle",
    "ValidationIssue",
    "ValidationReport",
    "WorkspaceCaseBundleEntry",
]
