"""Schemas for validation reports and persisted workspaces."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

_FORBIDDEN_CASE_NAME_CHARACTERS = frozenset('<>:"/\\|?*')
_WINDOWS_DEVICE_NAMES = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
    }
)


def validate_workspace_case_name(value: str) -> str:
    """Return one portable case name or reject it without normalization."""
    if not isinstance(value, str):
        raise ValueError("workspace case name must be a string")
    if not value:
        raise ValueError("workspace case name must be non-empty")
    if value != value.strip():
        raise ValueError(
            "workspace case name must not have leading or trailing whitespace"
        )
    if value in {".", ".."}:
        raise ValueError("workspace case name cannot be '.' or '..'")
    if value.endswith("."):
        raise ValueError("workspace case name must not end with a period")
    if any(character in _FORBIDDEN_CASE_NAME_CHARACTERS for character in value):
        raise ValueError("workspace case name contains a forbidden path character")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError("workspace case name contains a control character")
    if value.split(".", 1)[0].upper() in _WINDOWS_DEVICE_NAMES:
        raise ValueError("workspace case name is reserved by Windows")
    return value


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

    schema_version: Literal["3"]
    project_name: Optional[str] = None
    baseline_name: str = "baseline"
    cases: Dict[str, WorkspaceCaseBundleEntry]

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _reject_unsupported_schema(cls, value):
        if isinstance(value, Mapping):
            schema_version = value.get("schema_version")
            if schema_version != "3":
                raise ValueError(
                    "Unsupported workspace schema_version "
                    f"{schema_version!r}; expected '3'."
                )
            validate_workspace_case_name(value.get("baseline_name", "baseline"))
            cases = value.get("cases")
            if isinstance(cases, Mapping):
                for case_name in cases:
                    validate_workspace_case_name(case_name)
        return value


__all__ = [
    "ConfigurationFieldMetadata",
    "PinchWorkspaceBundle",
    "ValidationIssue",
    "ValidationReport",
    "WorkspaceCaseBundleEntry",
]
