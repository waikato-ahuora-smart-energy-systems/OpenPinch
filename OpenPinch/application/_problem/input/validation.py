"""Schema validation, issue formatting, and report assembly."""

from __future__ import annotations

import warnings
from typing import Any, Optional

from pydantic import ValidationError

from ....contracts.input import TargetInput
from ....contracts.workspace import ValidationIssue, ValidationReport
from .semantics import semantic_issues

ValidationContext = dict[str, list[dict[str, Any]]]


def validate_problem_inputs(
    problem_data: Any,
    *,
    context: Optional[ValidationContext] = None,
) -> TargetInput:
    """Validate one problem definition and raise user-facing errors on failure."""
    effective_context = context or {}
    try:
        validated = TargetInput.model_validate(problem_data)
    except ValidationError as exc:
        raise ValueError(
            format_schema_validation_error(
                exc,
                problem_data=problem_data,
                context=effective_context,
            )
        ) from exc

    validate_problem_semantics(validated, context=effective_context)
    return validated


def build_validation_report(
    problem_inputs: Any,
    *,
    context: Optional[ValidationContext] = None,
    source_kind: str = "target_input",
) -> ValidationReport:
    """Build a structured validation report without raising on invalid inputs."""
    effective_context = context or build_validation_context(
        problem_inputs,
        source_kind=source_kind,
    )
    issues: list[ValidationIssue] = []

    try:
        validated = TargetInput.model_validate(problem_inputs)
    except ValidationError as exc:
        for error in exc.errors():
            issues.append(schema_issue_to_view(error, context=effective_context))
        return ValidationReport(valid=False, issues=issues)

    issues.extend(semantic_issues(validated, context=effective_context))
    return ValidationReport(
        valid=not any(issue.severity == "error" for issue in issues),
        issues=issues,
    )


def build_validation_context(
    problem_data: Any,
    *,
    source_kind: str,
) -> ValidationContext:
    """Build record metadata used for readable validation messages."""
    if not isinstance(problem_data, dict):
        return {}

    context: ValidationContext = {}
    for section in ("streams", "utilities"):
        records = problem_data.get(section)
        if not isinstance(records, list):
            continue
        context[section] = [
            _build_record_context(section, index, record, source_kind=source_kind)
            for index, record in enumerate(records)
        ]
    return context


def schema_issue_to_view(
    error: dict[str, Any],
    *,
    context: ValidationContext,
) -> ValidationIssue:
    """Convert one Pydantic schema error into a workspace validation issue."""
    loc = tuple(error.get("loc", ()))
    section = loc[0] if loc and isinstance(loc[0], str) else None
    record_index = loc[1] if len(loc) > 1 and isinstance(loc[1], int) else None
    field = ".".join(str(part) for part in loc[2:]) if len(loc) > 2 else None
    return ValidationIssue(
        severity="error",
        path=path_from_loc(loc),
        message=error.get("msg", "Invalid value."),
        section=section,
        record_index=record_index,
        field=field,
        record_label=validation_record_label(section, record_index, context),
    )


def validate_problem_semantics(
    problem_inputs: TargetInput,
    *,
    context: ValidationContext,
) -> None:
    """Raise or warn using the shared semantic validation issue list."""
    issues = semantic_issues(problem_inputs, context=context)
    fatal_issues = [
        _format_validation_issue(issue) for issue in issues if issue.severity == "error"
    ]
    warning_issues = [
        _format_validation_issue(issue, include_warning_prefix=True)
        for issue in issues
        if issue.severity == "warning"
    ]

    if fatal_issues:
        raise ValueError(
            "Input validation failed with "
            f"{len(fatal_issues)} issue(s):\n" + "\n".join(fatal_issues)
        )

    if warning_issues:
        warnings.warn(
            "Input validation reported "
            f"{len(warning_issues)} warning(s):\n" + "\n".join(warning_issues),
            UserWarning,
            stacklevel=2,
        )


def format_schema_validation_error(
    exc: ValidationError,
    *,
    problem_data: Any,
    context: ValidationContext,
) -> str:
    """Format one schema-validation exception for user-facing error output."""
    lines = [f"Input validation failed with {len(exc.errors())} issue(s):"]
    for error in exc.errors():
        lines.append(
            format_single_validation_error(
                error,
                problem_data=problem_data,
                context=context,
            )
        )
    return "\n".join(lines)


def format_single_validation_error(
    error: dict[str, Any],
    *,
    problem_data: Any,
    context: ValidationContext,
) -> str:
    """Format one schema error entry using source-aware record labels."""
    del problem_data
    loc = tuple(error.get("loc", ()))
    message = error.get("msg", "Invalid value.")
    section = loc[0] if loc else None
    record_index = loc[1] if len(loc) > 1 and isinstance(loc[1], int) else None
    field_path = ".".join(str(part) for part in loc[2:]) if len(loc) > 2 else ""

    prefix = "Input"
    if isinstance(section, str) and record_index is not None:
        record_context = lookup_record_context(context, section, record_index)
        prefix = describe_record(section, record_index, record_context)
    elif loc:
        prefix = f"Field '{'.'.join(str(part) for part in loc)}'"

    rendered = f"- {prefix}"
    if field_path:
        rendered += f": field '{field_path}'"
    rendered += f" - {message}"
    return rendered


def path_from_loc(loc: tuple[Any, ...]) -> str:
    """Render a Pydantic location tuple into a dotted/indexed path."""
    parts = []
    for part in loc:
        if isinstance(part, int):
            if not parts:
                parts.append(f"[{part}]")
            else:
                parts[-1] = f"{parts[-1]}[{part}]"
        else:
            parts.append(str(part))
    return ".".join(parts)


def validation_record_label(
    section: Optional[str],
    record_index: Optional[int],
    context: ValidationContext,
) -> Optional[str]:
    """Return a concise human label for one stream or utility record."""
    if section is None or record_index is None:
        return None
    record = lookup_record_context(context, section, record_index)
    label = "Stream" if section == "streams" else "Utility"
    name = record.get("name")
    if name in (None, ""):
        return f"{label} {record_index + 1}"
    return f"{label} {record_index + 1} '{name}'"


def lookup_record_context(
    context: ValidationContext,
    section: str,
    record_index: int,
) -> dict[str, Any]:
    """Return record metadata for one section/index pair when available."""
    records = context.get(section, [])
    if 0 <= record_index < len(records):
        return records[record_index]
    return {"index": record_index, "section": section}


def describe_record(
    section: str,
    record_index: int,
    record_context: dict[str, Any],
) -> str:
    """Describe one record using name and source-location metadata."""
    label = "Stream" if section == "streams" else "Utility"
    name = record_context.get("name")
    row = record_context.get("row")
    sheet = record_context.get("sheet")
    entry = record_context.get("entry")

    description = f"{label} {record_index + 1}"
    if name not in (None, ""):
        description += f" '{name}'"
    if sheet and row:
        description += f" ({sheet} row {row})"
    elif entry:
        description += f" (entry {entry})"
    return description


def _build_record_context(
    section: str,
    index: int,
    record: Any,
    *,
    source_kind: str,
) -> dict[str, Any]:
    details: dict[str, Any] = {"index": index, "section": section}
    if isinstance(record, dict):
        details["name"] = record.get("name")
        details["zone"] = record.get("zone")

    if source_kind in {"excel", "csv"}:
        details["sheet"] = "Stream Data" if section == "streams" else "Utility Data"
        details["row"] = index + 3
    elif source_kind in {"json", "in_memory"}:
        details["entry"] = index + 1
    return details


def _format_validation_issue(
    issue: ValidationIssue,
    *,
    include_warning_prefix: bool = False,
) -> str:
    record_label = issue.record_label or issue.path or "Input"
    field_prefix = ""
    field_path = issue.path or issue.field
    if field_path:
        field_prefix = f"field '{field_path}' - "
    warning_prefix = "Warning: " if include_warning_prefix else ""
    return f"- {record_label}: {warning_prefix}{field_prefix}{issue.message}"
