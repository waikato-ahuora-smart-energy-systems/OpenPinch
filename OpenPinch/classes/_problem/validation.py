"""Shared validation helpers for problem and workspace payloads."""

from __future__ import annotations

import warnings
from typing import Any, Optional

from pydantic import ValidationError

from ...lib.enums import ST
from ...lib.schemas.io import TargetInput
from ...lib.schemas.workspace import ValidationIssue, ValidationReport
from ...utils.miscellaneous import get_value

ValidationContext = dict[str, list[dict[str, Any]]]


def validate_problem_payload(
    problem_data: Any,
    *,
    context: Optional[ValidationContext] = None,
) -> TargetInput:
    """Validate one problem payload and raise user-facing errors on failure."""
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

    _validate_problem_semantics(validated, context=effective_context)
    return validated


def build_validation_report(
    payload: Any,
    *,
    context: Optional[ValidationContext] = None,
    source_kind: str = "target_input",
) -> ValidationReport:
    """Build a structured validation report without raising on invalid payloads."""
    effective_context = context or build_validation_context(
        payload,
        source_kind=source_kind,
    )
    issues: list[ValidationIssue] = []

    try:
        validated = TargetInput.model_validate(payload)
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


def semantic_issues(
    payload: TargetInput,
    *,
    context: ValidationContext,
) -> list[ValidationIssue]:
    """Return semantic validation issues for one validated problem payload."""
    issues: list[ValidationIssue] = []

    if len(payload.streams) == 0:
        issues.append(
            ValidationIssue(
                severity="error",
                path="streams",
                message="At least one stream must be provided.",
            )
        )

    for index, stream in enumerate(payload.streams):
        label = validation_record_label("streams", index, context)
        t_supply = _maybe_get_value(stream.t_supply)
        t_target = _maybe_get_value(stream.t_target)
        if t_supply == t_target:
            issues.append(
                ValidationIssue(
                    severity="error",
                    path=f"streams[{index}].t_supply",
                    field="t_supply/t_target",
                    message="Supply and target temperatures must differ.",
                    section="streams",
                    record_index=index,
                    record_label=label,
                )
            )

        heat_flow = _maybe_get_value(getattr(stream, "heat_flow"))
        if heat_flow is not None and heat_flow < 0:
            issues.append(
                ValidationIssue(
                    severity="error",
                    path=f"streams[{index}].heat_flow",
                    field="heat_flow",
                    message="Value must be non-negative.",
                    section="streams",
                    record_index=index,
                    record_label=label,
                )
            )

        dt_cont = _maybe_get_value(getattr(stream, "dt_cont"))
        if dt_cont is not None and dt_cont < 0:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    path=f"streams[{index}].dt_cont",
                    field="dt_cont",
                    message="Value should be non-negative.",
                    section="streams",
                    record_index=index,
                    record_label=label,
                )
            )

        htc = _maybe_get_value(getattr(stream, "htc"))
        if htc is not None and htc <= 0:
            issues.append(
                ValidationIssue(
                    severity="error",
                    path=f"streams[{index}].htc",
                    field="htc",
                    message="Value must be positive.",
                    section="streams",
                    record_index=index,
                    record_label=label,
                )
            )

    for index, utility in enumerate(payload.utilities):
        label = validation_record_label("utilities", index, context)
        utility_type = str(utility.type)
        t_supply = _maybe_get_value(utility.t_supply)
        t_target = _maybe_get_value(utility.t_target)
        if (
            utility_type == ST.Hot.value
            and t_supply is not None
            and t_target is not None
            and t_supply < t_target
        ):
            issues.append(
                ValidationIssue(
                    severity="warning",
                    path=f"utilities[{index}].t_supply",
                    field="t_supply/t_target",
                    message="Hot utilities should have t_supply >= t_target.",
                    section="utilities",
                    record_index=index,
                    record_label=label,
                )
            )
        if (
            utility_type == ST.Cold.value
            and t_supply is not None
            and t_target is not None
            and t_supply > t_target
        ):
            issues.append(
                ValidationIssue(
                    severity="warning",
                    path=f"utilities[{index}].t_supply",
                    field="t_supply/t_target",
                    message="Cold utilities should have t_supply <= t_target.",
                    section="utilities",
                    record_index=index,
                    record_label=label,
                )
            )

        for field_name in ("dt_cont", "price", "heat_flow"):
            value = _maybe_get_value(getattr(utility, field_name))
            if value is not None and value < 0:
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        path=f"utilities[{index}].{field_name}",
                        field=field_name,
                        message="Value should be non-negative.",
                        section="utilities",
                        record_index=index,
                        record_label=label,
                    )
                )

        htc = _maybe_get_value(getattr(utility, "htc"))
        if htc is not None and htc <= 0:
            issues.append(
                ValidationIssue(
                    severity="error",
                    path=f"utilities[{index}].htc",
                    field="htc",
                    message="Value must be positive.",
                    section="utilities",
                    record_index=index,
                    record_label=label,
                )
            )

    return issues


def _validate_problem_semantics(
    payload: TargetInput,
    *,
    context: ValidationContext,
) -> None:
    """Raise or warn using the shared semantic validation issue list."""
    issues = semantic_issues(payload, context=context)
    fatal_issues = [
        _format_validation_issue(issue)
        for issue in issues
        if issue.severity == "error"
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
    if issue.field:
        field_prefix = f"field '{issue.field}' - "
    warning_prefix = "Warning: " if include_warning_prefix else ""
    return f"- {record_label}: {warning_prefix}{field_prefix}{issue.message}"


def _maybe_get_value(value: Any) -> Any:
    if value is None:
        return None
    return get_value(value)
