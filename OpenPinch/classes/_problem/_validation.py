"""Shared validation helpers for problem and workspace input data."""

from __future__ import annotations

import math
import warnings
from typing import Any, Optional

import numpy as np
from pydantic import ValidationError

from ...lib.config_metadata import CONFIG_FIELD_SPECS, input_unit_options_to_map
from ...lib.enums import ST
from ...lib.schemas.io import TargetInput
from ...lib.schemas.workspace import ValidationIssue, ValidationReport
from ...lib.unit_system import standardise_input_value
from ..value import Value

ValidationContext = dict[str, list[dict[str, Any]]]
_TEMPERATURE_EQUAL_TOL = 1e-12
_STREAM_VALUE_FIELDS = (
    "t_supply",
    "t_target",
    "p_supply",
    "p_target",
    "h_supply",
    "h_target",
    "heat_flow",
    "dt_cont",
    "htc",
)
_STREAM_OPTIONAL_VALUE_FIELDS = (
    "p_supply",
    "p_target",
    "h_supply",
    "h_target",
    "dt_cont",
    "htc",
)
_UTILITY_VALUE_FIELDS = (
    "t_supply",
    "t_target",
    "p_supply",
    "p_target",
    "h_supply",
    "h_target",
    "heat_flow",
    "dt_cont",
    "htc",
    "price",
)
_UTILITY_OPTIONAL_VALUE_FIELDS = (
    "t_target",
    "p_supply",
    "p_target",
    "h_supply",
    "h_target",
    "heat_flow",
    "dt_cont",
    "htc",
    "price",
)


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

    _validate_problem_semantics(validated, context=effective_context)
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


def semantic_issues(
    problem_inputs: TargetInput,
    *,
    context: ValidationContext,
) -> list[ValidationIssue]:
    """Return semantic validation issues for one validated problem definition."""
    issues: list[ValidationIssue] = []
    options = problem_inputs.options if isinstance(problem_inputs.options, dict) else {}
    input_unit_config = input_unit_options_to_map(
        {name: spec.default for name, spec in CONFIG_FIELD_SPECS.items()} | options
    )

    if len(problem_inputs.streams) == 0:
        issues.append(
            ValidationIssue(
                severity="error",
                path="streams",
                message="At least one stream must be provided.",
            )
        )

    for index, stream in enumerate(problem_inputs.streams):
        label = validation_record_label("streams", index, context)
        stream_values, stream_value_issues = _coerce_validation_values(
            stream,
            section="streams",
            record_index=index,
            record_label=label,
            field_names=_STREAM_VALUE_FIELDS,
            optional_field_names=_STREAM_OPTIONAL_VALUE_FIELDS,
            config=input_unit_config,
        )
        issues.extend(stream_value_issues)
        issues.extend(
            _validate_stream_record_states(
                stream_values,
                section="streams",
                record_index=index,
                record_label=label,
            )
        )

    for index, utility in enumerate(problem_inputs.utilities):
        label = validation_record_label("utilities", index, context)
        utility_values, utility_value_issues = _coerce_validation_values(
            utility,
            section="utilities",
            record_index=index,
            record_label=label,
            field_names=_UTILITY_VALUE_FIELDS,
            optional_field_names=_UTILITY_OPTIONAL_VALUE_FIELDS,
            config=input_unit_config,
        )
        issues.extend(utility_value_issues)
        issues.extend(
            _validate_utility_record_states(
                utility_values,
                section="utilities",
                record_index=index,
                record_label=label,
            )
        )

    return issues


def _validate_problem_semantics(
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
    if issue.field:
        field_prefix = f"field '{issue.field}' - "
    warning_prefix = "Warning: " if include_warning_prefix else ""
    return f"- {record_label}: {warning_prefix}{field_prefix}{issue.message}"


def _coerce_validation_values(
    record: Any,
    *,
    section: str,
    record_index: int,
    record_label: Optional[str],
    field_names: tuple[str, ...],
    optional_field_names: tuple[str, ...] = (),
    config: dict[str, dict[str, Any]] | None = None,
) -> tuple[
    dict[str, Value | None],
    dict[str, tuple[list[str] | None, np.ndarray | None]],
    list[ValidationIssue],
]:
    values: dict[str, Value | None] = {}
    issues: list[ValidationIssue] = []

    for field_name in field_names:
        raw_value = getattr(record, field_name, None)
        if raw_value is None:
            values[field_name] = None
            continue
        if field_name in optional_field_names and _raw_value_is_missing(raw_value):
            values[field_name] = None
            continue
        try:
            values[field_name] = standardise_input_value(
                raw_value,
                field_name=field_name,
                config=config,
            )
        except (TypeError, ValueError) as exc:
            values[field_name] = None
            issues.append(
                _build_issue(
                    severity="error",
                    section=section,
                    record_index=record_index,
                    record_label=record_label,
                    path_field=field_name,
                    field=field_name,
                    message=str(exc),
                )
            )

    return values, issues


def _raw_value_is_missing(raw_value: Any) -> bool:
    if raw_value is None:
        return True
    if hasattr(raw_value, "model_dump"):
        raw_value = raw_value.model_dump(mode="python")
    if not isinstance(raw_value, dict):
        return False
    if "value" in raw_value and "values" not in raw_value:
        return raw_value.get("value") is None
    if "values" in raw_value:
        magnitudes = raw_value.get("values")
        if magnitudes is None:
            return True
        return all(value is None for value in magnitudes)
    return False


def _validate_stream_record_states(
    values: dict[str, Value | None],
    *,
    section: str,
    record_index: int,
    record_label: Optional[str],
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for field_name in _STREAM_VALUE_FIELDS:
        issues.extend(
            _validate_value_finiteness(
                values.get(field_name),
                section=section,
                record_index=record_index,
                record_label=record_label,
                field_name=field_name,
            )
        )
    issues.extend(
        _validate_non_negative_states(
            values.get("heat_flow"),
            section=section,
            record_index=record_index,
            record_label=record_label,
            field_name="heat_flow",
            severity="error",
            message="Value must be non-negative.",
        )
    )
    issues.extend(
        _validate_non_negative_states(
            values.get("dt_cont"),
            section=section,
            record_index=record_index,
            record_label=record_label,
            field_name="dt_cont",
            severity="warning",
            message="Value should be non-negative.",
        )
    )
    issues.extend(
        _validate_non_negative_states(
            values.get("htc"),
            section=section,
            record_index=record_index,
            record_label=record_label,
            field_name="htc",
            severity="error",
            message="Value must be positive.",
        )
    )

    t_supply = values.get("t_supply")
    t_target = values.get("t_target")
    if t_supply is None or t_target is None:
        return issues

    classifications: dict[str, list[str]] = {
        ST.Hot.value: [],
        ST.Cold.value: [],
        ST.Neutral.value: [],
    }

    num_periods = 1
    for value in values.values():
        if isinstance(value, Value):
            num_periods = max(num_periods, len(value))

    for idx in range(num_periods):
        t_supply_value = t_supply[idx]
        t_target_value = t_target[idx]
        if t_supply_value is None or t_target_value is None:
            continue
        if not (math.isfinite(t_supply_value) and math.isfinite(t_target_value)):
            continue

        if math.isclose(
            t_supply_value,
            t_target_value,
            rel_tol=0.0,
            abs_tol=_TEMPERATURE_EQUAL_TOL,
        ):
            issues.append(
                _build_issue(
                    severity="error",
                    section=section,
                    record_index=record_index,
                    record_label=record_label,
                    path_field="t_supply",
                    field="t_supply/t_target",
                    message=_with_period_suffix(
                        "Supply and target temperatures must differ.",
                        idx,
                    ),
                )
            )
        elif t_supply_value > t_target_value:
            classifications[ST.Hot.value].append(str(idx))
        elif t_supply_value < t_target_value:
            classifications[ST.Cold.value].append(str(idx))

    active_classes = {
        stream_type: period_group
        for stream_type, period_group in classifications.items()
        if period_group
    }
    if len(active_classes) > 1:
        issues.append(
            _build_issue(
                severity="error",
                section=section,
                record_index=record_index,
                record_label=record_label,
                path_field="t_supply",
                field="t_supply/t_target",
                message=(
                    "Stream states must classify consistently. "
                    f"Hot={classifications[ST.Hot.value]}, "
                    f"Cold={classifications[ST.Cold.value]}, "
                    f"Neutral={classifications[ST.Neutral.value]}."
                ),
            )
        )

    return issues


def _validate_utility_record_states(
    values: dict[str, Value | None],
    *,
    section: str,
    record_index: int,
    record_label: Optional[str],
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for field_name in _UTILITY_VALUE_FIELDS:
        issues.extend(
            _validate_value_finiteness(
                values.get(field_name),
                section=section,
                record_index=record_index,
                record_label=record_label,
                field_name=field_name,
            )
        )

    for field_name in ("dt_cont", "price", "heat_flow"):
        issues.extend(
            _validate_non_negative_states(
                values.get(field_name),
                section=section,
                record_index=record_index,
                record_label=record_label,
                field_name=field_name,
                severity="warning",
                message="Value should be non-negative.",
            )
        )

    issues.extend(
        _validate_non_negative_states(
            values.get("htc"),
            section=section,
            record_index=record_index,
            record_label=record_label,
            field_name="htc",
            severity="error",
            message="Value must be positive.",
        )
    )

    t_supply = values.get("t_supply")
    t_target = values.get("t_target")
    if t_supply is None or t_target is None:
        return issues

    return issues


def _validate_value_finiteness(
    value: Value | None,
    *,
    section: str,
    record_index: int,
    record_label: Optional[str],
    field_name: str,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if value is None:
        return issues

    for idx in range(len(value.period_values) - 1):
        magnitude = value[idx]
        if magnitude is None:
            continue
        if math.isfinite(magnitude):
            continue
        issues.append(
            _build_issue(
                severity="error",
                section=section,
                record_index=record_index,
                record_label=record_label,
                path_field=field_name,
                field=field_name,
                message=_with_period_suffix("Value must be finite.", idx),
            )
        )
    return issues


def _validate_non_negative_states(
    value: Value | None,
    *,
    section: str,
    record_index: int,
    record_label: Optional[str],
    field_name: str,
    severity: str,
    message: str,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if value is None:
        return issues

    for idx in range(len(value.period_values) - 1):
        magnitude = value[idx]
        if magnitude is None or not math.isfinite(magnitude) or magnitude >= 0.0:
            continue
        issues.append(
            _build_issue(
                severity=severity,
                section=section,
                record_index=record_index,
                record_label=record_label,
                path_field=field_name,
                field=field_name,
                message=_with_period_suffix(message, idx),
            )
        )

    return issues


def _build_issue(
    *,
    severity: str,
    section: str,
    record_index: int,
    record_label: Optional[str],
    path_field: str,
    field: str,
    message: str,
) -> ValidationIssue:
    return ValidationIssue(
        severity=severity,
        path=f"{section}[{record_index}].{path_field}",
        message=message,
        section=section,
        record_index=record_index,
        field=field,
        record_label=record_label,
    )


def _period_suffix(period_id: str | None) -> str:
    if period_id is None:
        return ""
    return f" for period_id '{period_id}'"


def _with_period_suffix(message: str, idx: int | None) -> str:
    if idx is None:
        return message
    trimmed = message[:-1] if message.endswith(".") else message
    return trimmed + _period_suffix(idx) + "."
