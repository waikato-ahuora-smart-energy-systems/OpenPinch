"""Shared validation helpers for problem and workspace payloads."""

from __future__ import annotations

import math
import warnings
from typing import Any, Optional

import numpy as np
from pydantic import ValidationError

from ...classes.value import Value
from ...lib.enums import ST
from ...lib.schemas.io import TargetInput
from ...lib.schemas.workspace import ValidationIssue, ValidationReport
from ...utils.miscellaneous import get_value

ValidationContext = dict[str, list[dict[str, Any]]]
_STATE_WEIGHT_RTOL = 1e-12
_STATE_WEIGHT_ATOL = 1e-12
_TEMPERATURE_EQUAL_TOL = 1e-12


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
        stream_values, stream_value_issues = _coerce_validation_values(
            stream,
            section="streams",
            record_index=index,
            record_label=label,
            field_names=("t_supply", "t_target", "heat_flow", "dt_cont", "htc"),
            optional_field_names=("dt_cont", "htc"),
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

    for index, utility in enumerate(payload.utilities):
        label = validation_record_label("utilities", index, context)
        utility_type = str(utility.type)
        utility_values, utility_value_issues = _coerce_validation_values(
            utility,
            section="utilities",
            record_index=index,
            record_label=label,
            field_names=(
                "t_supply",
                "t_target",
                "heat_flow",
                "dt_cont",
                "htc",
                "price",
            ),
            optional_field_names=(
                "t_target",
                "heat_flow",
                "dt_cont",
                "htc",
                "price",
            ),
        )
        issues.extend(utility_value_issues)
        issues.extend(
            _validate_utility_record_states(
                utility_values,
                utility_type=utility_type,
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


def _maybe_get_value(value: Any) -> Any:
    if value is None:
        return None
    return get_value(value)


def _coerce_validation_values(
    record: Any,
    *,
    section: str,
    record_index: int,
    record_label: Optional[str],
    field_names: tuple[str, ...],
    optional_field_names: tuple[str, ...] = (),
) -> tuple[dict[str, Value | None], list[ValidationIssue]]:
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
            values[field_name] = Value(raw_value)
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
    issues.extend(
        _validate_value_finiteness(
            values.get("t_supply"),
            section=section,
            record_index=record_index,
            record_label=record_label,
            field_name="t_supply",
        )
    )
    issues.extend(
        _validate_value_finiteness(
            values.get("t_target"),
            section=section,
            record_index=record_index,
            record_label=record_label,
            field_name="t_target",
        )
    )
    issues.extend(
        _validate_value_finiteness(
            values.get("heat_flow"),
            section=section,
            record_index=record_index,
            record_label=record_label,
            field_name="heat_flow",
        )
    )
    issues.extend(
        _validate_value_finiteness(
            values.get("dt_cont"),
            section=section,
            record_index=record_index,
            record_label=record_label,
            field_name="dt_cont",
        )
    )
    issues.extend(
        _validate_value_finiteness(
            values.get("htc"),
            section=section,
            record_index=record_index,
            record_label=record_label,
            field_name="htc",
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

    state_ids, alignment_issues = _validate_state_alignment(
        values,
        section=section,
        record_index=record_index,
        record_label=record_label,
    )
    issues.extend(alignment_issues)

    t_supply = values.get("t_supply")
    t_target = values.get("t_target")
    if t_supply is None or t_target is None or alignment_issues:
        return issues

    state_labels = list(state_ids) if state_ids is not None else [None]
    heat_flow = values.get("heat_flow")
    classifications: dict[str, list[str]] = {
        ST.Hot.value: [],
        ST.Cold.value: [],
        ST.Both.value: [],
    }

    for state_id in state_labels:
        t_supply_value = _get_state_magnitude(
            t_supply, state_id=state_id, state_ids=state_ids
        )
        t_target_value = _get_state_magnitude(
            t_target, state_id=state_id, state_ids=state_ids
        )
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
                    message=_with_state_suffix(
                        "Supply and target temperatures must differ.",
                        state_id,
                    ),
                )
            )
            continue

        if t_supply_value > t_target_value:
            classifications[ST.Hot.value].append(_render_state_label(state_id))
        elif t_supply_value < t_target_value:
            classifications[ST.Cold.value].append(_render_state_label(state_id))
        else:
            heat_flow_value = _get_state_magnitude(
                heat_flow,
                state_id=state_id,
                state_ids=state_ids,
            )
            if heat_flow_value is None or math.isclose(
                heat_flow_value,
                0.0,
                rel_tol=0.0,
                abs_tol=_TEMPERATURE_EQUAL_TOL,
            ):
                classifications[ST.Both.value].append(_render_state_label(state_id))
            elif heat_flow_value < 0.0:
                classifications[ST.Hot.value].append(_render_state_label(state_id))
            else:
                classifications[ST.Cold.value].append(_render_state_label(state_id))

    active_classes = {
        stream_type: state_group
        for stream_type, state_group in classifications.items()
        if state_group
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
                    f"Neutral={classifications[ST.Both.value]}."
                ),
            )
        )

    return issues


def _validate_utility_record_states(
    values: dict[str, Value | None],
    *,
    utility_type: str,
    section: str,
    record_index: int,
    record_label: Optional[str],
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for field_name in ("t_supply", "t_target", "heat_flow", "dt_cont", "htc", "price"):
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

    state_ids, alignment_issues = _validate_state_alignment(
        values,
        section=section,
        record_index=record_index,
        record_label=record_label,
    )
    issues.extend(alignment_issues)

    t_supply = values.get("t_supply")
    t_target = values.get("t_target")
    if t_supply is None or t_target is None or alignment_issues:
        return issues

    for state_id in list(state_ids) if state_ids is not None else [None]:
        t_supply_value = _get_state_magnitude(
            t_supply, state_id=state_id, state_ids=state_ids
        )
        t_target_value = _get_state_magnitude(
            t_target, state_id=state_id, state_ids=state_ids
        )
        if t_supply_value is None or t_target_value is None:
            continue
        if not (math.isfinite(t_supply_value) and math.isfinite(t_target_value)):
            continue

        if utility_type == ST.Hot.value and t_supply_value < t_target_value:
            issues.append(
                _build_issue(
                    severity="warning",
                    section=section,
                    record_index=record_index,
                    record_label=record_label,
                    path_field="t_supply",
                    field="t_supply/t_target",
                    message=_with_state_suffix(
                        "Hot utilities should have t_supply >= t_target.",
                        state_id,
                    ),
                )
            )
        if utility_type == ST.Cold.value and t_supply_value > t_target_value:
            issues.append(
                _build_issue(
                    severity="warning",
                    section=section,
                    record_index=record_index,
                    record_label=record_label,
                    path_field="t_supply",
                    field="t_supply/t_target",
                    message=_with_state_suffix(
                        "Cold utilities should have t_supply <= t_target.",
                        state_id,
                    ),
                )
            )

    return issues


def _validate_state_alignment(
    values: dict[str, Value | None],
    *,
    section: str,
    record_index: int,
    record_label: Optional[str],
) -> tuple[list[str] | None, list[ValidationIssue]]:
    stateful_values = [
        (field_name, value)
        for field_name, value in values.items()
        if value is not None and value.state_ids is not None
    ]
    if not stateful_values:
        return None, []

    ref_name, ref_value = stateful_values[0]
    ref_state_ids = ref_value.state_ids
    ref_weights = ref_value.weights
    issues: list[ValidationIssue] = []

    for field_name, value in stateful_values[1:]:
        if value.state_ids != ref_state_ids:
            issues.append(
                _build_issue(
                    severity="error",
                    section=section,
                    record_index=record_index,
                    record_label=record_label,
                    path_field=field_name,
                    field=field_name,
                    message=f"state_ids for {field_name} must align with {ref_name}.",
                )
            )
            continue
        if not np.allclose(
            value.weights,
            ref_weights,
            rtol=_STATE_WEIGHT_RTOL,
            atol=_STATE_WEIGHT_ATOL,
        ):
            issues.append(
                _build_issue(
                    severity="error",
                    section=section,
                    record_index=record_index,
                    record_label=record_label,
                    path_field=field_name,
                    field=field_name,
                    message=f"weights for {field_name} must align with {ref_name}.",
                )
            )

    if issues:
        return None, issues
    return ref_state_ids, []


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

    for state_id, magnitude in _iter_state_magnitudes(value):
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
                message=_with_state_suffix("Value must be finite.", state_id),
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

    for state_id, magnitude in _iter_state_magnitudes(value):
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
                message=_with_state_suffix(message, state_id),
            )
        )

    return issues


def _validate_positive_states(
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

    for state_id, magnitude in _iter_state_magnitudes(value):
        if magnitude is None or not math.isfinite(magnitude) or magnitude > 0.0:
            continue
        issues.append(
            _build_issue(
                severity=severity,
                section=section,
                record_index=record_index,
                record_label=record_label,
                path_field=field_name,
                field=field_name,
                message=_with_state_suffix(message, state_id),
            )
        )

    return issues


def _iter_state_magnitudes(value: Value) -> list[tuple[str | None, float]]:
    if value.state_ids is None:
        return [(None, float(value.value))]
    return [(state_id, float(value[state_id].value)) for state_id in value.state_ids]


def _get_state_magnitude(
    value: Value | None,
    *,
    state_id: str | None,
    state_ids: list[str] | None,
) -> float | None:
    if value is None:
        return None
    if value.state_ids is None:
        return float(value.value)
    if state_id is None and state_ids is None:
        state_id = value.state_ids[0]
    return float(value[state_id].value)


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


def _state_suffix(state_id: str | None) -> str:
    if state_id is None:
        return ""
    return f" for state_id '{state_id}'"


def _render_state_label(state_id: str | None) -> str:
    return "default" if state_id is None else str(state_id)


def _with_state_suffix(message: str, state_id: str | None) -> str:
    if state_id is None:
        return message
    trimmed = message[:-1] if message.endswith(".") else message
    return trimmed + _state_suffix(state_id) + "."
