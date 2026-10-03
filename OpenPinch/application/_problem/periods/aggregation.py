"""Helpers for multi-period target summary aggregation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ....contracts.output import TargetOutput
from ....contracts.report_metrics import AggregationPolicy, metric_specifications
from ....contracts.report_units import split_report_value
from ....contracts.reporting import HeatUtility, PinchTemp, TargetResults
from ....domain._stream.value_state import resolve_period_weights
from ....domain.value import Value

if TYPE_CHECKING:
    from ...problem import PinchProblem

WEIGHTED_AVERAGE_PERIOD_ID = "weighted_average"
SUMMARY_PERIOD_MODES = frozenset(
    {"selected", "all", "weighted_average", "all_with_weighted_average"}
)


@dataclass(frozen=True)
class _TargetRunSpec:
    """Immutable targeting intent used for explicit period replay."""

    surface: str
    options: dict[str, object]
    zone_name: str | None = None
    include_subzones: bool = False


def record_target_run(
    problem: "PinchProblem",
    surface: str,
    *,
    options: dict[str, Any] | None = None,
    zone_name: str | None = None,
    include_subzones: bool = False,
) -> None:
    """Remember immutable targeting intent for later period replay."""
    if problem._suspend_target_run_recording:
        return
    problem._last_target_run_spec = _TargetRunSpec(
        surface=surface,
        options=deepcopy(dict(options or {})),
        zone_name=zone_name,
        include_subzones=bool(include_subzones),
    )


def target_run_spec_for_summary(problem: "PinchProblem") -> _TargetRunSpec:
    """Return recorded explicit targeting intent for report replay."""
    if problem._last_target_run_spec is None:
        raise RuntimeError(
            "No target analysis is available. Run a problem.target.<method>() "
            "workflow before requesting multi-period summaries."
        )
    return problem._last_target_run_spec


def period_options_for_replay(
    spec: _TargetRunSpec,
    *,
    period_id: str,
) -> dict[str, Any]:
    """Overlay one canonical period identity onto replay options."""
    runtime_options = deepcopy(dict(spec.options or {}))
    runtime_options.pop("period_id", None)
    runtime_options.pop("period_idx", None)
    runtime_options["period_id"] = period_id
    return runtime_options


def target_outputs_for_recorded_periods(
    problem: "PinchProblem",
) -> list[TargetOutput]:
    """Replay targeting against isolated zone copies and restore parent state."""
    period_ids = list(problem.period_ids)
    if not period_ids:
        raise ValueError("This problem has no canonical period_ids to target.")

    spec = target_run_spec_for_summary(problem)
    original_master_zone = problem._require_prepared_root_zone()
    previous_results = problem._results
    previous_recording_state = problem._suspend_target_run_recording
    previous_spec = problem._last_target_run_spec
    outputs: list[TargetOutput] = []
    try:
        problem._suspend_target_run_recording = True
        baseline_zone = deepcopy(original_master_zone)
        for period_id in period_ids:
            problem._master_zone = deepcopy(baseline_zone)
            outputs.append(problem._target_output_for_recorded_period(spec, period_id))
    finally:
        problem._master_zone = original_master_zone
        problem._suspend_target_run_recording = previous_recording_state
        problem._last_target_run_spec = previous_spec
        problem._results = previous_results
    return outputs


def target_output_for_recorded_period(
    problem: "PinchProblem",
    spec: _TargetRunSpec,
    period_id: str,
) -> TargetOutput:
    """Replay one recorded targeting surface for a selected period."""
    runtime_options = period_options_for_replay(spec, period_id=period_id)
    target_method = getattr(problem.target, spec.surface)
    target_method(
        zone=spec.zone_name,
        options=runtime_options,
        include_subzones=spec.include_subzones,
    )
    if problem._results is None:
        raise RuntimeError(
            f"Target accessor {spec.surface!r} did not produce report results."
        )
    return TargetOutput.model_validate(problem._results.model_dump(mode="python"))


def period_weights_for_summary(problem: "PinchProblem") -> list[float]:
    """Return period weights in canonical identity order."""
    master_zone = problem._require_prepared_root_zone()
    period_ids = list(master_zone.period_ids)
    resolved = resolve_period_weights(
        master_zone.period_ids,
        getattr(master_zone, "weights", None),
    )
    return [
        float(resolved[master_zone.period_ids[period_id]]) for period_id in period_ids
    ]


def summary_results(
    problem: "PinchProblem",
    *,
    periods: str,
) -> TargetOutput:
    """Resolve existing selected or explicitly generated all-period results."""
    if periods not in SUMMARY_PERIOD_MODES:
        raise ValueError(
            "periods must be one of: " + ", ".join(sorted(SUMMARY_PERIOD_MODES)) + "."
        )
    if periods == "selected":
        if problem._results is None:
            raise RuntimeError(
                "No target analysis is available. Run a problem.target.<method>() "
                "workflow before requesting results."
            )
        # A copy, so editing a returned report never changes the cached one.
        return _deep_copy(problem._results)

    outputs = list(problem._period_results.values())
    if not outputs:
        raise RuntimeError(
            "No all-period analysis is available. Run the matching "
            "problem.target.all_periods.<method>() workflow first."
        )
    weights = period_weights_for_summary(problem)
    return output_for_period_mode(outputs, weights, periods=periods)


def _deep_copy(value: Any) -> Any:
    """Return a deep copy of a pydantic model or any other object."""
    copier = getattr(value, "model_copy", None)
    if callable(copier):
        return copier(deep=True)
    return deepcopy(value)


def combine_period_outputs(outputs: Sequence[TargetOutput]) -> TargetOutput:
    """Return one output with period-specific target rows concatenated."""
    ordered_outputs = list(outputs)
    if not ordered_outputs:
        raise ValueError("At least one period output is required.")
    targets = [
        _deep_copy(target)
        for output in ordered_outputs
        for target in list(getattr(output, "targets", []) or [])
    ]
    return TargetOutput(name=ordered_outputs[0].name, period_id=None, targets=targets)


def weighted_average_output(
    outputs: Sequence[TargetOutput],
    weights: Sequence[float],
) -> TargetOutput:
    """Return weighted-average target rows for aligned period outputs."""
    ordered_outputs = list(outputs)
    resolved_weights = _normalise_weights(weights, expected_len=len(ordered_outputs))
    grouped_targets = _aligned_target_groups(ordered_outputs)
    targets = [
        _weighted_average_target(targets, resolved_weights)
        for targets in grouped_targets
    ]
    return TargetOutput(
        name=ordered_outputs[0].name,
        period_id=WEIGHTED_AVERAGE_PERIOD_ID,
        targets=targets,
    )


def output_for_period_mode(
    outputs: Sequence[TargetOutput],
    weights: Sequence[float],
    *,
    periods: str,
) -> TargetOutput:
    """Build a report output for one multi-period summary mode."""
    if periods == "all":
        return combine_period_outputs(outputs)
    weighted = weighted_average_output(outputs, weights)
    if periods == "weighted_average":
        return weighted
    if periods == "all_with_weighted_average":
        combined = combine_period_outputs(outputs)
        return TargetOutput(
            name=combined.name,
            period_id=None,
            targets=[*combined.targets, *weighted.targets],
        )
    raise ValueError(
        "periods must be one of: " + ", ".join(sorted(SUMMARY_PERIOD_MODES))
    )


def _normalise_weights(weights: Sequence[float], *, expected_len: int) -> np.ndarray:
    return resolve_period_weights(
        [str(period_idx) for period_idx in range(expected_len)],
        weights,
    )


_HPR_TARGET_METHODS = frozenset({"Heat Pump", "Refrigeration"})
_HPR_ZERO_NONE_FIELDS = (
    "hpr_cop",
    "hpr_eta_he",
    "hpr_success",
    "hpr_target_simulation_record",
    "hpr_hot_streams",
    "hpr_cold_streams",
)


def _fill_zero_load_hpr_rows(
    outputs: Sequence[TargetOutput],
) -> list[TargetOutput]:
    """Add a no-heat-pump row for each period whose HPR load was zero.

    A period whose HPR load is below the tolerance produces no HPR row, which
    would break every weighted summary. Its row is the period's own base
    (heat-exchange) row relabelled as the HPR row, with zero HPR duty, work
    and cost, so the period counts as running no heat pump.
    """
    templates: dict[tuple, TargetResults] = {}
    for output in outputs:
        for target in output.targets:
            if target.target_method in _HPR_TARGET_METHODS:
                templates.setdefault(_target_key(target), target)
    if not templates:
        return list(outputs)

    filled = []
    for output in outputs:
        present = {_target_key(target) for target in output.targets}
        additions = []
        for key, template in templates.items():
            if key in present:
                continue
            base = next(
                (
                    target
                    for target in output.targets
                    if target.scope == template.scope
                    and target.zone_type == template.zone_type
                    and target.integration_type == template.integration_type
                    and target.target_method not in _HPR_TARGET_METHODS
                ),
                None,
            )
            if base is not None:
                additions.append(_zero_load_hpr_row(base, template))
        if additions:
            output = output.model_copy(
                update={"targets": [*output.targets, *additions]}
            )
        filled.append(output)
    return filled


def _zero_load_hpr_row(base: TargetResults, template: TargetResults) -> TargetResults:
    data = base.model_dump(mode="python")
    for field in (
        "integration_type",
        "target_method",
        "row_type",
        "hpr_cycle",
        "hpr_simulation_backend",
        "hpr_shared_design",
    ):
        data[field] = getattr(template, field, None)
    metadata = {"hpr_cycle", "hpr_simulation_backend", "hpr_shared_design"}
    for field in type(template).model_fields:
        if not field.startswith("hpr_") or field in metadata:
            continue
        value = getattr(template, field, None)
        if field in _HPR_ZERO_NONE_FIELDS:
            data[field] = None
        elif isinstance(value, Value):
            data[field] = Value({"value": 0.0, "unit": value.to_dict()["unit"]})
        elif isinstance(value, tuple):
            data[field] = tuple(0.0 for _ in value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            data[field] = 0.0
        else:
            data[field] = None
    return type(template).model_validate(data)


def _aligned_target_groups(
    outputs: Sequence[TargetOutput],
) -> list[list[TargetResults]]:
    if not outputs:
        raise ValueError("At least one period output is required.")
    outputs = _fill_zero_load_hpr_rows(outputs)

    first_targets = list(outputs[0].targets)
    first_keys = [_target_key(target) for target in first_targets]
    if len(set(first_keys)) != len(first_keys):
        raise ValueError("Cannot aggregate duplicate target metadata and row types.")

    groups: list[list[TargetResults]] = [[target] for target in first_targets]
    expected_key_set = set(first_keys)
    for output in outputs[1:]:
        target_by_key = {_target_key(target): target for target in output.targets}
        if len(target_by_key) != len(output.targets):
            raise ValueError(
                "Cannot aggregate duplicate target metadata and row types."
            )
        if set(target_by_key) != expected_key_set:
            missing = sorted(expected_key_set - set(target_by_key))
            extra = sorted(set(target_by_key) - expected_key_set)
            details = []
            if missing:
                details.append(f"missing {missing}")
            if extra:
                details.append(f"extra {extra}")
            raise ValueError(
                "Period outputs must contain identical target rows"
                + (": " + "; ".join(details) if details else ".")
            )
        for index, key in enumerate(first_keys):
            groups[index].append(target_by_key[key])
    return groups


def _target_key(target: TargetResults) -> tuple[str, str, str, str, str | None]:
    return (
        target.scope,
        target.zone_type,
        target.integration_type,
        target.target_method,
        getattr(target, "row_type", None),
    )


def _weighted_average_target(
    targets: Sequence[TargetResults],
    weights: np.ndarray,
) -> TargetResults:
    first = targets[0]
    data = first.model_dump(mode="python")
    data["period_id"] = WEIGHTED_AVERAGE_PERIOD_ID
    data["period_idx"] = None
    for field, spec in metric_specifications(type(first)).items():
        if spec.aggregation is AggregationPolicy.WEIGHTED_MEAN:
            aggregate = (
                _weighted_numeric_attr
                if spec.representation == "scalar"
                else _weighted_report_value
            )
            data[field] = aggregate(targets, field, weights)
        elif spec.aggregation is AggregationPolicy.MAXIMUM:
            data[field] = _max_report_value(targets, field)
        elif spec.aggregation is AggregationPolicy.CONSENSUS:
            data[field] = _consensus_value(
                getattr(target, field, None) for target in targets
            )
        elif spec.aggregation is AggregationPolicy.EXCLUDE:
            data[field] = None
        elif field not in {
            "period_id",
            "degree_of_integration",
            "ETE",
            "hpr_cop",
            "hpr_eta_he",
            "hpr_success",
            "hpr_utility_annualized_capital_cost",
            "hpr_machine_capital_costs",
            "hpr_machine_annualized_capital_costs",
            "hpr_total_annualized_cost",
            "pinch_temp",
            "hot_utilities",
            "cold_utilities",
        }:
            raise ValueError(f"No derived aggregation implementation for {field!r}.")
    # Ratios over a year are ratios of totals, not averages of ratios.
    data["degree_of_integration"] = _ratio_of_totals(
        targets, "degree_of_integration", "Qr", weights, basis_is_numerator=True
    )
    data["ETE"] = _ratio_of_totals(
        targets, "ETE", "exergy_sources", weights, basis_is_numerator=False
    )
    data["hpr_cop"] = _ratio_of_totals(
        targets, "hpr_cop", "hpr_work", weights, basis_is_numerator=False
    )
    # eta_he is heat-engine work over its condenser heat; the rows carry
    # neither (hpr_work is the net cycle work), so take the weighted mean.
    data["hpr_eta_he"] = _ratio_of_totals(
        targets, "hpr_eta_he", None, weights, basis_is_numerator=True
    )
    ran = [
        t.hpr_success for t in targets if getattr(t, "hpr_success", None) is not None
    ]
    data["hpr_success"] = all(ran) if ran else None
    data["hpr_utility_annualized_capital_cost"] = _hpr_utility_annualized_capital(
        targets, data
    )
    _apply_peak_machine_capital(targets, data)
    data["hpr_total_annualized_cost"] = _total_hpr_annualized_cost(data)
    data["pinch_temp"] = PinchTemp(
        cold_temp=_weighted_report_value(
            targets,
            "pinch_temp.cold_temp",
            weights,
            allow_partial_missing=True,
        ),
        hot_temp=_weighted_report_value(
            targets,
            "pinch_temp.hot_temp",
            weights,
            allow_partial_missing=True,
        ),
    )
    data["hot_utilities"] = _weighted_utilities(targets, "hot_utilities", weights)
    data["cold_utilities"] = _weighted_utilities(targets, "cold_utilities", weights)
    return type(first).model_validate(data)


def _collect_report_values(
    targets: Sequence[TargetResults],
    attr_path: str,
) -> tuple[list[float], str | None, int]:
    """Return ``(values, unit, missing)`` for the scalar report values of targets."""
    values: list[float] = []
    unit = None
    missing = 0
    for target in targets:
        raw_value = _target_attr(target, attr_path)
        value, value_unit = split_report_value(
            raw_value,
            period_idx=getattr(target, "period_idx", None),
        )
        if value is None:
            missing += 1
            continue
        if isinstance(value, list):
            raise ValueError(f"Cannot aggregate array-valued field {attr_path!r}.")
        if value_unit is not None:
            if unit is None:
                unit = value_unit
            elif value_unit != unit:
                value = Value(value, value_unit).to(unit).value
        values.append(float(value))
    return values, unit, missing


def _weighted_report_value(
    targets: Sequence[TargetResults],
    attr_path: str,
    weights: np.ndarray,
    *,
    allow_partial_missing: bool = False,
) -> Value | float | None:
    values, unit, missing = _collect_report_values(targets, attr_path)
    if missing == len(targets):
        return None
    if missing:
        if allow_partial_missing:
            return None
        raise ValueError(f"Cannot aggregate partially missing field {attr_path!r}.")
    weighted = _weighted_average(values, weights)
    return Value(weighted, unit) if unit is not None else weighted


def _max_report_value(
    targets: Sequence[TargetResults],
    attr_path: str,
) -> Value | float | None:
    values, unit, missing = _collect_report_values(targets, attr_path)
    if missing == len(targets):
        return None
    if missing:
        raise ValueError(f"Cannot aggregate partially missing field {attr_path!r}.")
    maximum = max(values)
    return Value(maximum, unit) if unit is not None else maximum


def _ratio_of_totals(
    targets: Sequence[TargetResults],
    ratio_field: str,
    basis_field: str | None,
    weights: np.ndarray,
    *,
    basis_is_numerator: bool,
) -> Value | float | None:
    """Aggregate a ratio as weighted total numerator over total denominator.

    ``basis_field`` is the numerator (``basis_is_numerator``) or denominator of
    the ratio, e.g. COP = Q / W with basis ``hpr_work`` as denominator, so the
    seasonal COP is sum(w * COP * W) / sum(w * W). Periods without the ratio
    contribute nothing. Falls back to the weighted mean when the totals are 0
    or there is no ``basis_field``.
    """
    numerator = denominator = 0.0
    unit = None
    ratios: list[tuple[float, float]] = []
    for target, weight in zip(targets, weights):
        period_idx = getattr(target, "period_idx", None)
        ratio, ratio_unit = split_report_value(
            _target_attr(target, ratio_field), period_idx=period_idx
        )
        basis = None
        if basis_field is not None:
            basis, _basis_unit = split_report_value(
                _target_attr(target, basis_field), period_idx=period_idx
            )
        if ratio is None or isinstance(ratio, list):
            continue
        unit = unit or ratio_unit
        ratios.append((float(ratio), float(weight)))
        if basis is None or isinstance(basis, list):
            continue
        ratio, basis = float(ratio), abs(float(basis))
        if basis_is_numerator:
            numerator += weight * basis
            denominator += weight * basis / ratio if abs(ratio) > 0.0 else 0.0
        else:
            numerator += weight * ratio * basis
            denominator += weight * basis
    if not ratios:
        return None
    if denominator > 0.0:
        value = numerator / denominator
    else:
        total_weight = sum(weight for _ratio, weight in ratios)
        value = (
            sum(ratio * weight for ratio, weight in ratios) / total_weight
            if total_weight > 0.0
            else ratios[0][0]
        )
    return Value(value, unit) if unit is not None else value


def _apply_peak_machine_capital(
    targets: Sequence[TargetResults],
    data: dict[str, Any],
) -> None:
    """Size each heat-pump machine for its own peak period.

    Parallel machines can peak in different periods, so the peak of the period
    totals would undersize them. When every period comes from one shared design
    and carries the per-machine breakdown, each machine takes its own maximum
    and the HPR capital fields become the sum of those peaks. Independently
    optimised periods may number their machines differently, so they keep the
    peak of the totals.
    """
    capital = [getattr(target, "hpr_machine_capital_costs", None) for target in targets]
    annualized = [
        getattr(target, "hpr_machine_annualized_capital_costs", None)
        for target in targets
    ]
    shared = all(getattr(target, "hpr_shared_design", None) for target in targets)
    if (
        (not shared)
        or any(not value for value in (*capital, *annualized))
        or (len({len(value) for value in (*capital, *annualized)}) != 1)
    ):
        data["hpr_machine_capital_costs"] = None
        data["hpr_machine_annualized_capital_costs"] = None
        return
    peak_capital = tuple(float(v) for v in np.max(np.asarray(capital), axis=0))
    peak_annualized = tuple(float(v) for v in np.max(np.asarray(annualized), axis=0))
    data["hpr_machine_capital_costs"] = peak_capital
    data["hpr_machine_annualized_capital_costs"] = peak_annualized
    # The per-machine tuples are canonical $ and $/y; the target fields may
    # already be in a configured display unit, such as k$, so keep theirs.
    data["hpr_capital_cost"] = _in_unit_of(
        Value(sum(peak_capital), "$"), data.get("hpr_capital_cost")
    )
    data["hpr_annualized_capital_cost"] = _in_unit_of(
        Value(sum(peak_annualized), "$/y"), data.get("hpr_annualized_capital_cost")
    )


def _in_unit_of(value: Value, reference: Any) -> Value:
    """Return ``value`` in the unit of ``reference`` when that is a ``Value``."""
    if isinstance(reference, Value):
        return value.to(reference.unit)
    return value


def _hpr_utility_annualized_capital(
    targets: Sequence[TargetResults],
    data: Mapping[str, Any],
) -> Value | float | None:
    """Sum each default utility's peak capital across periods.

    The hot utility and the refrigeration plant are separate assets, each sized
    for its own peak period, so their maxima are taken separately and added.
    Results without the split fall back to the peak of the combined value.
    """
    parts = [
        data.get(field)
        for field in (
            "hpr_hot_utility_annualized_capital_cost",
            "hpr_refrigeration_annualized_capital_cost",
        )
    ]
    parts = [part for part in parts if part is not None]
    if not parts:
        return _max_report_value(targets, "hpr_utility_annualized_capital_cost")
    if all(isinstance(part, Value) for part in parts):
        total = parts[0]
        for part in parts[1:]:
            total = total + part
        return total
    if any(isinstance(part, Value) for part in parts):
        raise ValueError("HPR utility capital fields must use compatible units.")
    return float(sum(float(part) for part in parts))


def _total_hpr_annualized_cost(data: Mapping[str, Any]) -> Value | float | None:
    operating = data.get("hpr_operating_cost")
    annualized_capital = data.get("hpr_annualized_capital_cost")
    if operating is None and annualized_capital is None:
        return None
    if operating is None or annualized_capital is None:
        raise ValueError(
            "Cannot recompute HPR total annualized cost from a partial cost breakdown."
        )
    # Default-utility capital is optional: older results do not carry it.
    utility_capital = data.get("hpr_utility_annualized_capital_cost")
    parts = [operating, annualized_capital]
    if utility_capital is not None:
        parts.append(utility_capital)
    if all(isinstance(part, Value) for part in parts):
        total = parts[0]
        for part in parts[1:]:
            total = total + part
        return total
    if any(isinstance(part, Value) for part in parts):
        raise ValueError("HPR annualized cost fields must use compatible units.")
    return float(sum(float(part) for part in parts))


def _weighted_numeric_attr(
    targets: Sequence[TargetResults],
    attr_name: str,
    weights: np.ndarray,
) -> float | None:
    values = []
    missing = 0
    for target in targets:
        value = getattr(target, attr_name, None)
        if value is None:
            missing += 1
            values.append(None)
            continue
        values.append(float(value))
    if missing == len(targets):
        return None
    if missing:
        raise ValueError(f"Cannot aggregate partially missing field {attr_name!r}.")
    return _weighted_average(values, weights)


def _weighted_utilities(
    targets: Sequence[TargetResults],
    attr_name: str,
    weights: np.ndarray,
) -> list[HeatUtility]:
    ordered_names: list[str] = []
    per_target_maps = []
    unit_by_name: dict[str, str | None] = {}
    for target in targets:
        utility_map: dict[str, tuple[float, str | None]] = {}
        for utility in getattr(target, attr_name, []) or []:
            value, unit = split_report_value(
                utility.heat_flow,
                period_idx=getattr(target, "period_idx", None),
            )
            if value is None:
                continue
            if isinstance(value, list):
                raise ValueError(
                    f"Cannot aggregate array-valued utility {utility.name!r}."
                )
            name = str(utility.name)
            if name not in ordered_names:
                ordered_names.append(name)
            preferred_unit = unit_by_name.get(name)
            if unit is not None:
                if preferred_unit is None:
                    unit_by_name[name] = unit
                    preferred_unit = unit
                elif unit != preferred_unit:
                    value = Value(value, unit).to(preferred_unit).value
            previous_value, previous_unit = utility_map.get(name, (0.0, preferred_unit))
            utility_map[name] = (previous_value + float(value), previous_unit or unit)
        per_target_maps.append(utility_map)

    utilities = []
    for name in ordered_names:
        unit = unit_by_name.get(name)
        values = []
        for utility_map in per_target_maps:
            value, value_unit = utility_map.get(name, (0.0, unit))
            if unit is None and value_unit is not None:
                unit = value_unit
            values.append(float(value))
        weighted = _weighted_average(values, weights)
        utilities.append(
            HeatUtility(
                name=name,
                heat_flow=Value(weighted, unit) if unit is not None else weighted,
            )
        )
    return utilities


def _weighted_average(values: Sequence[float], weights: np.ndarray) -> float:
    return float(np.average(np.asarray(values, dtype=float), weights=weights))


def _target_attr(target: TargetResults, path: str) -> Any:
    current: Any = target
    for part in path.split("."):
        if isinstance(current, Mapping):
            current = current.get(part)
        else:
            current = getattr(current, part, None)
    return current


def _consensus_value(values) -> Any:
    non_null = [value for value in values if value is not None]
    if not non_null:
        return None
    first = non_null[0]
    try:
        return first if all(value == first for value in non_null[1:]) else None
    except Exception:
        return first if all(value is first for value in non_null[1:]) else None
