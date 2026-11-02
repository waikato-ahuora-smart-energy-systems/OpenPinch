# Application Problem Periods

## Overview
Multi-period summary support for `PinchProblem`. It records the last explicit targeting call so it
can be replayed per period on isolated zone copies, and combines per-period `TargetOutput` objects
into all-period, weighted-average, or combined summaries. Aggregation rules come from
`contracts.report_metrics` metric specifications. `OpenPinch/application/problem.py` imports the
module and its `_TargetRunSpec`.

## Interface
- `_TargetRunSpec` (`aggregation.py`): frozen record of surface, options, zone name and `include_subzones`.
- `record_target_run` (`aggregation.py`): store a deep-copied run spec on the problem.
- `target_run_spec_for_summary` (`aggregation.py`): the recorded spec; raises `RuntimeError` when none exists.
- `period_options_for_replay` (`aggregation.py`): spec options with one canonical `period_id` overlaid.
- `target_outputs_for_recorded_periods`, `target_output_for_recorded_period` (`aggregation.py`): replay the recorded surface for every period, or one.
- `period_weights_for_summary` (`aggregation.py`): weights of the given period ids, by identity.
- `summary_results` (`aggregation.py`): resolve `selected` results or cached all-period results into one `TargetOutput`.
- `combine_period_outputs`, `weighted_average_output`, `output_for_period_mode` (`aggregation.py`): concatenate, weight-average, or both.
- `WEIGHTED_AVERAGE_PERIOD_ID`, `SUMMARY_PERIOD_MODES` (`aggregation.py`): the `"weighted_average"` period id and the modes `selected`, `all`, `weighted_average`, `all_with_weighted_average`.
- Subdirectories: none.

## Invariants
- `periods` must be one of `SUMMARY_PERIOD_MODES`; `selected` returns a deep copy of the cached results. [aggregation.py:summary_results]
- Period outputs are ordered by the root zone's `period_ids`, not publication order. [aggregation.py:summary_results]
- Weighted averaging needs identical target rows (keyed by scope, zone type, integration type, method, row type) in every period, with no duplicates. [aggregation.py:_aligned_target_groups]
- A period without a heat pump or refrigeration row gets a zero-load row built from its own base row. [aggregation.py:_fill_zero_load_hpr_rows]
- Each metric is aggregated by its `AggregationPolicy` (weighted mean, maximum, consensus, exclude); a metric with none of these and not listed as derived raises `ValueError`. [aggregation.py:_weighted_average_target]
- Ratios (degree of integration, ETE, COP, efficiencies) are ratios of weighted totals, falling back to the weighted mean when the totals are zero. [aggregation.py:_ratio_of_totals]
- The weighted row leaves pinch temperatures empty; `hpr_success` is true only if every run that reports it succeeded, and is `None` when no run reports it. [aggregation.py:_weighted_average_target]
- Partially missing fields cannot be aggregated. [aggregation.py:_weighted_report_value]
- Shared-design heat-pump machines take their own peak capital; otherwise the per-machine fields are cleared. [aggregation.py:_apply_peak_machine_capital]
- Values are converted to the first period's unit before aggregation. [aggregation.py:_collect_report_values]

## Side Effects
- `record_target_run` assigns `problem._last_target_run_spec` unless `_suspend_target_run_recording` is set. [aggregation.py:record_target_run]
- Replay re-runs targeting through `problem.target.<surface>`, replacing `problem._master_zone` with deep copies per period and restoring master zone, results, spec and recording flag in a `finally` block. [aggregation.py:target_outputs_for_recorded_periods]
- `_ratio_of_totals` can emit a `UserWarning` for periods that lack the basis field. [aggregation.py:_ratio_of_totals]

## Verification
- Run: `uv run --no-sync pytest tests/application/test_multiperiod_summary.py tests/application/test_multiperiod_summary_properties.py`
