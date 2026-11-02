# Application Problem Output

## Overview
Shapes solved zone trees into serialisable results and report tables for `PinchProblem`.
`result_extraction.py` serialises targets, default utilities and graph data from a `Zone`;
`reporting.py` builds summary frames, typed report metrics, graph catalogues and cross-problem
comparisons from a `TargetOutput`. `OpenPinch/application/problem.py` is the consumer.

## Interface
- `extract_results` (`result_extraction.py`): dict with `name`, `period_id`, `targets`, `utilities` and `graphs` for a zone tree, optionally for one period.
- `build_problem_summary_frame` (`reporting.py`): compact summary DataFrame, or the detailed workbook frame when `detailed=True`.
- `build_report_metrics` (`reporting.py`): `ReportMetric` list for every summary target row.
- `build_graph_availability` (`reporting.py`): flat `GraphAvailability` records from graph data.
- `build_problem_report` (`reporting.py`): `ProblemReport` from validation, optional results and graph data.
- `build_graph_data` (`reporting.py`): JSON-safe graph dict from results, or `None`.
- `locate_summary_row` (`reporting.py`): select one summary row by scope, zone type, integration type and target method.
- `compare_problem_summaries` (`reporting.py`): base, other and change rows for the headline metrics of two summary frames.
- `format_res` (`reporting.py`): one utility or value as text with two decimals and a unit.
- Subdirectories: none.

## Invariants
- Only targets with `reportable` set are serialised, and with a `period_id` only targets of that period. [result_extraction.py:_get_report]
- Subzone targets follow their parent's in a depth-first walk. [result_extraction.py:_get_report]
- `_get_utilities` returns a two-item list `[HU, CU]` found by the names `HU` and `CU`; either is `None` when absent. [result_extraction.py:_get_utilities]
- Optional summary columns (area, cost, exergy) appear only when at least one target has a value. [reporting.py:_optional_metrics]
- In `locate_summary_row` an omitted `integration_type` or `target_method` defaults to Process or Heat Exchange, and an omitted `scope` selects the shallowest matching scope. [reporting.py:locate_summary_row]
- Comparisons convert the other problem's values to the base row's unit; a dimension or type mismatch gives `None` in the change row. [reporting.py:compare_problem_summaries]
- Comparison scopes are matched below each problem's root name. [reporting.py:_matching_scope]
- Text columns of a comparison frame are kept as `object` so missing entries stay `None`. [reporting.py:_text_columns_as_objects]
- `build_problem_report` copies targets, so edits to the report do not change the results. [reporting.py:build_problem_report]

## Side Effects
- `extract_results` reads the zone's graph data through `get_output_graph_data`. [result_extraction.py:extract_results]
- Otherwise none; frames and reports are built from copies of their inputs.

## Verification
- Run: `uv run --no-sync pytest tests/application/test_pinch_problem.py tests/application/test_summary_optional_columns.py`
- There is no `tests/application/_problem` directory; these are the nearest focused tests. Broader: `uv run --no-sync pytest tests/application`.
