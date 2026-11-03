# Energy Transfer Analysis

## Overview
Builds the energy transfer diagram and its surplus/deficit table for a zone. Each operation (a child zone's Direct Integration target, or the zone's own target) contributes a grand composite cascade; the cascades are placed on one shared temperature grid and stacked. `service.py` is the entry used by the application layer, `selection.py` picks and refreshes the base target, `diagram.py` assembles the payload, `cascade.py` holds the array maths and `serialization.py` the cleaning helpers.

## Interface
- `run_energy_transfer_analysis_service(zone, args, *, refresh_services, compute_func)` (`service.py`): picks a base target (Indirect Integration, then Direct Integration) and adds one `EnergyTransferTarget` to the zone.
- `compute_energy_transfer_target(base_target, source_targets)` (`diagram.py`): returns a typed `EnergyTransferTarget` carrying the diagram and table.
- `create_energy_transfer_diagram(source_targets, *, base_target, simplify)` (`diagram.py`): returns a dict with `temperatures`, `operations`, base target name/type and pinch temperatures.
- `create_heat_surplus_deficit_table(diagram)` (`diagram.py`): returns interval rows with `heat_surplus` and `heat_deficit`.
- `ENERGY_TRANSFER_TARGET_ORDER`, `normalize_base_target_type`, `candidate_order`, `ensure_base_target`, `source_targets`, `iter_source_zones` (`selection.py`): base-target selection and refresh.
- `compile_temperature_intervals`, `transpose_operation_cascades`, `simplify_constant_cp_intervals`, `characterise_cascades`, `stack_order`, `stack_cascades`, `pinch_weights`, `cross_pinch_flags`, `first_active_temperatures`, `empty_header`, `has_problem_table_values`, `interp_descending_temperature`, `MODE_ORDER`, `WEIGHTING_TEMPERATURE` (`cascade.py`): grid, cascade and ordering helpers.
- Subdirectories: none.

## Invariants
- The default base-target order is Indirect Integration then Direct Integration; a zone with no Indirect target and no subzones uses Direct Integration only. [selection.py:ENERGY_TRANSFER_TARGET_ORDER] [selection.py:candidate_order]
- An explicit `base_target_type` that cannot be produced raises `RuntimeError`; the implicit order falls through to the next type. [service.py:run_energy_transfer_analysis_service]
- The shared grid starts as the descending union of the source and base-table temperatures, rounded to the decimal places implied by `tol`; with `simplify` (the default) adjacent intervals of equal heat capacity are then merged, so the payload `temperatures` are a subset of that union. [cascade.py:compile_temperature_intervals]
- Interval heat has the sign of a change in `H_NET` down the interval (`-dT * CP_NET`): positive is surplus. [cascade.py:transpose_operation_cascades] [diagram.py:create_heat_surplus_deficit_table]
- Stack order is by mode (`C`, then `R`, then `H`), then descending sort key, then name. [cascade.py:stack_order]
- A source's own `ProblemTable` is preferred over its saved GCC graph, which is rounded for plotting. [diagram.py:get_source_problem_table]
- A base target without a `ProblemTable` raises `TypeError`. [diagram.py:get_base_problem_table]
- Values within `tol` of zero are cleaned to exactly 0.0 in the payload. [serialization.py:_clean_value]

## Side Effects
- Sets `zone._selected_energy_transfer_base_target_type` and adds a target to the zone. [service.py:run_energy_transfer_analysis_service]
- May call the supplied refresh services, which re-run Direct and Indirect Integration on the zone and its subzones. [selection.py:ensure_base_target]
- Records the selected period on the zone through `prepare_enrichment_run`. [service.py:run_energy_transfer_analysis_service]

## Verification
- Run: `uv run --no-sync pytest tests/analysis/test_energy_transfer_analysis.py`
