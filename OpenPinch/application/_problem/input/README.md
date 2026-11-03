# Application Problem Input

## Overview
Turns a user-supplied problem definition (a `TargetInput`, a dict, or a file source) into a validated
`Zone` tree with process streams and utilities attached. `loading.py` normalises sources and swaps
state on a `PinchProblem`, `validation.py` and `semantics.py` produce errors and reports,
`canonicalization.py` normalises the zone tree, and `construction.py`, `segments.py` and `utilities.py`
build the domain objects. `PinchProblem` (`OpenPinch/application/problem.py`), `commands.py`,
`workspace.py`, `targeting.py`, `utility_placement.py`, `residual_utility.py` and `case_derivation.py`
import from here.

## Interface
- `prepare_problem` (`construction.py`): builds the master `Zone` hierarchy, streams and utilities from schemas.
- `_find_extreme_process_temperatures` (`construction.py`): shifted hot-stream minimum and cold-stream maximum; imported by `utility_placement.py`.
- `canonical_problem_inputs` (`canonicalization.py`): deep-copied input dict with a normalised zone tree and rewritten stream zones.
- `_LoadedProblemSource`, `load_problem_source`, `prepare_in_memory_problem_source`, `normalize_problem_mapping` (`loading.py`): normalise a source into inputs plus validation context.
- `apply_loaded_source`, `rebuild_problem_state`, `replace_loaded_source`, `replace_problem_inputs` (`loading.py`): set or replace a problem's state; `replace_loaded_source` restores prior state on failure.
- `find_zone_tree_node` (`loading.py`): look up a zone-tree dict node by absolute or relative name.
- `validate_problem_inputs`, `validate_problem_semantics` (`validation.py`): raise `ValueError` on schema or semantic errors; warn on warnings.
- `build_validation_report`, `build_validation_context` (`validation.py`): non-raising `ValidationReport`, and per-record metadata for readable messages.
- `InputValidationWarning` (`validation.py`): `UserWarning` carrying the `ValidationIssue` list.
- Formatting helpers (`validation.py`): `collapse_union_errors`, `schema_issue_to_view`, `format_schema_validation_error`, `format_single_validation_error`, `path_from_loc`, `validation_record_label`, `lookup_record_context`, `describe_record`.
- `semantic_issues`, `segmented_record_issues` (`semantics.py`): semantic `ValidationIssue` lists for a whole input and for one nested record.
- `_get_hot_and_cold_utilities`, `_set_utilities_for_zone_and_subzones` (`utilities.py`): build default and given utilities as stream collections; copy them to a zone and its subzones.
- `_create_segmented_process_stream`, `_create_segmented_utility_stream` (`segments.py`): build parent streams from nested `segments` or a cumulative-heat `profile`.
- Subdirectories: none.

## Invariants
- At least one stream is required (unless a `residual_basis` is given). [semantics.py:semantic_issues], [canonicalization.py:_validate_streams_passed_in]
- The root zone of the tree cannot be a Utility Zone. [canonicalization.py:_validate_zone_tree_structure]
- Zone `dt_cont_multiplier` is finite and non-negative; a child inherits its parent's when unset. [canonicalization.py:_resolve_zone_dt_cont_multiplier]
- Multipliers are applied before streams are built, because each stream takes its zone's multiplier and default utilities are placed from shifted temperatures. [construction.py:prepare_problem]
- Streams are sorted by name before preparation; a stream without `dt_cont` takes `THERMAL_DT_CONT`. [construction.py:prepare_problem], [construction.py:_build_prepared_stream_collection]
- Validated temperatures are in degC; below -273.15 is an error. Temperatures are compared in canonical units. [semantics.py:_validate_above_absolute_zero], [construction.py:_validate_stream_temperatures]
- A near-isothermal stream is widened to `THERMAL_DT_PHASE_CHANGE`; its duty is unchanged. [construction.py:_widen_near_isothermal_target]
- Per-period arrays must have one entry per `PROBLEM_PERIOD_IDS` id. [semantics.py:_validate_period_lengths]
- Supplied `heat_flow` must agree with `heat_capacity_flowrate` x |dT| to a relative 1e-4. [semantics.py:_validate_heat_capacity_flowrate_agreement]
- Segment temperatures chain end to start, keep one direction, and carry positive heat flow and HTC; profile cumulative heat increases strictly. [semantics.py:_validate_segmented_stream_states], [semantics.py:_validate_temperature_heat_profile_states]
- Supplied parent `t_supply`, `t_target` and `heat_flow` must match the nested profile. [segments.py:_validate_supplied_parent_aggregates]
- A default utility (HU, CU, then HU_1, CU_1, ...) is added unless an active, uncapped utility of that type reaches the process temperatures; a capped utility never replaces it. Default-utility streams are marked by identity. [utilities.py:_complete_utility_data], [utilities.py:_add_default_utilities], [utilities.py:_create_utilities_list]
- Default utilities are placed for the largest zone multiplier. [construction.py:_largest_dt_cont_multiplier]
- A state replacement either completes or leaves all `_PROBLEM_STATE_FIELDS` as before. [loading.py:replace_loaded_source]
- Excel and CSV sources report records as sheet row `index + 3`. [validation.py:_build_record_context]

## Side Effects
- `load_problem_source` reads files or tuples of files through the adapters for non-in-memory sources. [loading.py:load_problem_source]
- `apply_loaded_source` and `replace_loaded_source` assign attributes on the parent `PinchProblem`. [loading.py:apply_loaded_source]
- `rebuild_problem_state` calls `problem._invalidate_analysis()`. [loading.py:rebuild_problem_state]
- `rebuild_problem_state` empties `_process_components`; `replace_problem_inputs` keeps them by replaying them on the rebuilt tree, and a failed replay restores the prior state. [loading.py:rebuild_problem_state] [loading.py:replace_problem_inputs]
- Zone-tree normalisation rewrites `stream.zone` on the stream models it is given and may append process nodes to the tree. [canonicalization.py:_rewrite_stream_zones_from_tree]
- `validate_problem_semantics` emits `InputValidationWarning`. [validation.py:validate_problem_semantics]
- Utility completion mutates deep copies of the given utility schemas (target, `dt_cont`, price, HTC, `active`). [utilities.py:_complete_utility_data]
- `_validate_config_data_completed` repairs the `Configuration` in place. [canonicalization.py:_validate_config_data_completed]

## Verification
- Run: `uv run --no-sync pytest tests/application/test_problem_validation.py tests/application/test_input_contract_fixes.py tests/application/test_input_audit_regressions.py`
- There is no `tests/application/_problem` directory; these files are the nearest focused tests. Broader: `uv run --no-sync pytest tests/application`.
