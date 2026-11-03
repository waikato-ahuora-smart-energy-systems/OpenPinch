# Process MVR Helpers

## Overview
Private helpers behind `ProcessMVRComponent` in `../process_mvr.py`. They select hot gas or vapour source streams in a zone tree, solve the direct gas MVR model for every period via `../direct_mvr`, build multiperiod replacement streams, track which zones hold the original and replacement streams, and sum compressor work.

## Interface
What `../process_mvr.py` imports:
- `add_replacement_streams_to_memberships`, `find_hot_stream_memberships` (`membership.py`): attach replacement streams wherever the original sits; find every hot-stream collection holding a stream by identity.
- `build_process_mvr_stream_record` (`replacement_streams.py`): solve all periods and return a `_ProcessMVRStreamRecord`.
- `normalise_source_selectors`, `match_source_streams` (`selection.py`): normalise selectors and resolve them to unique, validated hot streams.
- `_ProcessMVRStreamRecord` (`state.py`): original stream, memberships, replacement streams and per-period stage results.
- `work_for_zone` (`work.py`; imported as `calculate_work_for_zone`): compressor work for records inside a zone subtree and period.
- Internal modules: `values.py` (period-value and unit helpers), `state.py` also defines `_StreamMembership`.

## Invariants
- A source must be an active, hot, process stream with `fluid_phase` of gas, vapour or vapor and a `fluid_name` [selection.py:validate_process_mvr_source].
- Streams are matched by identity, key or display name; a detached `Stream` that is not in the tree is resolved by a unique name, and an ambiguous name raises `ValueError` [selection.py:match_one_selector].
- A source with a temperature drop under 1 degC is treated as a vapour and its pressure, when given, must be near saturation (rtol 1e-2, atol 0.5 kPa); larger drops require `p_supply` and a gas or supercritical state [selection.py:validate_process_mvr_source_phase, selection.py:validate_vapour_state].
- Replacement stream keys are `<original key>.<component_id>.<stream name>`, and replacements are added to every zone holding the original [membership.py:add_replacement_streams_to_memberships].
- Zone tree walks are pre-order and deterministic [membership.py:walk_zones].
- Zone membership is decided by address prefix (`zone/`) [membership.py:record_affects_zone].
- Replacement streams are segmented across periods after aligning the stage profiles, and are named `<source>_direct_MVR_H<stage>` [replacement_streams.py:build_multiperiod_stage_streams].
- Work for a period is looked up by id, then by index label, then falls back to the first stored period; an inactive component yields 0.0 [work.py:record_stage_results_for_period, work.py:work_for_zone].

## Side Effects
- Selection fills `supply_pressure` and `target_pressure` on the source stream when they are unset [selection.py:validate_process_mvr_source_phase].
- Replacement streams are added to the zones' hot-stream collections [membership.py:add_replacement_streams_to_memberships].
- CoolProp `PropsSI` calls for critical and saturation properties [selection.py:coolprop_value, selection.py:validate_vapour_state].

## Verification
- Run: `uv run --no-sync pytest tests/analysis/heat_pumps/test_process_components.py tests/analysis/heat_pumps/test_direct_mvr.py` (no test directory is dedicated to this package; these cover the component and the solver it drives).
