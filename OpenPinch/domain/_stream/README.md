# Stream Helpers

## Overview
Private helpers behind `OpenPinch.domain.stream.Stream`. They split the stream model into stateless value and period helpers, core-state completion and derived thermodynamics, segment profile validation and aggregation, parent-owned segment transactions, piecewise linearisation and the `StreamSegment` record. `Stream` imports `profile`, `segments`, `thermodynamics` and `value_state` as modules and `segment.StreamSegment` at the end of its module; `zone.py` and the analysis package import `value_state.resolve_period_weights`.

## Interface
- `StreamSegment` (`segment.py`): a `Stream` subclass owned by a parent stream; role, activity, fluid and period context are inherited from the parent, and value edits are forwarded to the parent's `update_segment`.
- `complete_core_state`, `derive_stream_state` (`thermodynamics.py`): fill missing core values and enforce the minimum span, then compute type, shifted and bounding temperatures, CP, resistance and cost; return `CompletedCoreState` and `DerivedStreamState`.
- `coerce_to_value`, `value_array`, `build_value`, `copy_value`, `period_vector_size`, `validate_num_periods` (`value_state.py`): value coercion, scalar-to-period broadcasting and period-count checks.
- `resolve_period_weights`, `normalise_period_ids`, `normalise_weights`, `is_period_value_data` (`value_state.py`): period model helpers.
- `temperature_heat_segment_specs`, `validate_segments`, `aggregate_segments`, `detached_segment` (`profile.py`): profile-to-segment construction data, chain validation, aggregate core values and detached copies; `SegmentSpec` and `SegmentAggregateState` are the result records.
- `normalise_segment_input(s)`, `replace`, `update_transaction`, `update_all_value_attributes`, `set_total_heat_flow_at_idx`, `sync_aggregate`, `validate`, `detached` (`segments.py`): parent-owned segment transactions.
- `get_piecewise_linearisation_for_streams`, `get_piecewise_data_points`, `normalise_temperature_heat_profile`, `align_temperature_heat_profiles`, `build_segmented_stream_from_profile` (`linearisation.py`): piecewise-linear approximation of temperature-enthalpy curves and profile conditioning.

## Invariants
- A stream with equal supply and target temperatures (within `1e-12`) gets its target moved by 0.01 degC: up for positive heat flow, down for negative. [thermodynamics.py:complete_core_state]
- Stream type is Hot if any period has supply above target, otherwise Cold if any period has supply below target, otherwise Neutral; shifted temperatures subtract the effective contribution for hot streams and add it for cold streams. [thermodynamics.py:derive_stream_state]
- `None` or empty period weights mean equal weights and one weight applies to all periods; otherwise every period needs exactly one weight (partial lists, excess values and unknown ids in a mapping raise). Weights must be finite and non-negative and sum to a positive value. [value_state.py:resolve_period_weights]
- Stream values do not accept `period_ids`; they are read in the problem's period order. [value_state.py:coerce_to_value]
- Multiperiod values must share one period count. [value_state.py:validate_num_periods]
- A segment profile needs finite positive heat flows (non-negative while targeting duties are allowed), finite positive HTCs, non-zero temperature span, one thermal direction across segments and periods, and each segment's supply equal to the previous target. [profile.py:validate_segments]
- Aggregate HTC is duty divided by the sum of duty over HTC; aggregate price is the duty-weighted mean in `$/MW/h`. [profile.py:aggregate_segments]
- Temperature-heat profiles need at least two finite points with strictly monotonic heat coordinates. [profile.py:temperature_heat_segment_specs] [linearisation.py:normalise_temperature_heat_profile]
- Period profiles align on the union of their cumulative duty fractions, and fractions within 1e-9 count as one breakpoint, so rounding cannot create a zero-duty segment; each profile's end points stay exact. [linearisation.py:align_temperature_heat_profiles]
- Segment replacement validates before attaching; if attachment or aggregate sync fails, the previous segment list and ownership are restored (aggregate values already assigned are not rolled back); segments still owned by another stream, as when a shallow copy shares its original's, keep that owner. [segments.py:replace]
- Updates to attached segments run on detached clones and commit through `replace_segments`; `is_active`, `is_process_stream`, `fluid_name` and `fluid_phase` cannot be set on a segment. [segments.py:update_transaction] [segment.py:StreamSegment]
- A segmented stream's targeting duty must be finite and at least `-tol`; it is applied by scaling child duties by cached fractions. [segments.py:set_total_heat_flow_at_idx]
- Piecewise breakpoints use Ramer-Douglas-Peucker, refined by SLSQP for hot or cold integrity when more than ten points remain, with plain RDP as the fallback when the refinement fails. [linearisation.py:get_piecewise_data_points]

## Side Effects
- `sync_aggregate` assigns the parent's value attributes and bumps its numeric revision; `replace` mutates the parent and its segments. [segments.py:sync_aggregate] [segments.py:replace]
- `set_total_heat_flow_at_idx` writes the parent's `_segment_targeting_fractions` cache and toggles `_allow_targeting_segment_duties` for the duration of the update. [segments.py:set_total_heat_flow_at_idx]
- Imports `scipy.optimize` at module load. [linearisation.py]

## Verification
- Run: `uv run --no-sync pytest tests/domain/test_stream_segments.py tests/domain/test_stream.py`
