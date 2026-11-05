# Heat Exchanger Helpers

## Overview
Private helpers owned by `OpenPinch.domain.heat_exchanger` and `OpenPinch.domain.heat_exchanger_network`. They hold the per-period state record, the local area-slice record and the validation rules for exchanger and network records. Only the rules module is shared with the input contract (`OpenPinch/contracts/input.py`); the period-state and area-slice checks are duplicated there.

## Interface
- `HeatExchangerPeriodState` (`period_state.py`): Pydantic record of one exchanger's duty, activity, split fractions, approach temperatures and source/sink temperatures for one ordered period; `source_mid_temperature` and `sink_mid_temperature` properties.
- `HeatExchangerAreaSlice` (`area.py`): frozen Pydantic record of one duty-aligned local area slice (period, segment identities, duty, temperatures, HTCs, LMTD, area).
- `segment_duty_by_period`, `segment_area_by_period` (`area.py`): sum slice duty or area per period id.
- `segment_design_area` (`area.py`): maximum period-total slice area, or `None` without slices.
- `validate_segment_design_area` (`area.py`): return the slice-derived design area when slices exist (raising if a supplied area disagrees), otherwise the supplied area.
- `optional_identity`, `optional_non_negative_finite` (`rules.py`): strip a non-empty identity or check a finite non-negative float, passing `None` through.
- `check_period_states`, `check_period_alignment` (`rules.py`): period ordering, uniqueness and cross-exchanger alignment checks.
- `check_direction_semantics` (`rules.py`): endpoint distinctness, kind-consistent stream roles and recovery stage presence.
- `check_summary_metrics` (`rules.py`): metric names and float values.
- `EXCHANGER_IDENTITY_MESSAGE`, `EXCHANGER_NUMERIC_MESSAGE`, `NETWORK_IDENTITY_MESSAGE`, `NETWORK_NUMERIC_MESSAGE`, `EXPECTED_STREAM_ROLES` (`rules.py`): shared error messages and the kind-to-roles table.

## Invariants
- A period state has a non-empty stripped `period_id`, non-negative `period_idx`, finite non-negative `duty`, split fractions in [0, 1], finite temperatures and finite non-negative approach temperatures. [period_state.py:HeatExchangerPeriodState]
- `period_states` must be ordered by contiguous `period_idx` starting at zero, with unique `period_id` values. [rules.py:check_period_states]
- Every exchanger in a network must share the first exchanger's ordered period ids. [rules.py:check_period_alignment]
- Recovery exchangers link Process to Process, hot utility exchangers link Utility to Process, cold utility exchangers link Process to Utility; source and sink streams must differ; recovery exchangers need a stage. [rules.py:check_direction_semantics]
- Area slices are immutable, with finite positive duty, HTCs, LMTD and area, finite temperatures and non-empty identities. [area.py:HeatExchangerAreaSlice]
- The design area is the maximum over periods of the summed slice areas, and a supplied area must match it within relative 1e-4 or absolute 1e-3. [area.py:segment_design_area] [area.py:validate_segment_design_area]
- Summary metric names are non-empty strings and float metric values are finite. [rules.py:check_summary_metrics]

## Side Effects
None.

## Verification
- Run: `uv run --no-sync pytest tests/domain/test_heat_exchanger_area_slices.py tests/domain/test_heat_exchanger_network.py`
