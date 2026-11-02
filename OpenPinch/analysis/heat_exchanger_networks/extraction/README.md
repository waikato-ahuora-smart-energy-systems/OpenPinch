# HEN Extraction

## Overview
Converts a solved equation model (and the solver arrays it was built from) into identity-labelled domain objects: a `HeatExchangerNetwork` of `HeatExchanger` records with per-period states, and the synthesis result contract. The solved model is read only through `getattr`, so both period-native and single-period model attributes are accepted. It sits between `../models` / `../solver` and `../results`.

## Interface
- `extract_heat_exchanger_network` (`service.py`): builds the network (recovery, hot-utility and cold-utility exchangers, costs, summary metrics, solver axis metadata and source metadata).
- `extract_network_synthesis_result` (`service.py`): wraps the network in a `HeatExchangerNetworkSynthesisResult` with objective values.
- Private helper modules, imported only by `service.py`:
  - `metadata.py`: period ids, costs, summary metrics, scalar coercion, axis identities.
  - `recovery.py`: process-to-process exchangers.
  - `utility.py`: heaters and coolers.
  - `period_state.py`: per-period duty, temperatures, approaches and split fractions.
  - `segment_area.py`: segment-resolved area contributions.

## Invariants
- Exchanger ids are `recovery:<hot>-><cold>:S<stage>`, `hot-utility:<utility>-><cold>` and `cold-utility:<hot>-><utility>`; recovery stages are one-based [recovery.py:_recovery_exchangers, utility.py:_hot_utility_exchangers].
- Inactive exchangers are dropped unless `include_inactive` is true; an exchanger is active when its duty and (if present) its binary exceed the tolerance [metadata.py:_is_active].
- Duties below minus `max(tolerance, 1e-4)` raise `ValueError`; smaller negatives are clipped to zero [metadata.py:_normalized_solver_duty].
- Period ids must be unique and match `N_periods` [metadata.py:_period_ids].
- At least one hot and one cold utility must exist on the solver axes [metadata.py:_utilities].
- Split fractions outside zero to one (beyond tolerance) raise `ValueError`; within tolerance they are clamped [period_state.py:_recovery_split_fraction].
- When segment area contributions exist, the aggregate `area` is set to `None` [recovery.py:_recovery_exchangers, utility.py:_hot_utility_exchangers].
- Recovery source inlet is read at stage boundary `k` of the hot stream and sink inlet at `k + 1` of the cold stream [period_state.py:_recovery_period_state].
- Without explicit branch outlet temperatures, the isothermal-mixing convention applies: outlets equal the stage outlet boundary [period_state.py:_hot_recovery_outlet, _cold_recovery_outlet].
- Capital cost prefers explicit model values, then component sums, then `TAC` minus utility cost when the two disagree [metadata.py:_capital_cost].
- Model codes `PDM`, `TDM`, `ESM` map to `HeatExchangerNetworkDesignMethod` values [metadata.py:_result_method].
- Segment slicing needs stage boundary temperatures and raises otherwise [segment_area.py:_recovery_segment_contributions].

## Side Effects
- None. The solved model and solver arrays are read, not mutated [service.py:extract_heat_exchanger_network].

## Verification
- Run: `uv run --no-sync pytest tests/analysis/heat_exchanger_networks/test_pinch_design_method.py tests/analysis/heat_exchanger_networks/test_multi_utility.py`
- There is no extraction-specific test file; these tests exercise extraction through the synthesis methods.
