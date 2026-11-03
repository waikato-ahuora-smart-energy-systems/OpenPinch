# HEN Extraction

## Overview
Converts a solved equation model (and the solver arrays it was built from) into identity-labelled domain objects: a `HeatExchangerNetwork` of `HeatExchanger` records with per-period states, and the synthesis result contract. The solved model is read only through `getattr`, so both period-native and single-period model attributes are accepted. It sits between `../models` / `../solver` and `../results`.

## Interface
- `extract_heat_exchanger_network` (`service.py`): builds the network (recovery, hot-utility and cold-utility exchangers, costs, summary metrics, solver axis metadata and source metadata).
- `extract_network_synthesis_result` (`service.py`): wraps the network in a `HeatExchangerNetworkSynthesisResult` with objective values.
- Private helper modules, imported only within this package:
  - `metadata.py`: period ids, costs, summary metrics, scalar coercion, axis identities.
  - `recovery.py`: process-to-process exchangers.
  - `utility.py`: heaters and coolers.
  - `period_state.py`: per-period duty, temperatures, approaches and split fractions.
  - `segment_area.py`: segment-resolved area contributions.

## Invariants
- Exchanger ids are `recovery:<hot>-><cold>:S<stage>`, `hot-utility:<utility>-><cold>` and `cold-utility:<hot>-><utility>`; recovery stages are one-based [recovery.py:_recovery_exchangers, utility.py:_hot_utility_exchangers].
- Exchangers inactive in every period are dropped unless `include_inactive` is true (only `extract_heat_exchanger_network` exposes it); an exchanger is active when its duty and (if present) its binary exceed the tolerance [metadata.py:_is_active].
- Duties below minus `max(tolerance, 1e-4)` raise `ValueError`; smaller negatives are clipped to zero [metadata.py:_normalized_solver_duty].
- Period ids must be unique and match `N_periods` [metadata.py:_period_ids].
- At least one hot and one cold utility must exist on the solver axes [metadata.py:_utilities].
- With explicit branch outlets, `X`/`Y` fractions outside zero to one raise `ValueError` and are otherwise clamped; under isothermal mixing the fraction is the duty share of the stage duty, capped at 1, or `None` [period_state.py:_recovery_split_fraction].
- When segment area contributions exist, the aggregate `area` is set to `None` [recovery.py:_recovery_exchangers, utility.py:_hot_utility_exchangers].
- Recovery source inlet is read at stage boundary `k` of the hot stream and sink inlet at `k + 1` of the cold stream [period_state.py:_recovery_period_state].
- Without explicit branch outlet temperatures, the isothermal-mixing convention applies: outlets equal the stage outlet boundary, falling back to inlet and duty over `cp` when no boundary exists [period_state.py:_hot_recovery_outlet, _cold_recovery_outlet].
- Capital cost is the explicit model value; otherwise `TAC` minus utility cost when `TAC` is at least the utility cost and the component sum is absent or disagrees; otherwise the component sum [metadata.py:_capital_cost].
- Model codes `PDM`, `TDM`, `ESM` map to `HeatExchangerNetworkDesignMethod` values [metadata.py:_result_method].
- Segment slicing of a segmented parent with positive duty needs stage boundary temperatures unless the model supplies segment area contributions, and raises otherwise [segment_area.py:_recovery_segment_contributions].

## Side Effects
- None. The solved model and solver arrays are read, not mutated [service.py:extract_heat_exchanger_network].

## Verification
- Run: `uv run --no-sync pytest tests/analysis/heat_exchanger_networks/test_pinch_design_method.py tests/analysis/heat_exchanger_networks/test_multi_utility.py`
- There is no extraction-specific test file; these tests exercise extraction through the synthesis methods.
