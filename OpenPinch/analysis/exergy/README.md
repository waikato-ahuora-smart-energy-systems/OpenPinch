# Exergy Targeting

## Overview
Enriches an existing thermal target with exergy outputs: the exergetic grand composite curve, exergy surplus and deficit net load profiles, source and sink totals, and the exergy transfer effectiveness of the balanced composite curves. It does not create base targets; it needs one already present for the requested period. All of the logic is in `service.py`.

## Interface
- `run_exergy_targeting_service(zone, args, *, apply_func)` (`service.py`): enriches a clone of the first compatible target and adds it to the zone.
- `apply_exergy_targeting(target, *, dt_cont_multiplier)` (`service.py`): writes `GCC_X` and `NLP_X` graphs and scalar exergy fields onto a target.
- `compute_exergetic_temperature(T, T_ref_in_C, units_of_T)` (`service.py`): `T_amb * (ratio - 1 - ln(ratio))` for a temperature against a reference.
- `build_exergy_gcc_curve(...)` (`service.py`): converts a GCC-like curve into exergetic temperature and `X_GCC` columns.
- `build_exergy_nlp_curves(...)` (`service.py`): aggregates exergy surplus and deficit profiles and their totals from hot and cold branches.
- Subdirectories: none.

## Invariants
- Default base-target order is Indirect Integration, Indirect Heat Pump, Direct Heat Pump, Direct Integration; an explicit type that is missing or stale for the period raises `RuntimeError`. [service.py:_EXERGY_TARGET_ORDER] [service.py:run_exergy_targeting_service]
- Exergy is computed on real temperatures: shifted curves are moved back interval by interval using the CP-weighted `dt_cont` of the zone's streams, falling back to `THERMAL_DT_CONT` times the zone multiplier. [service.py:_stream_interval_shifts] [service.py:apply_exergy_targeting]
- Total Site utility profiles (Indirect targets) and heat pump stream columns are already on real temperatures and are not shifted. [service.py:_resolve_target_exergy_spec]
- Absolute temperature must be above 0 K and the unit must be `"C"` or `"K"`. [service.py:compute_exergetic_temperature]
- Intervals are split where they cross the environment temperature. [service.py:_insert_breaks]
- The exergy transfer effectiveness is gained over released exergy, and 0 when nothing is released. [service.py:apply_exergy_targeting]
- Values within `tol` of zero are set to 0.0 in outputs. [service.py:_round_small_noise]
- The environment temperature is read from `config.environment.temperature`, defaulting to 15 degC. [service.py:_resolve_target_exergy_spec]

## Side Effects
- `apply_exergy_targeting` mutates the target's `graphs` and its `exergy_sources`, `exergy_sinks`, `exergy_released`, `ETE`, `exergy_req_min` and `exergy_des_min` fields. [service.py:apply_exergy_targeting]
- The service deep-copies the base target, applies the zone's `ENV_` settings to the copy, adds it to the zone without invalidating dependents, and sets `zone._selected_exergy_target_type`. [service.py:run_exergy_targeting_service]

## Verification
- Run: `uv run --no-sync pytest tests/analysis/test_exergy_analysis.py tests/analysis/test_exergy_targeting.py`
