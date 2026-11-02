# Direct Gas MVR

## Overview
Solves mechanical vapour recompression (MVR) of a hot gas or vapour process stream directly with CoolProp. Each compression stage raises the stream's pressure, cools it to the stream target and is returned as a linearised hot replacement stream plus a stage result with work and heat flow. `../process_mvr.py` and `../_process_mvr` consume it.

## Interface
`__init__.py` re-exports nothing; import from the modules.
- `solve_direct_gas_mvr_stream` (`execution.py`): solve all stages for one source stream at one period index and return `DirectGasMVRStreamSolveResult`.
- `coerce_positive_mvr_stage_count` (`execution.py`): validate a stage count as a positive integer.
- `DirectGasMVRSettings` (`models.py`): stage count, stage temperature lift or pressure ratio, liquid injection, efficiencies, `dt_diff_max`.
- `DirectGasMVRStageResult`, `DirectGasMVRStreamSolveResult` (`models.py`): per-stage accounting and the solve result.
- `DirectGasMVROutputUnits` (`models.py`): frozen output units.
- `DirectGasMVRFallbackDiagnostic`, `DirectGasMVRStageError` (`models.py`): named fallback evidence and a bounded stage error.
- Default constants `DEFAULT_*` (`models.py`); `execution.py` repeats them and adds `DEFAULT_MVR_STAGE_T_SAT_LIFT`, `MAX_DIRECT_MVR_STAGE_T_LIFT` and `MAX_DIRECT_MVR_STAGE_PRESSURE_RATIO`.
- `thermodynamics.py`: `source_enthalpies`, `state_property_at_temperature_pressure`, `find_pressure_for_actual_discharge`, `actual_outlet_at_pressure`, `build_cooling_temperature_enthalpy_curve`, `profile_enthalpy_values`.
- `units.py`: unit conversion helpers (`to_pascal`, `to_watt`, `to_kilowatt`, `convert_value`, `source_output_units`, `profile_to_output_units`, `stage_mass_flow`, and others).

## Invariants
- Specify at most one of `mvr_stage_t_lift` and `mvr_stage_pressure_ratio`; the default lift is 20 K, capped at 200 K, and the pressure ratio must be in (1, 20] [execution.py:_resolve_compression_target].
- `eta_mvr_comp` and `eta_motor` must be in (0, 1] [execution.py:_validate_efficiencies].
- The source needs positive duty, `p_supply`, finite states and `t_supply > t_target` [execution.py:solve_direct_gas_mvr_stream].
- Stage count must be a finite integer of at least 1; booleans are rejected [execution.py:coerce_positive_mvr_stage_count].
- Thermodynamic calculations run in SI (Pa, K, J/kg, W); results are converted to the source stream's units, falling back to degC, kPa, kJ/kg, kW [units.py:source_output_units, execution.py:_solve_compression_stage].
- Compressor work is `m_dot * (h_out - h_in) / eta_motor`; the actual outlet uses isentropic efficiency [execution.py:_solve_compression_stage, thermodynamics.py:actual_outlet_at_pressure].
- Liquid injection applies only when the target is below the outlet saturation temperature, which is below the discharge temperature (the saturated vapour enthalpy must also lie between the stage target and the outlet enthalpy); it falls back to a dry stage, recorded as a `dry_stage` diagnostic, when the saturation state is unavailable [execution.py:_solve_compression_stage].
- CoolProp failures are re-raised as `DirectGasMVRStageError`, which is picklable and bounds long context strings to 80 characters [execution.py:solve_direct_gas_mvr_stream, models.py:DirectGasMVRStageError].
- Fallback diagnostic summaries are 1 to 160 single-line characters [models.py:DirectGasMVRFallbackDiagnostic].
- Each stage's inlet is the previous stage's outlet pressure and the stream target temperature; the next stage's mass flow is the previous hot mass flow [execution.py:solve_direct_gas_mvr_stream].

## Side Effects
- CoolProp `PropsSI` calls throughout, except that the source states use the stream's own `supply_enthalpy` and `target_enthalpy` when both are present; otherwise they fall back to saturated vapour at supply and saturated liquid at target [thermodynamics.py:source_enthalpies, execution.py:_solve_compression_stage].
- None otherwise; the source `Stream` is not modified.

## Verification
- Run: `uv run --no-sync pytest tests/analysis/heat_pumps/test_direct_mvr.py`
