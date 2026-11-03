# Power Cogeneration

## Overview
Targets shaft power from steam turbines driven by a zone's hot utilities. A multi-stage solver (`steam_turbine.py`) works from stage temperatures and heat flows, either above the pinch (extraction from a given inlet) or below it (condensing against a sink). `service.py` selects a compatible thermal target, clones it with the zone's `POWER_` settings, and writes the work and efficiency onto the clone. Steam properties come from the sibling `thermodynamics` package.

## Interface
- `run_power_cogeneration_service(zone, args, *, refresh_services, cogeneration_func)` (`service.py`): adds a power-enriched copy of the first compatible target to the zone.
- `get_power_cogeneration_above_pinch(target, args)` (`service.py`): sets `work_target`, `turbine_efficiency_target`, `turbine_converged`, `turbine_diagnostics` and `turbine_result` on a target from its hot utilities, or returns the target untouched when no usable utility stage is found, in which case the service still adds the clone [service.py:_preprocess_utilities].
- `get_power_cogeneration_below_pinch(temperatures, heat_flows, *, config, T_sink)` (`service.py`): solves a below-pinch turbine; returns `(total_work, details)`.
- `MultiStageSteamTurbine` (`steam_turbine.py`): `solve(temperatures, heat_flows, *, mode, ...)` returns `(total_work, details)`; the `solved`, `result`, `stages` and `total_work` properties read the last result.
- Subdirectories: none.

## Invariants
- Default target order is Indirect Integration, Indirect Heat Pump, Indirect Refrigeration, Direct Heat Pump, Direct Refrigeration, Direct Integration; an explicit `base_target_type` is searched alone. [service.py:_COGENERATION_TARGET_ORDER]
- Pressures are in bar, temperatures in degC, heat flows in kW, and `load_frac` and `mech_eff` are clamped to [0, 1]. [service.py:_prepare_turbine_parameters] [steam_turbine.py:MultiStageSteamTurbine.solve]
- A hot utility is skipped as a turbine stage when it is at or above the critical temperature, has no heat flow, lies outside water's saturation range, or its saturation pressure exceeds the inlet pressure. [service.py:_preprocess_utilities]
- Stage heat flows must be finite, one-dimensional and equal in length to the temperatures; stages with heat flow at or below `tol` are dropped and the rest are ordered hottest first. [steam_turbine.py:MultiStageSteamTurbine._normalise_stage_inputs]
- Above-pinch mode needs `T_in` and `P_in`; below-pinch mode needs `T_sink`; any other mode raises `ValueError`. [steam_turbine.py:MultiStageSteamTurbine.solve]
- Stage efficiency is bounded between `min_eff` and 1; the shaft work is the fluid work times `mech_eff`, while the reported overall efficiency excludes mechanical losses. [steam_turbine.py:_apply_efficiency_limits] [steam_turbine.py:MultiStageSteamTurbine._solve_above_pinch]
- The extraction mass balance iterates at most 50 times and reports `converged`. [steam_turbine.py:_MASS_BALANCE_MAX_ITERATIONS]
- Each Varbanov model uses one size band and never switches on size; a wet back-pressure exit still switches type to condensing. A stage's fluid work outside the band adds one diagnostic per solve and does not change `converged`. [steam_turbine.py:_varbanov_band_diagnostics]
- A below-pinch sink temperature outside water's saturation range is clamped and a diagnostic is recorded. [steam_turbine.py:MultiStageSteamTurbine._solve_below_pinch]
- On a target's `turbine_result`, a stage's `source_index` is its position in the target's `hot_utilities`, not in the solver's filtered input. [service.py:_result_with_utility_indices]
- The solve result is validated against `TurbineSolveResult`; reading results before a solve raises `RuntimeError`. [steam_turbine.py:MultiStageSteamTurbine._require_solution]

## Side Effects
- A turbine solve that does not converge sets `turbine_converged` and `turbine_diagnostics` on the target and raises a `UserWarning`; the numeric targets are left as the last iteration. [service.py:get_power_cogeneration_above_pinch]
- `get_power_cogeneration_above_pinch` mutates the target it is given. [service.py:get_power_cogeneration_above_pinch]
- The service sets `zone._selected_cogeneration_target_type`, adds a target to the zone without invalidating dependents, and may call refresh services that re-run base targeting. [service.py:run_power_cogeneration_service]
- Every steam property lookup calls CoolProp through `thermodynamics.water`. [steam_turbine.py:MultiStageSteamTurbine._solve_above_pinch]

## Verification
- Run: `uv run --no-sync pytest tests/analysis/power` and `uv run --no-sync pytest tests/analysis/test_power_cogeneration_analysis.py`
