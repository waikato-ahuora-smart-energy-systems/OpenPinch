# HEN Solver Support

## Overview
Everything between a prepared problem and the equation models: solver arrays built from the problem, piecewise thermal profiles for segmented streams, the PDM above/below-pinch decomposition, GEKKO/Pyomo backend configuration, and optional-dependency guards. `../execution` calls these to build models; `../extraction` reads the arrays and profiles back when turning solutions into networks.

## Interface
- `problem_to_solver_arrays`, `PreparedSolverArrays`, `SEGMENT_PROFILE_VERSION`, `annual_cost_scales` (`arrays.py`): prepared-problem to padded, period-indexed arrays, axis maps, identities and unit conventions.
- `PiecewiseThermalProfile`, `profile_from_solver_arrays`, `utility_thermal_profile`, `duty_aligned_area_contributions`, `add_piecewise_temperature_mapping`, `add_piecewise_cost_mapping`, `add_piecewise_temperature_contribution_mapping` (`piecewise.py`): segment profiles, duty-aligned area slices, and model constraints mapping duty to temperature, cost or contribution.
- `build_pinch_design_decomposition`, `PinchDesignDecomposition`, `PinchDesignTarget`, `PinchLocation`, `StageSelection` (`pinch_design_decomposition.py`): per-period pinch targets and clipped active-stream fields for the PDM.
- `SolverRun`, `create_gekko_model`, `cleanup_gekko_model`, `require_solver_backend`, `configure_gekko_solver`, `solve_gekko_model`, `suppress_gekko_numpy_array_copy_deprecation`, `PYOMO_SOLVERS`, `GEKKO_SOLVERS`, `DEFAULT_IPOPT_GEKKO_OPTIONS` (`backend.py`): solver selection, option files and normalised solves.
- `require_synthesis_dependency`, `require_declared_synthesis_dependencies`, `require_solver_binary`, `SYNTHESIS_DEPENDENCIES`, `SynthesisDependency`, `SYNTHESIS_EXTRA`, `MissingSynthesisDependencyError`, `MissingSynthesisSolverError` (`dependencies.py`): optional package and executable guards.

## Invariants
- Arrays accept only a live prepared problem with a `Zone` master zone, a finite positive `dTmin`, and at least one hot and cold stream and utility [arrays.py:problem_to_solver_arrays].
- Units: temperatures K, heat capacity flowrate kW/K, HTC kW/m^2/K, utility price $/kW/y (price times `COSTING_ANNUAL_OP_TIME / 1000`), capital coefficients annualised by the capital recovery factor [arrays.py:annual_cost_scales, _solver_array_mapping].
- Stream and utility order follows input order; a default backup utility is dropped when real ones exist [arrays.py:_ordered_utility_items].
- Temperature contributions are scaled by `dTmin / (2 * THERMAL_DT_CONT)` and fall back to `dTmin / 2` when the scaled value is not positive [arrays.py:_temperature_contribution].
- Segmented parents carry a zero heat-capacity-flowrate sentinel; segment tensors are authoritative [arrays.py:_stream_heat_capacity_flowrate].
- A piecewise profile needs positive duties, CPs and HTCs, one direction and continuous temperatures [piecewise.py:PiecewiseThermalProfile.__post_init__].
- Exchanger duty beyond a profile raises, except for solver noise within a relative allowance that is dropped [piecewise.py:duty_aligned_area_contributions].
- The PDM decomposition works on a deep copy of the master zone, and its stage boundary count satisfies `K = S + 1` [pinch_design_decomposition.py:_zone_with_hen_dt_contribution, PinchDesignDecomposition].
- Shifted pinch temperature is the selected pinch (cold pinch when the hot utility target is zero, otherwise hot pinch) in K [pinch_design_decomposition.py:_shifted_pinch_temperature].
- Solver names must be in `PYOMO_SOLVERS | GEKKO_SOLVERS`; solver options are validated [backend.py:require_solver_backend, _normalise_solver_options].
- Pyomo solves that need an option file are serialised by a process-wide lock [backend.py:_solver_working_directory].

## Side Effects
- Pyomo solves with an option file (solver options supplied) change the process working directory under a per-process thread lock, restoring it afterwards [backend.py:_solver_working_directory].
- Writes `couenne.opt` or `ipopt.opt` into the GEKKO run directory and mutates the model's options [backend.py:_write_solver_option_file, configure_gekko_solver].
- Deletes a GEKKO run directory and recreates it on re-solve [backend.py:cleanup_gekko_model, _restore_run_directory].
- Imports optional packages (`pyomo`, `gekko`, `idaes`) lazily and may prepend the IDAES bin directory to `PATH` [dependencies.py:require_synthesis_dependency, require_solver_binary, _prepend_path].
- Launches solver executables through GEKKO/Pyomo [backend.py:solve_gekko_model].

## Verification
- Run: `uv run --no-sync pytest tests/analysis/heat_exchanger_networks/test_heat_exchanger_network_array_adapter.py tests/analysis/heat_exchanger_networks/test_segmented_streams.py tests/analysis/heat_exchanger_networks/test_solver_option_validation.py tests/analysis/heat_exchanger_networks/test_synthesis_dependency_boundaries.py`
