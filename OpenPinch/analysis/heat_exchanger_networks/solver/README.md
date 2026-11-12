# HEN Solver Support

## Overview
Everything between a prepared problem and the equation models: solver arrays built from the problem, piecewise thermal profiles for segmented streams, the PDM above/below-pinch decomposition, the native Pyomo model builder, Pyomo solver configuration, and optional-dependency guards. `../execution` calls these to build models; `../extraction` reads the arrays and profiles back when turning solutions into networks.

## Interface
- `problem_to_solver_arrays`, `PreparedSolverArrays`, `SEGMENT_PROFILE_VERSION`, `annual_cost_scales` (`arrays.py`): prepared-problem to padded, period-indexed arrays, axis maps, identities and unit conventions.
- `PiecewiseThermalProfile`, `profile_from_solver_arrays`, `utility_thermal_profile`, `duty_aligned_area_contributions`, `add_piecewise_temperature_mapping`, `add_piecewise_cost_mapping`, `add_piecewise_temperature_contribution_mapping` (`piecewise.py`): segment profiles, duty-aligned area slices, and model constraints mapping duty to temperature, cost or contribution.
- `build_pinch_design_decomposition`, `PinchDesignDecomposition`, `PinchDesignTarget`, `PinchLocation`, `StageSelection` (`pinch_design_decomposition.py`): per-period pinch targets and clipped active-stream fields for the PDM.
- `SolverRun`, `cleanup_model_run_directory`, `require_solver_backend`, `configure_pyomo_solver`, `solve_pyomo_model`, `PYOMO_SOLVERS`, `INTEGER_SOLVERS`, `PYOMO_SOLVER_MODULES`, `PYOMO_SOLVER_FACTORY_NAMES`, `SOLVE_TIME_OPTION_SOLVERS`, `DEFAULT_IPOPT_PYOMO_OPTIONS`, `CONVERGED_TERMINATION_CONDITIONS`, `PASSING_SOLVER_STATUSES`, `PASSING_TERMINATION_CONDITIONS` (`backend.py`): solver selection, option files and normalised solves of native Pyomo models.
- `PyomoModel` (`pyomo_model.py`): a Pyomo `ConcreteModel` built through a small modelling interface (`Var`, `Param`, `Equation`, `Equations`, `Intermediate`, `sum`, `Minimize`, `Maximize`), so the HEN equation helpers build native Pyomo components. Every HEN model uses it.
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
- A stream is active on a side of the pinch only if it reaches more than `tol` past its pinch temperature, so a stream that starts or ends at the pinch is not kept on the far side as a sliver [pinch_design_decomposition.py:_clip_stream_temperatures].
- Solver names must be in `PYOMO_SOLVERS` (`couenne`, `highs`, `ipopt-pyomo`); any other name, including the removed `apopt` and `ipopt-GEKKO`, raises `MissingSynthesisSolverError` listing the supported names. Solver options are validated [backend.py:require_solver_backend, _normalise_solver_options].
- Pyomo solves that need an option file are serialised by a process-wide lock [backend.py:_solver_working_directory].
- `SolverRun.converged` is false for a solve that stopped at a termination condition outside `CONVERGED_TERMINATION_CONDITIONS` (an iteration or time limit, for example), though such a solve can still pass, and true when none was recorded [backend.py:SolverRun.converged].
- A `PyomoModel` has at most one objective; a second `Minimize` or `Maximize` raises `ValueError`. Its parameters are fixed variables, so they can be changed between solves; `Intermediate` returns the expression itself, and names are not used [pyomo_model.py:PyomoModel].
- A native Pyomo solve fails, with status 0, on a solver status outside `PASSING_SOLVER_STATUSES` or a termination condition outside `PASSING_TERMINATION_CONDITIONS` ("unbounded" and "noSolution" are outside it, as they carry no solution) or on an exception; otherwise its status is 1, values are loaded onto the variables and `objective_value` is the objective's value [backend.py:solve_pyomo_model].
- `configure_pyomo_solver` accepts only `PYOMO_SOLVERS` [backend.py:configure_pyomo_solver].
- `ipopt-pyomo` defaults to `bound_push` and `bound_frac` of 1e-6 under the given options, and with `highs` takes its options at solve time (Ipopt on its command line), so neither writes an option file or takes the working-directory lock [backend.py:configure_pyomo_solver, solve_pyomo_model].
- `highs` needs the `highspy` package and no executable; it runs in process through Pyomo's `appsi_highs` interface, takes its options at solve time and writes no option file. It rejects nonlinear models [backend.py:require_solver_backend, configure_pyomo_solver, solve_pyomo_model].

## Side Effects
- Pyomo solves with an option file (solver options supplied) change the process working directory under a per-process thread lock, restoring it afterwards [backend.py:_solver_working_directory].
- For a `PyomoModel` with Couenne options, creates a temporary run directory as the model's `_path` and writes `couenne.opt` into it [backend.py:configure_pyomo_solver, _write_solver_option_file].
- Deletes a model's run directory and recreates it, with its option file, on re-solve [backend.py:cleanup_model_run_directory, _restore_run_directory].
- Imports optional packages (`pyomo`, `idaes`, `highspy`) lazily and may prepend the IDAES bin directory to `PATH` [dependencies.py:require_synthesis_dependency, require_solver_binary, _prepend_path].
- Launches solver executables, or HiGHS in process, through Pyomo [backend.py:solve_pyomo_model].
- `pyomo_model.py` imports `pyomo.environ` only when a `PyomoModel` is created or used [pyomo_model.py:_pyomo].

## Verification
- Run: `uv run --no-sync pytest tests/analysis/heat_exchanger_networks/test_heat_exchanger_network_array_adapter.py tests/analysis/heat_exchanger_networks/test_segmented_streams.py tests/analysis/heat_exchanger_networks/test_solver_option_validation.py tests/analysis/heat_exchanger_networks/test_synthesis_dependency_boundaries.py tests/analysis/heat_exchanger_networks/test_pyomo_model.py`
