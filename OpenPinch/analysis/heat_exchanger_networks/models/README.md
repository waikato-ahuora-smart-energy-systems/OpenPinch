# HEN Equation Models

## Overview
Concrete equation models for heat exchanger network synthesis, plus the internal problem shell that builds, solves and reduces them. `BaseHeatExchangerNetworkModel` holds shared setup; `StageWiseModel` and `PinchDecompModel` are the superstructure models; `FixedStructureModel` allocates duty on a user-fixed network. `../execution/executor.py` and `../duty_optimisation.py` import from here, and `../extraction` converts solved models to network results. The package `__init__` is a docstring only, so import model modules directly.

## Interface
- `BaseHeatExchangerNetworkModel` (`base.py`): abstract base. The constructor stores options, creates the GEKKO model, then calls abstract `setup`; also provides `optimise`, `cleanup`, `get_alpha_values`, `set_match_restrictions` and utility-cost helpers by delegating to `_base`. Subclasses implement `setup`, `set_preprocessing`, `set_stage_wise_superstructure`, `set_obj`, `get_post_process`.
- `StageWiseModel` (`stagewise.py`): TDM, ESM and amalgamated PDM model; delegates to `_stagewise` (evolution, verification, ranking, warm start).
- `PinchDecompModel` (`pinch_decomposition.py`): one pinch side; delegates to `_pinch_design`; `amalgamate_networks` joins two sides into a `StageWiseModel`. For the Pyomo solvers (`highs`, `couenne`, `ipopt-pyomo`) it is a native Pyomo model (`../solver/pyomo_model.py`); for `apopt` and `ipopt-GEKKO` a GEKKO model. A `highs` model that is not linear is solved with Couenne instead, without the HiGHS options, and logs a warning. The stage-wise and fixed-structure models stay on GEKKO.
- `StagePackedStageWiseModel` (`packed_stagewise.py`), `StagePackedPinchDecompModel` (`packed_pinch_design.py`): the above with stage-packing constraints added after the superstructure.
- `add_recovery_stage_packing_constraints` (`stage_packing.py`): forces active recovery stages to be the first stages and imposes a minimum duty on each active match and stage.
- `InternalHeatExchangerNetworkProblem`, `ModelSliceUnavailableError` (retained, never raised), `FrameworkName` (`problem.py`): problem dataclass with `load_model`, `get_solution`, `remove_unused_stages`, `extract_network`, `extract_result`.
- `FixedStructureModel`, `FixedStructureSpec`, `RecoveryMatch`, `UtilityMatch`, `ExchangerResult`, `ExchangerPeriodResult`, `FIXED_STRUCTURE_GOALS`, `HEATER`, `COOLER` (`fixed_structure.py`): fixed-network duty NLP and its specification and result types. Public functions: `prepare_fixed_structure`, `build_fixed_structure_equations`, `set_fixed_structure_objective`, `post_process_fixed_structure`, `initial_point`, `utility_chains`, `utility_free_components`, `utility_profile`, `stream_profile`, `stream_off`, `utility_side_temperatures`, `operating_cost_expression`, `capital_cost_expression`, `fixed_unit_cost`, `chen_lmtd`, `exact_lmtd`.
- `ThermalProfile`, `SMOOTHING_K`, `smooth_max0`, `smooth_clip` (`thermal_profiles.py`): piecewise heat-temperature profile with smooth solver expressions and exact reporting functions.
- Subdirectories: [_base](./_base/README.md) shared equation, area, cost and execution helpers; [_pinch_design](./_pinch_design/README.md) pinch-side preprocessing, equations, post-processing and amalgamation; [_stagewise](./_stagewise/README.md) stagewise preprocessing, equations, objectives, evolution, verification and warm start.

## Invariants
- If `setup` raises, the GEKKO run directory is removed before the exception propagates. [base.py:BaseHeatExchangerNetworkModel.__init__]
- `StageWiseModel.setup` order: blank parameters, solver-array parameters, preprocessing, match restrictions, superstructure, dQ/dA equations (TDM only), objective. `PinchDecompModel.setup` inserts `calculate_pinch` before preprocessing and has no dQ/dA step. [stagewise.py:StageWiseModel.setup] [pinch_decomposition.py:PinchDecompModel.setup]
- `FixedStructureModel` accepts only "total utility", "total area" and "total cost", and fixes `framework="ESM"`, `non_isothermal_model=True`, `integers=False`. [fixed_structure.py:FixedStructureModel.__init__]
- Fixed-structure stages are 0-based; hot streams pass stages 0 to S-1 and cold streams S-1 to 0; utility exchangers at one position run in series (heaters coldest to hottest utility, coolers warmest to coldest). [fixed_structure.py:utility_chains]
- Fixed-structure validation raises `ValueError` for out-of-range indices, a recovery match whose hot supply is within its approach of the cold supply, a utility that cannot meet its approach at any duty, and utility-free stream groups whose loads differ. [fixed_structure.py:_check_indices] [fixed_structure.py:_check_recovery_spans] [fixed_structure.py:_check_utility_reach] [fixed_structure.py:utility_free_components]
- A stream that is off in a period has its exchanger duties fixed at zero and its balances skipped. [fixed_structure.py:_period_variables]
- Solver areas use Chen's LMTD; reported design area uses the exact LMTD and is the maximum over periods. [fixed_structure.py:chen_lmtd] [fixed_structure.py:_design_area]
- `ThermalProfile` is frozen; its arrays must share one non-zero length and `side` must be "hot" or "cold". Smoothing applies only to symbolic arguments; plain numbers are exact. [thermal_profiles.py:ThermalProfile.__post_init__] [thermal_profiles.py:smooth_max0]
- Stage packing applies only when `integers` is true, `S > 1` and some stage has positive capacity. [stage_packing.py:add_recovery_stage_packing_constraints]
- PDM construction needs "above" and "below" decompositions; stage-wise construction needs an explicit stage count or a parent solution. [problem.py:InternalHeatExchangerNetworkProblem._build_pdm] [problem.py:InternalHeatExchangerNetworkProblem._build_stage_wise]
- The amalgamated PDM case is built, not solved, so it takes the termination condition of the first required side whose solve did not converge; otherwise it keeps its own solver run. [problem.py:InternalHeatExchangerNetworkProblem._carry_unconverged_side_solve]
- `get_solution` converts `ValueError` into `None` with `solution_failure_reason` set. [problem.py:InternalHeatExchangerNetworkProblem.get_solution]
- Stage reduction keeps stages whose share of peak recovery meets a threshold (0.1, 5, 8, 10 percent for at most 3, 5, 8, 10 stages and 0 percent for more than 10; PDM and TDM solves only) and returns the original case unless the reduced model solves and verifies. [problem.py:InternalHeatExchangerNetworkProblem._utilisation_threshold] [problem.py:InternalHeatExchangerNetworkProblem.remove_unused_stages]

## Side Effects
- Every model construction creates a GEKKO model, or for a PDM side with a Pyomo solver a native Pyomo model, and `optimise` runs a solver (see `_base`). [base.py:BaseHeatExchangerNetworkModel.setup_model] [pinch_decomposition.py:PinchDecompModel.setup_model]
- `get_solution` removes the GEKKO run directories of the above, below, unreduced and final cases in a `finally` block, except the amalgamated model when amalgamation raises. [problem.py:InternalHeatExchangerNetworkProblem.get_solution]
- `problem.py` sets `args`, `above`, `below`, `case` and `solution_failure_reason` on the dataclass instance. [problem.py:InternalHeatExchangerNetworkProblem.load_model]
- `remove_unused_stages` builds and solves a reduced `StageWiseModel` with the "apopt" solver. [problem.py:InternalHeatExchangerNetworkProblem.remove_unused_stages]
- `PinchDecompModel` copies the original stream temperatures to `*_OG` attributes. [pinch_decomposition.py:PinchDecompModel.get_model_parameters_from_solver_arrays]
- Stage packing adds binaries and equations to the model it receives. [stage_packing.py:add_recovery_stage_packing_constraints]
- `problem.py` imports extraction at module level and `PinchDecompModel` lazily; `_stagewise/verification.py` lazily imports extraction and reporting. [problem.py:InternalHeatExchangerNetworkProblem._model_factory]

## Verification
- Run: `uv run --no-sync pytest tests/analysis/heat_exchanger_networks/test_stage_packing.py tests/analysis/heat_exchanger_networks/test_duty_optimisation.py tests/analysis/heat_exchanger_networks/test_stage_reduction_regression.py`. There is no `tests/` directory mirroring `models/`, so these sit in the nearest parent, `tests/analysis/heat_exchanger_networks`.
