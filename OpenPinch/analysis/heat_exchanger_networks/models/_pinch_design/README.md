# Pinch-Decomposition Model Helpers

## Overview
Private helper modules for `PinchDecompModel` (`../pinch_decomposition.py`), which models one side (above or below the pinch) of a pinch-design problem. They read targets from a `PinchDesignDecomposition`, build the side superstructure and objective, extract results after a solve, and join the two solved sides into one `StageWiseModel`. They build on the shared helpers in `../_base`.

## Interface
- `calculate_pinch`, `set_preprocessing` (`preprocessing.py`): read period targets and shifted pinch temperatures, then derive stream limits, duties, U values, `Q_max`, and the `z_feasible`, `z_hu_feasible`, `z_cu_feasible` flags. Private helpers used by post-processing: `_active_period_flag`, `_lmtd_formula_allowed`, `_area_from_heat_load`.
- `set_stage_wise_superstructure`, `set_obj` (`equations.py`): side variables, balances, approach constraints, binaries and the objective. Parent also calls `_set_multiperiod_stage_wise_superstructure`.
- `used_units`, `exclude_units` (`equations.py`): a solved side's used recovery, heater and cooler units as a `{"z", "z_hu", "z_cu"}` pattern of flags, and a no-good cut that makes an unsolved side differ from such a pattern in at least one unit.
- `get_post_process` (`postprocess.py`): extracts duties, approaches, LMTD, areas, units and TAC after a successful solve. Parent also calls `_get_multiperiod_post_process` and `_active_binary_value`.
- `amalgamate_networks` (`amalgamation.py`): builds a `StageWiseModel` from solved above and below sides.
- Subdirectories: none.

## Invariants
- The decomposition's pinch location must equal the model's `pinch_loc`; target count and period identities must match the model's periods; every period needs a shifted pinch temperature. [preprocessing.py:calculate_pinch]
- A side is required only when its utility target (hot above, cold below) exceeds `tol` in some period. [preprocessing.py:calculate_pinch]
- Heaters are feasible only above the pinch and coolers only below, on streams active on that side. [preprocessing.py:set_preprocessing]
- Stage count is `max(sum(z_i_active), sum(z_j_active))` unless the decomposition carries a manual selection. [preprocessing.py:_set_multiperiod_preprocessing]
- A stream absent from this side in a period has zero duty and no recovery approach constraint in that period. [preprocessing.py:_set_multiperiod_preprocessing] [equations.py:_set_multiperiod_stage_wise_superstructure]
- Supported goals are "hot utility", "cold utility", "total utility", "heat recovery" and "min units"; any other raises `ValueError`. The hot and cold utility goals also constrain total duty to be at least the target. [equations.py:set_obj]
- A side has no coolers above the pinch and no heaters below it, so its utility is held at the target by its energy balance whatever the goal, "min units" included. [preprocessing.py] [equations.py:set_obj]
- `utility_cost_cap` (constructor argument): with a choice of utilities, "min units" then caps utility cost at this value and minimises the recovery matches alone, rather than adding them to a utility cost whose scale would hide them within a MIP gap. [equations.py:set_obj]
- The model has exactly one objective. With one hot and one cold utility it is the goal. With more than one utility on either side, the utility goals minimise utility cost alone; heat recovery minimises utility cost plus unrecovered hot-stream heat (total hot-stream load less recovery); min units minimises utility cost plus the recovery binaries. [equations.py:set_obj]
- With integers, each recovery, heater and cooler duty is switched by its binary through `Q <= M * z`, with M the duty's upper bound (`Q_max_period`, `Qtot_sh_period`, `Qtot_sc_period`), so the model is linear for unsegmented streams. [equations.py:_set_multiperiod_stage_wise_superstructure]
- Amalgamation stacks above-pinch stages first, then below-pinch stages; it raises `ValueError` if a required side did not solve, and requires equal period counts. [amalgamation.py:amalgamate_networks] [amalgamation.py:_copy_recovery_match]
- The amalgamated model is created with `integers=True`, `tol=1e-3` and goal "total utility" (reduced to "hot utility" or "cold utility" when one side is not required); its `z_allowed` is every match with duty above `owner.tol`. [amalgamation.py:amalgamate_networks]
- Post-processing LMTD and area use the shared exact-LMTD helper; design area is the maximum over periods. `TAC_model` is the solve's objective value. [postprocess.py:_get_multiperiod_post_process]
- Amalgamation reads solved side values through `_stagewise.verification._value`, so plain values and Pyomo components amalgamate alike. [amalgamation.py:amalgamate_networks]

## Side Effects
- `amalgamate_networks` constructs a `StageWiseModel` (a new Pyomo model) and writes solved values into its variables through `_base.execution._assign_value`. [amalgamation.py:amalgamate_networks] [amalgamation.py:_copy_recovery_match]
- When the above side is not required but the below side is, `amalgamate_networks` sets `above_case.S` and `above_case.K` to 0, mutating its argument. [amalgamation.py:amalgamate_networks]
- Post-processing sets many attributes on the model (`z`, `theta_*`, `LMTD_r`, `area_r`, `TAC`). [postprocess.py:_get_multiperiod_post_process]
- Imports `StageWiseModel` from `../stagewise` at module load and `_value` from `../_stagewise/verification`. [amalgamation.py] [postprocess.py]

## Verification
- Run: `uv run --no-sync pytest tests/analysis/heat_exchanger_networks/test_pinch_design_method.py tests/analysis/heat_exchanger_networks/test_multiperiod_pinch_design.py tests/analysis/heat_exchanger_networks/test_pyomo_pinch_design.py` (no directory mirrors `models/_pinch_design`). The last covers the native Pyomo PDM model.
