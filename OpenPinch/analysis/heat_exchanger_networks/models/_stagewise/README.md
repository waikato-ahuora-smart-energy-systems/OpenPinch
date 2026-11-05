# StageWise Model Helpers

## Overview
Private helper modules for `StageWiseModel` (`../stagewise.py`), which delegates each lifecycle step to one module here: preprocessing, superstructure equations, objectives, warm start, post-processing, verification and topology evolution. They rely on `../_base` for shared equations, areas and costs, and `../_pinch_design/postprocess.py` reuses `verification._value`.

## Interface
- `set_preprocessing` (`setup.py`): stage counts, total duties, U values, `Q_max`, big-M and feasibility flags.
- `set_stage_wise_superstructure`, `set_dqda_equations` (`equations.py`): variables, balances, approach constraints, isothermal or non-isothermal mixing, minimum dQ/dA. Parent also calls `_set_multiperiod_stage_wise_superstructure`.
- `set_obj` (`objectives.py`): objective for each supported goal, including total-cost forms.
- `set_initial_values_for_variables` (`warm_start.py`): copies a solved model's values into another.
- `get_post_process`, `get_lowest_benefit_HX`, `get_lowest_benefit_HX_candidates`, `get_max_benefit_HX`, `get_max_benefit_HX_candidates` (`postprocess.py`): result extraction, dQ/dA, dTAC/dA, alpha, and ranked remove or add candidates. Parent also calls `_get_multiperiod_post_process`.
- `verify` (`verification.py`): temperature, cost, area and network-feasibility checks; returns `(ok, reasons)`. `_value` unwraps GEKKO values.
- `get_net_benefit_evolution`, `get_n_minus_one_evolution`, `get_n_plus_one_evolution` (`evolution.py`): topology evolution. Parent also calls `_active_binary_value`.
- Subdirectories: none.

## Invariants
- Hot streams enter at stage boundary 0 (fixed parameter) and cold streams at boundary `S` (fixed parameter); with isothermal mixing (PDM, TDM) `theta_1` pairs `T_h[k]` with `T_c[k]` and `theta_2` pairs `T_h[k+1]` with `T_c[k+1]`; with non-isothermal mixing (ESM) `theta_1 = T_h[k] - T_c_out_y` and `theta_2 = T_h_out_x - T_c[k+1]` using the per-match branch outlets. [equations.py:_set_multiperiod_stage_wise_superstructure] [equations.py:_set_multiperiod_isothermal_approach_equations]
- Topology (`z`, `z_hu`, `z_cu`) is shared across periods; flows, temperatures and approaches are per period. [equations.py:_set_multiperiod_stage_wise_superstructure]
- Supported goals: "hot utility", "cold utility", "total utility", "utility costs", "heat recovery", "dQ/dA obj", "total cost", "variable total cost"; others raise `ValueError`. [objectives.py:set_obj]
- Solver area terms use Chen's LMTD approximation; multiperiod total cost uses one shared area per unit bounded below by each period's requirement. [objectives.py:_chen_mean] [objectives.py:_set_multiperiod_total_cost_objective]
- `verify` checks cost and area consistency only for the two total-cost goals; once those pass it also runs the reporting-layer network feasibility check. [verification.py:verify]
- Evolution candidates are usable only if solved and `verify()` passes; the best is adopted only if its TAC is below the original and it verifies. [evolution.py:_is_usable_evolution_candidate] [evolution.py:_update_with_best_model]
- With more than one add or remove branch, or a patience limit, evolution keeps at most `beam_width` branches ranked by TAC then topology signature, solves each topology once, and orders parallel results by (branch, kind, rank) so serial and parallel runs agree. [evolution.py:get_net_benefit_evolution] [evolution.py:_solve_evolution_candidates]
- With one add branch, one remove branch and no patience, the original single-path search runs. [evolution.py:get_net_benefit_evolution]
- Evolution children are relaxed (`integers=False`, `tol=1e-3`) and use the root model's solver. [evolution.py:_build_and_solve_evolution]
- Remove ranking is ascending net benefit over active matches; add ranking is descending alpha-weighted dQ/dA over inactive, feasible matches with positive value; ties break by index order. [postprocess.py:get_lowest_benefit_HX_candidates] [postprocess.py:get_max_benefit_HX_candidates]
- Warm start uses the last source period when the source has fewer periods. [warm_start.py:_set_multiperiod_initial_values]

## Side Effects
- Evolution builds and solves child `StageWiseModel` instances (GEKKO models), in a `ThreadPoolExecutor` when `max_parallel` exceeds one and `supported_workers` allows it. [evolution.py:_solve_evolution_candidates]
- `get_net_benefit_evolution` ends by calling `owner.cleanup()` on both search paths (only the early return when `owner.mSuccess != 1` skips it), and `_update_with_best_model` overwrites the owner's variables, `z_allowed`, alpha and cost attributes, then re-runs post-processing. [evolution.py:get_net_benefit_evolution] [evolution.py:_update_with_best_model]
- `verify` lazily imports and runs `extract_heat_exchanger_network` and `verify_network_feasibility`. [verification.py:_check_network_feasibility]
- Post-processing runs the post-optimisation alpha solve via `get_alpha_values`. [postprocess.py:_get_multiperiod_post_process]
- Evolution and verification log mostly at debug level; exceptions in a candidate solve are caught and logged. [evolution.py:_solve_evolution_candidate]

## Verification
- Run: `uv run --no-sync pytest tests/analysis/heat_exchanger_networks/test_stagewise_helpers.py` (no directory mirrors `models/_stagewise`; `test_network_evolution_method.py` also covers evolution).
