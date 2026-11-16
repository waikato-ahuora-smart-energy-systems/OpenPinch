# Heat Exchanger Network Analysis

## Overview
Heat exchanger network (HEN) synthesis, fixed-structure duty optimisation and controllability scoring. `service.py` dispatches the OpenHENS family of methods (pinch design, heuristic and MILP, thermal derivative, network evolution), `context.py` prepares settings and finalises the cached design, and the subpackages split planning, execution, extraction, reporting (including ranking), results, models, solver and targeting. Equation models live in `models`. The synthesis and duty-optimisation entry points are reached through `PinchProblem.design`; controllability is a standalone function.

## Interface
- `heat_exchanger_network_synthesis_service` and per-method `heat_exchanger_network_*_method_service` functions (`service.py`): dispatch one method and update the problem cache.
- `prepare_service_context`, `finalise_design_result`, `ensure_target_results`, `normalise_runtime_options`, `optional_text` (`context.py`): shared service-boundary helpers.
- `heat_exchanger_network_duty_optimisation_service`, `build_duty_optimisation_request`, `DutyOptimisationRequest`, `fixed_network_structure`, `FixedNetworkStructure`, `FixedStructureDutyExecutor`, `build_solution_network`, `verify_solution` (not in `__all__`), `duty_optimisation_task`, `DUTY_OBJECTIVE_GOALS`, `DEFAULT_FEASIBILITY_APPROACH` (`duty_optimisation.py`): optimise exchanger duties on a user-defined network for `"utility"`, `"area"`, `"cost"` or `"rating"` (installed recovery areas).
- `apply_overdesign`, `validate_margin`, `with_scaled_recovery_areas`, `OPTIMISE` (`overdesign.py`): an overdesign margin on the recovery areas, rated over every period, fixed or optimised.
- `with_stream_end_utilities` (`period_fit.py`): a network designed on one basis, plus a heater or cooler at each stream end that has none, so a fit over every period can close each period's balance.
- `quantify_heat_exchanger_network_controllability` (`controllability.py`): 0 to 1 steady-state controllability score for a solved `HeatExchangerNetwork`, from an interaction matrix derived from exchanger duties.
- `PreparedProblem`, `SynthesisProblem` (`prepared_problem.py`): structural protocols for the problem state analysis code reads.
- `WorkflowContractError` (`errors.py`): raised when a synthesis workflow breaks its task contract or no task outcome succeeds.
- `build_index_grid`, `ordered_mapping_keys` (`indexing.py`): nested index grids and index-ordered mapping keys.
- Subdirectories:
  - [execution](./execution/README.md): settings, search levels, executors and task-builder helpers.
  - [extraction](./extraction/README.md): solved model to domain network.
  - [basis](./basis/README.md): the median design basis, clustered operating periods and their synthetic single-period case.
  - [heuristic](./heuristic/README.md): rules-based (Linnhoff) pinch design topology proposals.
  - [models](./models/README.md): concrete HEN equation models.
  - [reporting](./reporting/README.md): feasibility checks, ranking and exports.
  - [results](./results/README.md): result assembly, seeds and selection.
  - [solver](./solver/README.md): solver arrays, piecewise profiles, backends and dependency guards.
  - [targeting](./targeting/README.md): method orchestration and topology helpers.

## Invariants
- Runtime options reaching this package must be a dict or `None` and must not contain `HENS_` keys; persistent controls are loaded with the problem. The `problem.design` accessors move `HENS_` keys from a call's `options` into that call's configuration before this check, so users may pass them there [context.py:normalise_runtime_options].
- Open-HENS and PDM reject `initial_networks`, and the heuristic pinch design method is not a method of its own (PDM runs it); `method=None` and `OpenHENS` run the full workflow at the configured search width and depth [service.py:heat_exchanger_network_synthesis_service].
- An accepted design is ranked, selected and verified; verification failures raise `RuntimeError` [context.py:finalise_design_result].
- Analysis code accepts the `PreparedProblem` protocol instead of importing the application layer [prepared_problem.py:PreparedProblem].
- Duty optimisation: `"area"` needs a utility cap, `"cost"` and `"utility"` reject caps, and every process stream needs at least one exchanger, checked at solve time (including after each idle-exchanger removal) so a violation surfaces as a failed task and `WorkflowContractError` from the service [duty_optimisation.py:build_duty_optimisation_request] [duty_optimisation.py:_require_every_stream_served].
- Overdesign scales recovery areas only, by `1 + OD/100`; heaters and coolers keep their design areas. `"optimise"` rates both ends of `[0, maximum]` and then runs a bounded `minimize_scalar` search (0.5 point tolerance, at most 15 more solves), choosing the cheapest point evaluated, a tie to the smaller margin. The metrics hold mode, percentage, each period's recovery and utility before and after, the cost at 0 and at the margin, and every `[percentage, cost]` point [overdesign.py:apply_overdesign].
- The `"rating"` objective needs a positive `area` on every exchanger and rejects utility caps. Each exchanger's capacity is scaled per period by the exact over Chen LMTD at its design approaches, so the installed area, sized with the exact LMTD, carries its design duty; a rated segmented exchanger keeps its installed area [duty_optimisation.py:build_duty_optimisation_request, _lmtd_factor, build_solution_network].
- Stream-end padding first drops any end utility of the network that cannot keep the approach at both ends in every period its stream runs, then gives each stream with no end utility the cheapest utility (period-weighted price) that can; a stream no utility reaches gets none, and staged utilities and recovery exchangers are kept [period_fit.py:with_stream_end_utilities].
- Exchangers left at zero duty in every period are removed one per pass, the idle one with the smallest `max_duty` first, and the allocation is re-solved until none are idle [duty_optimisation.py:FixedStructureDutyExecutor._solve].
- The numeric controllability options are validated, component scores lie in 0 to 1, and the composite uses fixed weights renormalised over the components that are not `None` [controllability.py:_validate_options, _COMPOSITE_WEIGHTS].

## Side Effects
- `finalise_design_result` assigns `problem._results` with the design attached [context.py:finalise_design_result].
- `ensure_target_results` runs direct heat integration through `problem.target` when no matching targets are cached [context.py:ensure_target_results].
- Synthesis runs solver models in a `ProcessPoolExecutor` through `LocalSynthesisExecutor` when more than one worker is available and the main script has a main guard (or the start method is fork); duty optimisation solves serially in-process through `FixedStructureDutyExecutor`. Both launch external solver executables via Pyomo [execution/executor.py:LocalSynthesisExecutor, duty_optimisation.py:FixedStructureDutyExecutor].
- Controllability and indexing are pure.

## Verification
- Run: `uv run --no-sync pytest tests/analysis/heat_exchanger_networks`
- Controllability: `uv run --no-sync pytest tests/analysis/test_heat_exchanger_network_controllability.py`
