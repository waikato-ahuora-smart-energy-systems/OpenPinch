# Heat Exchanger Network Analysis

## Overview
Heat exchanger network (HEN) synthesis, fixed-structure duty optimisation and controllability scoring. `service.py` dispatches the OpenHENS family of methods (pinch design, thermal derivative, network evolution), `context.py` prepares settings and finalises the cached design, and the subpackages split planning, execution, extraction, ranking and solver support. Equation models live in `models`. Entry points are reached through `PinchProblem.design`.

## Interface
- `heat_exchanger_network_synthesis_service` and per-method `heat_exchanger_network_*_method_service` functions (`service.py`): dispatch one method and update the problem cache.
- `prepare_service_context`, `finalise_design_result`, `ensure_target_results`, `normalise_runtime_options`, `optional_text` (`context.py`): shared service-boundary helpers.
- `heat_exchanger_network_duty_optimisation_service`, `build_duty_optimisation_request`, `DutyOptimisationRequest`, `fixed_network_structure`, `FixedNetworkStructure`, `FixedStructureDutyExecutor`, `build_solution_network`, `verify_solution`, `duty_optimisation_task`, `DUTY_OBJECTIVE_GOALS`, `DEFAULT_FEASIBILITY_APPROACH` (`duty_optimisation.py`): optimise exchanger duties on a user-defined network for `"utility"`, `"area"` or `"cost"`.
- `quantify_heat_exchanger_network_controllability` (`controllability.py`): 0 to 1 steady-state controllability score from an interaction matrix.
- `PreparedProblem`, `SynthesisProblem` (`prepared_problem.py`): structural protocols for the problem state analysis code reads.
- `WorkflowContractError` (`errors.py`): task fan-out contract violation.
- `build_index_grid`, `ordered_mapping_keys` (`indexing.py`): nested index grids and index-ordered mapping keys.
- Subdirectories:
  - [execution](./execution/README.md): settings, pathways, executors and task-builder helpers.
  - [extraction](./extraction/README.md): solved model to domain network.
  - [models](./models/README.md): concrete HEN equation models.
  - [reporting](./reporting/README.md): feasibility checks, ranking and exports.
  - [results](./results/README.md): result assembly, seeds and selection.
  - [solver](./solver/README.md): solver arrays, piecewise profiles, backends and dependency guards.
  - [targeting](./targeting/README.md): method orchestration and topology helpers.

## Invariants
- Runtime options must be a dict and must not contain `HENS_` keys; persistent controls are loaded with the problem [context.py:normalise_runtime_options].
- Open-HENS and PDM reject `initial_networks`; `method=None` runs tier 0 and `OpenHENS` runs tier 1 [service.py:heat_exchanger_network_synthesis_service].
- Quality tier must be an integer from 0 to 5 [service.py:_validate_quality_tier].
- An accepted design is ranked, selected and verified; verification failures raise `RuntimeError` [context.py:finalise_design_result].
- Analysis code accepts the `PreparedProblem` protocol instead of importing the application layer [prepared_problem.py:PreparedProblem].
- Duty optimisation: `"area"` needs a utility cap, `"cost"` and `"utility"` reject caps, and every process stream needs at least one exchanger [duty_optimisation.py:build_duty_optimisation_request, _require_every_stream_served].
- Exchangers left at zero duty in every period are removed and the allocation is re-solved [duty_optimisation.py:FixedStructureDutyExecutor._solve].
- Controllability option values are validated, component scores are clipped to 0 to 1, and the composite uses fixed weights [controllability.py:_validate_options, _COMPOSITE_WEIGHTS].

## Side Effects
- `finalise_design_result` assigns `problem._results` with the design attached [context.py:finalise_design_result].
- `ensure_target_results` runs direct heat integration through `problem.target` when no matching targets are cached [context.py:ensure_target_results].
- Synthesis runs solver models in a `ProcessPoolExecutor` through `LocalSynthesisExecutor` when more than one worker is available; duty optimisation solves serially in-process through `FixedStructureDutyExecutor`. Both launch external solver executables via GEKKO/Pyomo [execution/executor.py:LocalSynthesisExecutor, duty_optimisation.py:FixedStructureDutyExecutor].
- Controllability and indexing are pure.

## Verification
- Run: `uv run --no-sync pytest tests/analysis/heat_exchanger_networks`
- Controllability: `uv run --no-sync pytest tests/analysis/test_heat_exchanger_network_controllability.py`
