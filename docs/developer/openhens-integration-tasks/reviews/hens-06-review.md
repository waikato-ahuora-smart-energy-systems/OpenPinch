# HENS-06 Adversarial Re-review

## Findings

No findings. HENS-06 is cleared.

## Prior P1 Resolution

The prior P1 about concrete solves bypassing source stage reduction is resolved. `InternalHeatExchangerNetworkProblem.load_model(...)` and `get_solution(...)` now reject model loading/solving immediately (`OpenPinch/services/heat_exchanger_network_synthesis/models/problem.py:46-65`), and `_raise_deferred_model_error(...)` explains that concrete PDM/TDM/ESM construction remains unavailable until HENS-07 and stage-reduction/topology-evolution semantics remain deferred to HENS-08 (`OpenPinch/services/heat_exchanger_network_synthesis/models/problem.py:102-119`). There is no longer a concrete solve path in HENS-06 that can skip the source PDM/TDM stage-reduction behavior.

The prior P1 about the PDM factory path bypassing above/below-pinch construction semantics is also resolved. Factory registrations are now ignored and reported before invocation (`OpenPinch/services/heat_exchanger_network_synthesis/models/problem.py:107-119`), so HENS-06 no longer defines a partial PDM factory contract that could collapse source above/below-pinch construction into one call.

## Coverage And Evidence

- The migration-gate tests cover PDM, TDM, and ESM load attempts and require `ModelSliceUnavailableError` (`tests/test_heat_exchanger_network_synthesis_models.py:131-145`).
- The factory-not-called coverage passes PDM `pinch_decomposition` and TDM/ESM `stagewise` factories that would fail if invoked, then asserts the HENS-06 deferred-slice error instead (`tests/test_heat_exchanger_network_synthesis_models.py:148-173`).
- The implementation notes record the re-review fix and verification commands/results: model tests `13 passed`, focused HENS/import/API set `37 passed`, ruff passed, and diff check excluding `.DS_Store` passed (`docs/developer/openhens-integration-tasks/06-equation-kernel-base-move-and-solver-boundary.md:157-174`).

## Other Review Checks

- Scope remains limited to base equation-kernel/backend setup, private problem shell, and extraction. I found no moved full `StageWiseModel`, full `PinchDecompModel`, stage-reduction implementation, topology evolution implementation, visualization/grid-diagram move, public docs example, duplicate helper replacement, public OpenHENS compatibility API, public raw-input runner, or root-exported solver service in the HENS-06 surface.
- GEKKO/Pyomo imports remain lazy. The model package imports local helpers only, `create_gekko_model(...)` imports GEKKO through the optional dependency guard, and `SolverFactory` is reached only inside `configure_gekko_solver(...)` after the Pyomo backend path is requested.
- Missing optional package and missing solver-binary errors remain actionable for general users through `openpinch[synthesis]`, solver binary, and `PATH` guidance without assuming `rtk` or `uv`.
- Solved arrays crossing the HENS-06 extraction boundary become `HeatExchangerNetwork` / `HeatExchangerNetworkSynthesisResult` payloads with OpenPinch stream identities and source/sink direction. Raw solver arrays remain excluded from the tested serialized result payload.
- The changed docs/comments satisfy the README review requirement for this slice: they describe the stable private boundary, deferred source semantics, optional dependency contract, and migration constraints without creating misleading public contracts or narrating incidental implementation trivia.
- `.DS_Store` remains outside HENS-06 scope.

## Residual Risks

- I did not rerun tests in this read-only re-review; I inspected the source and the recorded worker evidence.
- HENS-06 intentionally does not prove live solver parity. Concrete PDM/stagewise semantics, source stage reduction, topology evolution, and solver-regression evidence remain for HENS-07/HENS-08 and later solver gates.
- The task Definition of Done remains unchecked in the task file. I did not update task checkboxes because this re-review was limited to the review artifact.
