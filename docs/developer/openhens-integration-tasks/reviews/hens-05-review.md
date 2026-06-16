# HENS-05 Adversarial Re-review

## Findings

No findings. HENS-05 is cleared.

## Prior Finding Resolution

- The previous blocker on public-looking `run_synthesis_workflow(...)` is resolved. The synthesis workflow module now exposes the implementation helper as private `_execute_synthesis_workflow(...)`, and a static search over the OpenPinch HENS service/classes/lib surfaces found no `run_synthesis_workflow` symbol. Refs: `OpenPinch/services/heat_exchanger_network_synthesis/workflow.py:183`, `OpenPinch/services/heat_exchanger_network_synthesis/service.py:11`, `OpenPinch/services/heat_exchanger_network_synthesis/service.py:45`.
- The service now imports and calls `_execute_synthesis_workflow(...)`, preserving the public route through `problem.design.heat_exchanger_network_synthesis(...)` and workspace dispatch rather than adding an alternate public runner. Refs: `OpenPinch/classes/_problem/_design_accessor.py:18`, `OpenPinch/classes/_problem/_design_accessor.py:33`, `OpenPinch/classes/_workspace/execution.py:68`, `OpenPinch/classes/_workspace/execution.py:78`.
- The negative test now checks both package export absence and the concrete workflow module attribute absence for the banned name. Refs: `tests/test_heat_exchanger_network_synthesis_workflow.py:225`, `tests/test_heat_exchanger_network_synthesis_workflow.py:233`, `tests/test_heat_exchanger_network_synthesis_workflow.py:234`.

## Clearable Scope Checks

- `heat_exchanger_network_synthesis_service(problem)` remains internal-facing, problem-rooted, and absent from the root/service package exports. Refs: `OpenPinch/services/heat_exchanger_network_synthesis/service.py:18`, `OpenPinch/services/heat_exchanger_network_synthesis/service.py:32`, `OpenPinch/services/heat_exchanger_network_synthesis/__init__.py:5`, `tests/test_heat_exchanger_network_synthesis_workflow.py:229`.
- `PinchProblem._results` / `problem.results` remains the canonical cache path, with the service writing a `TargetOutput` containing `design`. Refs: `OpenPinch/services/heat_exchanger_network_synthesis/service.py:39`, `OpenPinch/services/heat_exchanger_network_synthesis/service.py:58`.
- Optional exports still read from `problem.results.design`, identify OpenPinch problem/workspace variant identity, and stay outside the terminal workflow path. Refs: `OpenPinch/services/heat_exchanger_network_synthesis/exports.py:16`, `OpenPinch/services/heat_exchanger_network_synthesis/exports.py:25`, `OpenPinch/services/heat_exchanger_network_synthesis/exports.py:66`, `OpenPinch/services/heat_exchanger_network_synthesis/exports.py:99`.
- The HENS-05 Definition of Done remains unchecked pending review, and the Implementation Notes now truthfully record the blocker resolution plus focused test, ruff, and diff-check evidence. Refs: `docs/developer/openhens-integration-tasks/05-design-workflow-result-cache-and-fake-executor.md:124`, `docs/developer/openhens-integration-tasks/05-design-workflow-result-cache-and-fake-executor.md:163`.
- Documentation and comments reviewed in the changed HENS-05 surface describe stable OpenPinch ownership, public semantics, domain invariants, and migration constraints. They do not introduce OpenHENS compatibility aliases, command parity, wrapper-package language, or implementation-trivia comments that would obscure the contract. Refs: `OpenPinch/services/heat_exchanger_network_synthesis/workflow.py:1`, `OpenPinch/services/heat_exchanger_network_synthesis/service.py:25`, `OpenPinch/classes/_problem/_design_accessor.py:25`, `OpenPinch/services/heat_exchanger_network_synthesis/exports.py:1`.
- `.DS_Store` remains modified in the worktree and outside HENS-05 scope; I did not inspect or modify it.

## Residual Risks

- I did not rerun the worker's recorded tests because this re-review was read-only except for this findings file. The task notes record `37 passed` for the focused workflow/API/schema tests, ruff passing, and diff check excluding `.DS_Store` passing.
- The private helper is still importable by Python as a leading-underscore module attribute. That is acceptable for this HENS-05 contract because the banned public name is gone, the stable public route is documented through `PinchProblem.design` / workspace dispatch, and the package `__all__` remains empty for the synthesis helper package.

## Verdict

HENS-05 is cleared.
