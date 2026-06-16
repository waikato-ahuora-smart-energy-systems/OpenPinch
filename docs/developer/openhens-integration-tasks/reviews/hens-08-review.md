# HENS-08 Adversarial Re-review

## Findings

No findings. HENS-08 is cleared.

## Prior Finding Resolution

- The prior P1 is resolved. The task no longer marks OpenHENS runtime dependency removal complete; that requirement is explicitly unchecked until moved OpenPinch Four-stream parity is proven (`docs/developer/openhens-integration-tasks/08-stage-reduction-and-topology-evolution-move.md:71-72`). The notes now state that the dependency-boundary scan is boundary evidence only and that runtime dependency removal remains gated on moved parity (`docs/developer/openhens-integration-tasks/08-stage-reduction-and-topology-evolution-move.md:123-127`).
- The prior P2 is resolved. `LocalSynthesisExecutor` now passes `evolution=task.method == "energy_stage_refinement"` into every solve, so the executor-wide flag can no longer enable evolution for PDM/TDM (`OpenPinch/services/heat_exchanger_network_synthesis/workflow.py:183-186`). The regression test intentionally constructs `LocalSynthesisExecutor(evolution=True)` and asserts PDM/TDM receive `False` while ESM receives `True` (`tests/test_heat_exchanger_network_synthesis_models.py:488-564`).

## Re-review Result

- The remaining live Four-stream solver evidence is blocked by the documented missing `couenne` and `ipopt` binaries. Under this task's contract, that is acceptable because the task explicitly allows a Four-stream solver-capability blocker when it is documented exactly (`docs/developer/openhens-integration-tasks/08-stage-reduction-and-topology-evolution-move.md:35-36`), and the notes record both missing binaries and the exact rerun command (`docs/developer/openhens-integration-tasks/08-stage-reduction-and-topology-evolution-move.md:123-124`).
- The task does not overclaim solver parity: the Four-stream solver baseline remains unchecked (`docs/developer/openhens-integration-tasks/08-stage-reduction-and-topology-evolution-move.md:87`), the full moved-path parity and runtime-dependency-removal DoD items remain unchecked (`docs/developer/openhens-integration-tasks/08-stage-reduction-and-topology-evolution-move.md:98-105`), and the notes distinguish blocked solver evidence from completion evidence (`docs/developer/openhens-integration-tasks/08-stage-reduction-and-topology-evolution-move.md:123-127`).
- The focused evidence recorded by the worker is coherent for the non-live-solver parts of HENS-08: HENS model/HENS-08 tests passed with `18 passed`, workflow/adapter/dependency/API tests passed with `24 passed`, `ruff check` passed, and `git diff --check -- . ':!.DS_Store'` was clean (`docs/developer/openhens-integration-tasks/08-stage-reduction-and-topology-evolution-move.md:121-126`). I did not rerun these commands because this review is read-only apart from the review artifact.
- The docs/comments quality gate is satisfied. The updated notes describe stable private contracts, ESM-only evolution invariants, the migration evidence boundary, and the solver blocker without adding public OpenHENS contracts, alternate workflow routes, implementation trivia, or misleading completion claims (`docs/developer/openhens-integration-tasks/08-stage-reduction-and-topology-evolution-move.md:116-127`).
- `.DS_Store` remains modified in the worktree and outside HENS-08 scope. I did not touch it.

## Residual Risks

- Live moved OpenPinch Four-stream solver parity has not run in this environment. It should be rerun when `couenne` and `ipopt` are available on `PATH`.
- Runtime OpenHENS dependency removal remains gated on that moved-path parity and must not be marked complete until parity is proven.
- The current Four-stream snapshot test replays a saved source artifact through OpenPinch adapter/extraction code; it does not prove that the moved OpenPinch solver path can regenerate the artifact.

## Verdict

HENS-08 is cleared for this review pass. The remaining solver parity gap is an explicitly documented environmental blocker and residual risk, not a current review-blocking finding.
