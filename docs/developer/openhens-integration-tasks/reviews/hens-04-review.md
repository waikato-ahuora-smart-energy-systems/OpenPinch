# HENS-04 Adversarial Re-Review

## Findings

No findings. HENS-04 is cleared for the parity harness and focused adapter-convention slice.

## Prior Finding Resolution

### [Resolved] Native target extraction no longer depends on raw ProblemTable rows

The previous P1 finding was that the HENS-04 helper read HU target, CU target, and heat recovery directly from raw `ProblemTable` top/bottom row positions. That is resolved. `_target_snapshot_from_direct_target(...)` now consumes semantic `DirectIntegrationTarget` fields for HU target, CU target, heat recovery target, hot pinch, and cold pinch (`OpenPinch/services/heat_exchanger_network_synthesis/pinch_decomposition.py:119`, `OpenPinch/services/heat_exchanger_network_synthesis/pinch_decomposition.py:122`). The access contract recorded in the snapshot now names those semantic target fields rather than `ProblemTable` row/column access (`OpenPinch/services/heat_exchanger_network_synthesis/pinch_decomposition.py:138`), and the parity test asserts that contract (`tests/test_heat_exchanger_network_pinch_parity.py:81`).

This satisfies the HENS-04 row/column-access concern for the new native decomposition path: HENS-04 no longer introduces its own raw `ProblemTable` layout dependency, and the existing OpenPinch targeting service remains the owner of `ProblemTable` interpretation.

### [Resolved] Source OpenHENS parity is no longer skippable

The previous P1 finding was that the required source OpenHENS parity comparison skipped when `../OpenHENS` was absent or not importable. That is resolved. The test now pins source OpenHENS to commit `2afc14b7779482fc829edb1c3fa187b918d7fb19` (`tests/test_heat_exchanger_network_pinch_parity.py:23`, `tests/test_heat_exchanger_network_pinch_parity.py:26`) and raises explicit assertion failures for a missing checkout, wrong commit, non-git checkout, or non-importable `PinchDecompModel` (`tests/test_heat_exchanger_network_pinch_parity.py:245`, `tests/test_heat_exchanger_network_pinch_parity.py:252`, `tests/test_heat_exchanger_network_pinch_parity.py:261`, `tests/test_heat_exchanger_network_pinch_parity.py:271`).

I also confirmed the sibling checkout currently resolves to the pinned commit with `rtk git -C ../OpenHENS rev-parse HEAD`, and `rtk git -C ../OpenHENS status --short` reported no source worktree changes.

### [Resolved] Helper documentation now states the stable internal contract

The previous P2 finding was that the helper documentation read as temporary test plumbing rather than a stable internal migration contract. That is resolved. The module docstring now states that the helper is private to the synthesis service package, accepts a prepared `PinchProblem`, uses OpenPinch targeting for target semantics, returns structural PDM fields, and remains behind the HENS-04 replacement gate (`OpenPinch/services/heat_exchanger_network_synthesis/pinch_decomposition.py:1`). The public helper docstring now describes the stable input/output contract and explicitly says the helper is private so it cannot become a public synthesis entry point or bypass the gate (`OpenPinch/services/heat_exchanger_network_synthesis/pinch_decomposition.py:73`).

That satisfies the added documentation/comment review requirement for this HENS-04 slice: the comments describe durable OpenPinch ownership, domain invariants, and migration constraints rather than incidental mechanics.

## Scope Checks

- The changed implementation remains limited to the HENS-04 task document, review-criteria README edit, private pinch decomposition helper, parity tests, and this review file. I found no `StageWiseModel` or `GenericHENModel` move, no public design API, no solver benchmark move, and no LMTD/costing/helper replacement beyond the documented pinch gate.
- `OpenPinch/services/heat_exchanger_network_synthesis/__init__.py` still has an empty `__all__`, and the new helper is not exported from `OpenPinch/__init__.py` or `OpenPinch/services/__init__.py`.
- I found no new public OpenHENS compatibility surface, `CaseStudy`, `SynthesisStudy`, public `run_synthesis_workflow(...)`, raw-input runner, import shim, wrapper package, eager GEKKO/Pyomo import, or workflow bypass in the HENS-04 changes.
- OpenHENS private `pinch_classes` usage was not removed. Replacement remains gated, and the replacement checkboxes remain unchecked (`docs/developer/openhens-integration-tasks/04-pinch-target-parity-and-native-replacement-gate.md:69`, `docs/developer/openhens-integration-tasks/04-pinch-target-parity-and-native-replacement-gate.md:92`).
- `.DS_Store` remains dirty in the shared worktree, but it is outside the HENS-04 scope and should stay out of staging/publication.

## Coverage Checks

- The parity test matrix covers the required Four-stream and Nine-stream fixtures, the full required `dTmin` grid, and above/below pinch decompositions (`tests/test_heat_exchanger_network_pinch_parity.py:27`, `tests/test_heat_exchanger_network_pinch_parity.py:31`, `tests/test_heat_exchanger_network_pinch_parity.py:58`).
- The assertions compare HU target, CU target, heat recovery target, hot/cold/shifted pinch temperatures, active masks, clipped hot/cold stream temperatures, `S`, and `K` (`tests/test_heat_exchanger_network_pinch_parity.py:286`). Manual stage selection is covered at `(2, 3)` (`tests/test_heat_exchanger_network_pinch_parity.py:95`).
- Stream-order invariance remains meaningful: base and reordered native snapshots are compared, and the reordered native snapshot is also compared against source OpenHENS (`tests/test_heat_exchanger_network_pinch_parity.py:121`, `tests/test_heat_exchanger_network_pinch_parity.py:148`).
- The HU/CU zero-threshold cases remain uncovered because the required README case matrix has no such rows. That gap is explicitly documented and the threshold checkboxes remain unchecked (`docs/developer/openhens-integration-tasks/04-pinch-target-parity-and-native-replacement-gate.md:60`, `docs/developer/openhens-integration-tasks/04-pinch-target-parity-and-native-replacement-gate.md:142`). I do not consider that a blocker for this HENS-04 parity slice because native replacement remains gated.

## Residual Risks

- I did not rerun pytest or ruff because this re-review was constrained to read-only work except for this findings file. The implementation notes report `83 passed` for HENS pinch parity, `34 passed` for adapter/schema tests, ruff passed, and diff-check passed.
- I did run `rtk git diff --check -- . ':(exclude).DS_Store'`, which passed with no output.
- The parity tests now intentionally require the sibling OpenHENS checkout at the pinned commit. That is correct for this gate, but it means default OpenPinch-only environments must provision that checkout before running the required HENS-04 parity test.
- HU/CU threshold behavior is still not proven. Replacement of OpenHENS private `pinch_classes` should remain blocked until the acceptance matrix names threshold fixtures/grid rows or a later reviewed task explicitly resolves that gap.

## Verdict

HENS-04 is cleared for the current scope. The prior blockers are resolved, the implementation remains within the allowed parity-harness/focused-adapter boundary, and native replacement is still correctly gated.
