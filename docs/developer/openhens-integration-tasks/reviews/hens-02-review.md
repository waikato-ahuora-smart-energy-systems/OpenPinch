# HENS-02 Adversarial Re-Review

## Findings

No blocking findings remain for HENS-02.

## Prior Finding Resolution

### [Resolved] Persistent HEN options now validate on canonical paths

The previous P1 finding was that HENS option validation existed on standalone synthesis schemas but not on the persistent `TargetInput.options` / `Configuration` path. That is resolved.

`OpenPinch/lib/config_metadata.py` now defines canonical HENS value validation through `validate_configuration_options(...)` and `validate_configuration_option_value(...)`, covering positive approach-temperature and derivative-threshold grids, positive unique stage selections, method/output format choices, positive solve tolerance, positive integer counts, and run-id syntax (`OpenPinch/lib/config_metadata.py:299`, `OpenPinch/lib/config_metadata.py:315`). `Configuration` calls that validation before assigning option values (`OpenPinch/lib/config.py:64`, `OpenPinch/lib/config.py:81`). `TargetInput.options` also validates through the same helper during schema validation (`OpenPinch/lib/schemas/io.py:73`).

The tests now cover both valid normalization and invalid rejection through both public/canonical entry points: `Configuration(options=...)` and `TargetInput(..., options=...)` (`tests/test_lib/test_synthesis_schemas.py:257`, `tests/test_lib/test_synthesis_schemas.py:283`).

### [Resolved] `.DS_Store` remains excluded from the HENS-02 task slice

The previous P2 finding treated the dirty root `.DS_Store` as task scope leakage. The file is still modified in the working tree, but the implementation notes now document it as pre-existing/user-owned, unstaged, and excluded from the HENS-02 slice, with explicit command evidence (`docs/developer/openhens-integration-tasks/02-synthesis-schemas-and-network-domain.md:245`).

I independently confirmed there are no staged files, `.DS_Store` is tracked, and the non-`.DS_Store` diff is limited to HENS-02 implementation, test, and task-documentation files. On that basis, I do not consider `.DS_Store` a blocker for HENS-02 clearability, as long as it remains excluded from staging/publication.

## Scope Checks

- No HENS-02 changed file adds solver execution, public `PinchProblem.design`, raw input synthesis runners, fixture conversion, or export file writing.
- No public `OpenHENS`, `CaseStudy`, `SynthesisStudy`, public `run_synthesis_workflow`, import shim, OpenHENS field alias, or option-owner class was found in the new HENS-02 exports.
- HEN option concepts live in `CONFIG_FIELD_SPECS`, `Configuration`, and `TargetInput.options`, not alternate public option-owner classes.
- `HeatExchanger` and `HeatExchangerNetwork` continue to use source/sink stream identities, labelled accessors, totals, and private solver/source metadata fields that are excluded from public dumps.
- `HeatExchangerNetworkSynthesisResult` owns the network plus task/solver metadata, objective values, optional manifest, task outcomes, and diagnostic references; I did not find stream, utility, case, or workspace ownership on that result envelope.
- `TargetOutput.design` remains an optional synthesis result payload beside existing `targets`.
- The task DoD checkboxes remain unchecked pending this review, which is appropriate for this review-gated state.

## Residual Risks

- I did not rerun pytest or ruff because this review was constrained to read-only work except for this review file. The task notes report focused pytest at 47 passed, ruff passed, and `git diff --check -- ':!.DS_Store'` passed.
- `.DS_Store` remains dirty in the shared worktree. It is not a HENS-02 blocker after the exclusion evidence, but it must stay unstaged and out of any HENS-02 commit or PR.
- The domain model validates process-vs-utility source/sink roles and uses source/sink semantics for hot-to-cold direction. Full proof against real hot/cold stream classification remains a later adapter/extraction concern unless HENS-02 adds cross-reference validation against prepared stream metadata.

## Verdict

HENS-02 is cleared from this adversarial re-review.
