# Notebook Presentation Build and Test Summary

## Build Status

- **Build tool**: Hatchling through the repository distribution helper.
- **Build status**: Success.
- **Artifacts**:
  - `openpinch-0.5.3-py3-none-any.whl`
  - `openpinch-0.5.3.tar.gz`
- **Artifact location**: Clean temporary directory
  `/private/tmp/openpinch-notebook-dist.EAOp0d`.
- **Wheel contents**: All 18 improved packaged notebooks present.
- **Documentation**: Sphinx 9.1.0 built 53 sources with warnings treated as
  errors in `/private/tmp/openpinch-notebook-docs.DKTKE0`.

## Test Execution Summary

| Gate | Result | Status |
|---|---:|---|
| Focused notebook, PBT, base execution, and coverage | 22 passed, 3 optional-profile skips | Pass |
| Slow-HPR selected profile | 1 selected test passed, 2 unselected skips; 4 notebooks executed in 213.71 s | Pass |
| External-solver selected profile | 1 selected test passed, 2 unselected skips; 3 notebooks executed in 164.50 s | Pass |
| Interactive selected profile | 1 selected test passed, 2 unselected skips; 1 notebook executed in 8.55 s | Pass |
| Integrated packaging suite | 84 passed, 3 optional-profile skips | Pass |
| Complete fixed-seed non-solver suite | 2,184 passed, 3 skipped, 4 solver deselected in 163.54 s | Pass |
| Repository Ruff lint and format | All checks; 460 files formatted | Pass |
| Warning-as-error Sphinx build | 53 sources, zero warnings | Pass |
| Wheel and source archive | 2 OpenPinch 0.5.3 artifacts | Pass |
| Isolated installed-wheel smoke | Site-packages workflow, resource, root API, retired package, notebook, and CLI checks | Pass |
| Patch and generated-resource review | No blocking findings | Pass |

## Integration and End-to-End Evidence

- The generator produces exactly 18 source-only notebooks in two byte-identical
  passes.
- Every notebook contains one `Review the result` section, a following explicit
  display cell, and subsequent subject-specific interpretation.
- All displayed values come from already-calculated public workflow objects.
- All 18 notebooks execute under their declared base, slow-HPR, solver, or
  interactive profile.
- The built wheel discovers all notebook and sample resources from outside the
  checkout and completes a public targeting workflow.
- The package root remains exactly `PinchProblem` and `PinchWorkspace`.

## Performance Assessment

- Load, throughput, concurrency, and service response-time testing: N/A.
- Notebook runtime validation: Pass. The slow-HPR, solver, and interactive
  profiles remain consistent with their declared qualitative runtime bands.
- Presentation overhead: Limited to IPython display of cached or already-
  computed values; no target or design analysis is rerun by a review cell.

## Generated Instructions

- `build-and-test-plan.md`
- `build-instructions.md`
- `unit-test-instructions.md`
- `integration-test-instructions.md`
- `performance-test-instructions.md`
- `build-and-test-summary.md`

All files are under
`aidlc-docs/construction/notebook-presentation/build-and-test/`.

## Extension Compliance

| Extension rule | Status | Evidence |
|---|---|---|
| Security Baseline | N/A | Disabled by the user. |
| Resiliency Baseline | N/A | Disabled by the user. |
| PBT-02 Round trips | N/A | No inverse pair was introduced. |
| PBT-03 Invariants | Compliant | Source-only state, cell IDs, and review placement are property-tested. |
| PBT-07 Generator quality | Compliant | Strategy uses canonical tutorial identifiers. |
| PBT-08 Shrinking and reproducibility | Compliant | Shrinking enabled; all final suites use seed `20260715`. |
| PBT-09 Framework selection | Compliant | Hypothesis with pytest remains configured and executed. |

## Overall Status

- **Build**: Success.
- **Tests**: Pass.
- **Documentation**: Pass.
- **Distribution**: Pass.
- **Installed artifact**: Pass.
- **Operations**: N/A; no deployment or production runtime change requested.
- **Notebook improvement workflow**: Complete.
