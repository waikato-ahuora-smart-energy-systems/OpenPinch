# Experimental Discopt Integration Removal Requirements

## Intent Analysis

- **User request**: Remove the experimental Discopt integration.
- **Request type**: Focused removal and repository cleanup.
- **Scope**: HEN solver backend, solver model annotations, benchmark-only tooling,
  tests, and developer documentation.
- **Complexity**: Moderate because the experimental adapter spans runtime and
  benchmark boundaries inside an already modified worktree.

## Functional Requirements

- FR-01: Remove the private GEKKO-to-Pyomo Discopt solver bridge.
- FR-02: Remove `discopt` from accepted HEN solver names and integer-capable
  solver classifications.
- FR-03: Remove Discopt-specific dependency checks, solver configuration,
  execution branches, and result normalization.
- FR-04: Remove the dedicated Couenne/APOPT/Discopt benchmark entry point and
  its benchmark-specific test and Hypothesis strategy modules.
- FR-05: Remove Discopt-specific backend tests and benchmark tracer assertions.
- FR-06: Remove developer documentation that instructs users to install or run
  the experimental Discopt stack.
- FR-07: Preserve completed benchmark result artifacts and AI-DLC audit/history
  as historical evidence; these records must not remain executable integration
  surfaces or imply current support.
- FR-08: Retain solver-agnostic benchmark telemetry only where it remains useful,
  tested, and independent of Discopt-only result fields.

## Non-Functional Requirements

- NFR-01: Preserve current Couenne, IPOPT, and APOPT behavior.
- NFR-02: Do not modify unrelated heat-pump, Read the Docs, stream-segmentation,
  or other user-owned working-tree changes.
- NFR-03: Leave `pyproject.toml` and `uv.lock` free of Discopt dependencies.
- NFR-04: Repository-facing package code, scripts, tests, and developer
  documentation must contain no active Discopt integration after removal.
- NFR-05: Focused backend tests, the CI-selected non-solver suite, the 95% line
  coverage gate, solver-marked tests, Ruff checks, documentation build, and
  package build must remain successful.
- NFR-06: Property-based testing remains partially enforced. Removing the
  Discopt-only generated benchmark matrix does not require a replacement
  property test because no production pure function or serialization contract
  is added or changed.

## Acceptance Criteria

- Discopt cannot be selected through the HEN solver backend or unit-model type
  contracts.
- The private adapter and dedicated three-stack benchmark source files are gone.
- No Discopt dependency is declared by the package or lockfile.
- A repository search finds no active Discopt references outside retained
  historical AI-DLC and ignored benchmark-result evidence.
- Couenne, IPOPT, and APOPT regression behavior and project quality gates pass.

## Extension Compliance

- **Security Baseline**: Disabled; skipped.
- **Resiliency Baseline**: Disabled; skipped.
- **Property-Based Testing**: Partial; applicable removal impact assessed in
  NFR-06, with no new pure-function or serialization behavior requiring PBT.
