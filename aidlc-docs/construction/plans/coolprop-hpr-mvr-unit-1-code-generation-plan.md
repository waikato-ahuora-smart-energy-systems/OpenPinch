# Unit 1 Code Generation Plan

Unit: CoolProp HPR and MVR — Candidate Correctness and Detached HPR Results

Status: Approved under the user's authorization through completion; Part 2 ready.

This plan is the single source of truth for Unit 1 Code Generation.

## Unit Context

- **Requirements**: FR-1, FR-2, FR-3, and FR-8 primary; foundational
  contracts for FR-4, FR-5, and FR-7.
- **Verification**: VR-3 and VR-4 primary.
- **Dependencies**: existing HPR contracts, objective helpers, performance-map
  target records, multiperiod aggregation, and application transaction.
- **Interfaces produced**: penalty normalizer, evaluation mode, search budget,
  diagnostics/error contracts, generalized simulation records, detached output
  finalizer/guard.
- **Database entities**: none.
- **Deployment/infrastructure**: none.
- **Code location**: existing `OpenPinch/` and `tests/` paths at workspace
  root; Markdown summary only under `aidlc-docs/`.

## Part 1 Planning Checklist

- [x] Read Functional Design, NFR Requirements, NFR Design, unit mapping, and
  dependency artifacts.
- [x] Confirm brownfield workspace and existing source owners.
- [x] Inspect current HPR contracts, shared evaluators, adapter, target-record
  builders, multiperiod path, and focused tests.
- [x] Define the exact nine-step RED-GREEN-REFACTOR sequence below.
- [x] Map requirements, NFRs, and PBT obligations to steps.
- [x] Record approval under the user's explicit completion authorization.

## Part 2 Generation Steps

### Step 1 — RED Contract and Reliability Tests

- [x] Add focused failing examples and Hypothesis properties for budget,
  diagnostics, generalized loop/stage records, rectangular penalty
  normalization, model omission, deep-copy safety, and evaluation-mode artifact
  behavior.
- [x] Confirm failures are caused only by missing Unit 1 behavior.

### Step 2 — Foundational Reliability Contracts

- [x] Extend `OpenPinch/contracts/hpr.py` with budget, failure, evaluation,
  topology, loop, and stage contracts.
- [x] Generalize `HprTargetSimulationRecord` additively while preserving
  current single-stage inputs and JSON round trips.
- [x] Export only through the existing contract module convention.

### Step 3 — Shared Penalty Normalization

- [x] Add one pure shared normalizer at the HPR analysis boundary.
- [x] Route Carnot and simulated-vapour accounting plus VC+MVR finite failures
  through it.
- [x] Preserve sign until the owning accounting step clips inequality terms.

### Step 4 — Explicit Search and Final Evaluation Modes

- [x] Add an internal `search`/`final` mode to the adapter/objective seam.
- [x] Prevent search results from retaining model, figure, and simulation record.
- [x] Preserve identical core objective/accounting semantics across modes.

### Step 5 — Generalized Detached Simulation Evidence

- [x] Extend CoolProp target-record construction to supported cascade, parallel,
  and VC+MVR final results with ordered detached loop/stage facts.
- [x] Preserve TESPy and existing single-stage compatibility.
- [x] Reject topology-inconsistent evidence.

### Step 6 — Detached Public Result Finalization

- [x] Omit internal model/debug artifacts from public output translation.
- [x] Add recursive detached-object validation and `model=None`.
- [x] Recursively sanitize/finalize period outputs without altering period order.

### Step 7 — Compatibility and Property Test Completion

- [x] Complete U1-P1 through U1-P12 properties and explicit reproductions.
- [x] Update existing contract/adapter tests for the deliberate detached-model
  behavior.
- [x] Prove transaction deep-copy safety with representative real/fake engine
  objects.

### Step 8 — Focused Verification and Refactor

- [x] Run focused Unit 1 tests, existing HPR contract/adapter/target-record/
  multiperiod/application regressions, Ruff, formatting, compilation, and patch
  hygiene.
- [x] Refactor only while the focused gate remains green.
- [x] Verify no duplicate brownfield source files or dependency changes.

### Step 9 — Code Summary and Traceability

- [x] Generate
  `aidlc-docs/construction/coolprop-hpr-mvr-unit-1/code/code-generation-summary.md`.
- [x] Map FR/NFR/PBT obligations to changed code and passing evidence.
- [x] Mark Unit 1 Code Generation complete and continue to Unit 2 under the
  completion authorization.

## TDD and PBT Gates

- Each production behavior begins with a failing example or property.
- Hypothesis shrinking remains enabled; seeds follow repository convention.
- Explicit regressions pin the nested-penalty and live-model copy failures.
- Changed Unit 1 production lines target at least 95 percent statement/branch
  coverage.
- PBT complements rather than replaces examples.
