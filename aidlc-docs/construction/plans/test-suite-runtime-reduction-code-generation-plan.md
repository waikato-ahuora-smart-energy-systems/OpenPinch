# Test-Suite Runtime Reduction Code Generation Plan

This plan is the single source of truth for implementation. Every completed
step must be marked `[x]` in the same interaction in which its work is finished.

## Unit Context

- **Unit**: `test-suite-runtime-reduction`
- **Project**: Brownfield Python library rooted at
  `/Users/timothyw/Github_Local/OpenPinch`
- **Stories**: None; user stories were intentionally skipped for this internal
  test and CI refactor.
- **Public API changes**: None planned.
- **Production numerical changes**: None planned. A production testability seam
  may be changed only if a new failing regression proves it is necessary and
  the change preserves the public API.
- **Database entities**: None.
- **Infrastructure**: None. GitHub Actions workflow configuration is in scope.

## Requirements Traceability

| Finding | Requirements | Plan steps |
|---|---|---|
| Notebook 19 search budget | FR-01, NFR-PERF-02, NFR-DET-03 | 2, 4, 11, 12 |
| Repeated utility-placement solves | FR-02, NFR-DET-02 | 2, 5, 6, 11, 12 |
| TESPy duplication | FR-03, NFR-PERF-03 | 2, 3, 11, 12 |
| Optimizer benchmarks | FR-04, NFR-REL-01 | 2, 3, 8, 11, 12 |
| Duplicate Sphinx build | FR-05, NFR-MNT-01 | 2, 3, 9, 11, 12 |
| CoolProp audit overlap | FR-06, NFR-COR-01 | 2, 7, 11, 12 |
| Fresh-process startup cost | FR-07, NFR-REL-02 | 2, 10, 11, 12 |
| Overall runtime and coverage | NFR-PERF-01, NFR-COV-01 | 1, 11, 12, 13 |

## Dependencies and Interfaces

- The tutorial generator is authoritative for the packaged Notebook 19 JSON.
- `pytest.ini` owns marker registration.
- `.github/workflows/ci-develop.yml`, `ci-pull-request.yml`, and
  `ci-publish.yml` must retain equivalent lane ownership.
- `tests/packaging/test_packaging_metadata.py` owns workflow and marker
  contracts.
- Existing `PinchProblem`, `PinchWorkspace`, HPR/MVR services, optimisation
  backends, and Sphinx build scripts remain unchanged interfaces.
- Existing domain-specific Hypothesis strategies, shrinking, and fixed CI seed
  remain mandatory.

## Implementation Steps

### Step 1: Preserve the baseline and implementation boundary

- [x] Confirm the working tree contains only this workflow's documentation.
- [x] Record the exact baseline commands, Python version, seed, selected count,
  coverage result, overall runtime, and seven hotspot timings.
- [x] Inventory the current assertions and CI owners for every test that will be
  moved, consolidated, or batched.
- [x] Confirm no application source change is required before editing tests.

**Files**: Existing AI-DLC records only.

### Step 2: Add regression contracts before implementation

- [x] Extend `tests/packaging/test_packaging_metadata.py` to require central
  `docs` and `performance` marker declarations, the four-marker ordinary-lane
  exclusion, one dedicated owner per specialized marker, and finite timeouts.
- [x] Add static generator/notebook contracts for the approved Notebook 19
  upper bounds and canonical regeneration.
- [x] Add fixture mutation/order-isolation coverage before consolidating
  utility-placement evidence.
- [x] Add child-batch failure-attribution coverage or a directly testable helper
  contract before batching processes.
- [x] Run the new tests and record the expected pre-change failures.

**Files**: `tests/packaging/test_packaging_metadata.py`, relevant tutorial,
utility-placement, and architecture test files.

### Step 3: Establish marker and CI lane ownership

- [x] Register `performance` and `docs` in `pytest.ini`; retain `tespy`,
  `solver`, and `tutorial_profile` policies.
- [x] Change the ordinary command in all three workflows to exclude `solver`,
  `tespy`, `performance`, and `docs`.
- [x] Make `hpr-tespy-tests` execute the complete `tespy` selection once under
  a finite command timeout; remove the duplicate public smoke invocation.
- [x] Add a finite `performance-tests` job that executes the complete
  `performance` selection once in each workflow and participates in required
  downstream gates.
- [x] Change each `docs` job to execute the `docs` selection once instead of
  directly building the documentation a second time.
- [x] Update PR/release job dependency and gate assertions without weakening
  exact-develop-validation reuse.

**Files**: `pytest.ini`, `.github/workflows/ci-develop.yml`,
`.github/workflows/ci-pull-request.yml`, `.github/workflows/ci-publish.yml`,
`tests/packaging/test_packaging_metadata.py`.

### Step 4: Bound and regenerate Notebook 19

- [x] Probe decreasing fixed-seed iteration, evaluation, candidate, and run
  budgets using both process and site scopes.
- [x] Select the smallest stable budget satisfying feasibility, finite
  objective, physical constraints, and best-observed convergence.
- [x] Update only the authoritative generator and its related static contracts.
- [x] Regenerate Notebook 19 and verify source-only canonical equality.
- [x] Execute Notebook 19 end to end and confirm the 20-second focused target or
  document a justified compensating result under the overall target.

**Files**: `scripts/generate_tutorial_notebooks.py`,
`OpenPinch/tutorials/notebooks/19_utility_placement_optimisation.ipynb`, and
the existing tutorial generation/execution tests.

### Step 5: Introduce isolated utility-placement solved evidence

- [x] Identify identical process and site solve signatures in
  `tests/application/test_utility_placement.py`.
- [x] Add module-scoped deterministic source fixtures only for identical real
  solves that serve multiple non-optimizer assertions.
- [x] Give mutating consumers deep copies or immutable reconstructed payloads;
  keep source evidence inaccessible to mutation.
- [x] Retain an independent true optimizer integration case for each process
  and site scope.
- [x] Verify canonical source serialization before and after representative
  consumers and in reversed operation order.

**Files**: `tests/application/test_utility_placement.py` and, only if reusable
across files, a narrowly scoped module under `tests/support/`.

### Step 6: Shorten utility-placement property and batch setup

- [x] Replace optimizer execution inside ordering, forwarding, or scaling
  properties with existing deterministic seams when optimization is not the
  asserted behavior.
- [x] Preserve domain-specific generators, constrained inputs, shrinking, and
  seed reproducibility.
- [x] Retain explicit example tests for public process/site optimization.
- [x] Run property tests in multiple orders and measure the focused
  utility-placement file against the 75-second target.

**Files**: `tests/application/test_utility_placement.py`,
`tests/application/test_utility_placement_batch.py`, and existing strategies
under `tests/strategies/` if a reusable constrained strategy needs refinement.

### Step 7: Refine the CoolProp audit without reducing the E2E corpus

- [x] Keep all 54 parameterized cases in `tests/e2e/test_hpr_mvr.py` unchanged
  unless a new regression requires an assertion improvement.
- [x] Reduce the CoolProp audit's search budgets to the smallest fixed-seed
  values that retain representative heat-pump, refrigeration, and MVR success.
- [x] Use bounded deterministic evidence for serialization, topology,
  detachment, and atomic-failure assertions where an independent search is not
  their subject.
- [x] Preserve at least one independent real search for each service family and
  both relevant topology classes.
- [x] Measure `test_coolprop_hpr_audit.py` against the 30-second target.

**Files**: `tests/application/test_coolprop_hpr_audit.py`; production HPR files
only if a failing test proves a minimal testability correction is required.

### Step 8: Isolate and tune optimizer convergence benchmarks

- [x] Mark both cross-backend convergence benchmarks as `performance`.
- [x] Probe lower run, iteration, and evaluation budgets while preserving all
  four backend quality and relative-tolerance assertions.
- [x] Retain ordinary fast backend/service tests outside the marker.
- [x] Verify the dedicated performance selection is deterministic and completes
  inside its command timeout.

**Files**: `tests/optimisation/test_backends.py` and marker/workflow contracts.

### Step 9: Give the Sphinx smoke one owner

- [x] Mark `test_sphinx_build_smoke` as `docs`.
- [x] Preserve warnings-as-errors behavior, return-code diagnostics, and output
  `index.html` validation.
- [x] Verify each docs job invokes `pytest -m docs` once and no direct second
  build remains.

**Files**: `tests/packaging/test_docs_build.py`, three CI workflows, and
workflow-contract tests.

### Step 10: Batch compatible fresh-process checks

- [x] Replace the per-layer cold-import parameterization with one ordered child
  script that identifies the failing case and target module.
- [x] Batch retired-package import failures in one child process with equivalent
  per-package diagnostics.
- [x] Keep root export, default TESPy-cold selector, and any incompatible
  pristine-state contracts in separate child processes.
- [x] Prove the batch attribution contract and measure the combined focused
  architecture/API selection against the 20-second target.

**Files**: `tests/architecture/test_cold_imports.py`,
`tests/architecture/test_api_boundary.py`, and an optional narrow helper under
`tests/support/` if direct failure-attribution testing warrants it.

### Step 11: Run focused correctness and performance gates

- [x] Run marker/workflow metadata tests and pytest collection checks.
- [x] Run tutorial generator drift, Notebook 19 execution, and packaging tests.
- [x] Run utility-placement tests with order/isolation repetitions and duration
  reporting.
- [x] Run CoolProp audit plus the complete 54-case HPR/MVR E2E corpus.
- [x] Run fresh-process architecture/API tests with duration reporting.
- [x] Run Ruff and `git diff --check`; correct every in-scope finding.
- [x] Refine only the failing hotspot until its correctness and focused timing
  gate pass.

### Step 12: Run complete lane and coverage verification

- [x] Run the ordinary branch-aware selection with seed `20260715` and
  `fail-under=95`.
- [x] If coverage is below 95 percent, use the NFR Design fallback without
  duplicating expensive specialized tests.
- [x] Run complete `tespy`, `performance`, `docs`, and `solver`-appropriate
  selections with their intended dependencies and timeouts.
- [x] Run all relevant packaged notebook profiles, build distributions, and an
  isolated installed-wheel smoke if affected metadata or packaged artifacts
  changed.
- [x] Confirm no unexpected skip, deselection, flaky retry, or orphaned marker.

### Step 13: Re-profile, audit, and summarize

- [x] Re-run the uninstrumented serial ordinary selection with the exact
  baseline environment and seed.
- [x] Compare total and per-hotspot timings with the 600.89-second baseline and
  verify the 480-second acceptance threshold.
- [x] Audit assertion preservation, fixture isolation, generator authority,
  lane ownership, timeout behavior, coverage, and all seven finding closures.
- [x] Verify no duplicate brownfield files or unintended production API changes
  were introduced.
- [x] Mark every plan checkbox complete and create
  `aidlc-docs/construction/test-suite-runtime-reduction/code/code-generation-summary.md`.

## Property-Based Testing Compliance

- **PBT-01 and PBT-03**: Existing utility-placement invariants and metamorphic
  order/scale properties remain; fixture isolation adds state-preservation
  coverage.
- **PBT-07**: Existing constrained domain strategies remain centralized and are
  not replaced by unconstrained primitives.
- **PBT-08**: Shrinking remains enabled and CI seed `20260715` remains visible.
- **PBT-09**: Hypothesis remains the configured framework.
- **PBT-10**: Explicit process/site, CoolProp, E2E, TESPy, benchmark, docs, and
  import examples remain alongside generated properties.
- **PBT-02, PBT-04, PBT-05, PBT-06**: N/A unless implementation introduces a
  new reversible mapping, idempotent transform, independent oracle, or mutable
  state machine. Fixture source preservation is covered by direct state
  invariants rather than a state-machine model.

## Completion Conditions

Code Generation is complete only when all thirteen steps and every nested
checkbox are `[x]`, the generated-code summary records actual evidence, and the
user has reviewed the implementation before Build and Test begins.
