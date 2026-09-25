# HPR and MVR Benchmark E2E Code Generation Plan

This plan is the single source of truth for Code Generation. It implements the
approved requirements, Functional Design, NFR Requirements and NFR Design for
one brownfield test/reliability unit. User Stories were explicitly skipped; the
traceability target is FR-01 through FR-17 and VR-01 through VR-08.

## Unit context

- **Workspace root**: `/Users/timothyw/Github_Local/OpenPinch`
- **Project type**: Brownfield Python library.
- **Primary boundary**: public `PinchProblem.target` HPR methods and
  `PinchProblem.components.add_process_mvr` with real CoolProp behavior.
- **Standard corpus**: dynamically discovered
  `examples/stream_data/p_*.json`, currently 54 problems.
- **Test ownership**: `tests/e2e`; production code does not know benchmark
  fixtures.
- **Production dependencies**: existing HPR application, contract, analysis and
  optimization owners only. Production changes are conditional on a reproduced
  valid-input defect.
- **Database entities**: none.
- **Frontend/API/deployment artifacts**: none.
- **Infrastructure**: none; the existing ordinary non-solver pytest job is the
  execution environment.

## Expected file changes

| Path | Planned action |
|---|---|
| `tests/e2e/cases.py` | Create the single sorted standard-corpus owner. |
| `tests/e2e/test_main.py` | Replace local discovery with the shared owner. |
| `tests/e2e/hpr_benchmark.py` | Create immutable profiles, assignments, observations, outcome helpers and convergence analysis. |
| `tests/e2e/test_hpr_benchmark_helpers.py` | Create focused example/property tests for pure helpers if separation remains useful. |
| `tests/e2e/test_hpr_mvr.py` | Create the 54-case HPR matrix, six convergence sentinels and direct process-MVR e2e workflow. |
| `OpenPinch/analysis/heat_pumps/**` | Modify only if the matrix reproduces an analysis/optimization defect. |
| `OpenPinch/application/_problem/**` | Modify only if the matrix reproduces a public transaction/accessor defect. |
| `OpenPinch/contracts/hpr.py` | Modify only if the matrix proves a contract/serialization defect. |
| `tests/analysis/heat_pumps/**` or `tests/application/**` | Add one focused regression beside the owner of every confirmed production defect. |
| `aidlc-docs/construction/hpr-mvr-benchmark-e2e/code/code-generation-summary.md` | Record implementation, investigation evidence, traceability and limitations. |

The helper-test file may be folded into `test_hpr_mvr.py` if the resulting file
remains cohesive. No alternate `_new`, `_modified` or duplicate production file
may be created.

## Generation sequence

### Step 1 - Baseline and RED corpus contracts

- [x] Record the current 54-file sorted corpus, existing direct-target behavior,
  current focused HPR/MVR tests and clean baseline outcomes.
- [x] Add failing tests that require one shared corpus owner, exact path
  uniqueness, stable sorting, complete assignment and nine cases per profile.
- [x] Confirm the RED failures are caused only by the not-yet-created shared
  helpers.

**Traceability**: FR-01 through FR-03, VR-01, VR-02, NFR-01, NFR-05.

### Step 2 - Shared corpus and declarative assignment

- [x] Create `tests/e2e/cases.py` and update `tests/e2e/test_main.py` to import
  its immutable discovery result.
- [x] Create frozen profile and assignment values in
  `tests/e2e/hpr_benchmark.py` with explicit topology, placement, CoolProp
  inputs, one restart and provisional bounded controls.
- [x] Implement deterministic ordinal-modulo assignment and readable
  `<filename>::<profile_id>` IDs.
- [x] Make the Step 1 corpus and assignment tests pass without changing
  production code.

**Traceability**: FR-01 through FR-05, VR-01, VR-02, NFR-01, NFR-03, NFR-05.

### Step 3 - RED/GREEN pure validation and convergence helpers

- [x] Add failing examples and Hypothesis properties for exact-point
  deduplication, finite viable filtering, incumbent-best monotonicity,
  scale-aware improvement boundaries, selected-objective tolerance and budget
  enforcement.
- [x] Add failing tests for solved/no-op/typed-failure classification,
  transaction snapshots, exactly-one-target success and bounded diagnostics.
- [x] Implement minimal test-owned outcome and convergence helpers that satisfy
  independent reference oracles; do not duplicate production thermodynamics.
- [x] Add the observation-only delegate with guaranteed restoration, exact
  argument forwarding, one original call, search-mode filtering and detached
  trace values.
- [x] Run the fixed-seed helper/property gate and retain normal shrinking.

**Traceability**: FR-06 through FR-11, FR-16, VR-03 through VR-05,
NFR-01 through NFR-05, PBT-01 through PBT-10 as applicable.

### Step 4 - Exploratory real-CoolProp matrix

- [x] Run every one of the 54 assignments sequentially through the real public
  API, starting with a small explicit search allowance and increasing only when
  evidence requires it.
- [x] Select the smallest stable common or profile-specific allowance that
  produces robust search evidence and remains at or below 50 distinct search
  evaluations per call.
- [x] Record per-case/profile outcome, search/final counts, runtime as a
  diagnostic only, strict-success facts and every unexpected traceback.
- [x] Identify one stable named strict-success sentinel for each profile that
  has at least two viable points, material incumbent improvement and final
  selection of the best observed viable objective.
- [x] If any profile lacks a compliant sentinel within the bound, investigate
  the implementation or revise the design with user approval; do not weaken or
  skip the requirement.

**Traceability**: FR-04 through FR-11, FR-15, FR-16, VR-03 through VR-05,
VR-08, NFR-02 through NFR-04, NFR-06, NFR-07.

### Step 5 - Evidence-backed production corrections

- [x] For each unexpected exception or invalid public transition, add the
  smallest focused failing regression beside the owning component.
- [x] Correct only reproduced valid-input defects in existing production files,
  preserving public APIs and fatal exception tracebacks.
- [x] Run the focused regression plus the exposing benchmark assignment after
  each correction.
- [x] N/A because exploration found reproduced production defects; no
  speculative production changes were made.

**Traceability**: FR-07, FR-10, FR-15 through FR-17, VR-03, VR-08.

### Step 6 - Freeze the 54-case HPR e2e contract

- [x] Create `tests/e2e/test_hpr_mvr.py` with all assignments generated from
  the shared corpus and immutable profiles; do not embed a 54-name list.
- [x] Assert only solved, documented no-op or bounded `HPRTargetingError`
  outcomes for every case, with atomic state and case/profile diagnostics.
- [x] For solved outcomes, assert finite public accounting, detached CoolProp
  evidence, positive useful duty, internally consistent duties, no live engine,
  deep-copy safety and JSON serialization.
- [x] Freeze six explicit sentinel identities and require strict success plus
  bounded best-observed convergence for every profile.
- [x] Keep tests sequential, free of retries, sleeps, network and external
  solvers, with no wall-clock correctness assertion.

**Traceability**: FR-01 through FR-11, FR-15, FR-16, VR-01 through VR-05,
NFR-01 through NFR-07.

### Step 7 - Add the direct process-MVR e2e workflow

- [x] Load packaged `process_mvr.json`, call the public component accessor for
  `Evaporator vapour` with explicit bounded compression controls and validate
  positive finite work/duty, pressure lift, replacement streams, inventory,
  detachment and serialization.
- [x] Run downstream public direct heat integration and validate finite target
  state and serialization.
- [x] Retain and rerun the existing invalid-fluid, fallback and second-stage
  atomicity regressions as complementary coverage.

**Traceability**: FR-12 through FR-14, VR-06, NFR-03, NFR-06, NFR-07.

### Step 8 - Focused verification and refactoring

- [x] Run shared-corpus, helper/property, complete e2e HPR/MVR, focused HPR
  analysis/contract/application and existing direct-target tests with Hypothesis
  seed `20260715`.
- [x] Confirm the matrix is collected by the existing `not solver` selection,
  with no new marker, dependency or workflow change.
- [x] Refactor duplication only after behavior is green; preserve test-owned
  ownership and observation-only instrumentation.
- [x] Run Ruff lint/format, Python compilation, architecture/import checks and
  `git diff --check` for the affected surface.
- [x] Verify no duplicate brownfield files, generated fixture snapshots or
  unbounded diagnostic artifacts were introduced.

**Traceability**: VR-01 through VR-07 and all approved NFRs.

### Step 9 - Code summary and requirement closure

- [x] Create the Code Generation summary with modified/created files, selected
  budgets, six sentinel identities, per-profile outcome counts, confirmed fixes,
  exact focused test evidence and explicit limitations.
- [x] Map FR-01 through FR-17, VR-01 through VR-08 and applicable PBT rules to
  implementation/tests.
- [x] State that the convergence oracle proves bounded progress to the best
  observed candidate, not a global optimum or universal feasibility.
- [x] Mark every completed plan checkbox immediately and update AI-DLC state and
  audit records before presenting Code Generation for approval.

**Traceability**: VR-08 and complete workflow traceability.

## Property-Based Testing plan

- **PBT-01**: assignment, convergence, transition, cache, budget and
  serialization properties are identified.
- **PBT-02/PBT-03**: generated finite traces and corpus sizes exercise valid,
  invalid and boundary partitions with independent fold/reference oracles.
- **PBT-04**: exact-point deduplication is tested for repeated application;
  no new idempotent production operation is claimed.
- **PBT-05**: shared discovery is the corpus oracle; pure convergence results are
  compared with an independently expressed minimum fold.
- **PBT-06**: fresh problem state and existing stateful HPR cache properties
  cover mutable transitions.
- **PBT-07/PBT-08**: finite constrained strategies, shrinking and seed
  `20260715` remain active.
- **PBT-09**: existing pytest plus Hypothesis stack; no dependency change.
- **PBT-10**: properties complement all 54 real CoolProp examples, six real
  convergence sentinels and the real direct-MVR workflow.

## Approval and execution policy

Part 2 must not start until this complete nine-step plan is explicitly approved.
During execution, each checkbox is marked `[x]` in the same interaction in which
its work completes. Any production correction outside the conditional owners
above, any budget over 50, or inability to produce one sentinel per profile is a
scope/design blocker requiring user direction.
