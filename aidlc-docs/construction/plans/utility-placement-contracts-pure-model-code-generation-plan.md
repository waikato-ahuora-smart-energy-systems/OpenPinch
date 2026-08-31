# Utility Placement Contracts and Pure Model Code Generation Plan

## Plan Status

This is the single source of truth for Unit 1 Code Generation. Part 2 will not
begin until the entire plan is explicitly approved. Implementation must follow
the numbered sequence, and every completed checkbox must be updated in the
same interaction as the corresponding work.

## Unit Context and Readiness

- [x] **Step 1 - Reconcile the approved design and implementation readiness.**
  Confirmed that Functional Design, NFR Requirements, and NFR Design are
  approved; Infrastructure Design is correctly skipped; the package already
  owns Pint conversion through `OpenPinch.domain.value.Value.to`; and the new
  specialist production paths do not currently exist. Unit 1 has no dependency
  on Unit 2 or Unit 3 implementation. Unit 2 will consume its immutable model,
  bounds, codec, and error contracts.
- [x] **Step 2 - Fix the implementation boundary and protected files.** The
  implementation will create only the production and focused-test paths listed
  below, make narrow additions to three compatibility test files, and avoid
  unrelated dirty-worktree changes. Root `OpenPinch.__all__`, shared contract
  defaults, `pyproject.toml`, and `uv.lock` are protected compatibility
  surfaces and are not planned for modification.
- [x] **Step 3 - Obtain explicit approval of this entire Code Generation plan.**
  No production code or tests will be created before this checkbox is approved.

## Planned Production Paths

The following files will be created at the workspace root, never under
`aidlc-docs/`:

- `OpenPinch/contracts/utility_placement.py` - frozen extra-forbid request,
  template, coordinate, bound, diagnostic, and result contracts plus enums.
- `OpenPinch/analysis/utility_placement/__init__.py` - explicit specialist
  facade and `__all__`; no root-package export expansion.
- `OpenPinch/analysis/utility_placement/errors.py` - stable typed exception
  taxonomy and JSON-safe context.
- `OpenPinch/analysis/utility_placement/normalization.py` - count validation,
  deterministic template generation, canonical quantity conversion, and
  request/template blueprint normalization.
- `OpenPinch/analysis/utility_placement/bounds.py` - envelope intersection,
  caller narrowing, order propagation, feasibility checks, and deterministic
  starting-candidate construction.
- `OpenPinch/analysis/utility_placement/codec.py` - stable vector schema,
  encode/decode, dimension checks, and candidate verification.

No repository, database, remote API, frontend, worker, cache, queue,
infrastructure, or deployment source is applicable to this pure library unit.
The public API generated here is the specialist Python import surface only.

## Planned Test Paths

The plan will create:

- `tests/analysis/utility_placement/__init__.py`
- `tests/analysis/utility_placement/test_contracts.py`
- `tests/analysis/utility_placement/test_normalization.py`
- `tests/analysis/utility_placement/test_bounds.py`
- `tests/analysis/utility_placement/test_codec.py`
- `tests/analysis/utility_placement/test_properties.py`
- `tests/analysis/utility_placement/test_performance.py`
- `tests/strategies/utility_placement.py`

It will make only focused additions to:

- `tests/architecture/test_dependency_rules.py`
- `tests/architecture/test_api_boundary.py`
- `tests/packaging/test_repo_entrypoints.py`

Examples and property tests will remain separately identifiable. Every
production behavior slice below starts with a test that fails for the expected
missing or incorrect behavior, followed by the minimum implementation needed
to make it pass, then a green-only refactor checkpoint.

## Part 2 Execution Steps

- [x] **Step 4 - Capture a clean Unit 1 baseline.** Record the targeted path
  absence, current root-export contract, dependency direction, relevant package
  smoke behavior, and existing focused-test results. Record but do not alter
  unrelated working-tree changes. A pre-existing failure blocks only when it
  affects a Unit 1 boundary and cannot be isolated.
- [x] **Step 5 - RED: specify the contract, error, import, and compatibility
  boundary.** Create example tests for count validation including Boolean
  rejection, objective defaults, frozen extra-forbid models, stable enum and
  field values, JSON-safe inputs, canonical schema expectations, typed errors,
  direct specialist imports, unchanged root exports, dependency direction, and
  source/repository package entry points. Add foundational reusable Hypothesis
  strategies for valid and invalid counts, finite quantities, requests, and
  templates. Run the focused tests and record their expected failure before
  creating production modules.
- [x] **Step 6 - GREEN: implement the minimum public contracts and error
  vocabulary.** Create `utility_placement.py`, `errors.py`, and the minimum
  package facade required by Step 5. Use Pydantic v2 frozen extra-forbid models,
  reject non-finite values and Boolean counts, keep exception context primitive
  and stable, and expose no backend or mutable objects. Run the Step 5 tests to
  green without speculative Unit 2 optimization behavior.
- [x] **Step 7 - REFACTOR: harden the contract slice while green.** Remove
  duplication, add precise docstrings and annotations, centralize stable enum
  values and tolerance metadata, and rerun the focused example, architecture,
  and packaging tests. Do not expand `OpenPinch.__all__` or change existing
  shared defaults.
- [x] **Step 8 - RED: specify request and template normalization.** Add examples
  and properties for request-normalizer idempotence, request JSON round-trip,
  deterministic template inventory and names, hot/cold symmetry, caller order
  and placement rank, canonical finite values, template-normalizer idempotence,
  and conversion equivalence to `Value.to`. Include empty optional inputs,
  mixed compatible units, extreme finite values, dimension errors, and count
  boundaries. Run and record the expected failure before implementing the
  normalizer.
- [x] **Step 9 - GREEN: implement deterministic normalization and blueprints.**
  Add one testable conversion adapter over `Value.to`; normalize each public
  value once; generate count-only defaults with a 0.01 `delta_degC`
  isothermal span; preserve stable identity and order; and return immutable
  request/template blueprints. Satisfy Step 8 examples and properties using
  tuple-backed ordered data and ephemeral indexed lookups only.
- [x] **Step 10 - RED: specify bounds, feasibility, ordering, and starts.** Add
  examples and properties for all-period envelope intersection, period
  permutation invariance of effective bounds while retaining observable period
  order, the max-lower/min-upper oracle, caller-bounds narrowing only, fixed
  equal bounds, hot-descending and cold-ascending minimum separation,
  propagation invariants, infeasibility diagnostics, deterministic starts, and
  local feasibility checks for every generated start. Add the representative
  10-isothermal/10-sensible-per-side, 100-period performance fixture and a
  linear-scaling comparison now so it fails before the bounds implementation
  exists.
- [x] **Step 11 - GREEN: implement bounds and deterministic starting
  candidates.** Build stable coordinate indices once; intersect envelope and
  caller bounds in linear passes; propagate adjacent ordering constraints;
  retain equal coordinates; distinguish fatal model infeasibility from ordinary
  candidate invalidity; and create deterministic feasible starting points.
  Make Step 10 green without retries, caching, parallelism, or hidden recovery.
- [x] **Step 12 - RED: specify vector codec and candidate verification.** Add
  examples and properties for both encode/decode round trips, the dimension
  identity `2 * N_iso + 4 * N_sens`, coordinate uniqueness and family order,
  length and finiteness errors, quantity reconstruction, easy-to-check verifier
  oracles, the cross-component rule that every generated start passes the
  candidate verifier, and structured non-throwing candidate diagnostics. Cover
  zero sensible levels and fixed-coordinate vectors. Run and record the
  expected failure before codec production code.
- [x] **Step 13 - GREEN: implement the stable vector schema and codec.** Create
  the required coordinate families in exact order, precompute index metadata,
  encode/decode in one pass, and verify dimensions, finiteness, bounds, and
  ordering with centralized tolerances. Complete the explicit specialist facade
  and make Step 12 green.
- [x] **Step 14 - RED: specify nested results and stable error serialization.**
  Extend examples and properties to cover nested result JSON round-trip, metric
  and diagnostic retention, thermodynamic-default metadata, optional monetary
  and cogeneration fields without implementing their Unit 2 calculations,
  error-code/context invariants, safe primitive payloads, and schema snapshots.
  Run and record the expected failure before completing those contracts.
- [x] **Step 15 - GREEN: complete result, diagnostic, and exception contracts.**
  Add only the Unit 1 shapes needed by downstream units, preserve typed fatal
  errors versus ordinary candidate diagnostics, keep all payloads frozen and
  JSON-safe, and make Step 14 green. Do not add optimizer, objective evaluation,
  solver state, reporting, CLI, or workspace integration.
- [x] **Step 16 - REFACTOR: simplify the complete pure model while green.**
  Review component boundaries, imports, naming, schema duplication, conversion
  calls, algorithmic passes, and error routing. Refactor only behind passing
  tests. Verify no circular dependency, mutable global state, global Pint
  mutation, logging, retry, cache, I/O, or duplicate existing implementation
  has been introduced.
- [x] **Step 17 - Run Unit 1 quality, property, performance, and compatibility
  gates.** Run focused example tests; separately run property tests with
  Hypothesis seed `20260715` and shrinking enabled; run the p95 250 ms
  representative benchmark and linear-scaling assertion; run the touched
  architecture, package API, and installed/source entry-point smoke tests; run
  Ruff on every touched Python path; and collect at least 95 percent statement
  coverage for new production modules. Any failure returns to the owning RED /
  GREEN slice rather than weakening an invariant.
- [x] **Step 18 - Run proportional regression and packaging verification.** Run
  the relevant non-solver repository suite, build distribution artifacts using
  the existing repository mechanism, smoke-import the specialist API from the
  built wheel/source context, and verify that `pyproject.toml`, `uv.lock`, root
  exports, shared contracts, and existing configuration defaults are unchanged.
  Record any environmental skip precisely; do not claim a skipped gate passed.
- [x] **Step 19 - Generate the implementation summary and complete stage
  traceability.** Create
  `aidlc-docs/construction/utility-placement-contracts-pure-model/code/code-summary.md`
  with files created/modified, RED-GREEN-REFACTOR evidence, test and performance
  results, PBT evidence, design decisions, integration points, compatibility
  checks, limitations, and Unit 2 handoff. Mark the Unit 1 portions of UPO-01,
  UPO-08, UPO-09, and UPO-12 complete only when their acceptance coverage is
  demonstrated. Validate the summary before creation and then present the
  standardized generated-code approval choice.

## Requirement and Story Coverage

| Scope | Unit 1 code-generation coverage |
|---|---|
| UPO-01 | Validated count-based request, default objective metadata, deterministic template blueprint, and stable specialist import |
| UPO-08 | Unit 1 result, diagnostic, and serialization contract consumed by later evaluation/reporting |
| UPO-09 | Temperature-difference semantics, ordering, bounds, and constraint-ready coordinate model |
| UPO-12 | Unit conversion, count validation, degenerate bounds, infeasibility, candidate diagnostics, and regression coverage |
| Functional rules | All 43 approved rules allocated across Steps 5-16 and verified in Step 17 |
| U1-NFR-001 through U1-NFR-018 | Contract, algorithmic, performance, maintainability, compatibility, and test gates allocated across Steps 5-18 |

UPO-02 through UPO-07 and UPO-10 optimization/evaluation behavior belongs to
Unit 2. UPO-11 and the user-facing portions of UPO-08 belong to Unit 3. No
story is claimed complete beyond its Unit 1 slice.

## Artifact Applicability

| Code Generation artifact | Decision |
|---|---|
| Business logic and domain contracts | Applicable; generated in the six planned production files |
| Unit and property tests | Applicable; generated before each production slice under `tests/analysis/utility_placement/` and `tests/strategies/` |
| Python API boundary | Applicable; explicit specialist facade only |
| Web/remote API | N/A; no service transport is in Unit 1 |
| Repository layer | N/A; Unit 1 is pure and owns no persistence |
| Frontend components | N/A; presentation integration belongs to Unit 3 |
| Database artifacts or migrations | N/A; no persistence model exists |
| Deployment artifacts | N/A; no infrastructure or new runtime service exists |
| Public user documentation | Deferred to Unit 3; Unit 1 adds docstrings and the required implementation summary |
| Dependency or lockfile changes | N/A; approved dependencies are already present and version-compatible |

## Property-Based Testing Compliance Matrix

| Rule | Status in this plan | Evidence or rationale |
|---|---|---|
| PBT-01 Property Identification | Compliant | All 15 approved properties are allocated to Steps 8, 10, 12, and 14 |
| PBT-02 Generators and Strategies | Compliant | Step 5 creates reusable domain strategies; later RED steps extend them for valid, invalid, boundary, fixed-coordinate, and nested-result data |
| PBT-03 Invariants and Metamorphic Properties | Compliant | Inventory, symmetry, permutation, ordering, round-trip, dimension, verifier, and context invariants are explicit |
| PBT-04 Idempotence | Compliant | Request and template normalization idempotence is mandatory in Step 8; no other Unit 1 operation is a mutating normalizer |
| PBT-05 Oracle-Based Testing | Compliant | `Value.to`, max-lower/min-upper intersection, and easy candidate verification are independent oracles; structured-grid optimization belongs to Unit 2 |
| PBT-06 Model-Based Testing | N/A | Unit 1 has no state machine or operation sequence; its contracts are frozen pure values. Stateful workspace behavior is deferred to Unit 3 |
| PBT-07 Strategy Reuse | Compliant | Reusable strategies live in `tests/strategies/utility_placement.py` and are shared across focused property modules |
| PBT-08 Shrinking and Failure Reproduction | Compliant | Default shrinking remains enabled; fixed seed `20260715` is run separately; any discovered minimal counterexample becomes a named regression example |
| PBT-09 Framework Integration | Compliant | Existing Hypothesis/pytest dependencies and CI integration are reused unchanged and reverified in Step 17 |
| PBT-10 Test Balance | Compliant | Example tests and property tests remain separately identifiable and both are mandatory in every relevant TDD slice |

There are no blocking PBT findings in the plan. Security Baseline and
Resiliency Baseline remain disabled by approved Extension Configuration; their
full rules are not applied. Boundary validation and fail-fast behavior remain
ordinary Unit 1 requirements rather than extension activation.

## Completion Criteria

Part 2 is complete only when every execution checkbox is marked, all planned
files and only justified compatibility-test edits are accounted for, RED and
GREEN evidence exists for each production slice, all applicable PBT rules are
demonstrated, no unresolved Unit 1 test or quality failure remains, and the
implementation summary has been validated. The workflow will then request the
standardized choice between Request Changes and Continue to Next Stage.

## Plan Approval Question

How should Unit 1 Code Generation proceed?

A) Request Changes - revise the Code Generation plan before implementation

B) Approve Entire Plan - approve all numbered steps and begin Part 2 with the
strict TDD sequence

[Answer]: B
