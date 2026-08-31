# Unit 2 Placement Evaluation and Optimisation Service Code Generation Plan

## Plan Status

This is the single source of truth for Unit 2 Code Generation. The user has
approved routine checkpoints through completion unless an unexpected issue
requires a decision. Every production slice below remains strict RED-GREEN-
REFACTOR, and each checkbox must be updated in the interaction that completes
the work.

## Unit Context and Readiness

- [x] **Step 1 - Reconcile approved design and readiness.** Unit 1 contracts,
  pure model, codec, bounds, errors, and strategies are implemented and
  approved. Unit 2 Functional Design, 32 NFR Requirements, and 19 NFR patterns
  are complete; Infrastructure Design is skipped because no infrastructure
  boundary exists.
- [x] **Step 2 - Fix production/test ownership and protect the worktree.** The
  paths below extend the existing specialist owner and focused tests. Existing
  unrelated Excel/VBA, workbook, targeting, workflow, `pyproject.toml`, and
  `uv.lock` changes remain user-owned and must not be altered. Root exports,
  existing optimizer/target/turbine APIs, and dependency declarations are
  protected compatibility surfaces.
- [x] **Step 3 - Record whole-plan approval.** The user's exact instruction,
  "Approve through to completion unless something unexpected happens.",
  authorizes this plan and routine transitions. No unexpected issue is known.

## Planned Production Paths

Modified:

- `OpenPinch/contracts/utility_placement.py` - additive optimizer fields and
  Unit 2 internal/result validation refinements only.
- `OpenPinch/analysis/utility_placement/errors.py` - Unit 2 operational errors.
- `OpenPinch/analysis/utility_placement/__init__.py` - specialist Unit 2 exports.

Created:

- `OpenPinch/analysis/utility_placement/context.py`
- `OpenPinch/analysis/utility_placement/allocation.py`
- `OpenPinch/analysis/utility_placement/thermodynamics.py`
- `OpenPinch/analysis/utility_placement/economics.py`
- `OpenPinch/analysis/utility_placement/cogeneration.py`
- `OpenPinch/analysis/utility_placement/penalties.py`
- `OpenPinch/analysis/utility_placement/evaluation.py`
- `OpenPinch/analysis/utility_placement/optimisation.py`
- `OpenPinch/analysis/utility_placement/service.py`

Logical components may share only these planned files; no duplicate
`*_new.py`/`*_modified.py` variants are permitted. Application accessors,
presentation, CLI, and the notebook belong to Unit 3.

## Planned Test Paths

Created under `tests/analysis/utility_placement/`:

- `test_options.py`
- `test_context.py`
- `test_allocation.py`
- `test_thermodynamics.py`
- `test_economics.py`
- `test_cogeneration.py`
- `test_penalties.py`
- `test_evaluation.py`
- `test_optimisation.py`
- `test_service.py`
- `test_unit2_properties.py`
- `test_unit2_performance.py`

Modified narrowly:

- `tests/strategies/utility_placement.py`
- `tests/architecture/test_dependency_rules.py`
- `tests/architecture/test_api_boundary.py`
- `tests/packaging/test_repo_entrypoints.py`

## Part 2 Execution Steps

- [x] **Step 4 - Capture the Unit 2 baseline.** Run current Unit 1 focused,
  optimizer-service, target-utility, turbine, architecture, and packaging tests;
  record current root/specialist APIs and unrelated dirty paths.
- [x] **Step 5 - RED: optimizer options and operational error contract.** Add
  examples and JSON properties for backward-compatible option defaults, method,
  run count, cluster tolerance, local method, sorted JSON-safe overrides,
  existing-optimizer mapping, invalid method/options before callbacks, and all
  Unit 2 error subclasses/context. Confirm expected failure.
- [x] **Step 6 - GREEN: options reconciliation and Unit 2 errors.** Extend the
  frozen contract and implement public-optimizer option mapping/error
  translation with no private backend import or dependency change.
- [x] **Step 7 - RED: detached context and envelope population.** Add direct
  and Total Site numerical-snapshot examples/properties for period order,
  weights, residuals, ambient kelvin, profile completeness, exact blueprint
  coordinates, source-copy invariance, pickle round-trip, and invalid resolved
  scope/context failures. Confirm expected failure.
- [x] **Step 8 - GREEN: immutable context builder.** Implement frozen context,
  period/profile/entropy-slice/turbine-setting values, source normalization,
  Unit 1 feasibility-envelope population, and stable errors.
- [x] **Step 9 - RED: allocation reconstruction and coverage.** Add examples
  and properties for fresh replay, existing allocation-adapter invocation,
  stable level/interval keys, hot/cold conservation, combined tolerance,
  zero-duty levels, no clipping, all-period failure, and repeat non-mutation.
- [x] **Step 10 - GREEN: allocation and coverage.** Implement candidate-local
  reconstruction protocols, default numerical allocation over snapshot load
  profiles using the existing targeting ownership boundary, assignment slices,
  and structured infeasibility.
- [x] **Step 11 - RED: thermodynamic kernels.** Add hand-calculable hot/cold
  sensible and near-isothermal examples; decimal-oracle properties; sign,
  kelvin, finite, stable-sum, entropy-balance, non-negative generation, and
  exergy identities. Confirm expected failure.
- [x] **Step 12 - GREEN: thermodynamic evaluator.** Implement binary64
  `log1p`/limit kernels, real-interval attribution, canonical sums, noise
  handling, typed invariant failures, and public breakdown conversion.
- [x] **Step 13 - RED: monetary and cogeneration evaluators.** Add explicit
  purchase/credit/net examples and properties; eligible-hot filtering/order,
  no-eligible zero result, explicit-over-configuration settings, fresh turbine,
  negative net cost, and recoverable/run-level failure cases. Confirm failure.
- [x] **Step 14 - GREEN: monetary and turbine adapters.** Implement unit-
  normalized economics and a detached adapter over the existing multi-stage
  turbine without duplicating steam physics.
- [x] **Step 15 - RED: aggregation and scalar partition.** Add weighted-sum,
  zero-weight-feasibility, period-permutation tolerance, induction, feasible
  ordering, negative objective, normalized violation, and disjoint-range
  properties. Confirm expected failure.
- [x] **Step 16 - GREEN: aggregation and penalty mapping.** Implement canonical
  `fsum`, context-derived positive scale, monotone feasible transform, and
  bounded graded penalty with physical values retained separately.
- [x] **Step 17 - RED: evaluation session, compact memo, and diagnostics.** Add
  examples and command-model properties for exact keys, signed zero, one replay
  per process/session, evaluation-budget cap, compact records, top-ten stable
  diagnostics, pickle reconstruction, worker isolation, and all-period replay.
- [x] **Step 18 - GREEN: evaluation session and memo.** Implement pickle-safe
  frozen payloads, lazily recreated lock/memo, deterministic evaluation flow,
  compact bounded records, and diagnostic accumulator.
- [x] **Step 19 - RED: optimization coordination and grid oracle.** Add option
  forwarding, one optimizer call, start/backend union, exact deduplication,
  canonical parent full replay, feasible-only physical ordering/limit,
  termination translation, fixed-seed repeat, typed exhaustion, structured-grid
  oracle properties, and one tiny real dual-annealing regression.
- [x] **Step 20 - GREEN: optimization coordinator.** Implement only the public
  solver-neutral boundary, candidate normalization, parent re-evaluation, and
  typed error translation.
- [x] **Step 21 - RED: complete service and result assembly.** Add direct and
  Total Site, thermodynamic-default, monetary/cogeneration, multiperiod,
  alternatives, detached result JSON, source non-mutation, bounded diagnostics,
  and complete-or-typed-failure examples/properties. Confirm expected failure.
- [x] **Step 22 - GREEN: Unit 2 service facade.** Implement end-to-end context,
  model, evaluation, optimization, and frozen result assembly; complete
  specialist exports without root expansion.
- [x] **Step 23 - REFACTOR: complete numerical service while green.** Remove
  duplication; tighten type/docstring/module ownership; verify no private
  backend imports, duplicate target/turbine physics, hidden retries, global
  state, application/presentation dependency, or unbounded trace.
- [x] **Step 24 - Run properties and oracle gates.** Execute separately named
  Unit 2 properties with seed `20260715`, shrinking, decimal oracle,
  structured-grid oracle, memo command model, and fixed-seed reproducibility;
  capture any shrunk regression before proceeding.
- [x] **Step 25 - Run performance and focused quality gates.** Enforce p95 at
  50 ms for allocated kernels, 1 second for cold replay, and 1 ms for memo hit;
  run all focused Unit 1/2 tests, Ruff, branch coverage at 95%, architecture,
  root/API, and package-entrypoint tests.
- [x] **Step 26 - Run proportional regression and distributions.** Run the
  complete available non-solver suite, build wheel/source artifacts in an
  isolated temporary directory, confirm every specialist file is packaged, and
  smoke-import/run a small service case from the installed wheel. Do not touch
  unrelated `pyproject.toml`/`uv.lock` changes.
- [x] **Step 27 - Generate implementation summary and traceability.** Create
  `aidlc-docs/construction/utility-placement-evaluation-optimisation-service/code/code-summary.md`
  with file ownership, RED-GREEN-REFACTOR evidence, tests, properties,
  performance, coverage, compatibility, limitations, and Unit 3 handoff. Mark
  Unit 2 story slices complete only with evidence.
- [x] **Step 28 - Record generated-code approval and advance.** Under the
  user's continuing authorization, validate and approve Unit 2 generated code
  if no unexpected issue remains, then start Unit 3 Functional Design.

## Story and Requirement Coverage

| Scope | Unit 2 delivery |
|---|---|
| UPO-02 | Complete detached numerical default-thermodynamic workflow |
| UPO-03 | Exact per-period hot/cold coverage and zero-duty behavior |
| UPO-04 | Stable entropy/exergy objective and analytical evidence |
| UPO-05 | Thermal cost, eligible cogeneration, electricity credit, net cost |
| UPO-06 | Shared placement, raw period weights, all-period feasibility |
| UPO-07 | Feasible alternatives and physical deterministic ranking |
| UPO-08 | Candidate versus run-level typed failures and exhaustion |
| UPO-11 | Bounded seeded optimization and structured-grid oracle |
| UPO-12 | Strict TDD, full applicable PBT, quality and compatibility gates |
| FR-005, FR-007 through FR-012, FR-014 | All Unit 2 numerical/service ownership |
| U2-NFR-001 through U2-NFR-032 | All approved capacity, performance, numerical, concurrency, reliability, observability, compatibility, and test obligations |

Unit 3 retains public problem/all-period/workspace methods, observation,
reporting, documentation, and the single executable notebook. No CLI is
planned.

## PBT Compliance Matrix for Planning

| Rule | Status | Planned evidence |
|---|---|---|
| PBT-01 | Compliant | Every Functional Design property is assigned to Steps 5-24 |
| PBT-02 | Compliant | Options/result JSON round trips and pickle reconstruction properties |
| PBT-03 | Compliant | Coverage, entropy, monetary, aggregation, feasibility, ordering, isolation, and boundedness invariants |
| PBT-04 | Compliant | Replay and per-process exact memo idempotence |
| PBT-05 | Compliant | Decimal entropy and structured-grid optimization oracles |
| PBT-06 | Compliant | Memo command model plus isolated worker/parent state behavior |
| PBT-07 | Compliant | Reusable domain snapshots, allocations, branches, contexts, violations, commands, and tiny-model strategies |
| PBT-08 | Compliant | Fixed seed `20260715`, shrinking, and permanent regression capture |
| PBT-09 | Compliant | Existing Hypothesis/pytest dependency and CI integration |
| PBT-10 | Compliant | Every critical property is paired with explicit analytical/business examples |

There is no blocking PBT finding. Security and Resiliency remain disabled.

## Artifact Applicability

| Artifact | Decision |
|---|---|
| Business logic/service modules | Applicable in the planned specialist paths |
| Example/property/performance tests | Applicable and test-first |
| Public transport/API layer | N/A; Unit 3 owns in-process public accessors and no remote API exists |
| Repository/database/migrations | N/A; no persistence |
| Frontend/UI | N/A |
| Deployment/infrastructure | N/A |
| User documentation/notebook | Deferred to Unit 3 |
| Dependency/lock changes | N/A; existing stack is sufficient |
