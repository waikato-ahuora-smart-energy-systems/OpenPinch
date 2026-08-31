# Unit 1 Code Generation Summary

## Outcome

Unit 1, Placement Contracts and Pure Model, is implemented. It converts a
validated utility-placement request and a detached all-period feasibility
envelope into a complete immutable decision model. It owns deterministic
template generation, canonical quantities, intersected and ordered bounds,
starting candidates, stable vector encoding/decoding, independent candidate
verification, typed failures, and the detached nested result vocabulary that
Units 2 and 3 will populate and expose.

For count-generated utilities, matching hot/cold members share one temperature
interval and the cold endpoints exactly reverse the hot endpoints. The paired
schema has `N_iso + 2*N_sens` coordinates and independent hot/cold duties.
Explicit or inferred Hot/Cold inventories retain the independent
`2*N_iso + 4*N_sens` schema and existing side ordering.

It intentionally does not perform process targeting, duty allocation,
thermodynamic objective calculation,
numerical optimization, result ranking, workspace caching, presentation, or
CLI integration. Those remain allocated to Units 2 and 3.

## Production Files

| File | Responsibility |
|---|---|
| `OpenPinch/contracts/utility_placement.py` | Frozen extra-forbid enums, quantities, requests, templates, blueprints, envelopes, model/codec values, diagnostics, and nested results |
| `OpenPinch/analysis/utility_placement/errors.py` | Typed runtime/value-error taxonomy with stable primitive JSON-safe context |
| `OpenPinch/analysis/utility_placement/normalization.py` | Strict request handling, `Value.to` conversion adapter, deterministic generation, inventory/economics/kind validation, canonical blueprints |
| `OpenPinch/analysis/utility_placement/bounds.py` | Exact envelope coverage, max/min intersection, caller narrowing, positive-Kelvin checks, order propagation, deterministic starts |
| `OpenPinch/analysis/utility_placement/codec.py` | Stable coordinate schema, complete model build, one-pass encode/decode, structured candidate verification |
| `OpenPinch/analysis/utility_placement/__init__.py` | Explicit specialist facade without root-package export expansion |

The implementation uses the existing Pydantic v2, Pint/`Value`, pytest,
Hypothesis, coverage, and Ruff stack. No dependency, lockfile, database,
frontend, deployment, cache, queue, worker, logger, retry, file I/O, network,
or mutable global was added.

## Test Files

- `tests/analysis/utility_placement/test_contracts.py`
- `tests/analysis/utility_placement/test_normalization.py`
- `tests/analysis/utility_placement/test_bounds.py`
- `tests/analysis/utility_placement/test_codec.py`
- `tests/analysis/utility_placement/test_properties.py`
- `tests/analysis/utility_placement/test_performance.py`
- `tests/strategies/utility_placement.py`
- narrow additions to `tests/architecture/test_dependency_rules.py`,
  `tests/architecture/test_api_boundary.py`, and
  `tests/packaging/test_repo_entrypoints.py`

## RED-GREEN-REFACTOR Evidence

| Slice | RED evidence | GREEN/refactor evidence |
|---|---|---|
| Contracts/errors/imports | Collection failed because the specialist package did not exist | 25 focused tests passed; subsequent contract/architecture/API/packaging set passed 60 tests |
| Request/template normalization | Collection failed because `prepare_template_blueprints` and its module did not exist | 34 contract/normalization/property tests passed |
| Bounds/envelope/order/starts | Collection failed because envelope contracts and bounds functions did not exist | 14 bounds/property/performance tests passed after correcting the intended effective-bound permutation assertion |
| Schema/codec/verifier | Collection failed because model and codec exports did not exist | 16 codec/property tests passed |
| Nested results/errors | Collection failed because result contracts did not exist | 36 result/contract/property tests passed; validation-edge expansion retained green behavior |
| Complete refactor | No behavior was changed without a green test boundary | Ruff passed and the focused plus compatibility suite passed 110 tests |

Every production behavior slice was preceded by an observed missing-behavior
failure. Formatting, import organization, validation-edge coverage, and
precision-correct oracle assertions were completed behind green behavior.

## Verification Results

| Gate | Result |
|---|---|
| Focused example tests | 50 passed |
| Fixed-seed property tests | 11 passed with seed `20260715`; default shrinking enabled |
| Performance tests | 3 passed |
| Coupled representative performance | 5 tests passed for 10 isothermal plus 10 sensible pairs over 100 periods; the generated model has 30 decision coordinates and remains below the 250 ms requirement |
| Focused plus architecture/API/packaging gate | 110 passed |
| New-module statement coverage | 95 percent threshold passed; 72 focused tests in the coverage run |
| Ruff | Passed on every touched Unit 1 Python path |
| Broader non-solver analysis/contracts/architecture/packaging regression | 1,641 passed and 4 skipped in 201.56 seconds |
| Distribution build | `openpinch-0.5.4-py3-none-any.whl` and source distribution built successfully in an isolated temporary directory |
| Installed-wheel smoke | Specialist request normalization and four-template generation imported and executed successfully from the locally installed wheel |

The first coverage attempt used package source discovery and triggered a Python
3.14 NumPy double-load error before test collection. The successful coverage
gate omitted source pre-import discovery and selected the Unit 1 paths at report
time. Ordinary pytest, fixed-seed PBT, and the broader regression did not show
this tooling interaction.

## Property-Based Testing Compliance

| Rule | Result |
|---|---|
| PBT-01 | All 15 identified Unit 1 properties have example/property coverage |
| PBT-02 | Reusable strategies generate requests, quantities, intervals, identities, dimensions, bounds, and nested values |
| PBT-03 | Inventory, symmetry, canonical-value, permutation, max/min, ordering, dimension, round-trip, verifier, and context invariants pass |
| PBT-04 | Request and blueprint normalization idempotence pass |
| PBT-05 | `Value.to`, explicit max/min intersection, and easy candidate verification oracles pass; the structured-grid optimizer oracle remains allocated to Unit 2 |
| PBT-06 | N/A: Unit 1 owns frozen pure values and no state machine; workspace state remains Unit 3 |
| PBT-07 | Shared strategies are isolated in `tests/strategies/utility_placement.py` |
| PBT-08 | Shrinking remains enabled, seed `20260715` passes, and the float-associativity counterexample was retained as a tolerance-correct oracle assertion |
| PBT-09 | Existing Hypothesis/pytest/CI integration is reused without dependency changes |
| PBT-10 | Example and property tests remain separately identifiable and both are required gates |

There are no blocking PBT findings. Security and Resiliency extensions remain
disabled. Their full rules were not applied; ordinary boundary validation and
fail-fast behavior still satisfy the approved Unit 1 requirements.

## Compatibility and Protected Surfaces

- `OpenPinch.__all__` remains exactly `PinchProblem` and `PinchWorkspace`.
- The feature is available only through specialist imports.
- Analysis depends only on allowed analysis/contracts/domain owners; contracts
  do not depend on analysis or application.
- Existing shared contracts and configuration defaults were not changed.
- `pyproject.toml` and `uv.lock` were already dirty at the captured baseline for
  unrelated Excel/VBA tooling (`oletools`) and were not modified by Unit 1.
- All unrelated working-tree edits were preserved.
- The built wheel contains all six planned utility-placement production files.

## Story and Requirement Delivery

- UPO-01 Unit 1 primary scope is complete: strict counts, symmetric generated
  and explicit inventories, kind-specific temperature behavior, bounds,
  coordinates, and starts.
- UPO-08 Unit 1 supporting scope is complete: typed fatal errors and structured
  candidate diagnostics. Service and public recovery remain Units 2 and 3.
- UPO-09 Unit 1 primary scope is complete: frozen detached schemas, JSON
  round-trips, explicit units, stable specialist imports, and protected root
  exports.
- UPO-12 Unit 1 scope is complete: recorded TDD slices, full PBT compliance,
  deterministic seed, shrinking, coverage, performance, regression, and
  package evidence. The cross-cutting story remains open for Units 2 and 3.

All 43 Unit 1 business rules and U1-NFR-001 through U1-NFR-018 have a production
owner, example/property evidence, or explicit downstream-unit allocation.

## Unit 2 Handoff

Unit 2 should construct `PlacementFeasibilityEnvelope` values from direct or
Total Site targeting and consume the specialist facade:

1. normalize the request and prepare template blueprints;
2. construct the detached envelope with exact coordinate coverage per period;
3. build `UtilityPlacementModel`;
4. evaluate decoded candidates and return deterministic penalties for
   `CandidateVerification` failures;
5. populate the Unit 1 result contracts with duty, entropy/exergy, ranking,
   and termination values.

Unit 2 must not mutate the frozen request/model and must preserve the stable
coordinate and template identities defined here.
