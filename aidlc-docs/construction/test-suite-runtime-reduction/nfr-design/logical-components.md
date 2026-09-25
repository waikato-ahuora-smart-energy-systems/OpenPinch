# Test-Suite Runtime Reduction Logical Components

## Component Map

| Component | Existing owner | Planned responsibility | Dependencies |
|---|---|---|---|
| Tutorial search profile | `scripts/generate_tutorial_notebooks.py` | Own the smallest validated Notebook 19 budgets | Utility-placement public API |
| Generated tutorial | `OpenPinch/tutorials/notebooks/19_utility_placement_optimisation.ipynb` | Preserve process/site convergence evidence | Tutorial generator |
| Placement evidence fixtures | `tests/application/test_utility_placement.py` and support helpers if needed | Create deterministic solved evidence and isolate consumers | `PinchWorkspace`, copy/serialization boundary |
| Batch placement tests | `tests/application/test_utility_placement_batch.py` | Retain generated order/scaling invariants with bounded setup | Existing Hypothesis strategies |
| CoolProp audit | `tests/application/test_coolprop_hpr_audit.py` | Retain representative real search and bounded structural oracles | HPR public services and contracts |
| HPR/MVR E2E corpus | `tests/e2e/test_hpr_mvr.py` | Remain the complete 54-case real-service matrix | Standard benchmark fixtures |
| Optimizer benchmarks | `tests/optimisation/test_backends.py` | Own the two `performance` convergence cases | Existing optimisation backends |
| Documentation smoke | `tests/packaging/test_docs_build.py` | Own the single `docs` warning-strict build | `scripts/build_docs.py`, Sphinx |
| Fresh-process batches | `tests/architecture/test_cold_imports.py`, `test_api_boundary.py` | Batch compatible isolation checks with case attribution | Python child interpreter |
| Marker registry | `pytest.ini` | Declare `docs` and `performance`; retain `tespy` and `solver` | pytest |
| CI lane ownership | Three `.github/workflows/ci-*.yml` files | Run ordinary and specialized selections exactly once | `uv`, pytest, Coverage.py |
| Workflow guards | `tests/packaging/test_packaging_metadata.py` | Enforce marker exclusions, inclusions, timeouts, and job presence | Workflow source text |
| Verification record | `aidlc-docs/construction/test-suite-runtime-reduction/` | Record commands, timings, coverage, and exceptions | All test components |

## Dependency and Update Order

1. Add failing or tightened workflow, marker, generator-budget, and isolation
   contract tests.
2. Register markers and update CI lane selections plus dedicated jobs.
3. Bound the generator and regenerate Notebook 19.
4. Introduce placement fixture isolation and reduce redundant CoolProp work.
5. Batch compatible child-process checks.
6. Validate focused runtime and correctness, then refine budgets.
7. Run all specialized lanes, ordinary coverage, and final profile.

The order keeps generated artifacts downstream of their source and keeps CI
contract tests synchronized with workflow edits.

## Data and State Boundaries

- Solved problem objects never cross a process boundary.
- Module fixtures expose an immutable source or create per-test deep copies.
- Serialized dictionaries are copied before mutation.
- Marker selection is declarative and centrally registered.
- Child-process batches communicate only through exit status, stdout, and
  stderr; no persistent cache or temporary shared state is required.
- Notebook source remains deterministic JSON generated from one Python script.

## Failure Behavior

- Missing dedicated job or incorrect marker expression fails packaging tests.
- An optimizer budget that loses feasibility or convergence fails the focused
  real-execution test and is increased before broader verification.
- Mutation leakage fails the isolation regression with before/after payloads.
- Batched import failure reports the logical case and target module/package.
- Timeout fails the owning command/job; it is never converted to a skip.
- Coverage below 95 percent fails before completion and triggers the documented
  targeted-coverage or combine fallback.

## Infrastructure Assessment

No queue, cache server, database, load balancer, monitoring service, deployment
resource, or other infrastructure component is required. GitHub Actions files
are CI configuration owners, not runtime infrastructure. Availability,
disaster recovery, authentication, authorization, and data-encryption patterns
are N/A to this internal test-performance unit.

## Property-Based Testing Ownership

- Existing utility-placement domain strategies remain centralized under
  `tests/strategies/`.
- Existing generated order, scaling, and physical invariants remain in CI.
- The fixed CI seed remains `20260715`; shrinking remains enabled.
- Explicit example tests continue to pin process/site placement and real
  HPR/MVR cases.
- Any reduced-search failure found by generated data is retained as a concrete
  regression before further fixture consolidation.
