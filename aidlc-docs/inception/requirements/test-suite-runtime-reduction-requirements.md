# Test-Suite Runtime Reduction Requirements

## Intent Analysis

- **User request**: Address the top seven issues identified by the measured
  test-suite performance audit.
- **Request type**: Internal test and CI performance enhancement.
- **Scope estimate**: Multiple repository components: generated tutorials,
  utility-placement tests, HPR/MVR integration tests, optimisation benchmarks,
  architecture tests, documentation verification, and GitHub Actions.
- **Complexity**: Moderate. The edits are test-focused, but the 95 percent
  branch-coverage gate and the recently strengthened HPR/MVR evidence must not
  regress.
- **Requirements depth**: Standard. The audit supplies measured baselines and
  a precise seven-item scope, so no additional clarification is required.

## Baseline

The reproducible serial command using Hypothesis seed `20260715` completed with
3,455 passes, 6 skips, and 4 solver deselections in 600.89 seconds without
coverage. The equivalent branch-coverage run took 762.76 seconds and reported
96 percent coverage.

The measured hotspots were:

1. Notebook 19 used 100 iterations, 100 evaluations, 20 retained candidates,
   and 10 optimizer runs; its execution took 44.51 seconds.
2. Utility-placement application tests took 105.13 seconds, with additional
   batch and Notebook 19 placement work bringing the family to 172.56 seconds.
3. TESPy-marked tests ran in the ordinary lane and again in the dedicated TESPy
   lane; real TESPy tests consumed 52.74 seconds in the local serial profile.
4. Two cross-backend optimizer benchmarks took 33.42 seconds using four runs
   and broad iteration/evaluation budgets.
5. The pytest Sphinx smoke took 10.43 seconds while CI also built the same
   documentation in a dedicated job.
6. The CoolProp HPR audit and the 54-case HPR/MVR E2E suite overlapped in their
   expensive real-optimization coverage and together took 87.44 seconds.
7. Fresh-process architecture and API checks took 34.58 seconds, largely from
   launching a separate interpreter for every compatible import assertion.

## Functional Requirements

### FR-01: Bound Notebook 19

The generated Notebook 19 normal execution profile shall use the smallest
validated optimizer budget that still demonstrates both process-level and
site-level utility-placement convergence toward valid optimized solutions.
The generator remains authoritative, the notebook remains source-only, and
the tutorial must retain finite objective, feasibility, retargeting, and
detached-result evidence.

### FR-02: Consolidate utility-placement integration solves

Tests that verify read-only views, serialization, plotting, accessors, ordering,
or state contracts shall reuse immutable solved evidence or deterministic
candidate seams where doing so preserves the asserted contract. A compact set
of representative tests shall continue to execute the real optimizer for both
process and site placement. Tests that intentionally mutate a problem or result
shall receive isolated copies.

### FR-03: Remove TESPy lane duplication

The ordinary non-solver CI lane shall exclude `tespy` tests. Every TESPy-marked
test shall remain executed in a dedicated TESPy job with an explicit timeout,
including the public targeting smoke. The change shall be applied consistently
to pull-request, develop, and publish workflows and their workflow-contract
tests.

### FR-04: Separate convergence benchmarks from ordinary correctness tests

The two multi-backend convergence benchmarks shall be marked as performance
tests and removed from the ordinary lane. A dedicated CI lane shall continue to
run them. Ordinary optimisation-service smoke coverage shall remain bounded and
shall continue to establish backend availability and basic convergence.

### FR-05: Build documentation once per workflow

The Sphinx build smoke shall be marked as a documentation test, excluded from
the ordinary lane, and invoked by the dedicated documentation job instead of a
second direct build. Warnings-as-errors behavior and generated-page validation
shall be preserved.

### FR-06: De-duplicate CoolProp HPR audit work

The 54-case HPR/MVR E2E targeting corpus shall remain intact. Audit tests whose
primary purpose is records, serialization, topology classification, or failure
atomicity shall use bounded or deterministic evidence. Only the smallest
representative audit subset needed to verify real CoolProp optimization shall
perform independent optimizer searches. Existing success, convergence,
failure, detachment, and diagnostic assertions shall not be weakened.

### FR-07: Batch compatible fresh-process checks

Compatible cold-import and retired-package assertions shall run as batches in
one child interpreter per isolation boundary. Batch failures shall identify the
individual case that failed. Checks that specifically require a pristine,
independent interpreter state shall remain separate.

## Non-Functional Requirements

### NFR-01: Runtime

On the profiling host, the optimized ordinary non-solver lane shall complete at
least 20 percent faster than the 600.89-second uninstrumented baseline. The
target is 480 seconds or less using the same deterministic seed and equivalent
ordinary-lane selection.

### NFR-02: Coverage

The branch-aware ordinary CI lane shall retain the configured 95 percent
minimum. If excluding specialized tests makes the ordinary lane insufficient,
coverage data from dedicated lanes shall be collected and combined without
double-running the specialized tests.

### NFR-03: Determinism

Existing Hypothesis shrinking and the fixed CI seed `20260715` shall remain in
effect. Shared solved fixtures must not introduce order dependence or mutable
cross-test leakage.

### NFR-04: Robustness preservation

All 54 standard HPR/MVR E2E cases, the CoolProp audit contracts, process and site
utility-placement integration, real TESPy tests, optimizer convergence
benchmarks, and warning-strict documentation verification shall remain covered
by CI even when assigned to different lanes.

### NFR-05: Maintainability

Test markers and lane ownership shall be declared centrally in pytest
configuration, reflected in CI workflow contract tests, and documented where a
developer needs to reproduce a specialized lane locally.

## Acceptance Criteria

1. Notebook 19 is regenerated from its authoritative generator and its real
   execution is materially faster while both placement scopes still converge
   toward valid solutions.
2. Utility-placement focused tests pass with no shared-state or order-dependent
   failures, and their aggregate runtime is materially reduced.
3. Each CI workflow executes TESPy, performance, and documentation tests once,
   in the intended dedicated lane.
4. Both convergence benchmark tests still pass in the performance lane.
5. The dedicated documentation lane performs one warnings-as-errors build and
   validates the generated output.
6. All 54 HPR/MVR E2E cases and the refined CoolProp audit pass.
7. Batched architecture checks retain per-case diagnostics and their focused
   runtime is materially reduced.
8. The full optimized selection passes with no unexpected skips or deselections,
   and branch coverage remains at or above 95 percent.
9. Ruff, generator drift checks, CI workflow contract tests, notebook execution,
   and patch hygiene pass.

## Extension Compliance

- **Property-Based Testing**: Enabled and applicable. Existing domain-specific
  generators, shrinking, deterministic seed, and example-based regressions are
  retained. Test fixture consolidation must preserve all existing properties.
- **Security Baseline**: Disabled; N/A to this workflow.
- **Resiliency Baseline**: Disabled; N/A to this workflow.

## Key Requirements Summary

The implementation will reduce repeated work rather than discard verification:
bound one tutorial, reuse safe utility-placement evidence, assign TESPy,
performance, and documentation checks to single dedicated lanes, narrow
overlapping CoolProp searches, and batch compatible interpreter-isolation
checks. Completion requires measurable runtime improvement, unchanged HPR/MVR
robustness evidence, and the existing coverage gate.
