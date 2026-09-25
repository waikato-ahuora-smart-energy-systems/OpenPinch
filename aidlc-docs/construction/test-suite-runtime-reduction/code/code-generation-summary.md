# Test-Suite Runtime Reduction Code Generation Summary

## Outcome

The ordinary serial test selection now completes in 383.17 seconds on the
profiling host, compared with the 600.89-second baseline. This is a 217.72
second, 36.2 percent reduction and is 96.83 seconds below the 480-second
acceptance threshold.

The branch-aware ordinary run passed its 95 percent coverage gate. No OpenPinch
production Python module or public API changed.

## Implemented Changes

1. Regenerated Notebook 19 from its authoritative generator with a bounded
   1-candidate, 5-iteration, 10-evaluation, 1-run search. The tutorial exposes
   termination evidence for both process and site solutions.
2. Reused deterministic process and site utility-placement solutions through
   module-scoped, copy-on-consume fixtures. Mutation and order-isolation
   regressions protect the shared source evidence.
3. Assigned TESPy, performance, and documentation tests to one dedicated CI
   lane each in the develop, pull-request, and publish workflows. The ordinary
   lane excludes all three markers plus solver tests.
4. Reduced representative CoolProp audit budgets while preserving independent
   heat-pump, refrigeration, parallel/cascade, and MVR searches and leaving the
   complete HPR/MVR E2E corpus unchanged.
5. Moved the two four-backend convergence benchmarks to the performance lane
   and retained their quality and relative-runtime assertions under smaller
   deterministic budgets.
6. Made the marked Sphinx test the sole warning-strict documentation build
   owner in CI.
7. Batched compatible cold imports and retired-package checks while retaining
   logical case/module diagnostics and separate pristine-process checks.

## Timing Comparison

| Area | Baseline | Final evidence | Result |
|---|---:|---:|---|
| Ordinary serial selection | 600.89 s | 383.17 s | 36.2% faster |
| Notebook 19 | 44.51 s | 6.06 s | 86.4% faster |
| Utility-placement test file | 105.13 s | 65.63 s | 37.6% faster |
| CoolProp audit | 43.87 s | 22.82 s | 48.0% faster |
| Fresh-process focused set | 34.58 s | 18.53 s | 46.4% faster |
| TESPy in ordinary lane | 52.74 s | 0 s | Owned by dedicated lane |
| Optimizer benchmarks in ordinary lane | 33.42 s | 0 s | Owned by dedicated lane |
| Sphinx build in ordinary lane | 10.43 s | 0 s | Owned by dedicated lane |

The unchanged HPR/MVR E2E file contributed 55 passing cases in 41.95 seconds
during the final profile. The dedicated TESPy selection passed 58 tests in
58.23 seconds. The performance and documentation selections passed two and one
tests respectively; their combined verification completed in 45.41 seconds.

## Complete Verification

- Ordinary branch coverage: 3,390 passed, 6 expected optional-profile skips,
  65 intentionally deselected; configured 95 percent threshold passed in
  434.68 seconds.
- Ordinary uninstrumented profile: 3,390 passed, 6 expected optional-profile
  skips, 65 intentionally deselected in 383.17 seconds.
- Specialized collection: 58 TESPy tests, 2 performance tests, 1 docs test,
  and 4 solver tests. The solver execution passed 3 and retained 1 expected
  environment-dependent skip.
- CoolProp audit: 14 passed. Complete standard benchmark plus direct-MVR E2E
  file: 55 passed, including all 54 standard cases.
- Packaged artifacts: wheel and source distribution built successfully; both
  contain Notebook 19. An isolated core-wheel installation passed the package,
  CLI, resource, targeting, and checkout-isolation smoke.
- Final workflow/notebook contracts: 72 passed after formatting.
- Ruff, YAML parsing, canonical notebook drift, and `git diff --check` passed.

The six ordinary skips are the existing opt-in slow-HPR, solver-notebook, and
interactive-notebook profiles. The dedicated solver skip is the existing
optional external-solver case. No retry, expected-failure conversion, or hidden
marker orphan was introduced.

## Assertion and Design Audit

- The complete 54-case standard HPR/MVR parameter matrix remains unchanged.
- Real process-level and site-level utility-placement optimization remains,
  while derived-view consumers receive deep copies.
- Hypothesis generators and shrinking remain enabled. Only redundant example
  counts were reduced from six to three under seed `20260715`.
- Notebook 19 remains generator-owned and source-only; its process and site
  objectives are finite and feasible with zero fallback penalty.
- Each specialized marker is centrally registered, excluded from the ordinary
  lane, and selected exactly once by a timeout-bounded owning workflow job.
- Batched child-process failures identify the logical case and module/package.
- The diff contains test, tutorial generator/output, CI, pytest configuration,
  and AI-DLC documentation changes only. There are no duplicate brownfield
  files or unintended production API changes.

## Local Reproduction

```bash
uv run --no-sync pytest --hypothesis-seed=20260715 \
  -m "not solver and not tespy and not performance and not docs"
uv run --no-sync pytest --hypothesis-seed=20260715 -m tespy
uv run --no-sync pytest --hypothesis-seed=20260715 -m performance
uv run --no-sync pytest --hypothesis-seed=20260715 -m docs
uv run --no-sync pytest --hypothesis-seed=20260715 -m solver
```

## Extension Compliance

- Property-Based Testing: compliant. Domain strategies, shrinking, fixed seed,
  metamorphic checks, and example-based integration evidence remain.
- Security Baseline: disabled; N/A because the change adds no credentials,
  permissions, external service, or security boundary.
- Resiliency Baseline: disabled; N/A because the change adds no deployed
  service or runtime recovery behavior.
