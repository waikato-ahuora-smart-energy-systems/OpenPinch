# Test-Suite Runtime Reduction NFR Requirements

## Scope

These requirements apply to the `test-suite-runtime-reduction` unit. They govern
test execution, generated tutorial verification, CI lane ownership, and
profiling. They do not change OpenPinch runtime APIs or numerical behavior.

## Performance

### NFR-PERF-01: Ordinary lane budget

The uninstrumented ordinary lane shall complete in 480 seconds or less on the
profiling host using Hypothesis seed `20260715`. This is at least a 20 percent
improvement over the 600.89-second baseline.

The ordinary lane selection shall be equivalent to:
`not solver and not tespy and not performance and not docs`.

### NFR-PERF-02: Hotspot budgets

The following same-host focused targets guide refinement:

- generated Notebook 19 execution: 20 seconds or less, from 44.51 seconds;
- `tests/application/test_utility_placement.py`: 75 seconds or less, from
  105.13 seconds;
- `tests/application/test_coolprop_hpr_audit.py`: 30 seconds or less, from
  43.87 seconds;
- batched fresh-process architecture and API checks: 20 seconds or less, from
  34.58 seconds.

A focused target may be exceeded only when the overall 480-second target is met
and the implementation report identifies the compensating savings and reason.

### NFR-PERF-03: Specialized lane isolation

TESPy, performance benchmarks, and documentation builds shall not execute in
the ordinary lane. Each category shall execute exactly once in its dedicated
job for every applicable pull-request, develop, or publish workflow.

### NFR-PERF-04: Profiling consistency

Before-and-after timing shall use the same Python environment, test seed,
selection, and serial execution mode. Coverage-instrumented time shall be
reported separately and shall not be compared directly with the 600.89-second
uninstrumented baseline.

## Coverage and Correctness

### NFR-COV-01: Coverage floor

The ordinary branch-aware CI lane shall retain the configured 95 percent
minimum. Specialized lanes do not need instrumentation when the ordinary lane
alone satisfies the threshold.

If the ordinary lane falls below 95 percent after separation, fast
non-specialized tests shall close the specific coverage gaps. Expensive
specialized integration tests shall not be duplicated merely to raise the
percentage. Cross-job coverage combination is a fallback only if a specialized
engine owns branches that cannot be exercised without that engine.

### NFR-COR-01: Complete specialized evidence

The following evidence shall remain executable in CI:

- all 54 standard HPR/MVR E2E targeting cases;
- the refined CoolProp HPR audit and representative independent real searches;
- real process-level and site-level utility-placement optimization;
- every TESPy-marked test;
- both multi-backend convergence benchmarks;
- one warning-strict Sphinx build with output validation;
- all cold-import, retired-package, API-boundary, and optional-dependency
  assertions.

### NFR-COR-02: Assertion preservation

Runtime reductions shall remove repeated setup or search, not assertions. Each
existing behavior assertion must remain directly covered or be traceably moved
to a test consuming equivalent immutable evidence.

## Determinism and Isolation

### NFR-DET-01: Reproducibility

Hypothesis shall retain normal shrinking and seed `20260715` in CI. Optimizer
tests and tutorials shall retain explicit deterministic seeds where supported.

### NFR-DET-02: Shared evidence isolation

Reusable solved evidence shall be created once at module scope only when its
construction is deterministic. Every consumer that can mutate the problem,
result, target collection, options, or serialized payload shall receive a deep
copy or reconstruct from immutable serialized data.

An order-independence regression shall run representative consumers in more
than one sequence or explicitly verify the shared source remains unchanged.

### NFR-DET-03: Convergence evidence

Reduced search budgets shall still produce at least one finite feasible result
and observable improvement or best-observed convergence toward the objective.
Fixed expected optimizer coordinates are not required when equivalent physical
and objective invariants remain satisfied.

## Reliability and Diagnostics

### NFR-REL-01: Timeout boundaries

Dedicated external-engine and performance jobs shall have explicit finite job
or command timeouts. A timeout shall fail the owning job visibly rather than
skip or silently retry the test.

### NFR-REL-02: Batch attribution

Batched child-interpreter checks shall report the logical case name, module or
package, and captured child stderr for the first failure. Checks needing a
pristine state incompatible with batching shall keep their own process.

### NFR-REL-03: No hidden deselection

Workflow-contract tests shall parse all three CI workflows and verify that each
specialized marker is excluded from the ordinary lane and included in exactly
one dedicated lane.

## Maintainability

### NFR-MNT-01: Central marker registration

`tespy`, `performance`, and `docs` shall be registered in the central pytest
configuration with concise ownership descriptions. Test files shall use marker
names rather than filename-specific CI selections.

### NFR-MNT-02: Generator ownership

Notebook 19 changes shall originate in
`scripts/generate_tutorial_notebooks.py`; the packaged notebook shall be
regenerated and must pass canonical drift checks.

### NFR-MNT-03: Local reproducibility

Developer documentation or the build-and-test record shall provide commands for
the ordinary, TESPy, performance, documentation, and complete verification
lanes.

## Other Quality Attributes

- **Scalability**: Test runtime should grow approximately with the number of
  logical cases, not the number of redundant interpreter launches or repeated
  identical optimizer solves.
- **Availability**: N/A; OpenPinch has no deployed service in this workflow.
- **Security and compliance**: No secrets, permissions, external services, or
  protected data are added. Disabled Security Baseline remains N/A.
- **Usability and accessibility**: No UI changes. Notebook 19 must retain clear,
  compact process/site evidence for readers.
- **Operations**: N/A; no runtime monitoring, deployment, or rollback system is
  introduced.

## Verification Matrix

| Requirement | Verification |
|---|---|
| NFR-PERF-01 | Same-host serial ordinary-lane profile |
| NFR-PERF-02 | Focused pytest/notebook duration measurements |
| NFR-PERF-03 | Workflow inspection and workflow-contract tests |
| NFR-PERF-04 | Recorded commands, environment, and timing comparison |
| NFR-COV-01 | Branch-aware ordinary lane with `fail-under=95` |
| NFR-COR-01 | Dedicated marker lanes and focused HPR/MVR runs |
| NFR-COR-02 | Assertion inventory and targeted regressions |
| NFR-DET-01 | CI command inspection and Hypothesis execution |
| NFR-DET-02 | Mutation and order-independence regression |
| NFR-DET-03 | Real bounded optimizer and Notebook 19 execution |
| NFR-REL-01 | Workflow timeout assertions |
| NFR-REL-02 | Intentional batched-failure regression or helper unit test |
| NFR-REL-03 | Cross-workflow marker ownership tests |
| NFR-MNT-01 | Pytest configuration test and collection output |
| NFR-MNT-02 | Notebook generator drift test |
| NFR-MNT-03 | Build-and-test instruction review |

## Extension Compliance

- **Property-Based Testing**: Existing domain generators, shrinking, seed, and
  example regressions remain required. The fixture-isolation invariant is
  verified explicitly; no property test is removed solely for speed.
- **Security Baseline**: Disabled and N/A.
- **Resiliency Baseline**: Disabled and N/A.
