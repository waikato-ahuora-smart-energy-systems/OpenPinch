# Test-Suite Runtime Reduction NFR Design Patterns

## Pattern 1: Eliminate Repeated Work Before Parallelizing

Apply deterministic serial reductions first. Notebook 19 search budgets are
bounded, repeated utility-placement and CoolProp setup is consolidated, and
compatible interpreter checks are batched. No worker-pool dependency is added,
so timing improvements remain attributable and portable.

**Satisfies**: NFR-PERF-01, NFR-PERF-02, NFR-PERF-04.

## Pattern 2: One Marker, One CI Owner

The ordinary test job selects:
`not solver and not tespy and not performance and not docs`.

The dedicated owners are:

| Marker | Owning job | Command contract |
|---|---|---|
| `tespy` | `hpr-tespy-tests` | One timeout-guarded `pytest -m tespy` invocation |
| `performance` | `performance-tests` | One timeout-guarded `pytest -m performance` invocation |
| `docs` | `docs` | One `pytest -m docs` invocation that performs the warning-strict build |
| `solver` | Existing `solver-tests` | Existing external-solver selection, unchanged |

The mapping is identical across develop, pull-request, and publish workflows.
The existing develop-validation reuse mechanism may skip duplicate PR jobs only
when the exact source tree already passed develop; that is reuse of prior
evidence, not test loss.

**Satisfies**: NFR-PERF-03, NFR-REL-01, NFR-REL-03, NFR-MNT-01.

## Pattern 3: Copy-on-Consume Solved Evidence

Expensive deterministic utility-placement setup may execute once in a
module-scoped fixture. The retained source object is treated as immutable.
Read-only assertions may consume it directly; any targeting, result attachment,
plot mutation, serialization mutation, or option mutation receives a deep copy
or a fresh reconstruction from serialized input.

The fixture captures a canonical serialization before consumers run. An
isolation regression compares the source after representative operations and
verifies equivalent outcomes in reversed operation order.

Live session-scoped caches are prohibited because they obscure file-level
ownership and amplify order coupling.

**Satisfies**: NFR-DET-01, NFR-DET-02, NFR-COR-02.

## Pattern 4: Convergence-Preserving Budget Reduction

Search budgets are reduced iteratively and accepted only when fixed-seed real
execution retains:

1. at least one evaluated candidate;
2. a finite feasible best result;
3. physical constraint satisfaction;
4. a finite objective and best-observed improvement or convergence evidence;
5. both process and site workflows for Notebook 19;
6. the intended topology, result, failure, and detachment assertions in the
   CoolProp audit.

The design does not pin exact candidate coordinates, because multiple bounded
optimizers may find equivalent optima. Budgets are raised to the smallest
passing value if any invariant fails.

**Satisfies**: NFR-DET-03, NFR-COR-01, NFR-MNT-02.

## Pattern 5: Separate Behavioral Oracles from Integration Proof

The complete 54-case HPR/MVR E2E corpus remains the broad real-service oracle.
The CoolProp audit retains only the independent real searches needed to prove
representative heat-pump, refrigeration, and MVR behavior. Record,
serialization, classification, and failure-atomicity assertions use bounded
deterministic evidence where the optimizer path is not their subject.

Utility-placement tests follow the same split: a small process/site pair proves
the real optimizer integration, while derived view and persistence assertions
reuse isolated evidence.

**Satisfies**: NFR-COR-01, NFR-COR-02, NFR-PERF-02.

## Pattern 6: Batched Fresh-Process Contract

One child script receives an ordered mapping of logical cases. For each case it
imports or probes the declared modules, records the case name, and immediately
raises an assertion containing that case name and target when the contract
fails. The parent asserts one child return code and includes captured stderr.

Root export resolution and default TESPy-cold targeting remain separate because
they intentionally require distinct pristine interpreter states. Retired
package imports share one child process because every attempted import is
expected to fail without adding package state.

**Satisfies**: NFR-REL-02 and NFR-PERF-02.

## Pattern 7: Coverage-Protected Lane Separation

The ordinary lane remains branch-instrumented with `fail-under=95`. Marker
separation is first validated against this lane alone. If it fails, the missing
line and branch report determines the response:

1. add or retain fast non-specialized coverage when the behavior is independent
   of an optional engine;
2. otherwise upload named coverage data from the one owning specialized lane
   and combine it in a single report job;
3. never execute the same expensive test twice solely for coverage.

**Satisfies**: NFR-COV-01.

## Pattern 8: Fail-Visible Time Bounds

Dedicated GitHub Actions jobs retain job-level timeouts. TESPy and performance
commands also receive finite command-level timeouts on Linux. Timeout exit is a
hard failure. No retry, expected-failure conversion, or skip-on-timeout behavior
is allowed.

Notebook and focused local profiling commands are bounded by the test harness
or executor timeout and report the responsible notebook or test node.

**Satisfies**: NFR-REL-01.

## Pattern 9: Progressive Verification

Verification runs from cheapest to most diagnostic:

1. static marker and workflow contract tests;
2. generator drift and notebook compilation;
3. focused batched-process and fixture-isolation tests;
4. bounded Notebook 19 and CoolProp real execution;
5. complete 54-case HPR/MVR E2E suite;
6. TESPy, performance, and documentation marker lanes;
7. ordinary branch-coverage lane;
8. final uninstrumented serial profile and duration comparison.

This order exposes local design errors before expensive whole-suite execution.

**Satisfies**: all performance, correctness, reliability, and maintainability
requirements.

## Extension Compliance

Property-Based Testing remains complementary to example and integration tests.
No generator, shrinking, or seed behavior is weakened. Copy-on-consume
isolation and order invariance add an explicit state-related property around
the refactored fixture boundary. Security and Resiliency are disabled and N/A.
