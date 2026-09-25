# NFR Design Patterns

## 1. One shared corpus owner

Move standard-problem discovery into one test-owned helper under `tests/e2e`.
Both the existing direct-target suite and the HPR/MVR benchmark import that
helper. It discovers `examples/stream_data/p_*.json`, sorts by filename, and
returns an immutable sequence.

This prevents two fixture lists from drifting while keeping benchmark knowledge
out of production packages. A coverage assertion compares both consumers to the
same owner and pins the current count of 54 without embedding 54 paths.

## 2. Immutable declarative profiles

Represent the six HPR profiles as frozen test-owned values. Each value contains
the stable profile ID, public service selector, placement mode, topology,
working fluid, port/stage counts, load fraction, restart count and explicit
search limits.

Profiles contain data only. Invocation logic is shared, so topology and utility
differences cannot introduce six subtly different test implementations.

## 3. Stable round-robin assignment

Enumerate the sorted corpus and select `profiles[ordinal % 6]`. Materialize one
immutable pytest parameter per problem with ID `<filename>::<profile_id>`.

The assignment layer asserts uniqueness, completeness and nine current cases
per profile. Discovery and assignment are O(n), deterministic, and independent
of filesystem enumeration order.

## 4. Fresh-instance sequential isolation

Every parameter constructs a new public problem from its JSON file immediately
before invocation. No problem, HPR engine, optimization cache or diagnostic
collection is shared across cases.

The e2e module runs sequentially. This is required for unambiguous attribution
while the convergence observer temporarily wraps the shared evaluation
boundary, and it avoids multiplying expensive CoolProp work across processes.
The design does not depend on pytest-xdist scheduling.

## 5. Explicit hard budgets

Every profile specifies one restart plus finite iteration and evaluation
limits. Exploratory execution selects the smallest stable distinct-search
allowance that demonstrates the required real search behavior, with an absolute
ceiling of 50 per public call.

Exact-coordinate cache hits do not consume a distinct-search allowance. There
are no retries, sleeps, external solvers, network calls or timing assertions.
Elapsed time may be reported during investigation but is not a pass/fail oracle.

## 6. Public outcome adapter

A test-owned invocation adapter takes a fresh problem and immutable profile,
captures the public pre-state, invokes the real public method, and classifies
exactly one of:

- strict solved result;
- documented no-op;
- typed `HPRTargetingError`.

The adapter catches only `HPRTargetingError`. Any other exception preserves its
type and traceback and fails the named parameter. The adapter does not translate
production failures into test outcomes.

## 7. Transaction and contract validation

Independent assertion helpers validate outcome-specific invariants:

- no-op and typed failure preserve public JSON and target count;
- success appends exactly one target;
- a strict success has both success flags true, finite public accounting, a
  detached CoolProp simulation record, positive useful duty, consistent finite
  duties, no retained live model, and successful deep-copy and JSON round trips;
- typed diagnostics remain bounded, internally consistent, copyable and
  serializable.

These helpers inspect public contracts. They do not duplicate thermodynamic
calculations or assert unstable candidate coordinates.

## 8. Observation-only convergence probe

For a strict-success sentinel, temporarily wrap the real shared candidate
evaluation boundary. The wrapper delegates to the original function exactly
once with the original positional and keyword arguments and returns its result
unchanged. It records only search-mode calls after delegation.

The trace stores detached design-point tuples, success state and finite viable
objective values. It neither supplies candidates nor changes the optimizer,
CoolProp engine, objective, cache, result selection or exception behavior.
Restoration occurs in fixture cleanup even when the invocation raises.

Because the wrapper is process-global, convergence sentinels and the matrix are
kept sequential. The ordinary full-matrix cases need no wrapper unless
diagnostic evaluation counts are being observed.

## 9. Independent convergence oracle

A pure analyzer deduplicates exact design points in first-seen order and derives
the viable sequence, first viable objective, incumbent-best sequence, minimum
observed viable objective, selected-objective gap and distinct evaluation count.

The witness passes only when:

1. at least two distinct viable search points exist;
2. a later point improves the first viable objective by more than
   `1e-8 * max(1, abs(first_viable_objective))`;
3. the incumbent-best sequence is non-increasing;
4. the selected final objective equals the minimum viable objective within the
   same scale-aware tolerance; and
5. the distinct evaluation count is within the configured allowance.

The analyzer is independent of the optimizer and is unit/property tested with
generated traces. The real sentinel tests supply the integration proof. This is
bounded convergence toward the best observed candidate, not a global-optimum
claim.

## 10. Named strict-success sentinels

Exploratory evidence selects one stable named benchmark problem for each of the
six profiles. Those six cases must satisfy both the strict solved contract and
the convergence witness. They are explicit tests or clearly identified
parameters, not runtime skips selected from whatever happens to solve.

The remaining 48 assignments may validly solve, no-op or return a typed failure,
but cannot substitute for a missing profile sentinel.

## 11. Direct process-MVR scenario

Keep direct process MVR separate from the 54-case HPR rotation because the
standard inputs do not carry the required pressure-qualified gas streams. Load
packaged `process_mvr.json`, call the public component accessor for `Evaporator
vapour` with explicit bounded settings, validate the returned component and
replacement streams, then run public direct heat integration.

The scenario requires a real successful property workflow, finite work and
duty, a pressure lift, no retained live property engine, registered inventory,
finite downstream targets and serializable public state.

## 12. Evidence-led defect routing

The exploratory matrix records case, profile, outcome, evaluation count and
bounded diagnostic facts. Unexpected exceptions are reproduced in a focused
test before changing production code. A correction is made only in the owning
component, and both the focused regression and original e2e parameter remain.

Outcome counts are evidence for the tested environment, not frozen feasibility
snapshots. The suite does not weaken assertions or add retries to accommodate a
defect.

## 13. Stable failure reporting

Pytest IDs and assertion messages always include filename and profile. Failure
messages report the observed outcome, configured budget, search count and the
specific invariant that failed. Typed production diagnostics keep their closed,
bounded reason vocabulary; the harness does not stringify arbitrary live
objects.

## 14. Complementary property testing

The fixed 54-case matrix is example-based e2e coverage. Hypothesis is applied to
independent pure rules where generated cases add value:

- assignment completeness, range and repeatability over generated corpus sizes;
- convergence threshold boundaries, exact-point deduplication and
  best-so-far monotonicity;
- selected-objective tolerance acceptance and rejection;
- success/failure/no-op state-transition invariants where a model oracle is
  independent;
- existing cache, budget, diagnostic and serialization properties.

The fixed seed remains `20260715`; shrinking and ordinary CI discovery remain
enabled. Generated tests do not mock the real e2e solver or replace the six
real convergence sentinels.

## 15. Dependency and infrastructure containment

The design uses the existing pytest, Hypothesis, CoolProp, NumPy, SciPy and
Pydantic stack. It adds no runtime or development dependency, service, queue,
database, cache daemon, circuit breaker, deployment resource or CI workflow.
Test helpers remain under `tests/e2e`; production modules never import them.

## NFR traceability

| Requirement | Design patterns |
|---|---|
| NFR-BE-01 through NFR-BE-05 | 4, 5, 8 |
| NFR-RL-01 through NFR-RL-06 | 4, 6 through 13 |
| Determinism and reproducibility | 1 through 5, 8 through 10, 14 |
| Scalability | 1, 3 through 5 |
| Maintainability and diagnostics | 1, 2, 6 through 13 |
| Portability and CI compatibility | 4, 5, 14, 15 |
| Availability and operations | 15; not applicable to a local test harness |
| Security and privacy | 15; Security Baseline disabled |

## Extension compliance

- **Property-Based Testing**: compliant. Independent properties, generators,
  fixed-seed reproducibility, shrinking and complementary real examples are
  allocated to Code Generation.
- **Security Baseline**: disabled and skipped. Repository fixtures contain no
  credentials or personal data and the harness performs no network operation.
- **Resiliency Baseline**: disabled and skipped. Required reliability behavior
  is nevertheless covered by fail-fast typed outcomes and transaction checks;
  no retry or recovery infrastructure is introduced.
