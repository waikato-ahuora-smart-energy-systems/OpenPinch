# Unit 2 Technology Stack Decisions

## Decision Summary

Unit 2 uses the existing OpenPinch runtime, numerical, optimization, steam,
testing, and packaging stack. It adds source modules and tests but no runtime,
optional, development, build, or external-service dependency.

| Concern | Decision | Rationale/evidence |
|---|---|---|
| Runtime | Python `>=3.14.2` | Existing project and CI contract |
| Public contracts | Existing frozen Pydantic v2 specialist models | Preserves Unit 1 validation and JSON behavior |
| Units | Existing OpenPinch `Value`/Pint ownership | One canonical dimensional boundary; no currency conversion |
| Array/profile adapters | Existing NumPy | Existing target and turbine owners use copied arrays |
| Optimization | Existing `OpenPinch.optimisation` service and SciPy-backed methods | Solver-neutral bounded contract already supports method, seed, limits, clustering, and candidates |
| Steam cogeneration | Existing `MultiStageSteamTurbine` and CoolProp-backed thermophysical owners | Reuses canonical physics through a detached adapter |
| Production thermodynamics | Existing balanced-composite targeting plus binary64 `math.log1p`/`math.fsum` | Physical logarithmic entropy on aligned heat-load intervals with explicit finite/kelvin/balance checks |
| High-precision test oracle | Standard-library `decimal` | Adds no dependency and independently checks generated small branches/limits |
| Test runner | Existing pytest `>=9.0.3` | Current repository runner and markers |
| Property testing | Existing Hypothesis `>=6.135.0` | Domain strategies, shrinking, command models, and fixed seeds |
| Coverage/static quality | Existing coverage.py branch gate and Ruff | Current 95% and Python 3.14 gates |
| Build/distribution | Existing Hatchling configuration | Specialist modules ship in current wheel/source distributions |

## Production Ownership

Implementation remains under `OpenPinch.analysis.utility_placement`, with
modules split by one numerical or coordination responsibility:

- `context.py`: immutable direct/Total Site snapshot extraction and envelope
  population;
- `allocation.py`: reconstruction, existing target adapter, assignment slices,
  and coverage;
- `thermodynamics.py`: candidate-local balanced-composite entropy and exergy kernels;
- `economics.py`: thermal purchase, electricity credit, and net cost;
- `cogeneration.py`: eligible-level filtering and fresh turbine adaptation;
- `evaluation.py`: all-period replay, compact evaluation records, and memos;
- `penalties.py`: feasible transform and normalized violation penalties;
- `optimisation.py`: existing optimizer option mapping, candidate union,
  canonical parent re-evaluation, filtering, and ordering;
- `service.py`: end-to-end orchestration and result assembly; and
- `errors.py`: additive Unit 2 operational exception subclasses.

Files may be combined if implementation remains cohesive, but numerical
equations shall not move into application accessors or presentation. Unit 2
imports public optimizer service/models, never private backend modules.

## Optimizer Contract Reconciliation

The first TDD slice extends the frozen `UtilityPlacementOptions` with backward-
compatible defaults for method, run count, cluster tolerance, local method, and
immutable backend overrides. Unit 2 maps those values to existing
`OptimisationOptions`:

| Placement option | Existing optimizer option |
|---|---|
| `method` | `method` argument |
| `run_count` | `n_runs` |
| `iteration_limit` | `maxiter` |
| `evaluation_limit` | `maxfun` |
| `cluster_tolerance` | `cluster_tol` |
| `candidate_limit` | `max_minima` |
| `local_method` | `local_method` |
| `backend_options` | validated method-specific keyword values |

The existing service remains the sole owner of supported methods and method-
specific option names. The specialist contract restricts values to JSON-safe
immutable data and does not import private backends.

## Numerical Implementation

Production uses Python/NumPy binary64 values at existing numerical boundaries.
Candidate utilities and real process composites use existing NumPy cascade and
balanced-composite helpers. Sensible intervals use stable `math.log1p`,
isothermal intervals use `Q/T`, and terms use canonical `math.fsum`; penalty
separation uses the bounded monotone `math.atan` transform. Every kernel
validates finite values, positive kelvin, and balanced heat duty.

The entropy test oracle uses hand-calculable logarithmic and `Q/T` values on
the required Python runtime. It is test-only and independent of the production
implementation. Platform-specific `numpy.longdouble`, a second unit registry,
and a production arbitrary-precision dependency are rejected.

Coverage combines `PlacementTolerances.coverage` with its existing relative
tolerance. The formula and 1 kW scaling floor live in one named helper rather
than being repeated in adapters.

## Concurrency and Memoization

Optimizer objective payloads contain only pickle-safe frozen values. Each
process/session lazily creates its own lock-protected exact-coordinate mapping;
the lock and memo are excluded from serialized state and recreated after
unpickling. Concurrent same-process callbacks cannot corrupt or duplicate an
in-flight key. Separate worker processes do not share memory and may duplicate
physical work.

Worker memo records are compact and bounded by the worker evaluation budget.
The parent exact-deduplicates returned points and deterministic starts, then
fully re-evaluates retained coordinates in canonical order. This parent result
is the only source of public period evidence, making output independent of
worker completion order.

No manager process, shared database, process-global cache, filesystem cache, or
forced serial backend is introduced.

## Test Stack and Organization

### Example-based tests

Focused examples cover:

- optimizer option extension/mapping and backward-compatible JSON;
- direct and Total Site snapshot extraction and source non-mutation;
- allocation slices, full hot/cold coverage, and zero-duty levels;
- hand-calculable coincident, constant-gap, and breakpoint-refinement entropy/exergy;
- monetary purchase, eligible cogeneration, credit, and negative net cost;
- recoverable versus run-level target/turbine failures;
- raw weighted aggregation including a failed zero-weight period;
- feasible/infeasible scalar separation;
- exact memo behavior, isolated-worker simulation, and parent re-evaluation;
- candidate filtering, ordering, limit, and exhaustion; and
- complete detached result assembly and JSON round-trip.

### Property-based tests

Reusable Unit 1 strategies in `tests/strategies/utility_placement.py` are
extended with valid detached snapshots, allocation slices, entropy branches,
prices, eligible turbine inputs, multiperiod contexts, violation magnitudes,
memo command sequences, and tiny bounded grid-oracle cases. Property tests are
visibly named/grouped and do not replace the examples above.

Required generated evidence carries every PBT-01 property from Functional
Design: context/source invariance, replay idempotence, coverage, branch and
aggregate identities, high-precision oracles, filtering, weighted folds,
penalty separation, memo command-model equivalence, structured-grid comparison,
fixed-seed ordering, failures, and result JSON round-trip.

### Routine CI profiles

The routine fixed-seed non-solver job runs all injected-adapter examples and
properties, the high-precision entropy oracle, the structured-grid oracle, the
three performance gates, one tiny fixed-seed production dual-annealing
placement regression, and existing optimizer tests already selected by the
suite. Unit 2 does not add four redundant full placement regressions to every
CI job.

Hypothesis shrinking stays enabled. CI continues using
`--hypothesis-seed=20260715`; a shrunk defect becomes a permanent focused
example. No retry marker or health-check suppression may conceal a failure.

## Performance Evidence

A focused benchmark module owns three deterministic fixtures and reports p95
after at least three warm-ups and ten repetitions:

1. 40 total levels across 100 already-allocated periods for entropy, monetary,
   and aggregation kernels: at most 50 ms;
2. 16 total levels across 12 periods for one uncached thermodynamic target
   replay: at most 1 second; and
3. exact memo hit for fixture 2: at most 1 ms.

The full black-box solve is governed by explicit iteration/evaluation budgets,
not a universal latency assertion. Performance tests record platform/runtime
and avoid file/network I/O.

## Observability and Failure Translation

Structured failure codes and applicable reproducibility context are produced
directly rather than parsed from backend prose. Unexpected target/turbine/
optimizer exceptions retain a safe chained cause internally while public
messages remain stable and bounded. Rejected-candidate summaries use counts by
code and at most ten deterministic representatives. Full callback traces,
mutable source objects, worker identities, transformed scalars, and secrets do
not enter public result JSON.

## PBT-09 Compliance

| Verification criterion | Status | Evidence |
|---|---|---|
| Framework selected/documented | Compliant | Hypothesis with pytest is retained |
| Dependency present | Compliant | `hypothesis>=6.135.0` remains in the development group |
| Domain strategies | Compliant | Reusable Unit 2 strategy extensions are specified above |
| Automatic shrinking | Compliant | Default shrinking remains enabled |
| Seed reproduction | Compliant | Existing CI seed is `20260715` |
| Stateful/command model support | Compliant | Hypothesis supports the process-local memo command model |
| Runner/CI integration | Compliant | Existing pytest non-solver commands execute properties |
| Primary language coverage | Compliant | Unit 2 is Python-only |

PBT-09 has no blocking finding. PBT-01 properties remain mandatory inputs to
Code Generation planning. PBT-02 through PBT-08 and PBT-10 become blocking at
their designated later stages.

## Alternatives Rejected

| Alternative | Reason rejected |
|---|---|
| New optimization package | Existing solver-neutral methods already satisfy the bounded seeded contract |
| Production decimal/long-double entropy | Adds cost or platform variance without need; stable binary64 plus a high-precision test oracle is sufficient |
| Shared multiprocess memo service | Adds coordination complexity and a new state boundary for avoidable duplicate-work reduction |
| Force serial optimization | Removes existing backend capability and makes run-count options misleading |
| Retain all callback results/diagnostics | Memory grows with every expensive evaluation and exposes an unbounded trace |
| Full placement regression for every method in each CI job | Duplicates existing backend coverage and increases routine latency without proportional service evidence |
| New root export or CLI | Explicitly outside the approved scope |

## Compatibility Gates

Implementation is acceptable only when existing root imports, optimizer APIs,
target/turbine owners, dependency declarations, optional boundaries, source and
wheel contents, and non-solver behavior remain compatible. The additive options
fields must deserialize old payloads through defaults and round-trip new
payloads without backend-private values.
