# Unit 2 Non-Functional Requirements

## Scope

These requirements apply to the in-process Placement Evaluation and
Optimisation Service. Unit 2 owns detached numerical context preparation,
candidate replay, objective calculation, per-run memoization, optimizer
coordination, and result assembly. It does not own public problem/workspace
accessors, notebook delivery, a network service, persistent data, deployment,
or infrastructure. All seven NFR questions selected option A.

## Scalability and Capacity

### U2-NFR-001: No arbitrary domain-count ceiling

Unit 2 shall not introduce a product-level maximum for level or period counts.
Runtime is controlled by explicit optimizer budgets and available in-process
resources. Validation may reject only physically, structurally, numerically, or
dimensionally invalid inputs.

**Acceptance**: valid contexts beyond representative benchmark sizes reach
model construction rather than failing a hidden count cap.

### U2-NFR-002: Explicit computational scaling

Let `P` be selected periods, `L` total hot/cold levels, `D` decision
coordinates, `I` total process intervals per replay, `E` unique evaluations per
worker, and `W` optimizer worker processes. One cold candidate evaluation shall
scale with `O(P*(L+I))` plus existing target/turbine owner complexity. Exact
memo lookup shall be average `O(1)` over a `D`-value tuple. Parent candidate
normalization shall be `O(K log K)` for bounded retained candidate count `K`.

No Unit 2 loop may compare every period to every other period or every level to
every non-adjacent level.

### U2-NFR-003: Evaluation and memory bounds

Each worker-local memo contains at most the evaluations admitted by the
configured evaluation limit and stores compact callback records, not complete
per-level/per-period results. Full result evidence is retained only for the
bounded parent candidate union being canonically re-evaluated. Aggregate
rejected-candidate diagnostics contain counts by stable code and at most ten
representatives.

**Acceptance**: generated command sequences never grow a worker memo beyond its
budget; public result/diagnostic size is independent of total rejected callback
count except for bounded aggregate counters.

## Performance

### U2-NFR-004: Pure objective-kernel budget

On the project CI runner, evaluation of already allocated thermodynamic and
monetary evidence for 40 total levels across 100 periods shall complete at p95
within 50 ms. The fixture includes mixed sensible/isothermal levels, aligned
balanced-composite entropy evaluation, weighted aggregation, and cogeneration arithmetic
using a deterministic adapter result; it excludes targeting and a real turbine
solve.

### U2-NFR-005: Cold candidate-replay budget

One uncached thermodynamic candidate replay on deterministic fixture data with
16 total levels across 12 periods, including target reconstruction, existing
utility allocation, coverage, entropy, and aggregation but excluding the
optimizer, shall complete at p95 within 1 second on the project CI runner.

### U2-NFR-006: Memo-hit budget

An exact-coordinate memo hit returning an immutable compact evaluation shall
complete at p95 within 1 ms for the U2-NFR-005 fixture.

### U2-NFR-007: Benchmark protocol

Each performance gate uses at least three warm-ups and ten measured repetitions
in one process, reports p95, Python/platform identity, fixture dimensions, and
the slowest owned stage, and avoids file/network I/O. A failure is not retried
or hidden. There is no universal full-optimization wall-clock promise because
backend method and explicit evaluation budget are caller choices.

## Numerical Correctness

### U2-NFR-008: Combined duty-coverage criterion

For required duty `R`, coverage passes when
`abs(allocated - R) <= coverage_absolute + relative_tolerance * max(R, 1 kW)`.
Defaults are `coverage_absolute=1e-6 kW` and `relative_tolerance=1e-9`. Both are
finite and non-negative. Raw residuals are checked before zero normalization.

### U2-NFR-009: Binary64 production and high-precision oracle

Production kernels use finite binary64 Python `math`/NumPy values,
deterministic composite interpolation, stable logarithmic entropy evaluation,
and stable summation. Analytical tests cover exact sensible logarithmic and
isothermal-limit cases plus balanced heat-load transformations. No production
decimal or platform-dependent `longdouble` path is introduced.

### U2-NFR-010: Thermodynamic identities

For every feasible period and aggregate, utility plus process terms equal total
balanced-composite entropy generation within named tolerances. Sensible and
isothermal interval oracles, non-negativity, closer-temperature ranking, and
balanced-duty requirements hold. Exergy destruction equals ambient kelvin
times physical entropy generation, and every public quantity is finite with
the declared unit.

### U2-NFR-011: Monetary identities

Thermal cost equals the unit-normalized sum of allocated duty times template
price; electricity credit equals cogenerated work times electricity price; and
net monetary objective equals thermal cost minus credit. Negative net monetary
cost remains finite and valid.

### U2-NFR-012: Strict scalar feasibility separation

The monotone feasible transform always produces a value below one and preserves
physical-objective order. The normalized infeasible penalty always produces a
value at least one and below two. Final result selection independently proves
feasibility and never ranks on the transformed scalar.

### U2-NFR-013: Units and finite boundaries

All boundaries reuse Unit 1 canonical units and existing Pint ownership.
Unknown or incompatible units, non-finite values, and non-positive absolute
temperatures fail through typed placement errors or candidate diagnostics as
designed. Currency conversion remains out of scope.

## Reproducibility and Concurrency

### U2-NFR-014: Fixed-seed determinism

Equal request, context, method, seed, options, adapter behavior, and environment
shall produce equivalent physical objectives and the same candidate order
within documented numerical tolerances. Public order is physical aggregate
objective followed by exact coordinates.

### U2-NFR-015: Process-local memo isolation

Each execution process/session owns a concurrency-safe exact-coordinate memo.
The objective payload is pickle-safe. Worker processes may duplicate physical
evaluation of the same coordinate but cannot share or corrupt mutable memo
state.

### U2-NFR-016: Canonical parent re-evaluation

The parent exact-deduplicates backend points and deterministic initial points,
then performs one canonical full re-evaluation of each retained coordinate.
Public results therefore do not depend on worker completion order, worker-local
memo state, process identity, or unordered collection iteration.

### U2-NFR-017: No hidden nondeterminism

Unit 2 shall not use the system clock, locale, random global state, unordered
set/dict iteration for observable output, or non-seeded internal sampling.
Backend randomness is controlled only through forwarded method/seed/options.

## Reliability and Availability

### U2-NFR-018: Complete result or typed failure

Every service call returns one complete frozen result or raises one documented
typed placement exception. Partial period/candidate results and least-
infeasible success are prohibited.

### U2-NFR-019: Failure recoverability boundary

Candidate-correctable vector, targeting, coverage, thermodynamic, and turbine
conditions become structured infeasibility. Invalid context/settings, adapter
contract violations, invariant failures, and optimizer operational failures
raise typed run-level errors with chained causes where safe. Broad unexpected
exceptions are not suppressed.

### U2-NFR-020: No retry policy

Unit 2 calls the selected optimizer once and does not silently retry targeting,
turbine, or optimizer failures. Callers may submit a new request with different
validated options.

### U2-NFR-021: Availability and continuity are N/A

Uptime, failover, disaster recovery, backup/restore, RPO/RTO, health checks,
and deployment continuity are not applicable because Unit 2 is an in-process,
stateless, reconstructable library service with no persistent store.

## Security and Data Protection

### U2-NFR-022: Preserve the no-new-boundary scope

The Security extension remains disabled. Unit 2 adds no network, file,
credential, authentication, authorization, encryption, dynamic import, or code
execution boundary. Backend option names/values are validated data and are
never evaluated as code. Diagnostics omit source object representations,
secrets, and unbounded mutable payloads.

## Observability and Usability

### U2-NFR-023: Stable actionable diagnostics

Typed errors and candidate diagnostics carry a stable code/message and
applicable scope, objective, counts, method, seed, configured limits, period,
side, template, measured/required duty, tolerance, and adapter category.
Messages name corrective action without requiring backend-message parsing.

### U2-NFR-024: Bounded diagnostic summary

Solve-level rejected-candidate evidence contains aggregate counts by stable
failure code plus at most ten representatives in deterministic severity,
period, side, template, and code order. The service does not expose the full
callback trace.

### U2-NFR-025: Result interpretability

Every result exposes physical selected objectives, decompositions, units,
coverage residuals/tolerances, method, seed, termination counts, and ordered
alternatives. Backend-transformed scalars, internal penalties, memos, and
worker identities are not public result values.

## Compatibility and Maintainability

### U2-NFR-026: Additive contract reconciliation

New optimizer fields on `UtilityPlacementOptions` have backward-compatible
defaults. Existing Unit 1 constructor calls and JSON payloads remain valid. No
package-root export, existing target schema, configuration default, or
optional-dependency boundary changes.

### U2-NFR-027: Existing technology only

Unit 2 adds no dependency. It reuses supported Python, Pydantic/Pint,
NumPy/SciPy, CoolProp-based steam calculations, pytest, Hypothesis, coverage,
Ruff, and Hatchling declarations.

### U2-NFR-028: Ownership and dependency direction

Context, allocation, objective, turbine-adapter, evaluation, penalty,
optimization, and service modules remain under the specialist analysis owner.
They may consume contracts, domain/targeting, power, and public optimizer
owners, but not application accessors, presentation, private optimizer backend
modules, or Unit 3.

### U2-NFR-029: Static and documentation quality

New functions/classes have typed signatures and purposeful docstrings. Modules
remain cohesive. Ruff, architecture/import, stale-symbol, packaging, and
installed-specialist-import checks pass. Broad exception suppression,
unexplained numerical constants, and duplicate target/turbine physics are
prohibited.

## Test Quality and Technology

### U2-NFR-030: Complementary deterministic test profiles

Routine service behavior uses injected deterministic target, turbine, and
optimizer adapters plus the structured-grid oracle. Routine non-solver CI also
runs one small fixed-seed real dual-annealing placement regression. Broader
method behavior remains covered by the existing optimizer test suite; Unit 2
does not duplicate full regressions for all four methods in every job.

### U2-NFR-031: Coverage and regression gates

Delivery requires failing-first examples for every production slice, all
Functional Design properties with reusable domain strategies, fixed Hypothesis
seed `20260715`, shrinking enabled, permanent example regressions for shrunk
defects, branch coverage at the repository 95% gate, the complete available
non-solver suite, Ruff, architecture, packaging, and installed-wheel smoke.

### U2-NFR-032: PBT-09 framework capability

Hypothesis under the existing development dependency and pytest runner is the
required PBT stack. It must support domain composites, shrinking, fixed-seed
replay, command/state models for process-local memo behavior, and integration
with the existing CI command. No custom random harness is accepted.

## Traceability

| Approved concern | Unit 2 NFRs |
|---|---|
| NFR-001 numerical correctness | U2-NFR-008 through U2-NFR-013 |
| NFR-002 reproducibility | U2-NFR-014 through U2-NFR-017, U2-NFR-030 through U2-NFR-032 |
| NFR-003 bounded execution | U2-NFR-001 through U2-NFR-007, U2-NFR-020, U2-NFR-024 |
| NFR-004 compatibility | U2-NFR-026, U2-NFR-027, U2-NFR-029 through U2-NFR-031 |
| NFR-005 maintainability | U2-NFR-027 through U2-NFR-032 |
| NFR-006 observability | U2-NFR-018, U2-NFR-019, U2-NFR-023 through U2-NFR-025 |
| NFR-007 security/resiliency scope | U2-NFR-021, U2-NFR-022 |
| FR-007 through FR-012, FR-014 | U2-NFR-001 through U2-NFR-032 according to their numerical/service boundaries |
| PBT-09 | U2-NFR-032 |

## NFR Acceptance Summary

Unit 2 is acceptable when all 32 requirements have executable or reviewable
evidence, all three tiered performance gates pass, numerical and monetary
identities match their oracles, multiprocess runs remain isolated and
canonically reproducible, resource/diagnostic growth is bounded, the existing
technology/dependency boundary is preserved, PBT-09 has no blocking finding,
and all repository quality gates remain green.
