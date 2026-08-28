# Unit 2 NFR Design Patterns

## Pattern Catalogue

### U2-NFRP-01: Additive Frozen Options

Extend the Unit 1 frozen options contract with defaulted optimizer method,
run-count, clustering, local-method, and immutable backend-option fields.
Normalize JSON-safe values once, preserve old payload behavior, and delegate
method-specific validation to the public optimizer service.

**Implements**: U2-NFR-004, U2-NFR-014, U2-NFR-026, U2-NFR-027.

### U2-NFRP-02: Detached Snapshot Boundary

Convert an isolated direct or Total Site target source into frozen period
snapshots containing only finite tuples and metadata. Mutable zones, problem
tables, streams, targets, arrays, and configuration objects stop at the
builder. Candidate replay reconstructs fresh local objects from the snapshot.

**Implements**: U2-NFR-002, U2-NFR-013, U2-NFR-018, U2-NFR-028.

### U2-NFRP-03: Fresh Replay Sandbox

Each candidate/period replay receives newly reconstructed tables and utilities.
No target, duty, or turbine state survives a replay. The same adapter path is
used on success and failure, enabling repeatability and source non-mutation
properties.

**Implements**: U2-NFR-014, U2-NFR-018 through U2-NFR-020.

### U2-NFRP-04: Canonical Linear Folds

Process periods, sides, level ranks, and temperature intervals in explicit
canonical order. Use indexed maps for key lookup and `math.fsum` for numerical
reductions. Avoid period-pair and level-pair cross-products.

**Implements**: U2-NFR-002, U2-NFR-010, U2-NFR-011, U2-NFR-014,
U2-NFR-017.

### U2-NFRP-05: Central Combined Coverage Policy

One helper calculates `absolute + relative*max(required, 1 kW)` and compares
the raw allocation residual before zero normalization. Allocation, diagnostics,
properties, and result assembly reuse it.

**Implements**: U2-NFR-008, U2-NFR-013, U2-NFR-023.

### U2-NFRP-06: Stable Binary64 Profile-Gap Kernel

Use positive kelvin, a strictly monotonic union breakpoint grid, deterministic
linear interpolation, absolute horizontal profile separation, reciprocal-
temperature trapezoids, finite checks, and `math.fsum`. Inactive utility
coordinates do not contribute breakpoints.

**Implements**: U2-NFR-009, U2-NFR-010, U2-NFR-013.

### U2-NFRP-07: Independent Analytical Oracle

Test production entropy and limiting behavior against standard-library
high-precision decimal calculations on generated small branches. Oracle code
does not call the production kernel or reuse its intermediate formulas.

**Implements**: U2-NFR-009, U2-NFR-030 through U2-NFR-032.

### U2-NFRP-08: Detached Turbine Adapter

Filter eligible positive-duty hot levels, order them deterministically, convert
to copied arrays, create one fresh existing turbine per call, and retain only
finite work/summary values. Explicit placement settings override detached
configuration values.

**Implements**: U2-NFR-011, U2-NFR-018, U2-NFR-019, U2-NFR-027,
U2-NFR-028.

### U2-NFRP-09: Recoverability Classifier

Classify coordinate-dependent targeting, thermodynamic, and turbine failures as
candidate diagnostics. Classify invalid context/settings, adapter contract
violations, invariant defects, and optimizer operational failures as typed
run-level errors with safe chained causes. Never use broad suppression or
retry.

**Implements**: U2-NFR-018 through U2-NFR-020, U2-NFR-023.

### U2-NFRP-10: Strict Scalar Partition

Map every physical feasible objective monotonically into `(0,1)` and every
normalized infeasible violation into `[1,2)`. Retain physical objective and
diagnostics separately; the parent independently filters feasibility before
public ranking.

**Implements**: U2-NFR-012, U2-NFR-014, U2-NFR-025.

### U2-NFRP-11: Pickle-Safe Worker Session

Serialize only frozen model/context/adapter specifications. Lazily construct a
lock and compact exact-coordinate memo in each execution process; omit both
from serialized state and recreate them after unpickling. No worker shares
mutable state.

**Implements**: U2-NFR-003, U2-NFR-015, U2-NFR-017.

### U2-NFRP-12: Bounded Compact Memo

Memoize only coordinate, backend scalar, feasibility, physical aggregate,
violation, and a bounded diagnostic reference. Cap unique records at the
worker evaluation budget. Never retain complete period/level output for every
callback.

**Implements**: U2-NFR-003, U2-NFR-006, U2-NFR-015, U2-NFR-024.

### U2-NFRP-13: Canonical Parent Re-evaluation

Union backend candidates with Unit 1 starts, exact-deduplicate, order points
deterministically for evaluation, and fully replay retained candidates in the
parent. Only parent results become public evidence.

**Implements**: U2-NFR-014 through U2-NFR-017, U2-NFR-025.

### U2-NFRP-14: Bounded Diagnostic Reservoir

Maintain counts by stable failure code and a deterministic reservoir of at
most ten representative diagnostics ordered by severity, period, side,
template, and code. Do not retain callback history or backend prose.

**Implements**: U2-NFR-003, U2-NFR-023, U2-NFR-024.

### U2-NFRP-15: Injected Public Adapters

Context targeting, turbine calculation, and optimizer execution enter through
small callable protocols with production defaults pointing to existing public
owners. Tests use deterministic substitutes; production code never imports a
private backend.

**Implements**: U2-NFR-019, U2-NFR-027 through U2-NFR-030.

### U2-NFRP-16: Tiered Performance Harness

Separate already-allocated kernels, one uncached replay, and one memo hit so a
failure names the owned stage. Use three warm-ups, ten measurements, p95, fixed
fixtures, and runtime/platform reporting. Do not impose a backend-wide clock.

**Implements**: U2-NFR-004 through U2-NFR-007.

### U2-NFRP-17: Deterministic Test Pyramid

Begin each production slice with a failing analytical example or generated
property. Use injected adapters and structured-grid/decimal oracles for routine
coverage, plus one small fixed-seed real dual-annealing regression. Shrinking
stays enabled and minimal failures become examples.

**Implements**: U2-NFR-030 through U2-NFR-032.

### U2-NFRP-18: Compatibility Guardrails

Protect old options JSON through defaults, the two-symbol root facade, public
optimizer/target/turbine imports, dependency declarations, architecture
direction, package contents, and installed specialist imports.

**Implements**: U2-NFR-026 through U2-NFR-031.

### U2-NFRP-19: Complete-or-Typed Boundary

Assemble the frozen result only after complete parent feasibility and identity
checks. Any run-level failure returns no partial result. Termination and bounded
diagnostics retain enough context to reproduce the request.

**Implements**: U2-NFR-018 through U2-NFR-025.

## Resilience Decisions

| Concern | Applied pattern | Explicitly absent |
|---|---|---|
| Invalid context/options | Typed staged fail-fast validation | Retry or fallback |
| Candidate-correctable failure | Structured penalty plus continued bounded search | Exception-driven ordinary search |
| Target/turbine invariant failure | Typed error with safe cause | Broad suppression |
| Backend exhaustion | Typed no-feasible result with bounded evidence | Least-infeasible success |
| Worker failure | Existing optimizer error translation | Shared recovery queue |
| Remote dependency outage | N/A; no remote dependency | Timeout/circuit breaker |
| Persistent recovery | N/A; no persistent state | Backup/restore |

## Scalability and Performance Decisions

- Context and full public output are proportional to required period/level
  evidence; callback memo records are compact and budget-bounded.
- One candidate replay is a period fold over level/interval work, not an
  all-period comparison.
- Existing optimizer worker processes remain isolated; no synchronization
  service or cross-process cache is introduced.
- Parent re-evaluation is bounded by deterministic starts plus the backend's
  bounded candidate output, then truncated to the public candidate limit.
- The tiered p95 gates isolate pure kernels, cold replay, and hot memo lookup.

## Security Design Status

Security Baseline remains disabled. Input validation, frozen snapshots,
pickle-safe explicit state, bounded diagnostics, and no dynamic code evaluation
are ordinary safe-library design patterns. No auth, credential, encryption,
network, file, remote execution, or compliance component is introduced.

## Extension Compliance at NFR Design

- **PBT-01**: carried forward from compliant Functional Design, including the
  clarified per-process memo property.
- **PBT-09**: carried forward from compliant NFR Requirements.
- **PBT-02 through PBT-08 and PBT-10**: N/A for blocking enforcement at NFR
  Design; U2-NFRP-07, U2-NFRP-11 through U2-NFRP-17 preserve their future Code
  Generation obligations.
- **Security Baseline**: skipped because disabled.
- **Resiliency Baseline**: skipped because disabled; local typed-failure
  patterns do not enable the extension.

There is no blocking extension finding.
