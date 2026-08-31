# Unit 1 NFR Design Patterns

## Pattern Catalogue

### NFRP-01: Frozen Boundary Contracts

All specialist request, template, envelope, model-output, candidate, result, and
diagnostic contracts are frozen Pydantic values with forbidden extra fields.
Ordered collections become tuples at the boundary. Nested mappings become
ordered immutable entries or detached copies. No model holds a `Zone`, stream,
target, application accessor, callable, exception instance, NumPy array, or
backend result.

**Implements**: U1-NFR-007, U1-NFR-008, U1-NFR-010, U1-NFR-013.

### NFRP-02: Staged Fail-Fast Validation

The public pure pipeline has named stages: request, template blueprint, units,
economics, envelope, effective bounds, ordering, schema, starts, and
serialization. A stage receives complete immutable output from the prior stage
and either returns its own complete output or raises its typed failure. No
retry, fallback, partial object, or downstream validation occurs after failure.

Expected black-box candidate invalidity uses `CandidateVerification` rather
than exceptions; Unit 2 later turns those diagnostics into deterministic
penalties.

**Implements**: U1-NFR-007, U1-NFR-009, U1-NFR-011.

### NFRP-03: Deterministic Normalization

Normalization is a pure idempotent function of explicit input values and the
declared unit policy. It strips/validates identities, converts units once,
normalizes signed zero, preserves semantic collection order, and creates stable
generated identities. It reads no clock, locale, environment variable, random
source, mutable configuration singleton, or hash-set ordering.

**Implements**: U1-NFR-004 through U1-NFR-006.

### NFRP-04: Two-Phase Blueprint and Envelope Boundary

Unit 1 first turns counts/optional templates into a complete immutable
`TemplateBlueprintSet`. Unit 2 consumes those keys to construct a detached
physical envelope, then calls Unit 1 model construction with the request,
blueprints, and envelope. This data handshake permits generated templates while
preserving package dependency direction: Unit 2 imports Unit 1; Unit 1 never
imports Unit 2.

**Implements**: U1-NFR-006, U1-NFR-013, U1-NFR-015.

### NFRP-05: Ordered Tuple Plus Ephemeral Index

Canonical values retain tuples for observable order. Each operation may build a
local dictionary from stable keys to tuple positions for constant-time
agreement checks. The dictionary is not serialized, cached, or returned. This
provides deterministic order plus linear validation without mutable retained
state.

**Implements**: U1-NFR-001, U1-NFR-002, U1-NFR-006, U1-NFR-008.

### NFRP-06: Single-Pass Interval Reduction

For each coordinate, period intervals are reduced once to maximum lower and
minimum upper bounds while retaining the source period responsible for each
active limit. Caller narrowing is checked once after physical reduction.
There is no period-pair comparison.

**Implements**: U1-NFR-002, U1-NFR-003, U1-NFR-011.

### NFRP-07: Monotone Adjacent Constraint Propagation

Independent hot/cold ordering constraints and generated same-kind identity
constraints use forward/backward passes over adjacent ranks. Each update only
tightens a lower or upper bound. Generated cross-kind adjacency is checked on a
transient supply-temperature sort, allowing isothermal and sensible levels to
interleave without changing stable vector identities. A pass stops when no
value changes beyond bound tolerance. For a simple chain, the implementation
shall prefer the bounded fixed number of directional passes proven sufficient
by Functional Design; any defensive repeat loop has an explicit maximum
derived from level count and raises an invariant error if exceeded.

**Implements**: U1-NFR-002 through U1-NFR-004, U1-NFR-007.

### NFRP-08: Precomputed Decision Schema

Model construction creates the coordinate tuple and key-to-index lookup once.
Encoding, decoding, and verification reuse the schema and effective bounds.
They do not rerun Pydantic request validation, unit conversion, envelope
intersection, or order propagation. Fixed coordinates remain present so vector
dimension and identity are stable.

**Implements**: U1-NFR-002, U1-NFR-003, U1-NFR-006.

### NFRP-09: Canonical Unit Adapter

One small adapter converts `(value, source_unit, canonical_unit)` through
existing `OpenPinch.domain.value.Value.to`. Quantity metadata chooses the
canonical unit; callers cannot supply a conversion algorithm. The adapter
normalizes conversion failures into `UtilityPlacementUnitError` and is passed
as a callable to normalization for focused testing. Tests substitute the
callable, never the global Pint registry.

**Implements**: U1-NFR-004, U1-NFR-005, U1-NFR-010, U1-NFR-015.

### NFRP-10: Central Tolerance Policy

All comparisons receive one `PlacementTolerances` value. Default absolute,
bound, and ordering tolerance is the existing `tol=1e-6`; relative tolerance is
`1e-9`. Helpers distinguish approximate comparison from strict physical
positivity and normalize equal-within-tolerance intervals to a single
deterministic coordinate. No component defines a private epsilon.

**Implements**: U1-NFR-004 through U1-NFR-006, U1-NFR-016.

### NFRP-11: Structured Diagnostics, Silent Pure Core

Typed failures and candidate diagnostics carry stable codes plus applicable
field/template/period/quantity context. Pure components do not log, print,
warn, or mutate a diagnostics collector. Unit 2/3 boundaries decide whether and
how to log or present the returned context.

**Implements**: U1-NFR-006, U1-NFR-007, U1-NFR-011, U1-NFR-012.

### NFRP-12: Safe Primitive Serialization

Result validation rejects non-finite and private/runtime values before JSON.
Pydantic emits stable enum values, snake-case fields, explicit unit labels, and
ordered arrays. Contract snapshots test structure; round-trip properties test
generated contents with structural equality and named float tolerances.

**Implements**: U1-NFR-008, U1-NFR-010 through U1-NFR-013.

### NFRP-13: Explicit Compatibility Guardrails

Architecture tests protect dependency direction and the two-symbol root facade.
Schema snapshots protect specialist names/enum values. Focused regressions
protect shared contracts/configuration defaults. Built wheel/source smoke tests
import specialist modules without optional profiles.

**Implements**: U1-NFR-012 through U1-NFR-016.

### NFRP-14: Complementary Fixed-Seed Test Pyramid

Every production slice begins with a failing explicit scenario. PBT modules
exercise the PBT-01 round trips, invariants, idempotence, commutativity, oracle,
and easy-verification properties through reusable strategies. CI uses seed
`20260715` with shrinking enabled; shrunk failures become permanent examples.
Focused tests precede the full non-solver, 95% branch-coverage, Ruff,
architecture, packaging, and installed-import gates.

**Implements**: U1-NFR-016 through U1-NFR-018.

### NFRP-15: Representative Performance Evidence

A generated valid fixture contains 10 isothermal and 10 sensible levels per
side across 100 periods. A pure benchmark warms the pipeline, records at least
ten iterations, calculates p95, and asserts at most 250 ms on the project CI
runner. Separate fixtures vary `P` and `D` one at a time to detect superlinear
growth. The test reports runtime and fixture dimensions on failure.

**Implements**: U1-NFR-002, U1-NFR-003, U1-NFR-014, U1-NFR-017.

## Resilience Decisions

| Concern | Applied pattern | Explicitly absent |
|---|---|---|
| Invalid caller input | Staged typed fail-fast validation | Retry or fallback |
| Ordinary candidate violation | Structured verification diagnostics | Exception-driven search |
| Unit conversion failure | Typed adapter translation | Silent default units |
| Internal invariant failure | Typed model error with evidence | Partial model return |
| Dependency outage | N/A; no remote dependency | Circuit breaker/timeouts |
| Persistent recovery | N/A; no persistent state | Backup/restore/replay |

## Scalability and Performance Decisions

- Canonical storage is proportional to required input/output evidence; no
  unbounded internal history exists.
- Period processing is a reduction, not a cross-product between periods.
- Ordering checks are adjacent-chain operations, not all-pairs comparisons.
- Pydantic validates boundary values once; hot loops use already normalized
  floats and immutable internal structures.
- No caching is needed because hidden retained state would conflict with
  determinism/non-mutation and the pipeline has an explicit 250 ms budget.
- No parallelism is introduced because the work is small, linear, and pure;
  parallel coordination would add overhead and ordering complexity.

## Security Design Status

Security Baseline remains disabled. NFRP-01, NFRP-02, NFRP-09, NFRP-11, and
NFRP-12 are ordinary correctness and safe-library boundaries, not an enabled
security architecture. There is no credential, auth, network, file, remote
execution, or compliance component.

## Extension Compliance at NFR Design

- **PBT-01**: carried forward from compliant Functional Design.
- **PBT-09**: carried forward from compliant NFR Requirements.
- **PBT-02 through PBT-08 and PBT-10**: N/A for blocking enforcement at NFR
  Design; patterns preserve their future Code Generation obligations.
- **Security Baseline**: skipped because disabled.
- **Resiliency Baseline**: skipped because disabled; fail-fast behavior is an
  approved local correctness pattern, not extension enablement.
