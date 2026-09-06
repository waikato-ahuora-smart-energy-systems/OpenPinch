# Unit 2 NFR Design Patterns

## Pattern Overview

Unit 2 applies in-process library patterns. It introduces no database, queue,
network client, persistent cache, background worker, service replica, circuit
breaker, or deployment component. CoolProp is required; TESPy remains an
explicit optional leaf.

## NFRP-U2-001: Layered Dependency Firewall

The production dependency direction is:

1. Unit 1 request/map contracts into Unit 2 engine-neutral values;
2. engine-neutral values and protocol into orchestration;
3. CoolProp into the required working-fluid resolver and default adapter; and
4. TESPy into only the optional concrete TESPy adapter.

The package initializer and orchestration modules do not import concrete
adapters eagerly. The closed factory performs a local concrete import only
after normalized backend selection. It never uses open-ended dynamic plugin
loading.

**Satisfies**: NFR-U2-001, NFR-U2-005, NFR-U2-006.

## NFRP-U2-002: Dependency Upgrade and Compatibility Gate

Before Unit 2 production behavior is accepted, project metadata and the uv lock
advance to `CoolProp>=8`. The complete existing thermodynamic suite is the
compatibility gate. The neutral and legacy-compatible TESPy extras share
`tespy>=0.10.1.post2` without an upper bound; a blocking real-engine profile
detects incompatible future TESPy releases.

The base installed-wheel profile omits TESPy and proves cold imports. The
optional installed-wheel profile installs `.[tespy]` and treats a missing TESPy
dependency as failure, not skip.

**Satisfies**: NFR-U2-002 through NFR-U2-005, NFR-U2-023.

## NFRP-U2-003: Frozen Per-Call Context and Cache

One pure builder resolves target basis plus Unit 1 request into frozen slotted
dataclasses. It validates mode, backend, capacity, temperatures, tolerances,
fluid specification, and fixed assumptions before engine preparation.

Resolved working-fluid data, invariant thermodynamic inputs, characteristic
data, convergence settings, and the TESPy design snapshot live only in the
fresh call/session. Close releases them. There is no module-level state object,
LRU cache, or cross-call property-state reuse.

**Satisfies**: NFR-U2-006, NFR-U2-009, NFR-U2-011, NFR-U2-022, NFR-U2-026.

## NFRP-U2-004: Capability-Checked Working-Fluid Resolution

The resolver parses one selected pure fluid, opaque registered blend, or
explicit N-component molar mixture. It preserves source spelling for audit,
normalizes backend identity and fractions, retains component order, rejects
`REFPROP` before constructing an engine object, and never mutates CoolProp
interaction data.

HEOS is the default. Other normalized CoolProp-native backends proceed through
actual wrapper/state capability checks rather than a name allowlist. The
resolved value provides adapter-neutral composition facts; the TESPy adapter
alone formats explicit mixtures with `|molar`.

Dew (`Q = 1`) is the evaporation saturation anchor and bubble (`Q = 0`) is the
condensation anchor. Both adapters must prove those states at preparation or
return a typed failure.

**Satisfies**: NFR-U2-007 through NFR-U2-010, NFR-U2-021.

## NFRP-U2-005: Lazy Canonical Cartesian Iterator

A pure iterator enumerates canonical source, sink, and load tuples and yields
one `HprOperatingPoint` at a time. It calculates ordinal identities directly
from loop indexes and never materializes a second full operating-point
collection.

The coordinator retains successful Unit 1 points only while a successful map
remains possible. On the first diagnostic it may discard accumulated successful
points; it continues traversing coordinates solely to gather required results
or ordered diagnostics. This preserves all-or-nothing behavior and linear
memory.

**Satisfies**: NFR-U2-016 through NFR-U2-018.

## NFRP-U2-006: Closed Protocol Factory and Session Context

`HprPointSimulator` is a structural protocol over `prepare`, `simulate`, and
`close`. A closed backend factory returns exactly one fresh CoolProp or TESPy
session. The generation coordinator owns it through one context-manager
boundary so cleanup executes from fresh, prepared, failed, or fatal state.

Concrete engine state is private to the adapter. The coordinator depends only
on frozen values, normalized success results, and typed internal failures.

**Satisfies**: NFR-U2-005, NFR-U2-019, NFR-U2-022, NFR-U2-026.

## NFRP-U2-007: Explicit Point-Local and Session-Fatal Failure

Adapters translate expected failures into an internal point-failure value with
a stable category and `session_fatal` flag:

- invalid coordinate or property state and isolated non-convergence may be
  point-local when the design snapshot remains restorable;
- design-snapshot corruption, topology inconsistency, unknown engine mutation,
  and restoration failure are session-fatal; and
- unknown unclassified engine exceptions default to session-fatal.

A point-local failure is recorded once. The next point begins by restoring the
prepared design snapshot; the failed point is never retried. A fatal failure
records the current diagnostic and assigns deterministic
`session_unavailable` diagnostics to all remaining canonical coordinates.

**Satisfies**: NFR-U2-019 through NFR-U2-021.

## NFRP-U2-008: Atomic Diagnostic Accumulator

The coordinator maintains either a candidate complete point list or a
diagnostic list. Any diagnostic irrevocably selects the error outcome. It never
constructs or returns a partial map.

There is at most one primary diagnostic per requested coordinate plus bounded
preparation and cleanup diagnostics. Cleanup failure is appended last. Error
ordering therefore follows preparation, canonical point order, then cleanup.

**Satisfies**: NFR-U2-016, NFR-U2-019 through NFR-U2-021.

## NFRP-U2-009: Sanitized Boundary With Chained Cause

Stable diagnostics include only closed codes, backend/model identity, canonical
coordinate values, bounded engine-neutral messages, and selected finite scalar
details. They exclude raw engine representations, full exception text,
tracebacks, temporary paths, memory addresses, and partial results.

The original exception may be attached only through Python exception chaining
for in-process debugging. Unit 2 emits no automatic log entry; the application
or caller decides whether and how to log the raised exception.

**Satisfies**: NFR-U2-020, NFR-U2-021, NFR-U2-027.

## NFRP-U2-010: Existing-Cycle CoolProp Adapter

The default adapter delegates every point to the existing
`VapourCompressionCycle`. It does not duplicate property equations. It passes
the resolved fluid, dew/bubble temperatures, duty basis, efficiency, superheat,
subcooling, and internal-HX assumptions; converts watts to kilowatts once; and
returns only a normalized result.

Generated nominal conditions compare against a direct existing-cycle oracle.
Fixed-temperature load properties verify proportional duty/power and
load-invariant COP.

**Satisfies**: NFR-U2-007, NFR-U2-010, NFR-U2-012 through NFR-U2-015,
NFR-U2-024.

## NFRP-U2-011: Restorable TESPy Design Snapshot

The TESPy session builds one network, solves one global design condition, and
saves one design snapshot in its unique `TemporaryDirectory`. Before each
offdesign point, it restores the same design basis and clears point-varying
specifications. Previous-point results are not accepted as an implicit initial
state.

The adapter uses documented public network save/design/offdesign operations.
Successful close removes the snapshot directory. Cleanup runs even when
preparation or point evaluation fails.

**Satisfies**: NFR-U2-004, NFR-U2-019, NFR-U2-022, NFR-U2-024, NFR-U2-026.

## NFRP-U2-012: Versioned Deterministic Convergence Settings

One frozen settings value identified as
`openpinch-tespy-hpr-convergence-v1` supplies public TESPy solve arguments:

- `max_iter=50` and `min_iter=4`;
- `init_previous=False`;
- `use_cuda=False`;
- `print_results=False`;
- `robust_relax=False`;
- `oscillation_damping=False`; and
- `skip_postprocess=False`.

TESPy's internal Newton residual rule remains owned by the selected TESPy
version and is not mutated through private globals. OpenPinch separately
applies its named post-solve energy, useful-duty, and finite-value acceptance
tolerances. The settings identifier and values enter provenance. Schema `1.0`
has no caller tuning surface.

**Satisfies**: NFR-U2-004, NFR-U2-010 through NFR-U2-012, NFR-U2-021,
NFR-U2-026.

## NFRP-U2-013: Strict Versioned Characteristic Resource

The compressor characteristic is a package JSON resource with an exact closed
shape: schema version, characteristic identifier, abscissa meaning, ordinate
meaning, and ordered finite numeric points. A pure loader reads through
`importlib.resources`, rejects missing/extra/invalid content, constructs an
immutable value, and computes a SHA-256 digest over canonical committed bytes.

The TESPy session loads it once. Provenance records its identifier, complete
numeric points, and digest. A regeneration/snapshot test detects accidental
drift. The resource contains no executable expression or engine object.

**Satisfies**: NFR-U2-010, NFR-U2-026, NFR-U2-027.

## NFRP-U2-014: Layered Numerical Acceptance

Numerical checks execute in this order:

1. adapter convergence and finite extracted values;
2. nonnegative heat magnitudes and positive compressor power;
3. mode-specific requested useful-duty agreement;
4. source/power/sink energy closure;
5. COP recomputation from useful duty and power; and
6. strict Unit 1 map construction.

Each layer uses its named absolute/relative tolerance. Temperature matching,
energy balance, engine convergence, and regression comparison are never
interchanged. Native finite floats are retained without rounding.

**Satisfies**: NFR-U2-012 through NFR-U2-015.

## NFRP-U2-015: Deterministic Provenance Assembly

Only the engine-neutral provenance builder creates the final recursive JSON
mapping. It combines context values, working-fluid identity, exact selected
backend, engine versions, design condition, convergence settings,
characteristic data/digest, point counts, and explicit compressor-only power
boundary after every point succeeds.

Keys and collection order are fixed. Runtime timestamps, host details, random
identifiers, engine objects, and filesystem paths are excluded. Equal contexts
and results within one dependency environment therefore produce structurally
equal provenance.

**Satisfies**: NFR-U2-010, NFR-U2-011, NFR-U2-013, NFR-U2-027.

## NFRP-U2-016: Complementary Deterministic Test Pyramid

Testing separates responsibilities:

- examples pin pure, registered-blend, explicit binary/ternary mixture,
  REFPROP rejection, dependency failure, physical corruption, and cleanup;
- Hypothesis properties cover normalization, grid size/order, induction,
  physical invariants, oracle comparison, failure atomicity, and repeatability;
- a Hypothesis state machine models fresh, prepared, point-local-failed,
  fatal, and closed simulator states;
- real CoolProp tests own the legacy oracle;
- a small blocking TESPy profile owns actual design/offdesign and mixture
  integration; and
- architecture/package tests own dependency profiles and forbidden edges.

Hypothesis retains shrinking and CI seed `20260715`. A shrunk discovered defect
becomes a permanent example regression.

**Satisfies**: NFR-U2-014, NFR-U2-019 through NFR-U2-025, NFR-U2-028;
PBT-02, PBT-03, PBT-05 through PBT-10. PBT-04 is N/A.

## NFRP-U2-017: Linear Performance Guard

The fake simulator performs constant work per point. The performance regression
generates and validates 10,000 points under 5.0 seconds and 256 MiB additional
traced memory on the primary CI runner. Smaller increasing grids verify linear
call counts and reject accidental pairwise work.

Real engine tests have no latency assertion. Production has no point cap,
timeout, scheduler, or implicit parallel execution.

**Satisfies**: NFR-U2-016 through NFR-U2-018.

## NFRP-U2-018: Layered Quality Gates

Gates run from cheapest to broadest:

1. pure helper, fake-adapter, and property tests;
2. CoolProp oracle and thermodynamic regressions under CoolProp 8 or newer;
3. architecture, import, dependency-metadata, and resource-drift checks;
4. Ruff lint/format and patch hygiene;
5. blocking real-TESPy profile;
6. coverage, package build/validation, and isolated base/optional wheel smokes;
7. warning-strict integrated documentation in Unit 3; and
8. complete repository regression.

No later unit consumes a failing Unit 2 boundary.

**Satisfies**: NFR-U2-002 through NFR-U2-005, NFR-U2-023 through NFR-U2-028.

## Resilience, Scalability, Security, and Infrastructure Disposition

- Retries and backend fallback are prohibited; deterministic repair means a new
  caller invocation after correcting input, dependency, or model state.
- Point-local continuation is allowed only by restoring the original design
  snapshot, never by retrying the failed coordinate.
- Linear local traversal replaces queues, chunks, workers, or streaming partial
  maps. Caller-owned isolated processes are the parallel scaling boundary.
- Sanitized diagnostics and non-executable JSON resources provide ordinary
  library safety. Authentication, authorization, secrets, and compliance
  controls are N/A. The Security extension is disabled.
- There is no remote availability target, failover, disaster recovery, or
  persistent state. The Resiliency extension is disabled.
- Infrastructure Design remains N/A.

## Pattern Coverage Validation

NFRP-U2-001 through NFRP-U2-018 collectively cover NFR-U2-001 through
NFR-U2-028. PBT-09 uses the approved Hypothesis/pytest stack; PBT-06 is covered
by the internal lifecycle state model; PBT-04 is N/A. No design pattern adds an
external consumer dependency or broadens the first-release HPR topology.
