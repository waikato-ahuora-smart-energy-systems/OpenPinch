# Unit 1 Non-Functional Requirements

## Scope

These requirements apply to Placement Contracts and Pure Model only. The unit
is a local Python library module: it owns immutable schemas and deterministic
transformations, not optimisation runtime, process targeting, a network
service, persistent data, UI, or deployment infrastructure.

## Scalability and Capacity

### U1-NFR-001: No arbitrary public capacity ceiling

The first release shall not impose a product-level maximum on isothermal count,
sensible count, period count, candidate count in result contracts, or nested
diagnostic count. Validity remains governed by integer/range rules and by
available in-process resources. Unit 2 owns explicit optimisation budgets.

**Acceptance**: valid generated requests beyond representative test sizes are
not rejected solely by a hardcoded count cap; invalid/inconsistent collection
sizes still fail before model construction.

### U1-NFR-002: Linear pure-model scaling

Let `P` be selected periods, `L = 2*(N_iso + N_sens)` total hot/cold levels,
and `D = N_iso + 2*N_sens` generated-pair decision coordinates (or
`D = 2*N_iso + 4*N_sens` for independent explicit/inferred templates).
Request/template work shall
be `O(L)`, period-bound validation/intersection `O(P*D)`, order propagation
`O(L)`, vector encode/decode `O(D)`, and result serialization `O(serialized
result size)`. No transformation may compare every period with every other
period or every level with every non-adjacent level.

**Acceptance**: code review and a scaling benchmark show no worse than linear
growth in `P*D` after fixed overhead; doubling only periods or coordinates shall
not create quadratic growth.

### U1-NFR-003: Representative performance budget

On the project CI runner, the combined pipeline of request/template
normalization, feasibility-envelope validation and intersection, order-bound
propagation, vector schema construction, deterministic primary-start creation,
and independent start verification shall complete within 250 ms for:

- 10 isothermal plus 10 sensible levels per side (40 physical levels);
- 60 decision coordinates;
- 100 periods and 6,000 coordinate intervals;
- canonical units and a valid nonempty feasible region.

The benchmark shall use a warm-up plus at least ten measured repetitions; its
95th percentile must not exceed 250 ms. The fixture and runner details shall be
reported on failure. Serialization of a fully populated Unit 2 result is
measured separately because its size is not owned by this unit's model builder.

## Numerical Correctness and Reproducibility

### U1-NFR-004: Named tolerance policy

Defaults shall be:

- absolute comparison: `1e-6` in canonical quantity magnitude;
- bound comparison: `1e-6`;
- ordering comparison: `1e-6`;
- relative comparison: `1e-9`;
- coverage comparison: declared in the shared tolerance contract but finalized
  by Unit 2 Functional Design.

The first three reuse `OpenPinch.domain.configuration.tol`. Overrides shall be
finite and non-negative. Equality within bound tolerance normalizes to a fixed
coordinate. Positive physical constraints such as absolute temperature and
minimum separation remain strictly positive; tolerance is not permission to
cross a physical zero.

### U1-NFR-005: Unit correctness

Every public quantity shall accept configured bare-scalar interpretation or an
explicit compatible unit. Conversion shall use existing OpenPinch/Pint
ownership. Unknown and dimensionally incompatible units fail with typed field
context. Canonical values and result unit labels shall agree; no currency
conversion is attempted.

### U1-NFR-006: Deterministic values and ordering

Equal normalized inputs and envelopes shall produce equal templates, effective
bounds, coordinate schemas, primary starts, encoded vectors, decoded
placements, diagnostics, serialized collection order, and errors. Unit 1 shall
use no randomness, unordered set/dict iteration for public order, ambient
locale, system clock, or process identity.

## Reliability, Availability, and State

### U1-NFR-007: Deterministic completion or typed failure

Every public Unit 1 operation shall either return a complete frozen value or
raise the documented typed exception. It shall not expose a partially built
model, partially normalized caller collection, or mutable intermediate.
Expected candidate invalidity returns structured diagnostics rather than broad
exception suppression.

### U1-NFR-008: Source-state isolation

Success and failure shall leave all caller lists, mappings, Pydantic inputs,
streams, zones, targets, configurations, caches, and workspace state observably
unchanged. No returned contract may retain a mutable reference to those
objects. Memory use shall be `O(P*D + output size)` with no process-global
request/result cache.

### U1-NFR-009: Availability and continuity are N/A

Uptime, failover, disaster recovery, backup/restore, recovery point, recovery
time, health checks, and business continuity are not applicable because Unit 1
has no service process or persistent state. Its continuity contract is
reconstructability: equal inputs reproduce equal values or typed errors.

## Security and Data Protection

### U1-NFR-010: Existing validation boundary only

The Security Baseline extension remains disabled. Unit 1 shall add no
authentication, authorization, encryption, credential, network, file, dynamic
code execution, or remote-data boundary. Ordinary safe-library controls remain
required:

- Pydantic `extra="forbid"` and frozen specialist contracts;
- finite values, exact collection/key agreement, and compatible units;
- no callable, exception instance, mutable runtime object, or backend-private
  object in serializable contracts;
- no evaluation/import of caller-provided strings as code;
- no secrets or full mutable source payloads in diagnostics.

This requirement does not claim security-extension compliance; it preserves
the approved no-new-boundary scope.

## Observability and Usability

### U1-NFR-011: Actionable typed diagnostics

Validation errors shall expose a stable code and message plus field path,
template key, period identifier, units, measured/required values, and details
when applicable. Messages shall name the violated rule and corrective action.
They shall not include stack-dependent object representations or rely on
parsing backend messages.

### U1-NFR-012: Discoverable stable specialist API

Public specialist types shall be importable from their concrete owner module,
use documented snake-case fields and lowercase enum values, preserve collection
order, expose explicit unit labels, and provide schema/docstrings suitable for
IDE discovery. The package root remains exactly `PinchProblem` and
`PinchWorkspace`.

## Compatibility and Portability

### U1-NFR-013: Backward-compatible additive change

Unit 1 shall not change existing input/output schemas, target behavior, root
exports, configuration defaults, unit conventions, optional-dependency
boundaries, or installed-package imports. New fields in specialist schemas must
be optional or versioned under normal compatibility policy; removing or
renaming stable fields/enum values is a breaking change requiring explicit
approval and documentation.

### U1-NFR-014: Supported runtime and distribution

The unit shall run on the repository's supported Python 3.14.2+ environments and
ship in the existing wheel/source distribution. It shall use cross-platform
Python/Pint behavior and introduce no platform-specific path, process, binary,
or locale dependency.

## Maintainability and Test Quality

### U1-NFR-015: Ownership and dependency direction

Contracts belong to `OpenPinch.contracts.utility_placement`; pure normalization,
model, codec, and errors belong to the specialist analysis owner selected in
Application Design. Unit 1 may depend on contracts, domain value/unit owners,
configuration constants, enums, and standard library. It shall not import
application, presentation, targeting services, steam turbines, optimiser
backends, or Unit 2/3 implementations.

### U1-NFR-016: Static and documentation quality

New public/specialist functions and models shall have typed signatures,
purposeful docstrings, explicit `__all__` only at the specialist owner, and
focused modules aligned with one responsibility. Ruff and architecture/import
checks shall pass. Broad `except Exception` suppression and unexplained numeric
literals are prohibited.

### U1-NFR-017: Coverage and regression gates

Delivery requires:

- failing-first example tests for each implemented business slice;
- all PBT-01 properties with domain-valid reusable strategies;
- Hypothesis shrinking enabled and fixed seed `20260715` in CI;
- every discovered minimal counterexample retained as an example regression;
- focused Unit 1 tests and all cross-unit tests available at the time;
- the complete `not solver` suite;
- branch coverage satisfying the repository-wide 95% gate;
- Ruff, architecture/import, packaging, and specialist installed-import smoke.

No flaky retry may hide a property failure.

## Technology and Extension Requirement

### U1-NFR-018: PBT-09 framework capability

Hypothesis is the required Python property-testing framework and pytest is the
runner. The selected versions/declarations must support reusable composite
strategies, automatic shrinking, explicit seed replay, and pytest integration.
The dependency and fixed-seed CI configuration must exist before NFR
Requirements can be approved.

## Traceability

| Approved requirement | Unit 1 NFRs |
|---|---|
| NFR-001 numerical correctness | U1-NFR-004 through U1-NFR-006 |
| NFR-002 reproducibility | U1-NFR-006, U1-NFR-017, U1-NFR-018 |
| NFR-003 bounded execution | U1-NFR-001 through U1-NFR-003; optimiser budgets deferred to Unit 2 |
| NFR-004 compatibility | U1-NFR-012 through U1-NFR-014 |
| NFR-005 maintainability | U1-NFR-015 through U1-NFR-017 |
| NFR-006 observability | U1-NFR-007, U1-NFR-011 |
| NFR-007 approved security/resiliency scope | U1-NFR-009, U1-NFR-010 |
| PBT-09 | U1-NFR-018 |

## NFR Acceptance Summary

Unit 1 is acceptable when all 18 requirements have executable or reviewable
evidence, the representative model pipeline meets its CI budget, public values
are deterministic/detached/unit-correct, the specialist API is additive and
portable, the fixed-seed PBT and 95% coverage gates pass, and no disabled
extension is represented as enabled.
