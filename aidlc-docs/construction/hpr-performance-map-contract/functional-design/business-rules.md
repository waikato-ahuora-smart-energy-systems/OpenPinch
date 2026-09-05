# Unit 1 Business Rules

## Contract-wide rules

### BR-001: Closed alpha version

`schema_version` must equal `"1.0"`. Missing, differently typed, unknown, or
future values fail closed. Schema `1.0` remains labeled alpha until the two
golden fixtures pass the independent OpenUtility consumer tests.

### BR-002: Immutable strict values

Request, units, point, and map values are immutable after successful
construction. Extra fields are rejected at every contract object boundary.
NaN and positive or negative infinity are rejected wherever numbers occur,
including nested provenance.

### BR-003: Nonempty identity

`map_id`, `name`, `curve_id`, `thermodynamic_backend`, and `model_id` are
nonempty strings. Unit 1 does not reinterpret backend or model identifiers.

### BR-004: Exact canonical units

The unit mapping contains exactly five keys and values: source and sink
temperature use `degC`; source duty, sink duty, and electric power use `kW`.
Aliases, implicit defaults, missing keys, and additional unit keys are invalid.

## Request rules

### BR-005: Canonical coordinate request

Source temperatures, sink temperatures, and load fractions are nonempty finite
collections with no duplicates. Load fractions are greater than zero and at
most one. Each collection is stored in ascending order. Reordering the same
values does not change the canonical request.

### BR-006: Optional requested capacity

When present, request `reference_capacity` is finite and strictly positive. A
missing value means Unit 2 must derive the nominal useful capacity from the
supported target; zero does not mean missing.

## Point rules

### BR-007: Numeric domains

Temperatures are finite. `load_fraction` is in `(0, 1]`. `q_source`, `q_sink`,
and `electric_power` are finite and nonnegative. Because schema `1.0` producer
points are active part-load points, `electric_power` and `cop` are finite and
strictly positive.

### BR-008: Energy balance

For every point, the absolute residual of
`q_sink - q_source - electric_power` does not exceed
`energy_balance_tolerance`. Schema `1.0` has no hidden loss or auxiliary-energy
term outside `electric_power`.

### BR-009: Useful capacity and load

For heat pumps, useful duty is `q_sink`; for refrigeration it is `q_source`.
The absolute residual of useful duty minus
`load_fraction * reference_capacity` does not exceed
`energy_balance_tolerance`.

### BR-010: COP convention

Heat-pump mode requires `cop_convention="heating"` and
`cop = q_sink / electric_power`. Refrigeration mode requires
`cop_convention="cooling"` and `cop = q_source / electric_power`. The absolute
COP residual must not exceed `energy_balance_tolerance`, matching the settled
OpenUtility alpha consumer semantics.

## Mode and map rules

### BR-011: Mode-dependent capacity basis

Heat-pump mode requires `reference_capacity_basis="q_sink"`.
Refrigeration mode requires `reference_capacity_basis="q_source"`. No other
mode, capacity basis, or pairing is accepted.

### BR-012: Positive reference capacity

Map `reference_capacity` is finite and strictly positive. Point duties are
absolute `kW` values at that map capacity, not normalized per-unit values.
Downstream fixed-capacity scaling is not encoded into the map.

### BR-013: Ordered curve topology

`interpolation_topology` must equal `"ordered_part_load_curve"`. Each
`curve_id` contains one fixed source/sink temperature pair and points appear in
strictly increasing `load_fraction` order. This topology permits adjacent
breakpoint interpolation only and never permits interpolation across curves.

### BR-014: Identity and coordinate uniqueness

Point names are unique across a map. The tuple `(curve_id,
source_temperature, sink_temperature, load_fraction)` is unique. Repeating a
load on one curve, assigning multiple temperature pairs to one curve, or
duplicating a point name is invalid.

### BR-015: Complete successful map

A map contains at least one point and is returned only when every point and
curve passes validation. Unit 1 has no incomplete-map representation. Golden
fixtures contain at least three ordered points so adjacent-only consumer
interpolation can be tested.

## Tolerance rules

### BR-016: Separate tolerances

`energy_balance_tolerance` and `temperature_match_tolerance` are finite and
nonnegative and default to `1e-6`. Energy tolerance applies only to energy,
capacity, and COP consistency. Temperature tolerance applies only to matching
canonical `degC` coordinates in a consumer. Neither substitutes for the other.

### BR-017: Boundary comparison

A residual equal to its applicable tolerance passes. A residual greater than
the tolerance fails. Exact validation is available by setting a tolerance to
zero.

## Provenance rules

### BR-018: JSON-compatible structured provenance

`provenance` is a nonempty string-keyed object containing only JSON-compatible
null, boolean, string, finite-number, list, and nested-object values. Runtime
targets, engine sessions, NumPy arrays/scalars, bytes, paths, sets, and
non-string keys are rejected unless a producing unit explicitly converts them
to the declared JSON boundary first.

### BR-019: No lossy string coercion

Numbers, booleans, lists, nested objects, and null retain their JSON type.
Validation never converts an unsupported object to text merely to make it
serializable.

## Serialization and resource rules

### BR-020: Round-trip equality

Serializing a valid map to a JSON-compatible mapping or JSON text and validating
that representation reconstructs a value equal to the original canonical map.

### BR-021: Deterministic order and bytes

Equivalent canonical map values emit points and object keys under one stable
ordering/formatting policy. Golden resources are generated from validated
values, have no committed solver/runtime objects or outputs, and reproduce
byte-for-byte when regenerated with unchanged inputs.

### BR-022: Authoritative schema and fixtures

JSON Schema and the two golden fixtures are produced from the authoritative
contract. Each fixture validates both as an OpenPinch contract and against the
published JSON Schema. Consumers may vendor the resources, but OpenPinch never
imports the consumer.

## Error rules

### BR-023: Fail closed without fallback

Contract validation never changes an unsupported version, unit, mode, COP
convention, capacity basis, point, curve, provenance value, or tolerance to a
supported value. It returns no partial map on failure.

### BR-024: Actionable error location

Errors identify the offending field or collection element and distinguish
structural, point-physical, curve-topology, map-consistency, and provenance
failures. Message fragments used by tests describe the violated invariant, not
an implementation call stack.

## Golden fixture rules

### BR-025: Heat-pump fixture

The heat-pump fixture uses `mode="heat_pump"`, `q_sink` reference capacity,
heating COP, external `degC` coordinates, absolute `kW` values, and at least
three ascending breakpoints with nonconstant COP.

### BR-026: Refrigeration fixture

The refrigeration fixture uses `mode="refrigeration"`, `q_source` reference
capacity, cooling COP, external `degC` coordinates, absolute `kW` values, and
at least three ascending breakpoints with nonconstant COP.

### BR-027: Consumer-neutral content

Fixtures contain physical performance and producer provenance only. They omit
OpenUtility candidate names, thermal node assignments, costs, installed-size
decisions, operating periods, selection policy, and optimizer formulation data.

## Property-Based Testing compliance

- **PBT-01**: testable properties are identified in the business logic model.
- **PBT-02**: generated valid maps cover mapping and JSON round trips.
- **PBT-03**: generated maps cover physical, ordering, uniqueness, range, and
  type invariants.
- **PBT-04**: N/A; no idempotent mutation is claimed.
- **PBT-05**: JSON Schema validation is an oracle for representable structural
  rules; semantic model checks remain authoritative.
- **PBT-06**: N/A; all Unit 1 public values are immutable.
- **PBT-07**: reusable strategies generate structurally valid requests, points,
  nested provenance, and complete mode-specific maps.
- **PBT-08**: Hypothesis shrinking remains enabled and failures retain the
  repository's reproducible seed output.
- **PBT-09**: the existing Hypothesis/pytest toolchain is retained.
- **PBT-10**: golden fixtures and explicit invalid examples accompany all
  critical properties.

No Unit 1 Functional Design PBT finding is blocking.
