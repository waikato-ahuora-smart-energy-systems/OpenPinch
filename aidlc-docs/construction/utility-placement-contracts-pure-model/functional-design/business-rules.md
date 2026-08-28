# Unit 1 Business Rules

## Rule Precedence

Validation runs in this order so failures are deterministic:

1. request shape, enums, counts, period selection, tolerances, and options;
2. template identity, inventory, side, kind, and kind-specific fields;
3. units and finite canonical values;
4. monetary inputs and cogeneration metadata;
5. feasibility-envelope identity, periods, weights, bounds, and units;
6. physical/caller interval intersection and order-chain propagation;
7. vector schema, starting points, encode/decode, and candidate verification;
8. serialization boundary checks.

The first invalid stage raises one stable typed error. Validation does not
continue and replace it with a downstream symptom.

## Request Rules

### BR-001: Objective default and values

The absent objective becomes `thermodynamic`. Only `thermodynamic` and
`monetary` normalize successfully; normalization is case-sensitive after normal
enum/string handling so misspellings fail visibly.

### BR-002: Count types

Both counts must be true integers. Boolean, fractional, string-coerced,
NaN/infinity, and arbitrary numeric-protocol values are rejected unless the
existing Pydantic integer contract proves they are exact integers without lossy
coercion.

### BR-003: Count ranges and symmetry

`isothermal_level_count >= 2` and `sensible_level_count >= 0`. Each count
applies independently to both hot and cold sides. No side-specific count
override exists in the first release.

### BR-004: Period selection

Explicit period identifiers must be non-empty and unique in caller order. An
empty explicit selection is invalid. Resolution of omitted periods belongs to
Unit 3; Unit 1 only validates/retains the normalized selection supplied to it.

### BR-005: Tolerances

Absolute, relative, bound, coverage, and ordering tolerances are finite and
non-negative. Minimum separation and minimum sensible span are finite and
strictly positive. Their defaults are named values, never unexplained literals
inside transformations.

### BR-006: Bounded options

Candidate limit, iteration limit, and evaluation limit are positive integers;
seed is an integer and booleans are rejected. Backend-specific option-name
validation remains owned by the existing optimisation service in Unit 2.

## Template Rules

### BR-007: Complete collection or complete generation

For each side, omission generates the full inventory. A supplied collection
must itself contain exactly the requested inventory; missing members are never
filled implicitly.

### BR-008: Inventory agreement

Each side contains exactly `N_iso` templates with kind `isothermal` and exactly
`N_sens` with kind `sensible`. A side/kind mismatch names the expected and
observed counts.

### BR-009: Identity

Names are trimmed, non-empty strings and unique across both sides. Identity is
`(side, name)` for lookup, while global name uniqueness prevents ambiguous
human-facing diagnostics. Caller declaration order and `placement_rank` are
immutable.

### BR-010: Generated identity and metadata

Generated names follow `{side}_{kind}_{one_based_ordinal}` with kind rendered
as `iso` or `sensible`. Generated templates have no price, no fluid metadata,
and `cogeneration_eligible=false`. Generation is deterministic for equal counts
and envelope inputs.

### BR-011: Side and direction

Hot templates derive target temperature by subtracting span; cold templates add
span. Any explicit target-direction field must agree. A template cannot change
side during normalization or decoding.

### BR-012: Isothermal span

An isothermal span is fixed, finite, and strictly positive. The omitted value is
`0.01 delta_degC`. It has no lower/upper decision bounds and no span coordinate.

### BR-013: Sensible span

A supplied sensible span override has finite lower/upper bounds with lower at
least the named positive minimum sensible span and lower not greater than upper.
When omitted or generated, the complete physical span interval comes from the
feasibility envelope. Equal effective bounds are a supported fixed sensible
coordinate; it remains in the vector to preserve schema dimension.

### BR-014: Supply bounds

Template supply overrides are optional finite intervals. Equal bounds are
supported. Their dimensional unit must be absolute temperature, not a
temperature-difference unit.

### BR-015: Economics

Thermodynamic requests may omit all prices. Monetary requests require a finite,
non-negative price for every template and a finite, non-negative electricity
price. Zero is valid. Generated templates therefore require explicit economic
enrichment before a monetary request validates.

### BR-016: Cogeneration eligibility

Eligibility defaults false and is meaningful only for hot templates. A cold
template marked eligible fails validation. Unit 1 validates required declared
fluid/turbine input metadata structurally; Unit 2 owns physical turbine
compatibility.

## Unit and Numerical Rules

### BR-017: Scalar interpretation

A bare scalar is interpreted in the problem's configured input unit for its
quantity. An explicit value-with-unit overrides that default after dimensional
compatibility validation.

### BR-018: Canonical conversion

Supply temperatures normalize to `degC`, spans to `delta_degC`, duties/work to
`kW`, prices to the configured canonical price unit, entropy to `kW/K`, and cost
rate to the configured output utility-cost unit. Currency conversion is not
performed.

### BR-019: Finite and positive-Kelvin values

Every normalized numeric field is finite. Each possible supply and derived
target temperature represented by accepted bounds must remain strictly above
zero kelvin, allowing the named numerical tolerance only for comparisons, not
as permission to reach zero.

### BR-020: Signed zero

Every normalized public float equal to zero becomes positive `0.0`. This rule
does not convert negative non-zero values.

### BR-021: Float equality

Structural values compare exactly. Converted or calculated floats compare with
the request's named absolute/relative tolerances. Validation and tests must not
embed a separate tolerance.

## Feasibility-Envelope Rules

### BR-022: Boundary ownership

Unit 1 defines `PlacementFeasibilityEnvelope`; Unit 2 populates it. Unit 1 must
not import Unit 2 or any target service.

### BR-023: Period identity and weights

Envelope period identifiers are ordered and unique. Each weight is finite and
non-negative, and at least one weight is positive. Template-bound mappings
contain every selected period exactly once.

### BR-024: Template-key agreement

Envelope keys exactly equal normalized request template keys. Extra and missing
keys both fail. Isothermal span keys are prohibited; sensible span keys are
required.

### BR-025: Physical intervals

Every physical interval is finite and has lower no greater than upper. Equal
bounds are supported fixed coordinates. Bounds carry canonical compatible unit
metadata.

### BR-026: Period intersection

The physical intersection is maximum lower bound and minimum upper bound across
selected periods. It is independent of period iteration order and retains
source period detail for diagnostics.

### BR-027: Caller overrides narrow only

A caller lower bound may equal or exceed the physical lower bound; a caller
upper bound may equal or be below the physical upper bound. Expansion beyond
physical feasibility raises a template validation error instead of clipping.

### BR-028: Empty feasible region

An effective `lower > upper` beyond bound tolerance, a contradictory ordering
chain, or impossible positive-Kelvin derived target raises
`EmptyPlacementFeasibleRegionError` before optimiser execution. Equality within
tolerance normalizes to one fixed bound.

## Ordering and Vector Rules

### BR-029: Physical order

In `placement_rank`, adjacent hot supplies are strictly descending and adjacent
cold supplies strictly ascending by at least the named positive separation.
Kinds may be interleaved by explicit caller declaration.

### BR-030: No implicit repair

Normalization and candidate validation never reorder, merge, deduplicate, or
rename levels to satisfy BR-029. A failing candidate remains associated with
its original identities.

### BR-031: Coordinate sequence

Vector coordinates are hot isothermal supplies, hot sensible supply/span pairs,
cold isothermal supplies, then cold sensible supply/span pairs. Relative order
within a family follows caller declaration order.

### BR-032: Vector dimension

Dimension is exactly `2*N_iso + 4*N_sens`, including fixed supply or sensible
span coordinates with equal bounds.

### BR-033: Encode completeness

Encoding rejects missing, duplicate, or unknown template/field values.
Successful output contains one finite value per schema coordinate in BR-031.

### BR-034: Decode completeness

Decoding rejects a vector of incorrect length or a non-finite/out-of-bound
coordinate. It reconstructs every template once in original side declaration
order and derives targets using BR-011.

### BR-035: Fixed span preservation

Every decoded isothermal template retains its declared fixed span exactly after
canonical normalization. Sensible spans remain within their effective bounds.

### BR-036: Deterministic primary start

For the same normalized model, the primary start is identical. Hot supplies use
the deterministic hottest-feasible chain, cold supplies the coldest-feasible
chain, and sensible spans their interval midpoints.

### BR-037: Start verification

Every emitted start must pass the independent candidate verifier. Failure to
produce one from a model previously declared feasible is a model invariant
failure, not an optimiser exhaustion.

## Contract, Result, and Error Rules

### BR-038: Frozen detached values

Specialist public contracts are frozen and reject extra fields. They copy
caller collections and retain no mutable runtime or backend object.

### BR-039: Result inventory

Result contracts provide request/scope/termination metadata; one best candidate;
ordered alternatives; solved hot/cold levels; period/aggregate breakdowns;
coverage; entropy/exergy; thermal/cogeneration/electricity/net monetary values;
units; feasibility; and diagnostics. Optional values remain explicit `None`,
not invented zeros.

### BR-040: Serializable values only

JSON cannot contain callables, exception instances, NumPy objects, zones,
streams, targets, optimiser objects, or non-finite floats.

### BR-041: Typed root and compatibility

All specialist failures inherit `UtilityPlacementError`. Invalid caller-data
subclasses also inherit `ValueError`; operational subclasses introduced by
later units inherit only the specialist runtime root unless invalid input is
their cause.

### BR-042: Error context

Every typed failure has a stable code and message. Field path, template key,
period identity, objective, scope, method, seed, and details are populated only
when meaningful and must remain serializable.

### BR-043: Candidate diagnostics

An ordinary decoded-candidate failure is a structured diagnostic with code,
constraint, measured value, limit/tolerance, template, period, and side where
applicable. It is not thrown during expected black-box exploration.

## Scenario Matrix

| Scenario | Required outcome | Rules |
|---|---|---|
| Exactly two isothermal, zero sensible | Valid four-level request/model | BR-002, BR-003, BR-007, BR-008 |
| Boolean count | Request validation error before template work | BR-002 |
| Count-only thermodynamic request | Stable generated identities and no invented prices | BR-007, BR-010, BR-015 |
| Count-only monetary request | Missing-economics validation error | BR-010, BR-015 |
| Interleaved caller kinds | Identity/order retained; coordinate family order remains stable | BR-009, BR-029, BR-031 |
| One fixed supply bound | Equal bound retained as a coordinate | BR-014, BR-025, BR-032 |
| One zero-duty level later | Identity remains present; duty is outside Unit 1 | BR-030, BR-039 |
| Near-isothermal level | Fixed default/custom positive span, no span coordinate | BR-012, BR-032, BR-035 |
| One infeasible period bound | Empty-region error with period details | BR-023 through BR-028 |
| JSON round-trip | Structural equality plus named float tolerance | BR-020, BR-021, BR-038 through BR-040 |

## PBT-01 Rule-to-Property Traceability

| Rules | Property categories from the business-logic model |
|---|---|
| BR-001 through BR-006 | Request idempotence, request JSON round-trip, validation invariants |
| BR-007 through BR-016 | Generation/count/identity invariants, normalization idempotence |
| BR-017 through BR-021 | Unit-conversion oracle, finite/range invariants, signed-zero idempotence |
| BR-022 through BR-028 | Intersection commutativity/oracle and feasible-bound invariants |
| BR-029 through BR-037 | Vector round-trips, dimension/order invariants, easy start verification |
| BR-038 through BR-043 | Result JSON round-trip and error-context invariants |

PBT-01 is satisfied only when the companion business-logic property table and
domain-entity ownership table remain consistent with these rules. Example tests
remain mandatory for every scenario in the matrix.

## Requirement and Story Traceability

| Requirement/story | Owning rules or model section |
|---|---|
| FR-002, UPO-02 objective contract | BR-001, BR-015 |
| FR-003, UPO-01 counts | BR-002, BR-003, BR-007, BR-008 |
| FR-004, UPO-01 templates | BR-007 through BR-016 |
| FR-005 bounds and starts | BR-022 through BR-028, BR-036, BR-037 |
| FR-006 ordering/separation | BR-005, BR-029, BR-030 |
| FR-012, UPO-08 typed failures | BR-028, BR-041 through BR-043 |
| FR-014 result contract | BR-038 through BR-040 |
| UPO-09 detached serialization | BR-020, BR-021, BR-038 through BR-040 |
| UPO-12 TDD/PBT foundation | Scenario Matrix and PBT-01 traceability |
| NFR-001 numerical correctness | BR-005, BR-017 through BR-021, BR-025 through BR-028 |
| NFR-002 deterministic ordering | BR-009, BR-031, BR-032, BR-036 |
| NFR-004 compatibility | BR-017, BR-018, BR-038 through BR-041 |
| NFR-005 pure ownership | BR-022, BR-030, BR-038 |
| NFR-006 observability | BR-028, BR-041 through BR-043 |
