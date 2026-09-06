# Unit 1 Business Logic Model

## Purpose and boundary

Unit 1 defines a strict, immutable, engine-neutral contract for physical HPR
performance maps. It validates and serializes data; it does not calculate a
thermodynamic cycle, invoke CoolProp or TESPy, mutate a targeting result, or
construct optimization candidates and constraints.

## Contract construction flow

1. Accept a map-generation request or a plain map payload.
2. Reject unknown fields and unsupported schema versions before coercing
   domain values.
3. Validate scalar types, finite numeric values, nonempty identifiers, enums,
   exact unit declarations, and JSON-compatible provenance.
4. Validate each performance point independently.
5. Group points by `curve_id` and validate fixed-temperature curve structure,
   coordinate uniqueness, and strictly ascending load fractions.
6. Validate mode-dependent reference capacity, useful duty, COP, and the
   thermal/electricity balance across every point.
7. Construct one immutable map only after all checks pass. No partial map is
   observable.
8. Serialize the validated map to canonical JSON-compatible data, JSON text,
   JSON Schema, or checked-in golden fixture resources.

Textual flow: raw request or payload enters strict structural validation, then
point validation, curve validation, map-level physical validation, immutable
construction, and deterministic serialization. Any failure exits before the
immutable map is returned.

## Request normalization

`HprPerformanceMapRequest` describes a requested Cartesian operating grid. It
contains `map_id`, source temperatures, sink temperatures, load fractions, and
an optional reference capacity. Unit 1 applies these transformations:

- reject empty coordinate collections, duplicate values, non-finite values,
  and load fractions outside `(0, 1]`;
- sort each coordinate tuple in ascending numeric order so equivalent sets
  produce identical traversal input;
- require a positive finite reference capacity when supplied;
- preserve the caller's nonempty `map_id` exactly after validation.

Unit 2 consumes this canonical request. Unit 1 does not infer temperatures or
capacity from an HPR target.

## Map validation model

The complete producer-facing map field inventory is `schema_version`,
`map_id`, `mode`, `units`, `reference_capacity`,
`reference_capacity_basis`, `interpolation_topology`,
`thermodynamic_backend`, `model_id`, `provenance`, `points`,
`cop_convention`, `energy_balance_tolerance`, and
`temperature_match_tolerance`. Unit 1 treats this set as closed for schema
`1.0`.

### Structural validation

The accepted schema is exactly `1.0`. Unknown fields fail closed at map, units,
and point levels. Required fields cannot be silently defaulted except the two
documented tolerances, whose schema defaults are `1e-6`.

The `units` object contains exactly:

| Field | Required value |
|---|---|
| `source_temperature` | `degC` |
| `sink_temperature` | `degC` |
| `q_source` | `kW` |
| `q_sink` | `kW` |
| `electric_power` | `kW` |

The map contains at least one point. Point names are unique across the map.
Coordinate tuples `(curve_id, source_temperature, sink_temperature,
load_fraction)` are unique.

### Curve validation

Every `curve_id` identifies one fixed source/sink temperature pair. Its points
remain in strictly ascending `load_fraction` order. A curve may contain one or
more points; checked-in golden fixtures contain at least three so downstream
adjacent-segment behavior is testable. Unit 1 declares
`interpolation_topology="ordered_part_load_curve"` but does not implement an
interpolation formulation.

Maps containing multiple temperature pairs represent independent ordered
curves. Global point order is deterministic by source temperature, sink
temperature, load fraction, curve identifier, and point name. No contract rule
authorizes interpolation across curves.

### Physical validation

All duties and electric power are nonnegative finite magnitudes. Active points
have positive electric power. Every point satisfies, within
`energy_balance_tolerance`:

- `q_sink = q_source + electric_power`;
- useful duty equals `load_fraction * reference_capacity`; and
- declared COP equals useful duty divided by electric power.

Mode fixes the useful-duty rules:

| Mode | `reference_capacity_basis` | `cop_convention` | Useful duty |
|---|---|---|---|
| `heat_pump` | `q_sink` | `heating` | `q_sink` |
| `refrigeration` | `q_source` | `cooling` | `q_source` |

`energy_balance_tolerance` is a finite, nonnegative consistency tolerance used
for contract validation. `temperature_match_tolerance` is separately finite and
nonnegative and is carried for a consumer matching canonical `degC` node
coordinates. Temperature tolerance is never used to relax energy, capacity, or
COP checks.

## Structured provenance validation

`provenance` is a nonempty JSON object. Keys are strings. Values may be null,
booleans, strings, finite numbers, lists of JSON values, or nested string-keyed
objects. Tuples and immutable mappings may be normalized to their JSON list and
object forms during serialization. NaN, infinity, bytes, arbitrary Python
instances, engine objects, and non-string mapping keys fail validation.

Required provenance content is enforced when a generated map is assembled in
Unit 2. Unit 1 guarantees only safe JSON structure and non-emptiness so it can
also validate maps from another conforming producer.

## Deterministic serialization and resources

- Mapping serialization emits every schema field with stable field names and
  JSON-compatible values.
- JSON text uses one documented formatting policy and stable key/point order so
  identical maps produce identical bytes.
- Deserializing serialized map JSON reconstructs an equal immutable value.
- JSON Schema is generated from the same authoritative contract and committed
  as package data.
- Golden fixtures are validated map serializations, never independently edited
  examples. One fixture is a heat pump and one is refrigeration; each contains
  an ordered curve with at least three breakpoints.
- Fixture and schema resources are consumable with a standard JSON parser and
  no OpenPinch import.

## Error model

Contract failures use one public HPR performance-map validation category with
structured locations and messages from the underlying field/cross-field checks.
Unsupported schema version, invalid structure/type, physical inconsistency, and
invalid JSON provenance remain distinguishable by error location and stable
message fragments. Unit 1 never catches an invalid value and substitutes a
different mode, unit, version, convention, point, or tolerance.

## Testable properties (PBT-01)

| Property | Category | Owner and assertion |
|---|---|---|
| Valid map to mapping/JSON and back yields equality | Round-trip | Unit 1 generated heat-pump and refrigeration maps |
| Canonical request sorting preserves each input set and removes order dependence | Invariant | Unit 1 constrained request strategies |
| Map serialization preserves point count, coordinates, duties, mode, and provenance | Invariant | Unit 1 generated valid maps |
| Every accepted point satisfies energy, capacity, COP, range, and finiteness rules | Easy verification | Unit 1 domain-specific map strategies |
| Every accepted curve is grouped at one temperature pair with strictly increasing unique loads | Invariant | Unit 1 multi-curve strategies |
| Serialized values are accepted by a standard JSON parser and contain no runtime objects | Easy verification | Unit 1 nested JSON-value strategies |
| Schema validation and direct model validation accept/reject the same generated payload class | Oracle | Unit 1 JSON Schema comparison where representable |
| Reordering equivalent request coordinate inputs yields the same canonical request | Commutativity | Unit 1 permutation strategies |

PBT-04 is N/A because Unit 1 exposes no mutating idempotent operation. PBT-06 is
N/A because all contract values are immutable and no stateful command sequence
exists. Golden fixtures and explicit invalid payloads complement these
properties under PBT-10.

## Traceability

This model implements Unit 1 ownership of FR-1 through FR-4 and FR-9, supports
FR-7 typed failures, and supplies the plain contract required by FR-5, FR-6,
FR-8, and FR-10. It covers acceptance criteria 1, 2, 4, 6, and 9 through 13;
engine and public-API acceptance criteria remain with Units 2 and 3.
