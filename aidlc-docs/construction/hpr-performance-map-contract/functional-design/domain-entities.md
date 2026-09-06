# Unit 1 Domain Entities

## Entity overview

Unit 1 contains immutable value objects and published resources. It has no
mutable aggregate, database entity, engine session, optimizer object, or UI
state.

| Entity | Role | Owned by Unit 1 |
|---|---|---|
| `HprPerformanceMapRequest` | Canonical requested operating grid and optional useful capacity | Yes |
| `HprPerformanceMapUnits` | Closed schema `1.0` unit declarations | Yes |
| `HprPerformancePoint` | One physical active part-load point | Yes |
| `HprPerformanceMap` | Complete validated collection of independent ordered curves | Yes |
| `JsonValue` | Recursive JSON-compatible provenance value domain | Yes |
| Contract validation errors | Fail-closed structural and semantic diagnostics | Yes |
| JSON Schema resource | Machine-readable structural interchange contract | Yes |
| Heat-pump golden fixture | Canonical heating-mode interchange example | Yes |
| Refrigeration golden fixture | Canonical cooling-mode interchange example | Yes |
| HPR target/result/runtime object | Targeting and simulation context | No; existing owners/Units 2 and 3 |
| OpenUtility candidate or MILP model | Investment and dispatch decision model | No; external consumer |

## `HprPerformanceMapRequest`

| Field | Domain | Required | Meaning |
|---|---|---|---|
| `map_id` | Nonempty string | Yes | Stable requested map identity |
| `source_temperatures` | Nonempty ordered tuple of unique finite numbers | Yes | External source-service coordinates in `degC` |
| `sink_temperatures` | Nonempty ordered tuple of unique finite numbers | Yes | External sink-service coordinates in `degC` |
| `load_fractions` | Nonempty ordered tuple of unique numbers in `(0, 1]` | Yes | Active requested part-load states |
| `reference_capacity` | Positive finite number or null | No | Explicit useful capacity in `kW`, otherwise derived by Unit 2 |

The request is canonicalized before Unit 2 consumes it. It carries no engine,
cycle topology, refrigerant, target object, output path, or optimizer choice.

## `HprPerformanceMapUnits`

| Field | Schema `1.0` value |
|---|---|
| `source_temperature` | `degC` |
| `sink_temperature` | `degC` |
| `q_source` | `kW` |
| `q_sink` | `kW` |
| `electric_power` | `kW` |

This closed value object makes units explicit without attaching a unit-registry
object to serialized data.

## `HprPerformancePoint`

| Field | Domain | Meaning |
|---|---|---|
| `name` | Nonempty string, unique per map | Stable point identity |
| `curve_id` | Nonempty string | Membership in one fixed-temperature curve |
| `source_temperature` | Finite number | External source-service temperature in `degC` |
| `sink_temperature` | Finite number | External sink-service temperature in `degC` |
| `load_fraction` | Number in `(0, 1]` | Useful duty divided by reference capacity |
| `q_source` | Nonnegative finite number | Source-side heat magnitude in `kW` |
| `q_sink` | Nonnegative finite number | Sink-side heat magnitude in `kW` |
| `electric_power` | Positive finite number | Total declared external electrical input in `kW` |
| `cop` | Positive finite number | Heating or cooling COP selected by map mode |

A point does not contain an OpenUtility candidate, period, node, cost, selected
state, installed size, or interpolation weight.

## `HprPerformanceMap`

| Field | Domain | Meaning |
|---|---|---|
| `schema_version` | Literal `1.0` | Closed alpha interchange version |
| `map_id` | Nonempty string | Map identity used by downstream references |
| `mode` | `heat_pump` or `refrigeration` | Physical useful-service mode |
| `units` | `HprPerformanceMapUnits` | Exact canonical unit basis |
| `reference_capacity` | Positive finite number | Full-load useful duty in `kW` |
| `reference_capacity_basis` | `q_sink` or `q_source` as fixed by mode | Field defining useful duty |
| `interpolation_topology` | Literal `ordered_part_load_curve` | Adjacent-only curve semantics |
| `thermodynamic_backend` | Nonempty string | Producer-declared property/simulation backend |
| `model_id` | Nonempty string | Producer calculation/model identity |
| `provenance` | Nonempty `dict[str, JsonValue]` | Structured reproducibility metadata |
| `points` | Nonempty ordered tuple of points | Complete validated physical data |
| `cop_convention` | `heating` or `cooling` as fixed by mode | Interpretation of point COP |
| `energy_balance_tolerance` | Finite nonnegative number, default `1e-6` | Energy/capacity/COP validation tolerance |
| `temperature_match_tolerance` | Finite nonnegative number, default `1e-6` | Consumer coordinate-match tolerance in `degC` |

`thermodynamic_backend` remains open to nonempty producer identifiers at the
transport boundary. OpenPinch Units 2 and 3 initially emit normalized
`coolprop` or `tespy`; the contract does not prevent another conforming producer
from identifying its backend.

## `JsonValue`

The recursive provenance domain is:

- null;
- boolean;
- string;
- finite JSON number;
- ordered list of JSON values; or
- string-keyed object of JSON values.

Boolean values remain distinct from numbers. Nested values have no prescribed
depth in the domain, although ordinary resource and validation limits may be
defined later as non-functional safeguards. Unsupported runtime values are not
stringified.

## Relationships

- One request supplies the canonical coordinate sets later traversed by Unit 2.
- One map owns exactly one units object and one nonempty ordered point tuple.
- One point belongs to exactly one map and one `curve_id` within that map.
- One curve groups one or more points at exactly one source/sink temperature
  pair.
- One map may contain multiple independent curves at different temperature
  pairs.
- Every point's useful duty, capacity basis, and COP interpretation derive from
  its parent map mode.
- JSON Schema describes the serialized map shape; golden fixtures are concrete
  valid serialized maps.
- Unit 2 produces maps; Unit 3 exposes them; an external consumer reads only
  their serialized values.

## Lifecycle and state

1. Raw request or raw map data is untrusted and has no domain identity.
2. Successful validation constructs immutable canonical values.
3. Serialization creates detached JSON-compatible data; modifying that data
   cannot mutate the contract value.
4. JSON text, schema, and fixtures are passive resources with no session state.
5. Deserialization creates a new equal immutable value after full validation.
6. A validation failure creates no partially valid map and changes no existing
   value or resource.

## Error taxonomy

| Category | Examples | Required behavior |
|---|---|---|
| Structure/type | Missing/extra fields, wrong sequence/object type | Reject at precise location |
| Version/unit | Unknown version, missing/aliased units | Reject without conversion or fallback |
| Range/finiteness | NaN, infinity, negative duty, invalid load/tolerance | Reject offending field |
| Mode convention | Heating/cooling COP or capacity-basis mismatch | Reject map-level combination |
| Physical point | Energy, useful capacity, or COP residual too large | Reject point and complete map |
| Curve topology | Duplicate coordinate, mixed temperatures, unordered load | Reject curve and complete map |
| Provenance | Empty object, non-string key, unsupported nested value | Reject without string coercion |
| Resource drift | Fixture/schema bytes differ from authoritative generation | Fail the verification gate |

## Ownership constraints

- Unit 1 may depend on existing contract/JSON facilities but not HPR simulation,
  application accessors, TESPy, OpenUtility, Pyomo, or HiGHS.
- Unit 2 may import these Unit 1 entities and return a validated map.
- Unit 3 may expose the request and map but must not duplicate or weaken Unit 1
  validation.
- External consumers may reproduce the schema in their own types; type identity
  across packages is neither required nor allowed as a runtime dependency.
