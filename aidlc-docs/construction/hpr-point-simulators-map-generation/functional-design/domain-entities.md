# Unit 2 Domain Entities

## Entity Summary

| Entity | Role | Mutability | Visibility |
|---|---|---|---|
| `HprTargetMapBasis` | Engine-neutral target/configuration facts supplied by Unit 3 | Frozen | Internal analysis boundary |
| `HprWorkingFluidSpec` | Resolved pure, registered-blend, or explicit-mixture identity | Frozen | Internal analysis value |
| `HprMapGenerationContext` | Fully resolved valid generation inputs | Frozen | Internal analysis value |
| `HprOperatingPoint` | One canonical requested engine condition | Frozen | Internal analysis value |
| `HprPointSimulation` | Normalized converged engine result | Frozen | Internal analysis value |
| `HprSimulatorMetadata` | Backend preparation/model facts for provenance | Frozen | Internal analysis value |
| `HprSimulationDiagnostic` | Stable failure evidence | Frozen | Internal error value |
| `HprMapGenerationError` | Atomic generation failure containing diagnostics | Immutable exception state | Specialist analysis exception |
| `HprPointSimulator` | Internal prepare/simulate/close protocol | Session state | Internal protocol only |
| `HprPointSimulatorFactory` | Closed backend-to-session constructor | Stateless | Internal injectable seam |
| `HprPerformanceMapRequest` | Canonical grid and optional capacity | Frozen | Unit 1 contract input |
| `HprPerformanceMap` | Complete plain-data output | Frozen | Unit 1 contract output |

## HprTargetMapBasis

| Field | Type/constraint | Meaning |
|---|---|---|
| `target_id` | nonempty string | Stable source-target identity for diagnostics/provenance |
| `mode` | `heat_pump` or `refrigeration` | Useful-duty and COP convention |
| `simulation_backend` | `coolprop` or `tespy` | Exact simulator selection recorded by targeting |
| `cycle_id` | supported single-stage vapour-compression identifier | Cycle family, independent of optimizer backend |
| `model_id` | nonempty string | Exact physical model identity |
| `refrigerant_spec` | nonempty scalar string | One selected pure-fluid, registered-blend, or explicit mole-fraction mixture specification |
| `nominal_evaporating_temperature` | finite degrees Celsius | Refrigerant-side design temperature |
| `nominal_condensing_temperature` | finite degrees Celsius | Refrigerant-side design temperature |
| `nominal_useful_duty` | finite positive kilowatts | Default design capacity when request omits capacity |
| `source_approach_temperature` | finite nonnegative kelvin difference | External source to evaporating-temperature offset |
| `sink_approach_temperature` | finite nonnegative kelvin difference | Condensing to external-sink-temperature offset |
| `compressor_isentropic_efficiency` | finite `(0, 1]` | Design compressor efficiency |
| `superheat` | finite nonnegative kelvin difference | Compressor-inlet assumption |
| `subcooling` | finite nonnegative kelvin difference | Condenser-outlet assumption |
| `internal_hx_gas_temperature_change` | finite nonnegative kelvin difference | Existing cycle IHX assumption; zero means absent |
| `source_provenance` | nonempty recursive JSON mapping | Stable target/configuration facts only |

Unit 3 constructs this value from the current target plus its owning prepared
problem configuration. Unit 2 validates it but never looks back into the
application object graph.

## HprWorkingFluidSpec

| Field | Type/constraint | Meaning |
|---|---|---|
| `source_spec` | nonempty string | Exact selected OpenPinch/CoolProp input retained for compatibility and audit |
| `property_backend` | nonempty normalized string | Explicit provider backend or `HEOS` default |
| `kind` | `pure`, `registered_blend`, or `explicit_molar_mixture` | Closed interpretation of the specification |
| `registered_name` | string or null | Pure/registered provider token retained rather than reconstructed |
| `components` | tuple of distinct nonempty strings | One name for a pure fluid, explicit names for a custom mixture, or empty for an opaque registered blend |
| `mole_fractions` | same-length finite nonnegative tuple summing to one, or empty | `(1.0,)` for pure, normalized explicit molar composition, or empty for an opaque registered blend |
| `composition_basis` | `not_applicable`, `provider_defined`, or `molar` | Distinguishes pure identity, registered provider data, and explicit input |

Resolution may use a short-lived CoolProp state to validate provider support and
inspect explicit composition, but no state object is retained. A registered
blend keeps its opaque registered token in both adapters and does not require
component expansion. The TESPy adapter derives its pure/blend token or explicit
mixture token ending in `|molar` from this value. This entity represents one
loop fluid; it is not a cascade-stage list or TESPy connection-level secondary
mixture. Component tuples have no design-level maximum length; actual support
is decided by constructing the selected adapter's property wrapper and
evaluating the required operating states. Neither adapter may reject a working
fluid solely because it is a registered or explicit mixture.

## HprMapGenerationContext

| Field | Type/constraint | Meaning |
|---|---|---|
| `basis` | valid `HprTargetMapBasis` | Immutable thermodynamic basis |
| `working_fluid` | valid `HprWorkingFluidSpec` | Resolved single-loop fluid shared by the selected adapter and provenance |
| `request` | valid Unit 1 request | Canonical output grid |
| `reference_capacity` | finite positive kilowatts | Resolved fixed useful capacity |
| `reference_capacity_basis` | `q_sink` or `q_source` | Derived from mode |
| `cop_convention` | `heating` or `cooling` | Derived from mode |
| `nominal_source_temperature` | finite degrees Celsius | External source design coordinate |
| `nominal_sink_temperature` | finite degrees Celsius | External sink design coordinate |
| `characteristic_set_id` | fixed versioned string | `openpinch-single-stage-compressor-v1` for TESPy; steady-state identity for CoolProp |
| `energy_balance_tolerance` | finite nonnegative kilowatt tolerance | Copied into Unit 1 output and physical checks |
| `temperature_match_tolerance` | finite nonnegative degree-Celsius tolerance | Copied into Unit 1 output only |

Derived fields are created once. The context contains no engine session,
temporary directory, timestamp, or consumer option.

## HprOperatingPoint

| Field | Type/constraint | Meaning |
|---|---|---|
| `ordinal` | zero-based nonnegative integer | Global canonical grid position |
| `source_index` | zero-based nonnegative integer | Source-coordinate position |
| `sink_index` | zero-based nonnegative integer | Sink-coordinate position |
| `load_index` | zero-based nonnegative integer | Load-coordinate position |
| `curve_id` | deterministic nonempty string | Shared identity for one temperature pair |
| `name` | deterministic nonempty string | Unique point identity |
| `source_temperature` | finite degrees Celsius | External source coordinate |
| `sink_temperature` | finite degrees Celsius | External sink coordinate |
| `evaporating_temperature` | finite degrees Celsius | Engine input after approach translation |
| `condensing_temperature` | finite degrees Celsius | Engine input after approach translation |
| `load_fraction` | `(0, 1]` | Active part load |
| `requested_useful_duty` | finite positive kilowatts | Load fraction times fixed capacity |

The point has no mode-specific duplicate duty field. Mode lives in the context.

## HprPointSimulation

| Field | Type/constraint | Meaning |
|---|---|---|
| `q_source` | finite nonnegative kilowatts | Heat absorbed from source boundary |
| `q_sink` | finite nonnegative kilowatts | Heat delivered/rejected to sink boundary |
| `compressor_power` | finite positive kilowatts | Entire schema `electric_power` value |
| `converged` | boolean | Explicit engine success signal |
| `engine_details` | recursive JSON mapping | Bounded point-level convergence facts for local diagnostics only |

COP is intentionally absent. The generation service derives it. Engine details
are not copied wholesale into successful map points; stable material model facts
belong in map-level simulator metadata.

## HprSimulatorMetadata

| Field | Type/constraint | Meaning |
|---|---|---|
| `backend` | exact selected backend | Must match context |
| `engine_version` | nonempty string | Installed engine version |
| `model_id` | nonempty string | Must match declared physical model |
| `characteristic_set_id` | nonempty string | Versioned assumption identity |
| `design_converged` | true | Preparation cannot succeed otherwise |
| `design_details` | recursive JSON mapping | Nominal conditions, convergence limits, and stable model facts |
| `power_boundary` | `compressor_only` | First-release electric boundary |
| `modeled_auxiliaries` | empty tuple | Explicit first-release exclusion |

## HprSimulationDiagnostic

| Field | Type/constraint | Meaning |
|---|---|---|
| `code` | closed error code | Stable failure category |
| `backend` | normalized backend or null before resolution | Responsible simulator boundary |
| `model_id` | string or null | Physical model identity when known |
| `point_ordinal` | integer or null | Canonical failed coordinate position |
| `curve_id` | string or null | Curve identity when applicable |
| `source_temperature` | finite number or null | External failed coordinate |
| `sink_temperature` | finite number or null | External failed coordinate |
| `load_fraction` | finite number or null | Failed active load |
| `message` | nonempty bounded string | Actionable engine-neutral explanation |
| `details` | recursive JSON mapping | Stable scalar convergence/context facts |

Diagnostics compare structurally in tests. Full exception strings, stack traces,
engine representations, and filesystem locations are not stable details.

## HprMapGenerationError

The exception contains:

- a nonempty immutable tuple of `HprSimulationDiagnostic` values;
- a stable summary stating the backend and failure count; and
- an optional chained Python cause for local debugging only.

It contains no `HprPerformanceMap` because failure is atomic. Callers may inspect
diagnostics by attribute but do not receive a partially successful point list.

## HprPointSimulator Protocol

| Operation | Precondition | Result/state transition |
|---|---|---|
| `prepare(context)` | Fresh session | Returns `HprSimulatorMetadata`; session becomes prepared |
| `simulate(point)` | Prepared and not closed | Returns one normalized simulation or raises/reports one point failure |
| `close()` | Fresh, prepared, or failed; not already closed | Releases all resources; session becomes closed |

The protocol exposes no raw engine property. The generation service is its only
production coordinator. A test fake records lifecycle events and allows
deterministic failures by point ordinal.

## Relationships and Ownership

- Unit 3 target/configuration state produces `HprTargetMapBasis`.
- Unit 2 resolves the basis refrigerant specification into one
  `HprWorkingFluidSpec` without retaining a property-engine object.
- The Unit 1 request plus basis and working fluid produces one
  `HprMapGenerationContext`.
- One context expands to ordered `HprOperatingPoint` values.
- The closed factory selects one simulator session from context backend.
- One prepared simulator metadata value plus successful point simulations
  produces one Unit 1 performance map.
- Any preparation, point, normalization, or cleanup failure produces diagnostics
  inside one `HprMapGenerationError` instead of a map.

The dependency direction is Unit 1 contracts into Unit 2 analysis. CoolProp and
TESPy enter only their concrete adapters. No Unit 2 entity depends on Unit 3,
OpenUtility, Pyomo, or HiGHS.

## PBT Entity Coverage

- Generated bases, contexts, operating grids, normalized simulations, and
  diagnostics use reusable constrained strategies.
- JSON round trips apply to complete generated Unit 1 maps, not internal engine
  sessions.
- Cartesian size/order, identifier uniqueness, physical balance, mode-specific
  duty/COP, and failure atomicity are invariant properties.
- The existing CoolProp cycle is the independent oracle for nominal generated
  points.
- A generated fake session event model covers prepare/simulate/close state.
- Explicit heat-pump, refrigeration, missing-TESPy, non-convergence, and cleanup
  examples remain mandatory alongside properties.
- Pure-fluid, registered zeotropic-blend, valid explicit binary- and ternary-
  mixture, malformed-mixture, unsupported-engine, and dew/bubble-anchor
  examples remain mandatory. Generated explicit-mixture strategies cover
  arbitrary bounded component vectors with finite nonnegative fractions,
  positive totals, and deterministic normalization.
- TESPy examples verify successful registered-blend and explicit binary-mixture
  wrapper construction. A deterministic wrapper/state failure verifies typed
  diagnostics, no categorical mixture rejection, and no CoolProp fallback.

PBT-01 is compliant. No Functional Design PBT finding is blocking.
