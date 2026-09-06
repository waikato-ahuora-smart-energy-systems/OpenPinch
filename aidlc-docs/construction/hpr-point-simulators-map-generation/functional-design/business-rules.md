# Unit 2 Business Rules

## Context and Support Rules

| Rule | Requirement |
|---|---|
| BR-U2-001 | Unit 2 accepts only an immutable target map basis and Unit 1 request; it does not accept an application accessor or mutable problem. |
| BR-U2-002 | Backend identity is exactly `coolprop` or `tespy`; matching is normalized before context construction and no fallback is permitted. |
| BR-U2-003 | The base-install default is `coolprop`; selecting `tespy` is always explicit. |
| BR-U2-004 | Mode is exactly `heat_pump` or `refrigeration` and must agree with the target basis. |
| BR-U2-005 | Only a successful single-stage, single-source, single-sink vapour-compression basis with one selected working-fluid specification is supported. |
| BR-U2-006 | Carnot, Brayton, MVR, cascade, parallel multi-port, failed, and array-valued target bases are rejected before simulation. |
| BR-U2-007 | Request reference capacity overrides the nominal target duty when supplied; otherwise positive scalar nominal useful duty is required. |
| BR-U2-008 | Resolved reference capacity is fixed for the complete map and uses `q_sink` for heat pumps or `q_source` for refrigeration. |
| BR-U2-009 | Basis and request objects are never mutated or augmented with engine state. |

## Working-Fluid Rules

| Rule | Requirement |
|---|---|
| BR-U2-061 | One selected single-loop refrigerant specification may identify a pure fluid, a property-provider registered blend, or an explicit mole-fraction mixture of any component count; Unit 2 has no refrigerant or mixture allowlist. |
| BR-U2-062 | OpenPinch's existing `BACKEND::component[fraction]&...` syntax is authoritative for explicit mixtures; an omitted backend means `HEOS`. |
| BR-U2-063 | A targeting refrigerant list denotes candidates or one fluid per stage and is never interpreted as the components of one mixture. |
| BR-U2-064 | Context construction resolves a frozen working-fluid value while preserving the original source specification unchanged. |
| BR-U2-065 | Explicit fractions are finite and nonnegative, have a positive total, and are deterministically normalized as mole fractions. |
| BR-U2-066 | A registered blend retains its provider identity and is not reconstructed silently from resolved components. |
| BR-U2-067 | Zeotropic evaporation uses the saturated-vapour/dew anchor and condensation uses the saturated-liquid/bubble anchor, matching the current CoolProp cycle. |
| BR-U2-068 | TESPy receives an explicit mixture through its equivalent `|molar` wrapper syntax as one closed-loop working fluid; composition is not converted to connection mass fractions. |
| BR-U2-069 | A working fluid unsupported by the explicitly selected engine fails during context/preparation with no backend fallback and no partial map. |
| BR-U2-070 | Provenance records source specification, property backend, fluid kind, registered identity when applicable, explicit components and normalized mole fractions when applicable, composition basis, and saturation-anchor convention. |
| BR-U2-071 | Unit 2 never installs estimated mixing rules or mutates process-global binary interaction parameters; missing property data fails explicitly. |
| BR-U2-072 | The TESPy adapter accepts the same pure-fluid, registered-blend, and explicit N-component mixture input categories as the CoolProp adapter and never rejects solely on fluid kind or component count. |
| BR-U2-073 | TESPy working-fluid support is decided by actual wrapper construction and required property-state evaluation; engine limitations produce typed diagnostics without fallback or an OpenPinch allowlist. |

## Temperature and Design Rules

| Rule | Requirement |
|---|---|
| BR-U2-010 | Map temperatures are external service temperatures in degrees Celsius. |
| BR-U2-011 | Evaporating temperature equals source service temperature minus source approach. |
| BR-U2-012 | Condensing temperature equals sink service temperature plus sink approach. |
| BR-U2-013 | Source and sink approaches are nonnegative finite temperature differences and are separately recorded. |
| BR-U2-014 | Every operating point must have positive absolute internal temperatures and condensing temperature greater than evaporating temperature after all declared assumptions. |
| BR-U2-015 | TESPy prepares one global design point using target-derived nominal temperatures and resolved full reference capacity. |
| BR-U2-016 | Every requested TESPy coordinate, including a coordinate equal to the nominal point, is evaluated through the declared offdesign path after preparation. |
| BR-U2-017 | The design point is not automatically inserted into the output grid. |

## Grid and Identity Rules

| Rule | Requirement |
|---|---|
| BR-U2-018 | Traversal order is ascending source temperature, ascending sink temperature, then ascending load fraction. |
| BR-U2-019 | The generator emits exactly the Cartesian-product cardinality when every point succeeds. |
| BR-U2-020 | Useful duty is `load_fraction * reference_capacity` for every operating point. |
| BR-U2-021 | Curve and point identifiers derive from the map identifier and canonical coordinate ordinals, not object identity or locale-sensitive float text. |
| BR-U2-022 | A coordinate is simulated at most once in one generation request. |
| BR-U2-023 | Zero load is never simulated; the Unit 1 active-load interval `(0, 1]` remains authoritative. |

## CoolProp Rules

| Rule | Requirement |
|---|---|
| BR-U2-024 | The CoolProp adapter delegates to the existing single `VapourCompressionCycle`. |
| BR-U2-025 | Heat-pump points prescribe requested condenser/useful duty; refrigeration points prescribe requested evaporator/useful duty. |
| BR-U2-026 | Existing refrigerant, compressor efficiency, approach, superheat, subcooling, and internal-HX assumptions are passed explicitly. |
| BR-U2-027 | Current cycle values in watts are converted to kilowatts exactly once at the adapter boundary. |
| BR-U2-028 | No cycling degradation, PLF modifier, minimum-load penalty, or invented empirical curve is applied. |
| BR-U2-029 | At fixed source/sink temperatures, steady-state duty and compressor power scale together and COP is load-invariant within numerical tolerance. |
| BR-U2-030 | The nominal CoolProp point must agree with a direct existing-cycle oracle before the adapter is accepted. |

## TESPy Rules

| Rule | Requirement |
|---|---|
| BR-U2-031 | TESPy imports occur only inside the concrete optional leaf. |
| BR-U2-032 | The topology is one refrigerant-only loop with cycle closer, compressor, condenser-side heat rejection, expansion valve, and evaporator-side heat uptake. |
| BR-U2-033 | No secondary-fluid loop, pump, fan, motor loss, heat loss, or electrical auxiliary is modeled in schema `1.0`. |
| BR-U2-034 | `electric_power` is positive compressor power only; provenance explicitly lists an empty modeled-auxiliaries set. |
| BR-U2-035 | OpenPinch supplies an exact versioned compressor characteristic and never relies on an unrecorded TESPy runtime default. |
| BR-U2-036 | Pump and secondary-HX characteristic curves are marked not applicable for this topology; fixed approaches define the external boundary. |
| BR-U2-037 | Design and offdesign convergence are checked explicitly; a solver return without converged finite results is a failure. |
| BR-U2-038 | Every offdesign solve restores point-varying state from the prepared design basis before applying the next coordinate. |
| BR-U2-039 | Temporary design-state storage is session-private and removed on close. |
| BR-U2-040 | TESPy networks, components, connections, characteristic objects, and paths never enter the map, target basis, or diagnostics. |

## Normalization and Physical Rules

| Rule | Requirement |
|---|---|
| BR-U2-041 | Adapter output uses finite nonnegative `q_source` and `q_sink` magnitudes and finite positive compressor power in kilowatts. |
| BR-U2-042 | The service checks `q_sink = q_source + electric_power` within `energy_balance_tolerance`. |
| BR-U2-043 | Mode-specific useful duty must equal the requested useful duty within `energy_balance_tolerance`. |
| BR-U2-044 | The service calculates COP itself as useful duty divided by electric power. |
| BR-U2-045 | `temperature_match_tolerance` is never reused for energy, COP, convergence, or solver stopping. |
| BR-U2-046 | A complete result is constructed through the Unit 1 map model; adapters cannot bypass its validation. |

## Lifecycle and Failure Rules

| Rule | Requirement |
|---|---|
| BR-U2-047 | A fresh simulator is prepared once and closed once for every generation attempt. |
| BR-U2-048 | Preparation failure prevents point calls but still executes cleanup. |
| BR-U2-049 | After successful preparation, every canonical coordinate receives a simulation call or a deterministic `session_unavailable` diagnostic. |
| BR-U2-050 | Point exceptions, non-convergence, invalid results, and internal-lift failures become ordered structured diagnostics. |
| BR-U2-051 | One failed point invalidates the complete map; no partial map or placeholder point is returned. |
| BR-U2-052 | Cleanup failure invalidates an otherwise complete result and appears after point diagnostics. |
| BR-U2-053 | `HprMapGenerationError` carries an immutable nonempty diagnostic tuple and may chain, but does not serialize, the original exception. |
| BR-U2-054 | Diagnostic fields contain only stable codes, backend/model identity, coordinate/ordinal values when applicable, and a bounded message. |
| BR-U2-055 | No diagnostic contains a traceback, engine object, temporary path, partial map, optimization candidate, or consumer data. |

## Provenance Rules

| Rule | Requirement |
|---|---|
| BR-U2-056 | Provenance is assembled after complete success and contains only recursive JSON values accepted by Unit 1. |
| BR-U2-057 | Provenance records OpenPinch and selected engine versions, model/cycle, refrigerant, design condition, resolved capacity, and all material thermodynamic assumptions. |
| BR-U2-058 | TESPy provenance records the OpenPinch characteristic-set identifier and exact numeric characteristic data or a content digest tied to packaged data. |
| BR-U2-059 | Provenance records compressor-only electricity and explicitly excludes auxiliaries, secondary loops, and hidden heat losses. |
| BR-U2-060 | Equal deterministic contexts and engine results produce structurally equal provenance; timestamps, random identifiers, memory addresses, and temporary paths are forbidden. |

## Error Categories

| Code | Meaning |
|---|---|
| `unsupported_backend` | Backend selector is outside the closed registry |
| `unsupported_target` | Target mode, topology, success state, or scalar basis is unsupported |
| `invalid_context` | Capacity, temperature, refrigerant, or modeling assumption is invalid |
| `unsupported_working_fluid` | Selected engine cannot represent the resolved pure fluid, registered blend, or explicit mixture |
| `dependency_unavailable` | TESPy was explicitly selected but cannot be imported |
| `prepare_failed` | Simulator design preparation failed or did not converge |
| `invalid_operating_point` | Translated temperatures or requested duty are invalid before solve |
| `point_exception` | Adapter raised while evaluating one coordinate |
| `non_converged` | Engine returned without a converged operating point |
| `invalid_simulation` | Duties or power are non-finite or outside their domains |
| `energy_balance` | Source, sink, and power do not close within tolerance |
| `useful_duty_mismatch` | Simulated useful duty differs from the requested duty |
| `session_unavailable` | A prior fatal adapter failure prevents a later solve |
| `cleanup_failed` | Simulator/session cleanup failed |

## Requirement Traceability

| Requirement | Rules |
|---|---|
| FR-2 operating-point calculation | BR-U2-010 through BR-U2-014, BR-U2-020, BR-U2-041 through BR-U2-046, BR-U2-061 through BR-U2-073 |
| FR-3 deterministic grid | BR-U2-018 through BR-U2-023 |
| FR-4 provenance population | BR-U2-056 through BR-U2-060 |
| FR-5 simulator registry support | BR-U2-002, BR-U2-003, BR-U2-024, BR-U2-031 |
| FR-6 TESPy generation | BR-U2-015 through BR-U2-017, BR-U2-031 through BR-U2-040, BR-U2-067 through BR-U2-069, BR-U2-072, BR-U2-073 |
| FR-7 failure policy | BR-U2-047 through BR-U2-055 |
| FR-8 context/generator bridge | BR-U2-001 through BR-U2-009 |
| FR-9 downstream independence | BR-U2-040, BR-U2-055, BR-U2-059 |
| FR-10 optional isolation | BR-U2-003, BR-U2-031, dependency error category |

All seventy-three rules have a production owner in the Unit 2 entity/lifecycle model or
an explicit Unit 3 extraction boundary. No frontend rule is applicable.
