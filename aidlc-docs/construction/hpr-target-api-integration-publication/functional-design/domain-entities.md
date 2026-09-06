# Unit 3 Domain Entities

## Entity Overview

Unit 3 introduces only plain, immutable boundary values. Existing target and map
contracts retain their owners. No entity stores a live CoolProp or TESPy object.

| Entity | Purpose | Mutability | Owner |
|---|---|---|---|
| `HprSimulationBackend` | Closed normalized backend identity | Immutable literal | HPR contracts/analysis boundary |
| `HprTargetThermodynamicRequest` | One optimizer-candidate thermodynamic design request | Frozen | HPR analysis |
| `HprTargetThermodynamicResult` | Engine-neutral candidate result and profiles | Frozen | HPR analysis |
| `HprThermalProfilePoint` | One ordered temperature-enthalpy coordinate | Frozen | HPR analysis |
| `HprTargetSimulationRecord` | Winning target's detached simulation provenance and nominal facts | Frozen Pydantic value | HPR contract |
| `HprTargetMapBasis` | Unit 2 input derived from the winning record | Existing frozen dataclass | Performance-map analysis |
| `HprPerformanceMapCompatibilityError` | Typed unsupported-target failure | Immutable exception data | HPR application/analysis boundary |
| `HeatPumpTargetBase` | Existing public targeting result with additive backend field | Existing validated model | Domain targets |
| `HprPerformanceMapRequest` | Explicit grid and capacity request | Existing frozen Pydantic value | Public contract |
| `HprPerformanceMap` | Versioned downstream physical map | Existing frozen Pydantic value | Public contract |

## HprSimulationBackend

The closed domain is `coolprop` and `tespy`. Public string normalization occurs
once at the accessor boundary. All downstream values carry the normalized form.
The backend is independent of cycle identity and optimizer method.

## HprTargetThermodynamicRequest

This internal frozen value represents one nominal equipment candidate evaluated
inside HPR targeting.

| Field | Domain | Meaning |
|---|---|---|
| `mode` | `heat_pump` or `refrigeration` | Defines useful duty and COP convention |
| `cycle_id` | `single_stage_vapour_compression` | Stable supported topology identity |
| `model_id` | Nonempty string | OpenPinch topology/model revision |
| `working_fluid` | Resolved `HprWorkingFluidSpec` | Pure, registered blend, or explicit molar mixture |
| `evaporating_temperature` | Finite degrees Celsius | Refrigerant dew anchor |
| `condensing_temperature` | Finite degrees Celsius | Refrigerant bubble anchor |
| `useful_duty` | Finite positive kilowatts | Sink duty for heating or source duty for refrigeration |
| `source_approach_temperature` | Finite nonnegative kelvin difference | External source to evaporation boundary |
| `sink_approach_temperature` | Finite nonnegative kelvin difference | Condensation to external sink boundary |
| `compressor_isentropic_efficiency` | Greater than zero and at most one | Compressor model input |
| `superheat` | Finite nonnegative kelvin difference | Suction-side assumption |
| `subcooling` | Finite nonnegative kelvin difference | Liquid-side assumption |
| `internal_hx_gas_temperature_change` | Finite nonnegative kelvin difference | Internal-HX assumption |
| `candidate_id` | Stable ordinal/string | Diagnostic identity only, not physical input |

The request excludes load fractions because ordinary targeting evaluates a
nominal candidate. It excludes period weights, prices, and optimizer state; the
existing targeting pipeline owns those values.

## HprThermalProfilePoint

One profile point contains a finite temperature and a finite cumulative enthalpy
or duty coordinate. A profile contains at least two ordered points and carries
its hot/cold direction separately. Absolute origin is irrelevant; differences
must reproduce the associated exchanger duty. Profiles are detached and contain
no engine connection identity.

## HprTargetThermodynamicResult

This internal frozen result normalizes either backend before existing HPR
accounting.

| Field | Domain | Meaning |
|---|---|---|
| `backend` | `coolprop` or `tespy` | Actual evaluator used |
| `model_id` | Matches request | Stable topology/model identity |
| `converged` | Boolean | Whether all acceptance checks passed |
| `q_source` | Finite nonnegative kilowatts | Heat absorbed from source |
| `q_sink` | Finite nonnegative kilowatts | Heat rejected/delivered to sink |
| `compressor_power` | Finite positive kilowatts when converged | Compressor-only electrical input |
| `source_profile` | Ordered profile points | Evaporator-side thermal profile |
| `sink_profile` | Ordered profile points | Condenser-side thermal profile |
| `engine_version` | Nonempty string | Evaluator dependency version |
| `design_details` | JSON-compatible mapping | Convergence and physical assumptions |
| `failure` | Optional sanitized diagnostic | Present only for rejected candidate |

For a converged result, `q_sink` equals `q_source + compressor_power` within the
declared evaluation tolerance. Useful duty matches the request's mode-specific
duty. The result is sufficient to build existing OpenPinch HPR streams and
accounting without an engine model.

## HprTargetSimulationRecord

The winning record is a strict, frozen, extra-field-forbidden Pydantic value
owned with other HPR result contracts. It is not the external map schema.

| Field | Domain | Meaning |
|---|---|---|
| `simulation_backend` | Closed backend | Backend that determined target thermodynamics |
| `mode` | Heat pump or refrigeration | Target operating convention |
| `cycle_id` | `single_stage_vapour_compression` | Map-compatible cycle identity |
| `model_id` | Nonempty stable string | Topology/model revision |
| `refrigerant_spec` | Nonempty string | Exact selected OpenPinch fluid specification |
| `nominal_evaporating_temperature` | Finite degrees Celsius | Winning dew anchor |
| `nominal_condensing_temperature` | Finite degrees Celsius | Winning bubble anchor |
| `nominal_useful_duty` | Finite positive kilowatts | Winning useful duty |
| `source_approach_temperature` | Finite nonnegative delta | Source boundary assumption |
| `sink_approach_temperature` | Finite nonnegative delta | Sink boundary assumption |
| `compressor_isentropic_efficiency` | Greater than zero and at most one | Winning compressor assumption |
| `superheat` | Finite nonnegative delta | Winning superheat |
| `subcooling` | Finite nonnegative delta | Winning subcooling |
| `internal_hx_gas_temperature_change` | Finite nonnegative delta | Winning internal-HX assumption |
| `evaporator_count` | Exactly one for schema `1.0` compatibility | Topology evidence |
| `condenser_count` | Exactly one for schema `1.0` compatibility | Topology evidence |
| `period_id` | String or null | Scalar source-period identity |
| `engine_version` | Nonempty string | CoolProp or TESPy version |
| `power_boundary` | `compressor_only` | Electrical accounting boundary |
| `assumptions` | Nonempty JSON-compatible mapping | Structured model/convergence provenance |

The record permits future non-map-compatible targets to carry backend provenance,
but the initial builder creates a complete record only for supported single-stage
targets. Missing records remain explicit rather than being reconstructed from a
mutable configuration after the fact.

## Existing HeatPumpTargetInputs and HeatPumpTargetOutputs

`HeatPumpTargetInputs` gains a normalized `simulation_backend` field defaulting
to `coolprop`. It is transient numerical intent and may carry internal evaluator
injection only through a separate function boundary, never through serialized
output.

`HeatPumpTargetOutputs` gains:

- normalized `simulation_backend`, defaulting to `coolprop` for compatible
  legacy construction; and
- optional `target_simulation_record`, populated for successful supported
  vapour-compression results.

The output's existing numerical fields remain authoritative. Its existing
`model` field may retain the current CoolProp model on the default path for
compatibility, but a TESPy target stores no TESPy model there. TESPy-specific
details reside only in the plain record and existing normalized numerical
fields.

## Existing HeatPumpTargetBase

The public domain target gains `hpr_simulation_backend`, whose default is
`coolprop` for backward-compatible validation of existing fixtures. Its
`hpr_details` contains the normalized output and optional winning simulation
record. No independent mutable copy of the full record is added to the domain
target.

`DirectHeatPumpTarget`, `IndirectHeatPumpTarget`,
`DirectRefrigerationTarget`, and `IndirectRefrigerationTarget` inherit the field
without new subtypes.

## HprTargetMapBasis Relationship

The Unit 3 pure builder maps the target simulation record to the existing Unit 2
`HprTargetMapBasis` field for field. It adds a stable target identifier and
deep-copies structured provenance. Backend, mode, cycle, model, fluid,
temperatures, capacity, approaches, efficiency, superheat, subcooling, and
internal-HX assumptions must be equal across both values.

The mapping is one-way. Unit 2 never imports targets or reconstructs a public
target from a basis.

## HprPerformanceMapCompatibilityError

This typed `ValueError` subtype carries only a stable code and user-facing
message. Initial codes distinguish failed target, unsupported target type,
missing simulation record, multi-port topology, aggregate target, non-scalar
nominal data, and record inconsistency. It does not contain target or engine
objects.

## Relationships and Ownership

- The accessor creates normalized backend intent.
- Target preprocessing carries that intent to the selected thermodynamic
  evaluator.
- The evaluator turns a candidate request into a normalized candidate result.
- Existing targeting logic turns the winning result into
  `HeatPumpTargetOutputs` plus `HprTargetSimulationRecord`.
- Existing service translation turns the output into `HeatPumpTargetBase` with
  matching `hpr_simulation_backend`.
- The pure Unit 3 builder maps the target record to `HprTargetMapBasis`.
- Unit 2 combines that basis with `HprPerformanceMapRequest` and returns
  `HprPerformanceMap`.
- External consumers receive only `HprPerformanceMap` mappings or JSON.

Dependencies point from application to analysis/contracts/domain and from
analysis to contracts/domain. TESPy stays in one concrete optional analysis
leaf. No contracts or domain entity imports application, TESPy, OpenUtility,
Pyomo, or HiGHS.

## Testable Properties

PBT-01 applies to backend normalization, wrapper propagation, candidate-result
physics, candidate order independence, target-record JSON round trips,
target-to-basis determinism and idempotence, non-mutation, unsupported-context
rejection, and refrigerant-composition preservation. The complete property list
and categories are defined in `business-logic-model.md` and are mandatory inputs
to Code Generation planning.

No reusable mutable public entity is introduced, so stateful public-model PBT is
N/A. Session lifecycle remains an internal orchestration property with a fake
evaluator model and explicit success/failure sequences.

## Extension Compliance

- **PBT-01**: Compliant. Entity transformations, invariants, round trips,
  idempotence, oracle comparisons, and lifecycle verification are explicitly
  identified.
- **Security Baseline**: Disabled; not applicable.
- **Resiliency Baseline**: Disabled; not applicable.
