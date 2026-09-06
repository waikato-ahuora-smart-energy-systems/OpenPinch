# Unit 2 Business Logic Model

## Purpose and Boundary

Unit 2 converts one validated HPR generation context into one complete Unit 1
`HprPerformanceMap`. It owns internal point simulation and deterministic grid
orchestration. CoolProp is the default simulator; TESPy is an explicitly
selected optional simulator.

This unit does not add or alter public target methods. Unit 3 will extract a
plain target basis from the current vapour-compression heat-pump or
refrigeration result and call Unit 2. This refines the conceptual Application
Design signature: the numerical analysis layer receives an immutable
`HprTargetMapBasis`, not an application accessor or mutable problem.

Unit 2 does not own targeting optimization, OpenUtility candidates, capacity
investment, periods, dispatch, interpolation constraints, Pyomo, or HiGHS.

## Approved Functional Decisions

| Topic | Decision |
|---|---|
| CoolProp part load | Preserve current steady-state cycle physics; useful duty scales mass flow and COP remains constant across load fractions at fixed temperatures |
| TESPy topology | One refrigerant-only single-stage compressor/condenser/expansion/evaporator loop |
| TESPy design basis | One target-derived global design point; every requested grid coordinate is an offdesign condition for the same equipment |
| Characteristics | OpenPinch owns one explicit versioned characteristic set and records it in provenance |
| Electrical boundary | Compressor power only; no pumps or electrical auxiliaries are modeled |
| Point failures | Attempt the complete canonical grid, collect ordered diagnostics, clean up once, and return no partial map |
| Working fluid | Preserve current single-loop CoolProp support for pure fluids, registered blends, and explicit mole-fraction mixtures; translate the same resolved specification to TESPy |

The selected refrigerant-only topology has no secondary-fluid pumps and no
two-sided heat-exchanger conductance model. Consequently, pump and secondary
heat-exchanger characteristic curves are not applicable in the first release.
The owned characteristic set contains the compressor part-load efficiency line;
fixed source and sink approach temperatures define the external-to-refrigerant
boundary. Provenance states these exclusions explicitly.

## Inputs

### Target map basis

Unit 3 will derive one `HprTargetMapBasis` from a successful current HPR target
and its prepared configuration. The basis contains only immutable scalar and
JSON-compatible metadata:

- operating mode and normalized backend;
- supported single-stage vapour-compression cycle identity;
- one refrigerant specification and model identity;
- nominal evaporating and condensing temperatures;
- nominal useful duty;
- source and sink approach temperatures;
- compressor isentropic efficiency;
- superheat, subcooling, and internal-heat-exchanger assumptions; and
- target/provenance identifiers required to explain the design point.

No stream collection, NumPy array, CoolProp state, TESPy object, problem,
accessor, optimizer result, or mutable target enters the basis.

### Working-fluid normalization

One target contributes one working-fluid specification to the single-loop map.
That specification may be:

- a pure fluid, such as `R134a`;
- a property-provider registered blend, such as `R407C`; or
- the explicit mole-fraction syntax already accepted by OpenPinch, such as
  `HEOS::R32[0.5]&R125[0.5]`.

Explicit mixtures may contain any positive number of components; Unit 2 has no
refrigerant-name, pair, or component-count allowlist. “Supported” means the
installed property backend can construct the composition and evaluate every
state required by the existing vapour-compression cycle at the requested
conditions. Missing binary interaction data, unsupported flash behavior, and
out-of-domain states therefore fail validation or simulation rather than being
silently approximated.

A list of refrigerant strings in the HPR targeting configuration represents
candidate fluids or one fluid per cycle stage. It is not itself a mixture. Unit
3 extracts the one specification used by the supported successful single-loop
target. Cascade and parallel multi-loop maps remain excluded.

Context construction resolves the string through the existing OpenPinch
CoolProp fluid parser into a frozen `HprWorkingFluidSpec`. It preserves the
provider backend and registered-blend identity. Pure fluids and explicit
mixtures record component names and normalized mole fractions; provider-managed
registered blends remain opaque identities. Explicit fractions must be finite
and nonnegative with a positive total; normalization does not mutate the source
string. A registered blend is not silently reconstructed because that could
change provider-specific mixture data.

Unit 2 does not mutate CoolProp's process-global interaction-parameter library
or install an estimated mixing rule. A future caller-supplied property-model
extension would require a separate versioned input and provenance contract.

The two adapters use the same physical composition:

- CoolProp receives the existing OpenPinch specification unchanged.
- TESPy receives a pure or registered blend as one fluid token. An explicit
  mixture is translated to TESPy's equivalent
  `BACKEND::component[fraction]&...|molar` wrapper syntax and is still treated
  as one closed-loop working fluid, not as a secondary-fluid connection mix.

The TESPy adapter accepts the same three specification categories and imposes
no mixture-specific allowlist or component-count limit. It must attempt wrapper
construction and the required dew/bubble property states; it may not reject an
input merely because `kind` is `registered_blend` or
`explicit_molar_mixture`. A failure originating in the installed TESPy property
wrapper or selected CoolProp/REFPROP backend is returned as a typed
`unsupported_working_fluid` or `prepare_failed` diagnostic, not relabeled as an
OpenPinch policy restriction and not sent through an automatic CoolProp
fallback.

For zeotropic fluids, evaporation is anchored at saturated vapour/dew
temperature (`Q = 1`) and condensation at saturated liquid/bubble temperature
(`Q = 0`), matching the current `VapourCompressionCycle`. Fixed external
approaches apply to those anchors. This scalar-boundary model preserves the
current targeting semantics but does not claim to model a complete secondary
heat-exchanger temperature profile or enforce its minimum approach throughout
the refrigerant glide.

### Map request

The Unit 1 `HprPerformanceMapRequest` supplies the map identifier, canonical
source-temperature coordinates, canonical sink-temperature coordinates,
canonical active load fractions, and optional reference capacity.

If request capacity is present, it is the deliberate equipment design capacity.
Otherwise the scalar nominal useful duty from the target basis is used. The
resolved capacity is fixed for the entire map.

## Context Construction

`build_hpr_map_generation_context(basis, request)` performs these steps without
simulation:

1. Require `coolprop` or `tespy` as the normalized backend.
2. Require heat-pump or refrigeration mode and a successful supported
   single-stage vapour-compression basis.
3. Resolve and validate the selected target's one working-fluid specification,
   including explicit mixture composition and backend support.
4. Reject analytic Carnot, Brayton, MVR, cascade, parallel multi-port, failed,
   array-valued, and ambiguous targets.
5. Resolve reference capacity from the request or nominal target useful duty.
6. Convert the nominal refrigerant temperatures to nominal external service
   temperatures:
   - nominal source temperature is evaporating temperature plus source approach;
   - nominal sink temperature is condensing temperature minus sink approach.
7. Require positive absolute temperatures and positive internal lift after
   applying approach, superheat, subcooling, and internal-HX assumptions.
8. Attach the Unit 1 energy-balance and temperature-match tolerances without
   reusing either for engine convergence.
9. Return a frozen `HprMapGenerationContext`; do not mutate the basis or request.

The design condition may lie outside the requested output grid. Preparing a
design condition does not add an undeclared point to the map.

## Canonical Operating-Point Expansion

The pure expansion function traverses the already canonical request as:

1. ascending source temperature;
2. ascending sink temperature within source temperature; and
3. ascending load fraction within that temperature pair.

For each coordinate:

- `q_useful_requested = load_fraction * reference_capacity`;
- evaporating temperature equals external source temperature minus the source
  approach;
- condensing temperature equals external sink temperature plus the sink
  approach;
- `curve_id` uses map identity plus zero-based source and sink ordinals; and
- point name uses the curve identity plus the zero-based load ordinal.

Ordinal-based identifiers avoid platform-dependent float formatting while the
point fields retain the actual coordinates. The number of points is exactly
`source_count * sink_count * load_count`. Expansion rejects an internally
invalid lift before an engine call and records that coordinate as a diagnostic
during generation.

## Simulator Selection and Lifecycle

`get_hpr_point_simulator(backend)` uses an exact closed registry:

- `coolprop` returns the default adapter already available in the base install;
- `tespy` lazily imports the optional TESPy leaf; and
- any other value fails before session preparation.

The generation service owns one lifecycle:

1. obtain one fresh simulator context manager;
2. call `prepare(context)` exactly once;
3. simulate each canonical operating point exactly once unless preparation
   failed;
4. close exactly once in a `finally` path; and
5. discard the simulator and all engine objects after the call.

A failed point must not poison the next coordinate. Each adapter resets all
point-varying specifications from the prepared design state before solving the
next point. If the adapter reports that recovery is impossible, remaining
coordinates receive deterministic `session_unavailable` diagnostics without
fabricating simulation values.

## CoolProp Simulation

The CoolProp adapter delegates to the existing single
`VapourCompressionCycle` calculation rather than reimplementing refrigerant
properties.

For each point it:

1. supplies translated dew-anchored evaporating and bubble-anchored condensing
   temperatures plus the resolved working-fluid specification, compressor
   efficiency, superheat, subcooling, internal-HX, and
   approach assumptions;
2. supplies requested sink duty for heat-pump mode or requested source duty for
   refrigeration mode;
3. requires a solved cycle and finite positive compressor work;
4. extracts evaporator duty, condenser duty, and compressor work;
5. converts the current watt-based cycle outputs to map kilowatts; and
6. returns a normalized `HprPointSimulation` with no engine object.

At a fixed temperature pair, the current steady-state model scales refrigerant
mass flow with duty. It does not introduce cycling loss or a part-load factor,
so COP is invariant with load fraction within numerical tolerance. The nominal
full-load result is compared with the existing cycle calculation as the oracle.

## TESPy Design and Offdesign Simulation

The optional adapter owns a closed refrigerant loop consisting of a cycle
closer, compressor, condenser-side heat rejection component, expansion valve,
and evaporator-side heat uptake component. It models no secondary fluid circuit,
pump, fan, motor loss, or parasitic electrical load.

Preparation:

1. lazily import TESPy and translate an unavailable dependency to the standard
   optional-dependency message;
2. translate and preflight the resolved pure, registered-blend, or explicit
   molar-mixture specification and build a fresh network for that one working
   fluid;
3. apply the target-derived nominal evaporating and condensing conditions;
4. apply full resolved useful capacity, compressor efficiency, superheat, and
   subcooling;
5. solve exactly one design point and require convergence;
6. persist only the private design state needed for offdesign solves; and
7. return normalized simulator metadata for final provenance.

Offdesign evaluation:

1. restore the design snapshot before every point;
2. apply translated evaporating/condensing temperatures and requested useful
   duty;
3. apply the OpenPinch-owned compressor-efficiency characteristic identified as
   `openpinch-single-stage-compressor-v1`;
4. solve in offdesign mode with deterministic convergence settings;
5. require a converged network and finite extracted values; and
6. return nonnegative evaporator duty, condenser duty, and compressor power in
   kilowatts.

The characteristic's numeric coordinates are a versioned package-owned model
asset, not a runtime lookup of TESPy defaults. Code Generation must snapshot and
test the exact values before calling the first release physically validated.
The first-release model records fixed approach temperatures instead of claiming
heat-exchanger `kA` offdesign behavior.

Temporary design-state files, if required by the installed TESPy API, live in a
session-private temporary directory and are removed during close. No TESPy
network or path is serialized.

## Result Normalization and Map Assembly

The grid service, not an adapter, is authoritative for public point fields.
For every successful normalized simulation it:

1. checks finite, nonnegative source and sink duties and positive compressor
   power;
2. checks `q_sink = q_source + electric_power` using the Unit 1 energy tolerance;
3. selects useful duty by mode and checks it against requested useful duty;
4. computes COP from useful duty divided by compressor power rather than trusting
   a backend-provided COP;
5. constructs the immutable Unit 1 point with the canonical identity; and
6. appends it in canonical order.

Map provenance is assembled only after a complete successful traversal. It
contains OpenPinch and engine versions, mode, cycle/model and refrigerant,
nominal design coordinates and capacity, approaches, superheat/subcooling,
compressor efficiency, characteristic-set identifier, convergence policy,
point counts, and the exact statement `electric_power: compressor_only` with
modeled auxiliaries empty.

Working-fluid provenance contains the original specification, property backend,
kind, canonical registered identity when applicable, explicit components and
normalized mole fractions when applicable, composition basis, and the
dew/bubble anchor convention. Provider and engine versions make opaque
registered-blend identity auditable. No engine object enters provenance.

The resulting `HprPerformanceMap` is validated by Unit 1. Adapter values never
bypass the Unit 1 physical and topology checks.

## Failure and Cleanup Flow

Preparation failures produce one `prepare_failed` or `dependency_unavailable`
diagnostic and no point calls. Once prepared, the generator attempts the full
canonical grid. Each failed coordinate produces one ordered structured
diagnostic and no placeholder point.

After traversal, close runs exactly once. A cleanup failure is appended after
point diagnostics. If any diagnostic exists, the service raises one
`HprMapGenerationError` containing the full immutable diagnostic tuple and
returns no map. If no diagnostic exists, it constructs and returns the complete
map.

The exception retains a Python cause only for local debugging. Its diagnostic
payload contains plain stable fields and never serializes an engine object,
traceback, temporary path, or partial map.

## Testable Properties

| Component | Category | Property |
|---|---|---|
| Context builder | Invariant | Valid basis/request inputs are not mutated and resolved mode/backend/capacity remain stable |
| Context builder | Idempotence | N/A: construction is pure but does not accept its own context output as input |
| Operating-point expansion | Invariant | Output size equals the Cartesian-product cardinality and every requested coordinate occurs exactly once |
| Operating-point expansion | Ordering | Output is ordered by source, sink, then load regardless of valid request input permutation before Unit 1 normalization |
| Operating-point expansion | Induction | Adding one valid load fraction adds exactly one point to every source/sink curve without changing existing coordinate identities |
| Point normalization | Easy verification | Every accepted point satisfies balance, requested useful duty, mode-specific COP, finite/range, and Unit 1 validation checks |
| CoolProp adapter | Oracle | Generated nominal duties and power equal the existing `VapourCompressionCycle` calculation within the declared numerical tolerance |
| CoolProp adapter | Invariant | At fixed temperatures, duty and power scale with load while COP remains constant within tolerance |
| Working-fluid normalization | Invariant | Pure fluids, registered blends, and valid explicit mixtures of arbitrary component count resolve deterministically without changing the source specification |
| Working-fluid normalization | Round-trip | Explicit component order, normalized mole fractions, composition basis, and provider backend survive context/provenance serialization |
| Mixture adapters | Differential | CoolProp and TESPy preparation receive equivalent backend, component, fraction, and dew/bubble anchor semantics for every supported explicit mixture |
| TESPy fluid acceptance | Example | A registered zeotropic blend and explicit binary molar mixture construct through TESPy with the preserved provider backend and saturation anchors |
| TESPy fluid acceptance | Failure oracle | A wrapper/property-state limitation yields the typed backend diagnostic and never a categorical mixture rejection or fallback |
| Simulator registry | Invariant | Exact backend identity selects one adapter; unsupported values never fall back |
| Session lifecycle | Stateful model | A fake event log observes one prepare, one ordered call per point, and one close; failures never permit use after close |
| Failure aggregation | Invariant | Diagnostics preserve failed-coordinate order and any failure prevents map return |
| Map generation | Round-trip | Every generated complete map survives Unit 1 JSON serialization/deserialization unchanged |
| Map generation | Repeatability | Fresh deterministic simulators with equal contexts produce equal map values and provenance |

PBT-01 is satisfied by the table. These properties must be carried into Code
Generation. PBT-02, PBT-03, PBT-05, PBT-06, PBT-07, PBT-08, and PBT-10 are
applicable. PBT-04 is N/A because no Unit 2 operation claims repeat application
to the same mutable object is idempotent. PBT-09 continues to use the existing
Hypothesis/pytest stack and is finalized in NFR Requirements.
