# Unit 3 Code Generation Summary

## Outcome

Unit 3, HPR Target API Integration and Publication, is complete and ready for
review. The current vapour-compression heat-pump and refrigeration targeting
methods now accept `simulation_backend`. Omitting the selector or choosing
`coolprop` preserves the existing CoolProp calculation path. Explicit `tespy`
selection replaces the thermodynamic candidate evaluations inside the supported
single-stage scalar targeting path and records the winning design as detached,
plain data.

The public target accessor can generate a Unit 1 performance map from that
winning target record. The map remains a versioned Pydantic/plain-JSON contract;
OpenUtility, Pyomo, and HiGHS are neither imported nor required by OpenPinch.

## Created Production Files

| File | Responsibility |
|---|---|
| `OpenPinch/analysis/heat_pumps/performance_maps/targeting_models.py` | Frozen evaluator requests, results, profiles, metadata, failures, and protocol |
| `OpenPinch/analysis/heat_pumps/performance_maps/targeting.py` | Backend normalization, TESPy preflight, result validation, lifecycle coordination, and the exact bounded cache |
| `OpenPinch/analysis/heat_pumps/performance_maps/target_records.py` | Detached CoolProp and TESPy winning-record construction |
| `OpenPinch/analysis/heat_pumps/performance_maps/target_basis.py` | Pure target-record to map-basis compatibility boundary |

The existing Unit 2 TESPy adapter was extended in place for targeting design
evaluation. No duplicate replacement module was created.

## Modified Production Interfaces

- `OpenPinch/application/_problem/accessors/target.py` adds the optional backend
  selector to current vapour-compression heat-pump and refrigeration targeting
  and adds the explicit target-owned `hpr_performance_map` bridge.
- `OpenPinch/analysis/heat_pumps/service.py`, the cascade vapour-compression
  cycle and targeting handler route an injected evaluator through one complete
  optimization call while leaving CoolProp paths unchanged.
- `OpenPinch/contracts/hpr.py`, `OpenPinch/contracts/reporting.py`, and
  `OpenPinch/domain/targets.py` carry the normalized backend and frozen winning
  simulation record without an engine object.
- Multi-period preparation, execution, aggregation, targeting context, and
  reporting preserve scalar selected-period and independent all-period records.
  Shared-vector TESPy targeting is rejected before optimizer setup.
- Packaging metadata, resources, optional-install smokes, and pull-request,
  develop, and publish workflows include the target integration and installed
  artifact profiles.

## Design Decisions

- CoolProp is the omitted-selector default and remains the compatibility oracle.
- TESPy supports one evaporator, one condenser, one refrigerant loop, and the
  existing single-stage cascade vapour-compression target entry point.
- Each unique optimizer candidate receives a fresh TESPy design evaluation.
  Candidate results are never computed as offdesign points relative to another
  optimizer candidate.
- A call-local, exact `OrderedDict` LRU retains at most 512 candidate results.
  Keys preserve native floating-point values and complete physical identity;
  there is no quantization, tolerance bucketing, persistent cache, or cross-call
  reuse.
- Candidate-local failures are returned as bounded detached diagnostics and the
  session remains usable. Fatal setup, unexpected evaluation, restoration, or
  cleanup failures abort. There is no retry, fallback to CoolProp, or fabricated
  success.
- The winning target record supplies the design basis for Unit 2 TESPy
  offdesign map generation. Map creation is atomic and target-owned; callers
  cannot override the target backend.
- Pure fluids, registered blends, and explicit N-component molar mixtures use
  the common CoolProp-backed fluid resolver. Explicit mixtures retain component
  order and molar fractions. REFPROP is rejected. Evaporation uses the dew
  anchor and condensation uses the bubble anchor.
- The electrical boundary is compressor-only. Pumps, fans, motors, secondary
  circuits, cycling degradation, and plant dispatch are not silently invented.

## Requirement Traceability

| Requirement | Evidence |
|---|---|
| FR-5 | Existing public HPR targeting methods expose deterministic CoolProp/TESPy selection; omitted and explicit CoolProp results are oracle-equivalent |
| FR-6 | Unit 2 map generation is invoked from the target-owned basis; TESPy design/offdesign behavior is preserved |
| FR-7 | Typed local/fatal failures, atomic maps, exactly-once cleanup, and no fallback are tested |
| FR-8 | Frozen winning records and pure target-to-basis conversion connect targeting to map export |
| FR-9 | Maps serialize to ordinary schema-versioned data with no downstream optimizer dependency |
| FR-10 | TESPy imports remain confined to the concrete leaf and the core wheel smoke proves TESPy absent |
| NFR-U3-001 through NFR-U3-033 | Runtime, API, compatibility, isolation, determinism, cache, memory, performance, lifecycle, replay, documentation, PBT, coverage, regression, artifact parity, and ownership gates are closed |

## TDD and Property-Based Testing Evidence

Implementation followed the approved RED/GREEN sequence. Contracts were first
added for selector defaults, preflight rejection, engine-neutral evaluator
models, lifecycle transitions, cache behavior, real and fake TESPy evaluation,
target integration, winning records, the public map bridge, period/batch
semantics, documentation, cold imports, packaging, and installed artifacts.
Production changes were then made against those observable failures.

Hypothesis seed `20260715` is fixed in local and CI profiles. Generated tests
cover selector normalization, backend equivalence, request/result physics,
profile ordering and duties, mixture identity, cache key completeness, exact
hit/eviction behavior, callback/solve bounds, lifecycle state transitions,
candidate order independence, target-record round trips, incompatible map
contexts, period ordering, failure isolation, and public map serialization.
Normal shrinking remains enabled.

All applicable PBT-01 through PBT-10 rules are satisfied. PBT-04 is represented
by explicit repeatability, isolation, restoration, and immutable-value
properties because the evaluator protocol does not promise idempotent mutation.
There are no blocking Property-Based Testing findings. Security and Resiliency
extensions remain disabled and are N/A.

## Verification Results

| Gate | Result |
|---|---|
| Complete fixed-seed Unit 3 example/property selection | Passed |
| Focused heat-pump, application, contract, architecture, packaging, and resource selection | 871 passed |
| Broad non-solver regression profile | 3,007 passed before four Unit 3 compatibility findings; affected closure slice 7 passed after fixes |
| Combined changed-path statement and branch coverage | 97 percent; required minimum 95 percent |
| Exact cache capacity and memory | 512 entries; maximum-size fake profile below 64 MiB traced Python memory |
| Repeated ownership cleanup | Ten fake calls and at least three guarded real calls release private engine state |
| Public real TESPy source smoke | Target plus two-point map completes below the 300-second guard |
| Documentation | 55-source Sphinx build passes with warnings as errors |
| Static and structural quality | Ruff lint, changed-surface format, compilation, workflow parsing, architecture, resource, and diff hygiene pass |
| Repository-wide formatting | Fourteen unrelated pre-existing files remain outside this unit's patch |

## Distribution Evidence

Fresh OpenPinch 0.6.4 artifacts were built under
`/private/tmp/openpinch-step17.cUgtgY/artifacts`:

| Artifact | SHA-256 |
|---|---|
| `openpinch-0.6.4.tar.gz` | `78ff433120b332acc788b70916d879ed6da463f988266cf4b06b598a03bbd040` |
| `openpinch-0.6.4-py3-none-any.whl` | `03f0557f2df029862a7bf08fa9fe3daee9abb70c224ac4abf17849b644a85069` |

Both archives have zero duplicate members. The source checkout, source
distribution, and wheel contain byte-identical schema, heat-pump fixture,
refrigeration fixture, and compressor characteristic resources. Relevant
resource SHA-256 values are:

| Resource | SHA-256 |
|---|---|
| `heat-pump-1.0.json` | `b32dd243df836b16d5c3e6fa62dd8a0a45fa35d833edae52335c750eb1ee2856` |
| `refrigeration-1.0.json` | `e67acaed0e49c0a8394bb482e82321132c6f4c3bb801e414d1ed40f6255e54ae` |
| `schema-1.0.json` | `1d8e92432b0998c033607b4a2eb6d7b5639f1f46dafdb59586a881c0bb697d8b` |
| `openpinch-single-stage-compressor-v1.json` | `f7e1864476a243366aac8721a41aa09e4b909025414428d59b4ae7df9683b8af` |

An isolated Python 3.14 core install contains 18 packages, excludes TESPy, and
passes in 3.11 seconds. A second 33-package environment with TESPy 0.11.2 runs
the installed wheel's explicit public TESPy target, winning record, basis, and
two-point map smoke in 3.71 seconds. Both resolve OpenPinch from isolated
site-packages. Their public target/map signatures and resource digests match the
checkout.

## Documentation

The heat-pump workflow guide, fundamentals, reference API, schemas and config,
CLI/resources, capability matrix, support policy, README, release notes, and
tutorial coverage inventory now describe the selector default, supported TESPy
boundary, mixtures, dew/bubble anchors, target-to-map workflow, period and
concurrency rules, failure policy, install extra, limitations, and plain-data
consumer boundary.

## Known Limitations

- TESPy targeting is limited to scalar, single-stage, one-by-one topology.
- Shared-vector multi-period TESPy optimization is rejected; callers may target
  periods independently in canonical order.
- A nonzero internal heat exchanger and integrated expander are unsupported in
  the TESPy topology.
- Thermodynamic support for a fluid or mixture still depends on the installed
  CoolProp/TESPy property state and solvable phase envelope.
- Target candidate evaluation and map generation are sequential within one
  call. Independent process-level calls may be parallelized by the caller.
- Map schema 1.0 fixes installed useful capacity at each temperature cell and
  leaves adjacent interpolation, commitment, dispatch, electricity overlays,
  and piecewise MILP formulation to the downstream optimizer.

## OpenUtility Boundary

OpenUtility should consume only `HprPerformanceMap.model_dump(mode="json")` or
equivalent exported JSON. OpenPinch owns pinch targeting and optional
thermodynamic map generation. OpenUtility owns multi-period dispatch,
electricity and thermal balances, piecewise-linear/SOS2 or segment-binary MILP
formulations, Pyomo models, and HiGHS solves. Neither package needs to import the
other for this boundary to work.
