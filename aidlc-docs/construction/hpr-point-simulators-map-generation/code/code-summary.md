# Unit 2 Code Generation Summary

## Outcome

Unit 2, HPR Point Simulators and Map Generation, is complete. OpenPinch now
generates complete Unit 1 heat-pump and refrigeration performance maps through
an engine-neutral coordinator. The existing `VapourCompressionCycle` remains
the default CoolProp-backed calculation owner. TESPy is available only after
explicit selection and only through its optional concrete adapter.

The implementation preserves the OpenUtility boundary: it returns the
versioned plain-data `HprPerformanceMap` and imports no OpenUtility, Pyomo,
HiGHS, application-layer target, or optimization object. Unit 3 remains
responsible for connecting these internals to the current HPR targeting
methods.

## Created Production and Resource Files

| File | Responsibility |
|---|---|
| `OpenPinch/analysis/heat_pumps/performance_maps/models.py` | Frozen target basis, fluid, context, point, result, metadata, and diagnostic values |
| `OpenPinch/analysis/heat_pumps/performance_maps/fluids.py` | Pure, registered-blend, and explicit molar-mixture parsing, dew/bubble preflight, and TESPy wrapper translation |
| `OpenPinch/analysis/heat_pumps/performance_maps/context.py` | Pure validated target/request context construction |
| `OpenPinch/analysis/heat_pumps/performance_maps/points.py` | Lazy canonical Cartesian operating-point traversal |
| `OpenPinch/analysis/heat_pumps/performance_maps/protocols.py` | Structural point-simulator and factory contracts |
| `OpenPinch/analysis/heat_pumps/performance_maps/errors.py` | Typed simulator failures, bounded diagnostics, and atomic aggregate error |
| `OpenPinch/analysis/heat_pumps/performance_maps/factory.py` | Closed, lazy backend selection with one fresh session per call |
| `OpenPinch/analysis/heat_pumps/performance_maps/generation.py` | Prepare, traverse, validate, aggregate, clean up, and construct the complete Unit 1 map |
| `OpenPinch/analysis/heat_pumps/performance_maps/provenance.py` | Deterministic JSON-only map provenance |
| `OpenPinch/analysis/heat_pumps/performance_maps/settings.py` | Frozen `openpinch-tespy-hpr-convergence-v1` public solve arguments |
| `OpenPinch/analysis/heat_pumps/performance_maps/resources.py` | Strict canonical characteristic loader and SHA-256 identity |
| `OpenPinch/analysis/heat_pumps/performance_maps/adapters/coolprop.py` | Default adapter delegating each point to `VapourCompressionCycle` |
| `OpenPinch/analysis/heat_pumps/performance_maps/adapters/tespy.py` | Optional single-stage design/offdesign TESPy session |
| `OpenPinch/data/heat_pumps/performance_maps/openpinch-single-stage-compressor-v1.json` | Canonical 17-point relative compressor-efficiency characteristic |

Package marker files were added under
`OpenPinch/analysis/heat_pumps/performance_maps/adapters/` and
`OpenPinch/data/heat_pumps/performance_maps/`. No new package-root export was
added.

## Dependency, Packaging, and CI Changes

- Required CoolProp advanced from the prior major version to `CoolProp>=8`.
- The neutral `tespy` and compatible `brayton_cycle` extras both declare
  `tespy>=0.10.1.post2`; `full`, the development group, and `uv.lock` are
  aligned.
- The verified development environment uses CoolProp 8.0.0 and TESPy 0.11.2.
- `pytest.ini` declares the guarded `tespy` marker.
- Pull-request, develop, and publish workflows have blocking real-TESPy jobs,
  isolated optional-install coverage, and installed-wheel TESPy smoke jobs.
- The pull-request and publish gates require the TESPy source and artifact
  results. The base artifact smoke proves TESPy remains absent.
- Source and wheel resource tests pin the characteristic at 458 bytes with
  SHA-256
  `f7e1864476a243366aac8721a41aa09e4b909025414428d59b4ae7df9683b8af`.

## Physical and Numerical Model

- Heat-pump points prescribe condenser duty; refrigeration points prescribe
  evaporator duty. All public duties and compressor power are kilowatts.
- CoolProp uses the current cycle assumptions, dew/bubble saturation anchors,
  and compressor-only electrical boundary. At fixed temperatures, part load
  changes steady-state mass flow without an invented cycling or PLF penalty.
- TESPy uses one closed refrigerant loop: cycle closer, compressor,
  condenser-side heat rejection, expansion valve, and evaporator-side heat
  uptake.
- TESPy solves one target-derived full-capacity design point. Every requested
  coordinate is then solved as offdesign after restoring the same private
  design and initialization snapshot.
- The OpenPinch-owned compressor characteristic is a relative-mass-flow to
  relative-isentropic-efficiency line. Heat-exchanger part-load curves,
  secondary-fluid circuits, pumps, fans, motor losses, and auxiliaries are not
  modeled in schema `1.0`.
- The fixed TESPy solve policy uses maximum 50 iterations, minimum 4 iterations,
  previous-state initialization disabled, CUDA disabled, result printing
  disabled, robust relaxation disabled, oscillation damping disabled, and
  postprocessing enabled.
- The coordinator independently checks convergence, finite domains, sign,
  useful duty, energy closure, and derived COP before constructing any public
  map. Any diagnostic makes the operation atomic: no partial map is returned.

## Fluid Support

Both adapters accept the same specification categories without an OpenPinch
allowlist:

- pure fluids such as R134a;
- provider-registered blends such as R407C; and
- explicit binary, ternary, or larger N-component molar mixtures using current
  CoolProp component and fraction syntax.

Explicit mixtures retain component order and normalized mole fractions and are
passed to TESPy as one `|molar` CoolProp wrapper token. REFPROP is deliberately
rejected. Actual support is decided by constructing and evaluating the selected
property state at the required dew/bubble conditions; engine limitations become
typed diagnostics and never trigger CoolProp-cycle fallback.

## Tests

Created tests include reusable Hypothesis strategies, deterministic fake-engine
examples, generated map/fluid properties, a fresh/prepared/fatal/closed state
machine, direct CoolProp cycle oracles, and guarded real TESPy tests. They cover
heat-pump and refrigeration modes; pure fluid, registered blend, explicit
binary mixture, and explicit ternary mixture; preparation, point-local,
session-fatal, restoration, non-convergence, cleanup, and corrupt-result paths.

## Verification Results

| Gate | Result |
|---|---|
| Seeded Unit 2 example, property, lifecycle, CoolProp, and TESPy suite | 125 passed with Hypothesis seed `20260715` |
| New Unit 2 statement and branch coverage | 99 percent combined; no path exclusions |
| Real TESPy blocking profile | 41 passed; 2,809 unrelated tests deselected |
| Existing thermodynamic, stream, and HPR regression boundary | 600 passed under CoolProp 8.0.0 |
| Packaging and architecture boundary | 186 passed, 3 expected platform/profile skips |
| Ten-thousand-point fake map | Passed the 5-second and 256-MiB limits |
| Increasing Cartesian grids | Exact one simulator call per valid coordinate |
| Ruff and patch hygiene | Focused lint, format, and `git diff --check` passed |
| Distribution build | `openpinch-0.6.4-py3-none-any.whl` and `openpinch-0.6.4.tar.gz` built successfully |
| Resource archives | Wheel and source distribution contain byte-identical characteristic data with the pinned SHA-256 |
| Installed core wheel | Passed full artifact smoke in an isolated Python 3.14.2 environment with TESPy absent |
| Installed TESPy wheel extra | Passed artifact smoke in a second isolated environment with TESPy 0.11.2 |

## Property-Based Testing Compliance

| Rule | Result |
|---|---|
| PBT-01 | Deterministic examples cover physical and lifecycle cases; generated tests cover broad grids, fluids, failures, and round trips |
| PBT-02 | Reusable Unit 2 basis, request, mixture, simulation, and failure strategies are provided |
| PBT-03 | Cartesian size/order, physical closure, capacity basis, COP, and immutability invariants pass |
| PBT-04 | N/A: the simulator protocol does not claim idempotent mutation; repeatability and restoration are tested separately |
| PBT-05 | CoolProp adapter results are compared with the existing direct cycle; TESPy is checked against prescribed duty and energy closure |
| PBT-06 | Lifecycle state transitions and invalid sequences are generated by the Hypothesis state machine |
| PBT-07 | Reusable strategies remain under `tests/strategies/` and deterministic fakes under the observable analysis owner |
| PBT-08 | Seed `20260715` passes with normal shrinking enabled |
| PBT-09 | Existing pytest, Hypothesis, coverage, and CI integration is reused |
| PBT-10 | Example, property, stateful, CoolProp, TESPy, architecture, and packaging responsibilities remain separately identifiable |

There are no blocking Property-Based Testing findings. Security and Resiliency
extensions remain disabled and are N/A for this stage.

## Known Limitations and Deferred Unit 3 Work

- The TESPy adapter is intentionally a single-stage vapour-compression model.
  A nonzero internal heat-exchanger duty assumption is rejected because that
  component is absent from the declared topology.
- TESPy fluid support depends on the installed CoolProp wrapper and the requested
  operating state. A valid input category is not a promise that every component
  combination has a solvable phase envelope.
- The first release has fixed convergence settings and no caller tuning,
  parallel execution, timeout, persistent cache, or thermodynamic-engine object
  in the public result.
- Unit 3 must add the `simulation_backend` selector to the current
  vapour-compression heat-pump and refrigeration targeting methods, detach
  `HprTargetMapBasis`, expose explicit map generation/export, and update public
  HPR documentation.
- OpenUtility remains independent and consumes only the Unit 1 mapping or JSON.
  Multi-period dispatch, piecewise MILP formulation, electricity overlays,
  Pyomo, and HiGHS remain outside OpenPinch.
