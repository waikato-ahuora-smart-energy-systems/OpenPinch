# Unit 2 Technology-Stack Decisions

## Decision Summary

| Concern | Decision | Rationale |
|---|---|---|
| Runtime | Existing Python `>=3.14.2` | Matches the package baseline and current HPR implementation |
| Required property library | `CoolProp>=8` | User-approved baseline for pure, registered-blend, and explicit-mixture properties |
| Default property backend | HEOS | Portable CoolProp-native backend used by required CI |
| Additional property backends | Capability-checked CoolProp-native backends; REFPROP explicitly rejected | Avoids a licensed external dependency while retaining native CoolProp extensibility |
| Optional simulator | `tespy>=0.10.1.post2` without an upper bound | Establishes the tested design/offdesign and mixture-wrapper API floor while CI detects future incompatibility |
| Optional extras | New `tespy`; compatible `brayton_cycle`; retained `full` | Makes shared TESPy capability discoverable without breaking the existing extra |
| Internal values | Frozen slotted standard-library dataclasses | Lightweight immutable analysis values with no second validation framework at the engine boundary |
| Simulator abstraction | `typing.Protocol` plus explicit context-managed lifecycle | Supports deterministic fakes without inheritance or engine leakage |
| Public map validation | Existing Unit 1 Pydantic models | Preserves the single strict output-contract owner |
| Temporary TESPy state | `tempfile.TemporaryDirectory` | Unique per-call cleanup without persistent filesystem assumptions |
| Version reporting | `importlib.metadata` | Standard-library package version provenance without importing package internals |
| Example tests | Existing pytest | Repository-standard fixtures, monkeypatching, markers, and subprocess checks |
| Property tests | Existing Hypothesis with pytest | PBT-09 custom strategies, shrinking, state machines, and seed replay |
| Performance checks | `time.perf_counter` and `tracemalloc` | Coarse dependency-free time and peak-memory regression evidence |
| Static/style checks | Existing Ruff and architecture tests | Enforces formatting and dependency direction without new tooling |
| Build/dependency workflow | Existing Hatchling and uv lock | Supports extras, reproducible profiles, wheel smoke, and dependency audits |

## Required CoolProp Baseline

`CoolProp>=8` becomes a base runtime requirement because the default HPR path
and working-fluid validation use it. Code Generation must update project
metadata and the lockfile before implementing the Unit 2 adapter, then run the
complete thermodynamic suite. The upgrade is not isolated to new map tests:
existing stream, vapour-compression, MVR, targeting, and property calculations
are compatibility gates.

HEOS remains the default. Unit 2 rejects a normalized `REFPROP` prefix before
simulator preparation. Other CoolProp-native backend names are not placed in a
hard-coded allowlist; their wrapper and required-state capability determine
acceptance. This retains native alternatives such as cubic equations of state
without creating a dependency on licensed REFPROP.

Unit 2 does not call global CoolProp functions that install estimated mixing
rules or overwrite binary interaction data. Any future custom-interaction model
requires a separately versioned, immutable input/provenance design.

## TESPy Packaging and Compatibility

The new `tespy` extra and compatible `brayton_cycle` alias contain the same
`tespy>=0.10.1.post2` requirement. `full` and the development dependency group
retain that compatible floor. There is no upper bound: compatibility is an
executed test claim, not an assumption that every future release is safe.

Only the concrete TESPy adapter imports TESPy. It uses public components,
connections, networks, characteristic APIs, and documented design/offdesign
storage. Unit 1, orchestration, diagnostics, and the default adapter do not
import or type against TESPy classes.

Pure fluids and registered blends are supplied to one TESPy connection fluid
token. Existing OpenPinch explicit molar composition becomes
`BACKEND::component[fraction]&...|molar` and is supplied as one closed-loop
fluid. TESPy's separate connection-level mass-fraction mixture model is not used
for the refrigerant loop.

## Internal Models and Lifecycle

Frozen slotted dataclasses own `HprTargetMapBasis`, `HprWorkingFluidSpec`,
`HprMapGenerationContext`, `HprOperatingPoint`, `HprPointSimulation`, simulator
metadata, and diagnostics. Their constructors remain internal and are fed by
explicit validation/build functions. Pydantic remains authoritative only at
the Unit 1 request/map interchange boundary; duplicating those public models is
rejected.

`HprPointSimulator` is a structural `Protocol` with prepare, simulate, and close
semantics coordinated through a context manager. Each factory call returns one
fresh session. The core never branches on concrete engine types after
selection, which keeps fake state-machine and failure injection tests simple.

TESPy design-state storage uses a unique `TemporaryDirectory` owned by the
session. Cleanup executes through `finally`/context-manager behavior and no path
appears in successful provenance or stable diagnostics.

## Numerical and Performance Tooling

The generator preserves native finite floating-point values. `math.isfinite`
and named absolute/relative tolerance helpers validate results; decimal
quantization, NumPy rounding, and byte-level thermodynamic comparisons are
rejected.

The core traverses plain tuples and builds bounded lists before constructing the
Unit 1 tuple-backed map. It does not add Pandas, NumPy, a task scheduler, or a
benchmark framework. `perf_counter` and `tracemalloc` provide the coarse
10,000-point fake-engine regression. Real TESPy tests have no elapsed-time
assertion because solver behavior varies by fluid, platform, and engine version.

Production exposes neither point caps nor per-map timeouts. External callers
that require parallel offline generation use isolated processes. Unit 2 adds no
global lock, thread pool, process pool, or persistent worker.

## Test Organization and CI Profiles

- `tests/analysis/heat_pumps/test_hpr_map_generation.py` owns deterministic
  examples with a fake adapter.
- `tests/analysis/heat_pumps/test_hpr_map_generation_properties.py` owns grid,
  physical, mixture-normalization, failure, and repeatability properties.
- `tests/analysis/heat_pumps/test_hpr_simulator_stateful.py` owns the Hypothesis
  prepare/simulate/close model.
- `tests/analysis/heat_pumps/test_hpr_coolprop_simulator.py` owns existing-cycle
  oracle and pure/blend/mixture examples.
- `tests/analysis/heat_pumps/test_hpr_tespy_simulator.py` owns the small real
  design/offdesign and mixture integration profile.
- `tests/strategies/hpr_map_generation.py` owns reusable valid basis, working
  fluid, grid, simulation, and failure strategies.
- `tests/architecture/` owns blocked-TESPy, forbidden-import, and layer rules.
- `tests/packaging/` owns dependency floors, extras, profile commands, and wheel
  installation behavior.

The ordinary base profile installs the package without optional TESPy support
and must pass cold imports plus CoolProp tests. The blocking TESPy profile
installs `.[tespy]`; missing TESPy is a profile failure, not a skip. Real engine
tests stay small, while generated fake tests provide broad input coverage.

Hypothesis uses the repository CI seed `20260715`, automatic shrinking, and
pytest integration. The stateful test models protocol state and generated
command sequences; it never invokes a real thermodynamic engine.

## Rejected Alternatives

| Alternative | Decision | Reason |
|---|---|---|
| Reuse only the `brayton_cycle` extra | Reject | HPR users should not need an unrelated feature name to discover TESPy support |
| Add an HPR-only duplicate TESPy extra | Reject | Duplicates one shared engine capability and invites version drift |
| Pin one exact TESPy version | Reject | Appropriate for the lockfile, not the public library requirement |
| Add a TESPy upper bound | Reject for first release | Blocking compatibility tests are the approved forward-version gate |
| Permit REFPROP | Reject | User explicitly excluded the licensed external property provider |
| Restrict all HPR properties to HEOS | Reject | User retained capability-checked CoolProp-native alternatives |
| Fall back to CoolProp after TESPy failure | Reject | Hides selected-model failure and corrupts provenance meaning |
| Convert mixtures to connection mass fractions | Reject | Changes the refrigerant property model and composition basis |
| Quantize map floats | Reject | Masks engine variation and discards scientifically meaningful precision |
| Add a production grid cap or timeout | Reject | Approved policy uses linear behavior without an arbitrary engine-independent limit |
| Add a global TESPy lock | Reject | Creates hidden process-wide coupling without guaranteeing external-engine safety |
| Promise thread-safe engines | Reject | OpenPinch cannot guarantee third-party global-state behavior |
| Use real TESPy for broad property tests | Reject | Slow and environment-sensitive; deterministic fakes own orchestration coverage |
| Add a benchmark dependency | Reject | Standard-library timing and memory tools are sufficient for the coarse guard |

## NFR and Extension Traceability

- Python, CoolProp, TESPy, extras, uv, and Hatchling satisfy NFR-U2-001 through
  NFR-U2-005 and NFR-U2-023.
- Frozen dataclasses, `typing.Protocol`, Unit 1 Pydantic contracts, and
  architecture checks satisfy NFR-U2-006, NFR-U2-011, NFR-U2-015,
  NFR-U2-019 through NFR-U2-022, and NFR-U2-026.
- CoolProp/TESPy wrappers and explicit working-fluid translation satisfy
  NFR-U2-007 through NFR-U2-010 and NFR-U2-014.
- Named tolerances, standard floating-point checks, `perf_counter`, and
  `tracemalloc` satisfy NFR-U2-012, NFR-U2-013, and NFR-U2-016 through
  NFR-U2-018.
- pytest, Hypothesis, architecture/packaging suites, Ruff, documentation, and
  build gates satisfy NFR-U2-023 through NFR-U2-028.
- Hypothesis is the selected PBT-09 framework. PBT-06 is now applicable to the
  internal session lifecycle; the remaining applicable properties retain
  shrinking and seed reproducibility. PBT-04 is N/A. No PBT NFR finding is
  blocking.
- Security and Resiliency extensions are disabled; no technology is added only
  for those extensions.
