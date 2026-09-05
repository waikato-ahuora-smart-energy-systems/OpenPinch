# Unit 3 Technology-Stack Decisions

## Decision Summary

| Concern | Decision | Rationale |
|---|---|---|
| Runtime | Existing Python `>=3.14.2` | Matches the package and completed Units 1/2 |
| Required thermodynamics | Existing `CoolProp>=8` | Preserves the default HPR implementation and fluid capability gate |
| Optional thermodynamics | Existing `tespy>=0.10.1.post2`, current lock 0.11.2 | Reuses the tested optional network simulator without a second dependency policy |
| Public/internal records | Existing Pydantic plus frozen slotted dataclasses | Strict JSON records at contract boundaries and lightweight engine-neutral values internally |
| Evaluator abstraction | `typing.Protocol` and explicit factory/session lifecycle | Permits deterministic fakes and isolates engine imports |
| Candidate cache | Call-local `collections.OrderedDict`, maximum 512 exact keys | Provides bounded LRU behavior without a cache dependency or cross-call state |
| Cache key | Frozen normalized candidate request | Includes all physical inputs and forbids approximate equivalence |
| Default oracle | Existing CoolProp objective/cycle path | Establishes no-regression behavior rather than a rewritten reference |
| Map bridge | Existing Unit 2 basis builder and generator | Avoids a duplicate map implementation |
| Serialization | Existing Pydantic JSON/model APIs | Keeps target records and map contracts strict and testable |
| Example testing | Existing pytest | Repository-standard fixtures, monkeypatching, subprocesses, and markers |
| Property testing | Existing Hypothesis with pytest | Satisfies PBT-09 custom strategies, shrinking, state machines, and seed replay |
| Timing/memory | `time.perf_counter` and `tracemalloc` | Dependency-free trend and coarse regression evidence |
| Static/docs | Existing Ruff, architecture tests, and Sphinx | Enforces style, dependencies, public inventory, and warning-clean docs |
| Build/dependency | Existing Hatchling and uv | Preserves locked extras, source/wheel parity, and isolated smokes |

## Runtime and Dependency Policy

Unit 3 introduces no runtime dependency. CoolProp remains required and TESPy
remains available through the neutral `tespy` extra, compatible
`brayton_cycle` alias, and `full`. The tested public minimum remains
`tespy>=0.10.1.post2` without an upper bound; the repository lock currently
proves TESPy 0.11.2 compatibility.

The concrete TESPy leaf may use public TESPy network, component, connection,
characteristic, and design-solve APIs already established by Unit 2. Application
accessors, target records, evaluator protocols, cache values, map-basis builders,
and documentation helpers must not import or type against TESPy classes.

REFPROP remains excluded. HEOS is the required CI property backend. Other
CoolProp-native backends remain capability checked rather than allowlisted.

## Targeting Evaluator Architecture

An engine-neutral `HprTargetThermodynamicEvaluator` protocol defines the
targeting-time lifecycle and one candidate design evaluation. A closed factory
receives the normalized backend and returns a fresh call-owned evaluator.

The CoolProp factory branch wraps the existing vapour-compression objective
cycle without replacing its calculations. The TESPy branch lazily loads the
existing optional leaf and reuses the Unit 2 topology/resource helpers where
possible. Shared helpers may move within the concrete optional leaf, but TESPy
imports may not move upward into engine-neutral or targeting modules.

The optimizer receives evaluator behavior through an explicit injected closure
or callable boundary. A runtime evaluator object is not serialized into
`HeatPumpTargetInputs`, copied into `HeatPumpTargetOutputs`, or attached to the
domain target.

## Exact Bounded Cache

The TESPy evaluator coordinator uses a call-local `collections.OrderedDict` with
a constant maximum of 512 entries. Standard-library ownership makes insertion,
lookup, move-to-end, and oldest-entry eviction explicit and testable.

The cache key is the frozen normalized candidate request or an exact tuple of all
its fields. It includes mode, cycle/model identity, working-fluid backend and
composition, evaporating and condensing anchors, useful duty, both approaches,
compressor efficiency, superheat, subcooling, and internal-HX assumption.
Diagnostic ordinals and object identities are excluded because they are not
physical inputs.

No float rounding, tolerance bucket, string formatting, partial-key lookup,
process-global decorator cache, disk cache, or cross-call memoization is used.
All key floats are finite and normalized before lookup. The cache stores only
frozen engine-neutral success values or sanitized candidate-local failures.

`functools.lru_cache` is rejected because explicit call lifetime, selective
failure caching, counters, and teardown assertions are clearer with an owned
coordinator value. Third-party cache libraries are unnecessary.

## Contract and Domain Placement

The frozen, extra-field-forbidden `HprTargetSimulationRecord` belongs with
existing HPR contracts because it must serialize without analysis or engine
imports. Engine-neutral candidate request/result/profile values remain frozen
slotted dataclasses in HPR analysis. The domain target adds only
`hpr_simulation_backend`; detailed record ownership remains in normalized HPR
output details.

Unit 3 converts the winning record to Unit 2's existing `HprTargetMapBasis` in a
pure analysis builder. It calls the existing generator directly. No second map
request, map model, schema, resource loader, or serializer is added.

## Numerical and Performance Tooling

`math.isfinite`, named absolute/relative tolerance helpers, and existing Pint or
unit conventions validate numerical values. Native floats are retained.

`time.perf_counter` records the marked public target-and-map smoke durations.
The test asserts the approved 300-second total only on the supported TESPy
profile and records target/map subdivisions for trend review. The CI job also
has a finite outer timeout so a stalled solver cannot block indefinitely.

`tracemalloc` verifies the 64-MiB Python cache overhead using deterministic fake
maximum-size profiles. It does not claim to measure native TESPy allocations.
Repeated-call tests use explicit counters, temporary-directory inventories, and
weak references where objects support them.

The real performance smoke uses a deterministic bounded candidate-search test
fixture so it exercises public selector dispatch, one genuine TESPy targeting
design evaluation, target translation, basis extraction, and genuine TESPy map
design/offdesign work without making the external global optimizer's stochastic
search length part of the wall-clock contract.

## Test Organization

Planned ownership is:

- `tests/application/test_hpr_performance_map_accessor.py` for public selector,
  compatibility, map follow-up, non-mutation, and explicit examples;
- `tests/application/test_hpr_performance_map_accessor_properties.py` for
  selector, replay, target-to-basis, rejection, and non-mutation properties;
- `tests/analysis/heat_pumps/test_hpr_target_evaluator.py` for deterministic fake
  evaluation, validation, caching, failure, and lifecycle examples;
- `tests/analysis/heat_pumps/test_hpr_target_evaluator_properties.py` for exact
  cache, permutation, physics, and mixture properties;
- `tests/analysis/heat_pumps/test_hpr_target_evaluator_stateful.py` for the
  internal session/cache model;
- `tests/analysis/heat_pumps/test_hpr_tespy_targeting.py` for small real TESPy
  pure/blend/explicit-mixture and failure integration;
- `tests/strategies/hpr_targeting.py` for reusable selector, candidate,
  simulation-record, profile, failure-sequence, and supported-target strategies;
- `tests/architecture/` for TESPy and forbidden-package firewalls;
- `tests/packaging/` for extras, source/wheel parity, resources, and installed
  artifact smokes; and
- documentation/API inventory suites for signatures, examples, limitations, and
  no-root/no-CLI guarantees.

Existing file owners may be extended instead of creating a duplicate file when
their current scope matches exactly. Code Generation planning must confirm live
repository ownership before naming final paths.

Hypothesis remains configured through pytest with seed `20260715`, automatic
shrinking, and reusable strategies. Real TESPy calls do not run inside broad
generated properties; deterministic fakes prove combinatorial behavior and real
examples prove adapter integration.

## CI and Artifact Profiles

The base profile installs the core wheel without TESPy and runs default
CoolProp, cold-import, target-record, basis, map-contract, and documentation
examples. It must prove that merely importing or using default targeting does
not attempt a TESPy import.

The blocking TESPy source profile installs the neutral extra and runs focused
real targeting plus Unit 2 design/offdesign tests. The blocking TESPy artifact
profile installs the built wheel with its extra and executes the marked public
single-stage target-and-map smoke under the 300-second test threshold and finite
job timeout.

Pull-request and release gates retain both profiles. A declared TESPy profile
may not skip because TESPy is absent. Environment-specific non-convergence is a
failure to diagnose, not an automatic waiver or CoolProp fallback.

## Rejected Alternatives

| Alternative | Decision | Reason |
|---|---|---|
| Treat selector as map-only provenance | Reject | User selected full TESPy participation in ordinary targeting |
| Nominal verification after CoolProp optimization | Reject | Returned target would remain selected by CoolProp rather than TESPy |
| Rewrite the default CoolProp objective through a new adapter | Reject | Adds regression risk without value; existing path is the oracle |
| Fresh TESPy solve for exact duplicates | Reject | User selected safe exact call-local memoization |
| Approximate/tolerance cache | Reject | Nearby candidates are physically distinct and must not share results |
| Process-global or persistent cache | Reject | Risks stale engine/version/model state and cross-call coupling |
| Unbounded cache | Reject | Optimizer search size could create unbounded retained profiles/results |
| Cache fatal failures | Reject | Lifecycle/configuration failures invalidate the session rather than a key |
| Sixty-second universal target | Reject | TESPy and CI environments are too variable for that portability claim |
| No real wall-clock guard | Reject | A stalled or severe integration regression needs a finite release gate |
| Real TESPy in broad PBT | Reject | Slow, environment-sensitive, and hostile to shrinking |
| New benchmark/cache framework | Reject | Standard-library tools meet the approved requirements |
| Shared-vector multiperiod TESPy now | Reject | Requires a separate equipment-design/offdesign lifecycle design |
| OpenUtility integration dependency | Reject | Plain schema/fixture data is the intended package boundary |

## NFR and Extension Traceability

- Existing Python, CoolProp, TESPy, extras, uv, and Hatchling satisfy
  NFR-U3-001 and NFR-U3-004 through NFR-U3-006.
- Current target methods, Pydantic records, frozen dataclasses, protocol/factory,
  and Unit 2 bridge satisfy NFR-U3-002, NFR-U3-007 through NFR-U3-012,
  NFR-U3-022, NFR-U3-023, NFR-U3-028, and NFR-U3-033.
- `OrderedDict`, exact frozen keys, `perf_counter`, `tracemalloc`, counters, and
  weak references satisfy NFR-U3-013 through NFR-U3-018.
- Explicit lifecycle, typed failure classification, no-fallback assertions, and
  bounded diagnostics satisfy NFR-U3-019 through NFR-U3-024.
- Existing replay/batch infrastructure, Sphinx, schema resources, and isolated
  examples satisfy NFR-U3-025 through NFR-U3-027.
- pytest, Hypothesis, Ruff, coverage, architecture tests, Sphinx, build tooling,
  and isolated source/wheel profiles satisfy NFR-U3-029 through NFR-U3-032.
- **PBT-09** is compliant through existing Hypothesis/pytest selection,
  dependency, shrinking, strategies, and seed replay. All Functional Design
  properties are assigned to example/PBT owners. No PBT finding is blocking.
- Security and Resiliency extensions remain disabled; no technology is added
  solely for them.
