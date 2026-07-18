# Workflow Argument Simplification Map

## Objective

Make every core and advanced `PinchProblem` and `PinchWorkspace` workflow
discoverable without requiring a process engineer to memorize OpenPinch-owned
string values. Method names express algorithm families; booleans express only
independent binary engineering decisions; named values express quantities; and
configuration supplies omitted numerical assumptions rather than choosing
which workflow runs.

## Argument Design Rules

1. Closed OpenPinch choices are not accepted as ordinary strings on the normal
   public path.
2. A choice that changes the algorithm, supported arguments, result semantics,
   or validity rules receives a dedicated callable.
3. A boolean is used only when both values are meaningful for the callable and
   neither value makes another argument combination impossible.
4. Mutually exclusive load specifications are separate optional arguments and
   are validated as exactly one effective choice.
5. Omitted named arguments use advanced `options`, stored configuration, and
   library defaults in that order. Explicit values never mutate configuration.
6. Strings remain appropriate for open-world identities and external resource
   names: zone and period identifiers, stream and case names, file paths,
   refrigerants and fluids, solver names, and output filenames.
7. User-facing validation names the conflicting arguments and explains the
   valid call shape. It does not ask the user to discover an internal enum
   value.
8. The package root remains limited to `PinchProblem` and `PinchWorkspace`; the
   simplified API must not require root-level option-enum imports.

## Heat-Pump and Refrigeration Decision

A single generic `heat_pump()` method with `is_simulated_cycle`, `is_cascade_cycle`,
and `has_mvr` booleans is rejected. `has_mvr` is meaningful only for one
simulated cascade family, Brayton has different controls, and MVR refrigeration
is unsupported. That design creates valid-looking but impossible boolean
combinations.

The canonical methods are therefore:

```python
problem.target.carnot_heat_pump(
    is_utility_heat_pump=False,
    is_cascade_cycle=True,
    load_fraction=0.25,
    condensers=1,
    evaporators=1,
)

problem.target.vapour_compression_heat_pump(
    is_utility_heat_pump=False,
    is_cascade_cycle=True,
    refrigerants=["water", "ammonia"],
    load_fraction=0.25,
)

problem.target.brayton_heat_pump(
    is_utility_heat_pump=False,
    load_fraction=0.25,
)

problem.target.mvr_heat_pump(
    is_utility_heat_pump=False,
    mvr_fluids=["water"],
    load_fraction=0.25,
)
```

`carnot_heat_pump()` and `carnot_refrigeration()` are the fast Carnot
screening models.
`vapour_compression_*()` means a CoolProp-backed simulated cycle.
`brayton_*()` and `mvr_heat_pump()` expose specialized models whose argument
sets differ materially. There is no `mvr_refrigeration()` callable because the
backend does not support that combination.

### HPR common signature policy

The HPR methods share relevant named arguments rather than a placement or
cycle string:

- `is_utility_heat_pump` or `is_utility_refrigeration` selects process versus
  utility integration. `False` is direct/process integration and `True` is
  indirect/utility integration.
- `is_cascade_cycle` selects cascade versus parallel topology only on Carnot
  and vapour-compression methods.
- Exactly one effective load specification is allowed:
  `load_fraction`, `load_duty`, or `period_loads`.
- Common counts and numerical controls use engineering names such as
  `condensers`, `evaporators`, `compressor_efficiency`, `motor_efficiency`,
  `expander_efficiency`, `minimum_approach_temperature`, and
  `maximum_restarts`.
- Simulated methods additionally accept open-world `refrigerants`, optional
  initialization, sorting, integrated-expander, and optimizer controls.
- MVR accepts MVR fluids, MVR compressor efficiency, and MVR stage controls.
- Brayton exposes only Brayton-relevant arguments and rejects multiperiod
  dispatch until that backend supports it.

## Targeting Surface

| Current choice or argument | Canonical interaction | Reason |
|---|---|---|
| `zone_name: str` | `zone: str | Zone | None` | Accept an identity or prepared object and use one term everywhere. |
| `period_id: str` | Keep `period_id` | It is a user-defined identity, not a closed answer. |
| `include_subzones: bool` | Keep | It is an independent scope decision. |
| `placement="direct" | "indirect"` | `is_utility_heat_pump` or `is_utility_refrigeration` | The binary distinction is meaningful for every member of the relevant family. |
| `cycle="..."` | Dedicated HPR methods | Cycle families have different constraints and dependencies. |
| `load_mode="fraction" | "duty" | Exactly one of `load_fraction`, `load_duty`, or `period_loads` | The supplied argument states both meaning and value. |
| base-target type strings | `base_target=<returned target>` | Exergy, cogeneration, and transfer consume the result the engineer already selected. |
| turbine-model strings | Dedicated cogeneration methods | Correlations and fixed-isentropic analysis are algorithms, not labels. |
| generic all-period method string | Mirrored `target.all_periods.*` methods | IDE completion replaces string dispatch. |

The complete target vocabulary is:

- heat integration: `direct_heat_integration()`,
  `indirect_heat_integration()`, `total_site_heat_integration()`, and
  `all_heat_integration()`;
- area and cost: `heat_exchanger_area_and_cost()`;
- HPR: `carnot_heat_pump()`, `carnot_refrigeration()`,
  `vapour_compression_heat_pump()`,
  `vapour_compression_refrigeration()`, `brayton_heat_pump()`,
  `brayton_refrigeration()`, and `mvr_heat_pump()`;
- cogeneration: `cogeneration()` for the Medina-Flores correlation,
  `sun_smith_cogeneration()`, `varbanov_cogeneration()`, and
  `isentropic_cogeneration(efficiency=...)`;
- post-analysis: `exergy(base_target=...)` and
  `energy_transfer(base_target=...)`.

When `base_target` is omitted, the method uses its one documented default
prerequisite; configuration never supplies a target-type string.

### Multiperiod targeting

The current string-dispatched replay is replaced by a discoverable mirrored
accessor:

```python
period_heat = problem.target.all_periods.all_heat_integration(workers=1)
period_hpr = problem.target.all_periods.carnot_heat_pump(
    is_utility_heat_pump=True,
    is_cascade_cycle=False,
    period_loads={"winter": 0.30, "summer": 0.15},
    workers=2,
)
period_power = problem.target.all_periods.cogeneration(workers=1)
```

Each supported selected-period method has a same-named all-period method where
the backend genuinely supports shared or repeated period execution. `workers`
replaces backend strings for normal use: `1` is serial and values greater than
one use the documented process backend. Unsupported combinations, including
multiperiod Brayton, are absent rather than failing after string dispatch.

## Design and Component Surfaces

| Surface | Canonical arguments | Removed closed choices |
|---|---|---|
| `design.heat_exchanger_network()` | approach temperatures, stages, `pack_stages: bool | None`, solver resource name, numerical tolerances | method and stage-packing strings |
| `design.enhanced_heat_exchanger_network()` | the standard arguments plus `quality_tier` | implicit enhanced-method aliasing |
| `design.multiperiod_heat_exchanger_network()` | shared-design controls, period weights, numerical HEN controls | `periods="all"` |
| `design.open_hens()` | OpenHENS-specific numerical and solver controls | HEN method string |
| `design.pinch_design()` | pinch-design controls | HEN method string |
| `design.thermal_derivative()` | initial networks and derivative controls | HEN method string |
| `design.network_evolution()` | initial networks and evolution controls | HEN method string |
| `components.add_process_mvr()` | stream identities or objects, component ID, stages, injection flag, lift or ratio, efficiencies, period ID | no OpenPinch closed strings |

`pack_stages=None` means use the stored configuration or automatic library
default; `True` and `False` explicitly request packed or unpacked stages.
Solver names remain strings because they identify external resources.
`workspace_variant` is renamed `case_name` wherever a design needs a workspace
identity.

## Results, Reports, and Output

Observation never triggers analysis. Closed formatting and aggregation strings
are replaced as follows:

```python
problem.summary_frame(
    detailed=True,
    include_periods=True,
    include_weighted_average=True,
)
```

The same two aggregation booleans apply to `metrics()`, `report()`, and
`export_excel()`. Their four combinations represent selected only, all
periods, weighted average only, and all periods plus weighted average.
`detailed` replaces format strings where a compact versus detailed view is the
only distinction. Any `solve` argument is removed.

Named plot methods select the graph. `plot.data()` returns the full graph-data
catalog. Export accepts plot method references rather than graph-type strings:

```python
problem.plot.export(
    "figures",
    plots=(problem.plot.composite_curve, problem.plot.grand_composite_curve),
)
```

Omitting `plots` exports every available graph. `zone` accepts a name or
`Zone`; `index` is an integer; `show` and `return_graph_data` remain genuine
booleans. `index_name` remains a filename identity.

## Workspace Surface

`scenario()` creates and returns a `PinchProblem`; it no longer accepts
`solve=True` or a workflow string:

```python
tight_dt = workspace.scenario("tight_dt", dt_cont_multiplier=0.75)
tight_dt.target.direct_heat_integration()
```

For several cases, `workspace.cases(names)` returns a batch view whose
`target`, `design`, `summary_frame`, `metrics`, `report`, and export operations
mirror the single-problem vocabulary:

```python
study = workspace.cases(["baseline", "tight_dt"])
study.target.direct_heat_integration()
comparison = workspace.compare_cases("baseline", "tight_dt")
```

This replaces `run_cases(..., target="...")` and prevents a growing string
dispatcher. Active-case forwarding retains the same signatures as
`PinchProblem`. Case names, project names, paths, and bundle names remain
strings because they are user-owned identities.

## Complete Argument Review

| Public area | Simplification outcome |
|---|---|
| construction and loading | Source paths and project identity remain open-world strings; no workflow runs. |
| validation and configuration | No closed-choice arguments; persistent numerical options remain advanced. |
| heat integration | Descriptive methods, `zone`, `include_subzones`, and period identity only. |
| HPR | Seven model-specific callables, binary placement/topology flags, mutually exclusive named load values. |
| cogeneration | Four model-specific callables; base target passed as an object. |
| exergy and energy transfer | Base target passed as an object; no target-type code. |
| area and cost | Friendly price, hours, exchanger-cost, discount-rate, and service-life kwargs. |
| multiperiod target execution | Same-named methods under `all_periods`; integer workers; no method/backend string. |
| process MVR | Engineering quantities and open-world stream/fluid identities. |
| HEN synthesis | Dedicated methods for algorithm and period scope; binary stage packing. |
| selected HEN | One-based rank and period identity; utility name remains an identity. |
| summaries and metrics | Two aggregation booleans; no format, period-mode, or solve strings. |
| plots | Named plot methods and callable export selection; no graph-type string. |
| files, Excel, dashboard | Paths, filenames, and sheet identities remain strings; side effects stay explicit. |
| workspace cases | Scenario returns a problem; batch accessor mirrors workflows; no workflow string. |
| serialization and inspection | No execution or closed-choice arguments. |

## Validation and Configuration Contract

- Passing two load specifications raises before targeting and names both
  arguments.
- Config fallback may provide one omitted load specification, but an explicit
  named load clears conflicting stored load modes for that invocation.
- Topology flags are accepted only by methods that support both topologies.
- Model-specific kwargs on the wrong callable are rejected as unknown.
- `base_target` must belong to the same problem, zone, and compatible period.
- Batch case execution reports errors by case while preserving deterministic
  input order.
- All resolved arguments and their provenance are stored in result metadata.
- Removed closed selector strings and obsolete config selector keys are
  rejected; no compatibility aliases are required.

## Tutorial and RTD Consequences

The tutorial suite expands from seventeen to eighteen notebooks so specialized
HPR models are taught without overloading one example:

- tutorial 08 covers `carnot_heat_pump()` and `carnot_refrigeration()`;
- tutorial 09 covers vapour-compression and Brayton heat pump/refrigeration;
- tutorial 10 covers supported multiperiod HPR methods;
- tutorial 11 covers process MVR and `mvr_heat_pump()`;
- tutorials 12 and 13 cover selected-period and multiperiod cogeneration;
- later tutorial numbers shift by one, ending with results and exports at 18.

The canonical RTD coverage manifest must track every specialized method,
all-period mirror, binary semantic mode, load-specification form, configuration
fallback, and invalid-combination guard. Each public callable has at least one
executable tutorial owner; each advanced argument family has an RTD API entry
even when its expensive numerical example is confined to a slow test profile.

## Verification Requirements

1. Signature tests prove normal workflows expose no OpenPinch-owned closed
   string selectors.
2. Contract tests cover every valid HPR callable and reject cross-family or
   impossible combinations.
3. Property-based tests generate load forms, placement, topology, and omitted
   versus explicit values while preserving fixed-seed CI reproducibility.
4. Multiperiod tests cover every supported mirrored method and prove absent
   methods for unsupported algorithms.
5. Workspace batch tests prove identical results and ordering versus explicit
   per-case calls without string dispatch.
6. Static checks reject retired argument names, selector config keys, internal
   enum imports in tutorials, and stale RTD examples.
7. The live public-surface inventory, tutorial manifest, and generated RTD
   coverage page must agree exactly.
