# HPR correction interfaces

These signatures are proposed application contracts. Low-level function
placement may be adjusted during Code Generation without changing the public
behavior. Type definitions remain at their proper owners; the package root
continues to export only `PinchProblem` and `PinchWorkspace`.

## Public workflow

The user-selected entry point is
`problem.target.residual_utility(base_target=heat_pump)`. It belongs to the
existing target accessor and returns the detached residual case described below.

```python
class _TargetAccessor:
    def residual_utility(
        self,
        *,
        base_target: HeatPumpTargetBase,
        project_name: str | None = None,
    ) -> PinchProblem: ...
```

The target supplies its mode, zone, scalar period and snapshot. Do not add
redundant zone/period/mode selectors or infer a target from the last method run.
Require a successful eligible local result with matching provenance and result
digest. Reject foreign, stale, unavailable, weighted or unsupported aggregate
references before constructing a case. A valid retained scalar result may be
used only when the application can verify it against an authoritative retained
record without replay. A mutated result is not a source of silently recomputed
residual data.

The returned problem is initially unsolved. Creation detaches values and
validates input only. Utilities retain temperatures, prices, heat-transfer
contributions and capacity limits; inherited solved duties are reset for
allocation. Original case and HPR result remain unchanged.

```python
heat_pump = problem.target.carnot_heat_pump(
    load_fraction=0.25,
    condensers=1,
    evaporators=1,
    options={"COSTING_HPR_PRICE_RATIO_COLD_TO_ELE": 0.1},
)
load = heat_pump.hpr_load
profile = heat_pump.hpr_residual.profile

residual = problem.target.residual_utility(base_target=heat_pump)
leftover = residual.target.direct_heat_integration()
residual.plot.net_load_profiles()

optimized = residual.target.utility_placement(isothermal=2)
optimized_leftover = optimized.target.direct_heat_integration()
```

This example is proposed syntax and is not executable against the current
package. The optimized result is conditional on the frozen HPR result.

## Existing HPR calls and observations

Keep `carnot_heat_pump()` and `carnot_refrigeration()` names and load arguments.
`load_fraction` is a finite fraction within zero through one of the selected
background service duty. Explicit duty remains an alternative. The HPR result
reports available, selected and achieved duties separately through `hpr_load`.
Call/configuration precedence is unchanged; shared validation is consistent.

`hpr_residual.profile` is the precise residual representation, including any
new pocket breakpoints. Legacy problem-table projections must also be finite,
but must not truncate or replace that canonical grid.

## Public plots

```python
class PlotAccessor:
    def grand_composite_curve_with_heat_pump(
        self, *, target=None, zone_name=None, index=0,
        show=False, return_graph_data=False,
    ): ...

    def net_load_profiles_with_heat_pump(
        self, *, target=None, zone_name=None, index=0,
        show=False, return_graph_data=False,
    ): ...

    def grand_composite_curve_with_refrigeration(
        self, *, target=None, zone_name=None, index=0,
        show=False, return_graph_data=False,
    ): ...

    def net_load_profiles_with_refrigeration(
        self, *, target=None, zone_name=None, index=0,
        show=False, return_graph_data=False,
    ): ...
```

`target` accepts a solved HPR target object, with period derived from that
target. Each method accepts only its matching mode. An omitted target is valid
when exactly one matching eligible target remains after zone filtering;
otherwise raise a useful unavailable/ambiguous-selection error. Multiple
periods require explicit scalar target selection. `index` remains a graph index
within the selected target, not a way to pick a different target. Reject a
conflicting zone selector. Never fall back from refrigeration to heat pumping.

Examples: `problem.plot.grand_composite_curve_with_heat_pump(target=heat_pump)`
and `problem.plot.net_load_profiles_with_refrigeration(target=refrigeration)`.
Both use cached data. Plotting a retained target is supported only when its
matching graph snapshot is available and verifiable without execution.

## Residual case supported operations

| Operation | Contract |
|---|---|
| `validate()`, `to_problem_json()`, construction from canonical input | Preserve and validate residual_basis and utility definitions. |
| `target.direct_heat_integration()` | Allocate current utilities to the frozen profile; return a residual-classified utility target. |
| `target.utility_placement()` | Optimize utility levels on the frozen residual and return another residual case. |
| `summary_frame()`, reporting, standard residual GCC/net-load plots | Read completed residual utility results and identify origin/period/basis. |
| Other engineering methods requiring original physical streams | Fail explicitly for a residual-only case unless later implemented against its typed basis. |

In particular, new heat recovery, HPR resizing, process components, HEN design
and changes to process temperature shifts cannot silently reinterpret this case.
Current-case utility changes may invalidate residual utility results, but never
change the frozen thermal basis. Reloading a complete new source follows the
ordinary problem lifecycle. Ordinary source cases retain all existing behavior.

## Owner-level interfaces

| Owner | Proposed method | Input and output |
|---|---|---|
| HPR analysis | `build_hpr_load_summary(...)` | Selected service, verified cycle/ambient/accounting result -> immutable HPRLoadSummary |
| HPR analysis | `build_hpr_residual_data(...)` | Corrected background, physical boundary, result and period -> immutable HPRResidualData |
| Application | `bind_hpr_residual_snapshot(data, provenance, result_digest)` | Verified numerical data and local provenance -> final HPRResidualSnapshot |
| Application | `resolve_hpr_target_reference(problem, target)` | Live/retained result -> validated local reference and cached snapshot |
| Application | `create_hpr_residual_case(problem, base_target, project_name=None)` | Validated reference -> detached PinchProblem |
| Residual analysis | `allocate_residual_utilities(snapshot, hot_utilities, cold_utilities, options)` | Frozen profile and utility definitions -> allocations, summary and residual graphs |
| Placement application adapter | `build_residual_placement_context(snapshot, request)` | Residual snapshot and existing request -> immutable placement context |
| Placement allocation adapter | `allocate(period, placement)` | Existing candidate protocol -> AllocationAdapterResult on the frozen basis |
| Graph application resolver | `resolve_hpr_graph_selection(problem, target, mode, zone_name)` | Explicit/unique target -> graph-set identity; no solver calls |

Detailed parameter records, result validation predicates and tolerances will be
specified in the relevant Functional Design units.
