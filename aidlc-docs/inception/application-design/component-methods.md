# Component Methods

- `Stream(..., segments=...) -> Stream`: construct a parent with validated ordered children.
- `Stream.from_temperature_heat_profile(...) -> Stream`: linearize and normalize profile data.
- `Stream.replace_segments(segments) -> None`: atomically replace a complete profile.
- `Stream.update_segment(index, **changes) -> None`: atomically update one child and revalidate the profile.
- `StreamCollection.segment_numeric_view(idx=None) -> StreamCollectionNumericView`: return expanded thermal rows with parent metadata.
- `StreamCollection.to_dict(idx=None, expand_segments=False) -> dict`: choose parent or expanded reporting.
- `problem_to_solver_arrays(...) -> PreparedSolverArrays`: add padded segment-profile tensors without changing parent axes.
- `partition_exchanger_duty_by_segments(...) -> tuple[HeatExchangerSegmentAreaContribution, ...]`: form ordered duty-aligned slices.

## Package Usability Refactor Method Families

### Problem lifecycle

- `PinchProblem(source=None, *, project_name="Site")`: prepare one study.
- `load(source) -> Zone`: replace input and invalidate derived state.
- `validate() -> TargetInput` and `validation_report() -> ValidationReport`:
  validate without analysis.
- `to_problem_json() -> dict[str, object]`: serialize canonical input.
- `update_options(options, *, replace=False) -> None` and
  `set_dt_cont_multiplier(value) -> None`: persist assumptions and invalidate
  derived state.

### Target accessor

All focused methods accept keyword-only `zone: str | Zone | None`,
`include_subzones: bool`, `period_id: str | None`, and advanced
`options: Mapping[str, object] | None` where relevant.

- `direct_heat_integration(...) -> BaseTargetModel`
- `indirect_heat_integration(...) -> BaseTargetModel`
- `total_site_heat_integration(...) -> BaseTargetModel`
- `all_heat_integration(...) -> TargetOutput`
- `heat_exchanger_area_and_cost(..., utility_price=None,
  annual_operating_hours=None, exchanger_fixed_cost=None,
  area_cost_coefficient=None, area_cost_exponent=None, discount_rate=None,
  service_life_years=None) -> BaseTargetModel`
- `carnot_heat_pump(..., is_utility_heat_pump=False,
  is_cascade_cycle=True, load_fraction=None, load_duty=None,
  period_loads=None, condensers=None, evaporators=None,
  compressor_efficiency=None, motor_efficiency=None,
  expander_efficiency=None, minimum_approach_temperature=None,
  maximum_restarts=None) -> BaseTargetModel`
- `carnot_refrigeration(...) -> BaseTargetModel`: symmetric Carnot
  refrigeration signature using `is_utility_refrigeration`.
- `vapour_compression_heat_pump(..., refrigerants=None,
  initialize_from_carnot=None, sort_refrigerants=None,
  allow_integrated_expander=None) -> BaseTargetModel`
- `vapour_compression_refrigeration(...) -> BaseTargetModel`: symmetric
  simulated-refrigeration signature.
- `brayton_heat_pump(...) -> BaseTargetModel` and
  `brayton_refrigeration(...) -> BaseTargetModel`: Brayton-only controls.
- `mvr_heat_pump(..., mvr_fluids=None, mvr_compressor_efficiency=None,
  mvr_stages=None) -> BaseTargetModel`
- `cogeneration(..., base_target=None) -> BaseTargetModel`
- `sun_smith_cogeneration(..., base_target=None) -> BaseTargetModel`
- `varbanov_cogeneration(..., base_target=None) -> BaseTargetModel`
- `isentropic_cogeneration(..., efficiency, base_target=None)
  -> BaseTargetModel`
- `exergy(..., base_target=None) -> BaseTargetModel`
- `energy_transfer(..., base_target=None) -> BaseTargetModel`

`target.all_periods` mirrors each supported selected-period method and adds
`workers: int = 1`; unsupported methods, including multiperiod Brayton, are not
attributes.

### Components and design

- `components.add_process_mvr(source_streams, *, component_id, stages=None,
  liquid_injection=None, stage_temperature_lift=None,
  stage_pressure_ratio=None, compressor_efficiency=None,
  motor_efficiency=None, period_id=None, options=None) -> ProcessMVRComponent`
- `design.heat_exchanger_network(*, approach_temperatures=None, stages=None,
  pack_stages=None, initial_networks=None, solver=None, period_id=None,
  case_name=None, options=None) -> HENDesignView`
- `design.enhanced_heat_exchanger_network(*, quality_tier=2, ...)`
- `design.multiperiod_heat_exchanger_network(*, period_weights=None, ...)`
- `design.open_hens(...)`, `design.pinch_design(...)`,
  `design.thermal_derivative(initial_networks=None, ...)`, and
  `design.network_evolution(initial_networks=None, ...)`
- `HENDesignView.top(count)`, `network(rank)`, and `grid(rank)` use one-based
  ranks without mutating transport schemas.

### Observation and output

- `summary_frame(*, detailed=False, include_periods=False,
  include_weighted_average=False) -> pandas.DataFrame`
- `metrics(...) -> list[ReportMetric]`, `report(...) -> ProblemReport`, and
  `export_excel(destination, ...) -> Path` use the same aggregation booleans.
- `plot.catalog()`, `plot.data()`, and named plot methods consume cached graph
  data; plot indices are integers.
- `plot.export(destination, *, plots=None, zone=None) -> list[Path]` and
  `plot.export_gallery(...) -> Path` accept plot method references, with
  `plots=None` meaning all available plots.

### Workspace

- `scenario(name, *, base=None, options=None, replace_options=False,
  dt_cont_multiplier=None, activate=False) -> PinchProblem` creates but does
  not solve.
- `cases(names=None) -> WorkspaceCaseBatch` returns an ordered batch view.
- `WorkspaceCaseBatch.target.<method>(...)` and
  `WorkspaceCaseBatch.design.<method>(...)` mirror problem methods and return
  ordered per-case outcomes with structured failures.
- Active-case summary, metric, report, plot, export, configuration, and
  validation forwarding retains the corresponding `PinchProblem` signature.

## Repository Issue Remediation Methods

The names below are internal design signatures; they do not add root exports.

- `validate_workspace_case_name(value: str) -> str` returns the original valid
  identifier or raises `ValueError` for an unsafe/non-portable identifier.
- `PinchProblem.problem_data -> TargetInput | JsonDict | None` returns a deep,
  detached snapshot without changing the existing public type family.
- `PinchProblem.set_dt_cont_multiplier(value, *, zone_name=None) -> Zone` first
  resolves `_require_prepared_root_zone()` and retains its existing success
  contract.
- `_resolve_batch_export_directory(root: Path, case_name: str) -> Path` composes,
  resolves, and verifies one contained case directory.
- `_reserve_workbook_path(project_name: str, out_dir: str) -> Path` atomically
  reserves a readable `.xlsx` path.
- `_load_supported_openhens(openhens_root: Path) -> ContextManager[OpenHENSAPI]`
  yields verified modules/callables and restores interpreter import state.
- `_run_source_openhens(..., openhens_factory: Callable[..., Any])` consumes the
  verified factory and performs no ambient `openhens` import.

## Utility Placement Optimisation Methods

The signatures below define interface shape; detailed equations, tolerances,
and validation algorithms remain for per-unit Functional Design.

### Public application surfaces

```python
_TargetAccessor.utility_placement(
    *,
    isothermal: int | None = None,
    sensible: int | None = None,
    zone: str | Zone | None = None,
    period_ids: Sequence[str] | None = None,
    options: UtilityPlacementOptions | Mapping[str, object] | None = None,
) -> PinchProblem

_AllPeriodsTargetAccessor.utility_placement(
    **same_keyword_arguments_except_period_ids,
) -> PinchProblem

_CaseBatchTargetAccessor.utility_placement(**kwargs) -> CaseBatchResult
_CaseBatchAllPeriodsTargetAccessor.utility_placement(**kwargs) -> CaseBatchResult

PinchWorkspace.add(
    case: PinchProblem,
    *,
    name: str,
    activate: bool = False,
) -> PinchProblem
```

`period_ids=None` selects the problem's canonical ordered periods, or its
single scalar context when no period axis exists. The all-period method is an
explicit alias for that shared-placement behavior; it must not use the generic
per-period `_run` loop because that would produce independent placements.
Public keyword arguments are normalized immediately to an immutable
`UtilityPlacementRequest` so that specialist callers can also construct and
validate the request directly. Omitted counts infer templates from existing
problem utilities; supplied counts generate paired hot/cold templates. Omitted
zone uses the master zone, and the selected zone type resolves direct, Total
Site, or aggregate indirect scope without a public target selector.

### Contract construction and serialization

```python
UtilityPlacementRequest.from_public_arguments(...) -> UtilityPlacementRequest
UtilityPlacementRequest.model_dump_json(...) -> str
UtilityPlacementResult.model_validate_json(value: str) -> UtilityPlacementResult
UtilityPlacementResult.best -> UtilityPlacementCandidate
```

The internal request owns counts, ordered templates, scope, selected periods,
weights, named tolerances, and optimiser options. `UtilityLevelTemplate`
separates side and kind from temperature/span bounds, fixed span, and optional
fluid metadata. Nested evidence contracts
carry only serializable values, units, enums, tuples/lists, mappings, and typed
diagnostics.

### Application context boundary

```python
build_utility_placement_context(
    *,
    execution_zone: Zone,
    request: UtilityPlacementRequest,
    direct_service: Callable[..., Zone],
    total_site_service: Callable[..., Zone],
) -> UtilityPlacementContext

build_optimized_utility_case(
    source: PinchProblem,
    result: UtilityPlacementResult,
) -> PinchProblem
```

The builder receives an isolated execution-zone copy and returns detached
period profiles plus metadata. The case builder replaces utilities only on a
new normal problem and stores detached evidence there; it never changes source
inputs, configuration, canonical heat targets, workspace selection, or the
legacy `TargetOutput` cache.

### Template and vector model

```python
normalise_utility_templates(
    request: UtilityPlacementRequest,
    context: UtilityPlacementContext,
) -> UtilityTemplateSet

derive_placement_bounds(
    templates: UtilityTemplateSet,
    context: UtilityPlacementContext,
) -> tuple[tuple[float, float], ...]

build_initial_candidates(
    templates: UtilityTemplateSet,
    bounds: Sequence[tuple[float, float]],
) -> tuple[tuple[float, ...], ...]

encode_placement(candidate: DecodedPlacement) -> tuple[float, ...]
decode_placement(
    point: Sequence[float],
    templates: UtilityTemplateSet,
) -> DecodedPlacement
```

Encoding order is deterministic by side, template kind, and declared template
order. Isothermal spans are fixed metadata and therefore do not consume a span
coordinate; each sensible level consumes supply-temperature and span
coordinates.

### Candidate and objective evaluation

```python
evaluate_placement_candidate(
    point: Sequence[float],
    *,
    model: UtilityPlacementModel,
    context: UtilityPlacementContext,
) -> CandidateEvaluation

evaluate_period_coverage(
    placement: DecodedPlacement,
    period: PlacementPeriodContext,
) -> PeriodCoverage

evaluate_entropy_objective(
    coverage: PeriodCoverage,
    *,
    ambient_temperature: float,
    tolerances: UtilityPlacementTolerances,
) -> ThermodynamicCostBreakdown

```

Candidate evaluation returns structured infeasibility rather than throwing for
ordinary candidate rejection. Invalid requests, empty feasible bounds,
targeting failures, non-finite objectives, and solve
exhaustion use distinct placement exception subclasses.

### Service and optimisation coordination

```python
run_utility_placement_service(
    context: UtilityPlacementContext,
    request: UtilityPlacementRequest,
    *,
    minimise: Callable[..., OptimisationResult] = run_multistart_minimisation,
) -> UtilityPlacementResult

build_optimisation_problem(
    model: UtilityPlacementModel,
    context: UtilityPlacementContext,
) -> OptimisationProblem

normalise_placement_candidates(
    result: OptimisationResult,
    evaluations: Mapping[tuple[float, ...], CandidateEvaluation],
    *,
    limit: int,
) -> tuple[UtilityPlacementCandidate, ...]
```

The default callable dependencies remain injectable for focused tests. The
coordinator delegates backend execution to the existing optimiser and sorts
successful candidates by objective value followed by the decoded coordinate
tuple.

### Normal-case observation and registration

```python
optimized_case.utility_placement_result -> UtilityPlacementResult
optimized_case.target.direct_heat_integration(...) -> BaseTargetModel
optimized_case.summary_frame(...) -> DataFrame
optimized_case.plot.grand_composite_curve(...) -> Figure
workspace.add(optimized_case, name=..., activate=False) -> PinchProblem
```

The placement call returns the normal case. Targeting, summaries, reports, and
plots remain explicit ordinary case operations. Registration preserves only
placement evidence in addition to canonical case input.

### Executable notebook contract

`OpenPinch/data/notebooks/19_utility_placement_optimisation.ipynb` is generated
by `scripts/generate_tutorial_notebooks.py`, registered by the canonical
tutorial-coverage manifest, and executed under its declared notebook profile.
Its code cells call the public target accessor once with concise counts, add the
returned normal case to the workspace, then use ordinary target, summary, GCC,
and Total Site Profile methods. It has no CLI invocation, nested result-to-input
conversion, or placement-specific presentation method.
