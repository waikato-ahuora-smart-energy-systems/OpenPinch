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
