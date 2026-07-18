# PinchProblem and PinchWorkspace Tutorial Coverage Map

## Purpose

This map compares the live `PinchProblem` and `PinchWorkspace` functionality
with the ten existing tutorials, records the clean-break disposition of every
public surface, and assigns every supported canonical operation to an
executable tutorial.

Tutorial coverage is measured against the final supported process-engineer
surface, not against every historical spelling. A live operation may leave the
denominator only when the implementation removes or privatizes it and a stale
symbol test proves that it is no longer public.

The denominator includes methods and properties on `PinchProblem`,
`PinchWorkspace`, their accessors, and application-owned objects deliberately
returned for continued interaction, such as the process-MVR component and HEN
design view. It does not require a tutorial cell for every field on immutable
Pydantic result records; tutorials must instead interpret the representative
engineering outputs identified by each workflow contract.

Coverage applies to each operation and each documented semantic mode that
changes workflow behavior. It does not require the Cartesian product of every
keyword argument.

## Coverage Status

- **Covered**: an existing notebook calls or inspects the operation in a
  meaningful engineering workflow.
- **Partial**: a related workflow appears, but the operation is dynamic,
  guarded, private, stale, or not interpreted.
- **Missing**: no existing notebook exercises the operation.
- **Replace**: preserve the functionality through the listed canonical surface
  and teach only that surface.
- **Retire**: remove or privatize redundant, ambiguous, or non-process-engineer
  functionality; do not create tutorial coverage for the old spelling.

## Current Tutorial Baseline

The live callable/property inventory contains 66 named
`PinchProblem`-accessible members
(26 direct problem members, 11 target members, 6 design methods, 4
selected-network helpers, 1 component entry, and 18 plot members), 37 named
`PinchWorkspace` members, and 7 supported interactions on the returned
process-MVR component. Constructors and mutable data attributes such as
`results_dir` are tracked separately. These counts include duplicates and
operations marked for replacement or retirement, so they are an audit total
rather than the final coverage denominator.

| Existing tutorial | Material currently exercised | Important gaps or problems |
|---|---|---|
| 01 First Solve | construction, validation, generic targeting, area/cost, summary, graph catalog, CC, SCC, GCC, workspace copying/comparison | ambiguous target callable, stale area/cost name, scenario material overloads the first solve |
| 02 Total Site | workspace case selection, generic targeting, cogeneration, GCC, Total Site profiles, SUGCC, graph data | private scalar resolver, generic target prerequisite, direct and Total Site operations are not explicit |
| 03 Multiperiod | period IDs, current all-period replay, direct/indirect targeting, summaries, serialization, master-zone weights | obsolete replay surface, private result scans, no heat-pump, cogeneration, or synthesis multiperiod workflow |
| 04 Heat Pump | direct and indirect heat-pump dispatch, config mutation, HPR plots, case comparison | dynamic dispatch, internal enums, private scalar resolver, broad exception handling |
| 05 Process MVR | component addition, activation, direct/indirect retargeting, component inspection, GCC, case comparison | old component namespace, internal zone enum, manual copying |
| 06 VC and MVR | direct heat-pump targeting, HPR plots, config mutation | deep analysis imports, internal thermodynamic classes, important target cells disabled by a flag |
| 07 HEN Synthesis | main synthesis entry, selected-network utility, result inspection | private presentation import, stale selection methods, no advanced method comparison or multiperiod synthesis |
| 08 Energy Transfer | energy-transfer target, catalog, energy-transfer diagram | unnecessary sample copying and temporary directory |
| 09 Schema and Bundles | validation, target, summary, graph and Excel export, variant CRUD/solve/compare, bundle save/load | retiring service, concrete contracts import, frontend variant vocabulary, no generic reports/dashboard/gallery |
| 10 Multiperiod HPR | direct heat-pump result with shared period detail and weighted summaries | internal enum and scalar resolver, no explicit all-period HPR replay surface |

The current suite is therefore not a valid percentage baseline: seven notebooks
have confirmed runtime failures, several calls are obsolete, and guarded or
dynamic code cannot count as executable coverage.

## Live-to-Canonical PinchProblem Map

| Feature | Live surface | Existing coverage | Canonical disposition | Tutorial owner |
|---|---|---|---|---|
| Construct and reload | constructor, `load` | constructor covered in 01, 03, 07, 08, 09, 10; `load` missing | Keep both | 01 and 17 |
| Validate | `validate`, `validation_report` | `validate` in 01, 07, 09; report missing | Keep both | 01 |
| Configuration | `update_options`, `set_dt_cont_multiplier`; no public `config` | both covered indirectly, but only with flat keys | Keep mutation methods, add read-only `config`, use friendly kwargs | 04 |
| Input serialization | `from_json`, `to_problem_json`, `canonical_problem_json` | only `to_problem_json` in 03 | Keep constructor plus `to_problem_json`; retire both redundant alternatives | 17 |
| Identity and source | mutable `project_name`, `problem_filepath`, `problem_data` | `project_name` in 07; others missing | Keep read-only inspection; set the project name through construction/load rather than property assignment | 01 and 17 |
| Prepared model | `master_zone`, `hot_streams`, `cold_streams`, `hot_utilities`, `cold_utilities`, `process_components` | `master_zone` and components partial in 03, 05, 06, 07; collection properties missing | Keep as advanced read-only inspection | 03 and 10 |
| Period state | `period_ids`, `target_all_periods` | both in 03; period IDs in 10 | Keep `period_ids`; replace replay with mirrored `target.all_periods.*` methods and explicit `period_results` | 06, 10, 13, and 17 |
| Bulk targeting | callable `target()` | heavily used in 01, 02, 04, 09 | Replace with `target.all_heat_integration`; retire callable behavior | 01 |
| Focused heat integration | `target.direct_heat_integration`, `target.indirect_heat_integration` | 03 and 05 | Keep, add outcome-oriented `target.total_site_heat_integration` peer | 02 |
| Heat pumps | `target.direct_heat_pump`, `target.indirect_heat_pump` | 04, 06, 10 | Replace with Carnot, vapour-compression, Brayton, and MVR callables using binary placement/topology flags only where applicable | 08 through 11 |
| Refrigeration | `target.direct_refrigeration`, `target.indirect_refrigeration` | missing | Replace with Carnot, vapour-compression, and Brayton callables; MVR refrigeration remains unsupported | 08 and 09 |
| Cogeneration | `target.cogeneration` | selected-period use in 02 | Keep the default correlation and add specialized correlation/isentropic callables; pass base-target objects rather than type strings | 12 and 13 |
| Area and cost | `target.area_cost` | 01 | Rename to `target.heat_exchanger_area_and_cost` | 07 |
| Exergy | `target.exergy` | missing | Keep | 07 |
| Energy transfer | `target.energy_transfer` | 08 | Keep | 13 |
| Process MVR | `add_component.process_mvr`, `process_components` | 05 | Rename namespace to `components.add_process_mvr`; retain read-only inventory | 10 |
| Process-MVR lifecycle and inspection | returned component `activate`, `deactivate`, `original_streams`, `replacement_streams`, `stage_results_by_period`, `affected_zone_paths`, `work_for_zone` | activation, stream replacement, settings, and period stages partial in 05 | Keep as the supported application-owned component interaction | 10 |
| Main HEN synthesis | `design.heat_exchanger_network_synthesis` | 07 | Rename to `design.heat_exchanger_network` | 14 and 16 |
| Enhanced HEN synthesis | `design.enhanced_synthesis_method` | missing | Rename to `design.enhanced_heat_exchanger_network` | 16 |
| Advanced HEN methods | `design.open_hens_method`, `pinch_design_method`, `thermal_derivative_method`, `network_evolution_method` | missing | Remove `_method` suffixes and keep all four | 15 |
| Selected HEN metrics | `design.network.total_heat_recovery`, `total_hot_utility`, `total_cold_utility`, `utility` | only `utility` in 07 | Keep on the application-owned design view | 14 and 16 |
| Ranked HEN selection | schema methods and private grid service | stale/private use in 07 | Add application-owned `top`, `network`, and `grid` operations | 14 and 16 |
| Result state | `results` | inspected in 07 | Keep read-only; add explicit `period_results` | 01, 06, 09, 12, and 16 |
| Summary and comparison | `summary_frame`, `metrics`, `report`, `compare_to` | summary common; metrics, report, and direct comparison missing | Keep, remove implicit execution | 01, 04, and 17 |
| Plot inventory and data | callable `plot()`, `plot.catalog`, `plot.get_graph_data` | callable missing; catalog common; graph data in 02 | Retire callable, keep `catalog`, rename graph data to `data` | 01 and 17 |
| Base heat-integration plots | CC, SCC, BCC, GCC, real GCC | CC, SCC, GCC in 01; BCC and real GCC missing | Keep all named methods | 01 |
| Exergy plots | exergetic GCC and exergetic net-load profiles | missing | Keep both | 07 |
| HPR plots | GCC with heat pump, net-load profiles, net-load profiles with heat pump | heat-pump variants in 04 and 06; base net-load missing | Keep all three | 08 |
| Total Site plots | Total Site profiles and SUGCC | 02 | Keep both | 02 |
| Energy-transfer plot | energy-transfer diagram | 08 | Keep | 13 |
| Plot files | `plot.export`, `plot.export_gallery` | export in 09; gallery missing | Keep both as explicit side effects | 17 |
| Other outputs | `export_excel`, `show_dashboard`, mutable `results_dir` | Excel in 09; dashboard missing | Keep explicit methods with destination arguments; retire mutable directory state | 17 |

## Live-to-Canonical PinchWorkspace Map

| Feature | Live surface | Existing coverage | Canonical disposition | Tutorial owner |
|---|---|---|---|---|
| Construct and load | constructor, `from_json`, `load`, `load_bundle` | constructor common; bundle load in 09; direct load and `from_json` missing | Keep constructor, `load`, and `load_bundle`; retire redundant `from_json` | 04 and 05 |
| Workspace identity | mutable `project_name`/`baseline_name`, `active_case_name`, `list_cases`, `list_variants` | list methods in 02, 05, 09; identity properties missing | Keep read-only case identity, selection, and case vocabulary; retire `list_variants` | 04 |
| Case access | `case`, `use_case`, `copy_case` | all covered across 02, 04, 05, 06 | Keep `case` and `use_case`; privatize copying behind `scenario` | 04 |
| Scenario creation | `scenario`; manual `copy_case` plus mutation | `scenario` missing | Keep `scenario` as canonical | 04 |
| Multi-case execution | `solve_variant`; no case-oriented batch method | variant solve in 09 | Replace with `cases(names)` returning mirrored target/design/report/export accessors | 04 |
| Case comparison | `compare_cases`, `compare_to`, `compare_variants` | pair comparison common; variant comparison in 09; `compare_to` missing | Keep `compare_cases` and `compare_to`; retire variant comparison | 04 and 05 |
| Case input | `get_case_input`, `to_problem_json`, `get_variant_input`, `set_variant_input`, `input_view` | variant setter/view in 09; canonical case methods missing | Keep `to_problem_json`; use `load(..., case_name=...)` for replacement; retire duplicate and frontend variant surfaces | 05 |
| Case validation | `validate`, `validation_report`, `validate_variant` | only variant validation in 09 | Keep case-oriented methods; retire variant spelling | 05 |
| Active-case targeting and plotting | `target`, `plot` | used indirectly through returned cases, not workspace forwarding | Keep forwarding and teach when active-case shorthand is appropriate | 04 |
| Active-case state | `problem_data`, `problem_filepath`, `results`, `master_zone` | missing as workspace properties | Keep read-only forwarding | 05 |
| Active-case reporting | `summary_frame`, `metrics`, `report` | summary in 01 and 05; metrics/report missing | Keep forwarding with no hidden solve | 05 and 17 |
| Active-case outputs | `export_excel`, `show_dashboard` | missing | Keep explicit forwarding | 17 |
| Active-case configuration | `set_dt_cont_multiplier`, `update_options` | both covered indirectly | Keep forwarding with predictable invalidation | 04 |
| Workspace persistence | `save_bundle`, `load_bundle` | both in 09 | Keep using case vocabulary in outputs | 05 |
| Configuration metadata | `configuration_field_metadata` | missing | Retire from the process-engineer surface with other frontend variant APIs | removal tests only |

## Canonical 100 Percent Coverage Manifest

Every operation below must be present in executable notebook code or an
explicitly guarded interactive cell. The coverage test must compare this
manifest with the live canonical inventory.

### PinchProblem

| Canonical public operations | Primary tutorial |
|---|---|
| constructor, `load`, `validate`, `validation_report`, `project_name` | 01 First Solve and Core Curves |
| `target.all_heat_integration`, `results`, `summary_frame`, `metrics`, `report` | 01 First Solve and Core Curves |
| `plot.composite_curve`, `shifted_composite_curve`, `balanced_composite_curve`, `grand_composite_curve`, `real_grand_composite_curve` | 01 First Solve and Core Curves |
| `target.direct_heat_integration`, `indirect_heat_integration`, `total_site_heat_integration` | 02 Focused Direct and Total Site |
| `plot.total_site_profiles`, `site_utility_grand_composite_curve` | 02 Focused Direct and Total Site |
| `hot_streams`, `cold_streams`, `hot_utilities`, `cold_utilities`, `master_zone` | 03 Multi-Segment Streams |
| read-only `config`, `update_options`, `set_dt_cont_multiplier`, `compare_to` | 04 Workspace Cases and Scenarios |
| `period_ids`, `period_results`, `target.all_periods.all_heat_integration` | 06 Multiperiod Heat Integration |
| `target.heat_exchanger_area_and_cost`, `target.exergy` | 07 Area, Cost, and Exergy |
| `plot.exergetic_grand_composite_curve`, `exergetic_net_load_profiles` | 07 Area, Cost, and Exergy |
| `target.carnot_heat_pump`, `target.carnot_refrigeration`, named load forms, placement/topology flags, and HPR plots | 08 Carnot Heat Pump and Refrigeration |
| `target.vapour_compression_heat_pump`, `vapour_compression_refrigeration`, `brayton_heat_pump`, `brayton_refrigeration` | 09 Vapour Compression and Brayton HPR |
| supported mirrored `target.all_periods.*` HPR methods and weighted/shared HPR results | 10 Multiperiod Heat Pumps |
| `components.add_process_mvr`, `process_components`, and `target.mvr_heat_pump` | 11 Process MVR and VC Cascade |
| returned process-MVR component `activate`, `deactivate`, stream collections, period stage results, affected zones, and zone work | 11 Process MVR and VC Cascade |
| default and specialized cogeneration methods | 12 Cogeneration |
| mirrored `target.all_periods.*` cogeneration methods | 13 Multiperiod Cogeneration |
| `target.energy_transfer`, `plot.energy_transfer_diagram` | 14 Energy Transfer |
| `design.heat_exchanger_network`, `top`, `network`, `grid` | 15 HEN Synthesis and Selection |
| selected-network heat recovery, hot utility, cold utility, and named utility metrics | 15 HEN Synthesis and Selection |
| `design.enhanced_heat_exchanger_network`, `open_hens`, `pinch_design`, `thermal_derivative`, `network_evolution` | 16 Advanced HEN Methods |
| `design.multiperiod_heat_exchanger_network`, `period_results`, shared-design ranking and metrics | 17 Multiperiod HEN Synthesis |
| `problem_filepath`, `problem_data`, `to_problem_json`, `plot.catalog`, `plot.data`, callable-selected plot exports | 18 Results, Reports, Plots, and Exports |
| `export_excel`, guarded `show_dashboard`, no-hidden-execution behavior | 18 Results, Reports, Plots, and Exports |

### PinchWorkspace

| Canonical public operations | Primary tutorial |
|---|---|
| constructor, `load`, `project_name`, `baseline_name`, `active_case_name`, `list_cases` | 04 Workspace Cases and Scenarios |
| `case`, `use_case`, `scenario`, `cases`, `compare_cases`, `compare_to` | 04 Workspace Cases and Scenarios |
| active-case `target`, `plot`, `set_dt_cont_multiplier`, `update_options` | 04 Workspace Cases and Scenarios |
| `load_bundle`, `save_bundle`, `to_problem_json`, `validate`, `validation_report` | 05 Workspace Data and Persistence |
| active-case `problem_data`, `problem_filepath`, `results`, `master_zone` | 05 Workspace Data and Persistence |
| active-case `summary_frame`, `metrics`, `report` | 05 Workspace Data and Persistence |
| active-case `export_excel`, `show_dashboard` | 18 Results, Reports, Plots, and Exports |

### Required Semantic Modes

| Behavior dimension | Required tutorial evidence |
|---|---|
| Input sources | packaged sample name and JSON path in 01/18; in-memory mapping and multi-segment data in 03; spreadsheet/CSV source pair in 03 or 18 |
| Validation | raising `validate` and non-raising `validation_report` in both problem and workspace contexts |
| Zone scope | default/root selection, explicit `zone`, and `include_subzones` in focused targeting tutorials |
| Configuration precedence | method kwargs, advanced `options`, stored config, and default fallback, including falsey and nullable values without persistent mutation |
| Heat-integration scope | focused direct, focused indirect/Total Site, dependency-aware all-zone traversal, selected period, and all periods |
| HPR model and placement | Carnot, vapour-compression, Brayton, and MVR heat-pump families; supported refrigeration families; process and utility placement; cascade and parallel topology where applicable |
| HPR loading | fraction, duty, and period mapping; config fallback; conflicting forms rejected |
| Period aggregation | selected, all, weighted average, and all-plus-weighted summaries with stable ordering |
| Workspace selection | active-case shorthand and explicit named-case access |
| Workspace execution | one case and a named batch view, success and structured per-case failure, deterministic comparison, no workflow strings |
| Component lifecycle | MVR creation, activation, deactivation, inspection, invalidation, and retargeting |
| HEN synthesis | default method, quality tier, each advanced method, seeded method input, ranked selection, selected metrics, and multiperiod shared design |
| Plot selection | catalog and serialized data, figure and raw-data return, zone/method/index selection, callable-selected file export, and gallery export |
| Output side effects | explicit Excel/plot/gallery paths and an opt-in dashboard cell that never blocks automated execution |
| State behavior | prepared, targeted, designed, and invalidated states; read/report/plot/export operations never execute targeting or design |

## Tutorial Allocation

The ten existing notebooks are rewritten and redistributed, and eight focused
notebooks are added. Existing notebook numbers are not treated as compatibility
contracts; final names follow the learning progression.

| Number | Tutorial | Source decision |
|---|---|---|
| 01 | First Solve and Core Curves | rewrite existing 01 |
| 02 | Focused Direct and Total Site | rewrite existing 02 |
| 03 | Multi-Segment Streams | new |
| 04 | Workspace Cases and Scenarios | consolidate scenario material from existing 01, 03, 04, and 05 |
| 05 | Workspace Data and Persistence | retain useful bundle material from existing 09; remove service/variant teaching |
| 06 | Multiperiod Heat Integration | rewrite existing 03 |
| 07 | Area, Cost, and Exergy | split area/cost from existing 01 and add exergy |
| 08 | Carnot Heat Pump and Refrigeration | rewrite selected material from existing 04 and add refrigeration |
| 09 | Vapour Compression and Brayton HPR | new focused model-family tutorial using material from existing 06 |
| 10 | Multiperiod Heat Pumps | rewrite existing 10 |
| 11 | Process MVR and VC Cascade | combine supported material from existing 05 and 06 |
| 12 | Cogeneration | extract and expand the cogeneration section from existing 02 |
| 13 | Multiperiod Cogeneration | new |
| 14 | Energy Transfer | rewrite existing 08 |
| 15 | HEN Synthesis and Selection | rewrite existing 07 |
| 16 | Advanced HEN Methods | new |
| 17 | Multiperiod HEN Synthesis | new |
| 18 | Results, Reports, Plots, and Exports | rewrite non-service output material from existing 09 |

## Enforcement

1. Generate the canonical public inventory from `PinchProblem`,
   `PinchWorkspace`, their target, component, design, selected-network, and plot
   accessors, and supported application-owned return views.
2. Store exact operation names, tutorial owners, execution profiles, and
   coverage modes in a machine-readable manifest.
3. Parse tutorial code with AST checks. A call or property reference counts only
   when it appears in an executable cell; Markdown-only mentions do not count.
4. Require every canonical operation to have at least one tutorial owner and
   every manifest owner to exist.
5. Require retiring operations to be absent from the live public inventory and
   all tutorials before excluding them from the denominator.
6. Execute base, slow HPR, solver-backed HEN, and guarded interactive profiles
   honestly. A skipped environment never reports coverage as executed.
7. Report both operation coverage and notebook execution coverage. Acceptance
   requires 100 percent for both.

## Read the Docs Publication Contract

- Publish the user-facing map as
  `docs/examples/tutorial-coverage-map.rst` under the Examples section.
- Store the canonical table rows in `docs/_data/tutorial-coverage.csv` with one
  record per supported operation or semantic mode. Required columns are owner,
  operation, classification, semantic mode, primary tutorial, secondary
  tutorials, execution profile, optional dependency, and coverage status.
- Render the RTD table directly from that CSV with Sphinx `csv-table`; do not
  maintain a second hand-copied operation table in RST.
- Link the page from `docs/examples/index.rst`, the notebook-series page,
  `docs/api/pinchproblem.rst`, `docs/api/pinchworkspace.rst`, and the overview
  capability matrix.
- Explain the denominator, replacement/retirement policy, semantic-mode rules,
  and base/slow/solver/interactive execution profiles on the RTD page.
- Link every tutorial owner to its notebook-series entry and every API owner to
  the relevant public API page.
- Generate the same 100 percent coverage result in CI and Sphinx. RTD must fail
  if the CSV contains a stale operation, references a missing tutorial, omits a
  supported operation/mode, or reports a skipped profile as executed.
- Keep internal planning commentary and historical live-surface counts out of
  the public page. RTD shows the released canonical surface, coverage policy,
  current coverage result, and any deliberately unavailable optional profile.

## Extension Compliance

- **Security Baseline**: Disabled; N/A to the tutorial coverage map.
- **Resiliency Baseline**: Disabled; N/A to the tutorial coverage map.
- **Partial Property-Based Testing**: Applicable during construction to public
  inventory drift, period replay, multi-segment invariants, and deterministic
  tutorial-manifest resolution.
