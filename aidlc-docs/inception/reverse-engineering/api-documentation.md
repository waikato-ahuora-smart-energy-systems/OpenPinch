# API Documentation

## Public Workflow Boundary

OpenPinch is an in-process Python library and has no REST or HTTP API. The
package root exports exactly:

```python
from OpenPinch import PinchProblem, PinchWorkspace
```

- `PinchProblem(source=None, project_name="Site")` owns one study.
- `PinchWorkspace(source=None, project_name="Site", baseline_name="baseline")`
  owns named cases and delegates the active case's problem surface.

Contracts, domain classes, resources, and specialist services are imported from
their concrete modules when advanced integrations need them; they are not added
to the root namespace.

## `PinchProblem` Families

### Lifecycle and observation

- `load(source)`, `validate()`, and `validation_report()` load or inspect input.
- `problem_data` returns a detached snapshot; `to_problem_json()` returns the
  canonical serialized input.
- `period_ids`, `config`, `master_zone`, `results`, stream/utility collections,
  and `process_components` expose current state without choosing an analysis.
- `update_options(...)` and `set_dt_cont_multiplier(...)` are explicit mutations
  that invalidate dependent state.

### Targeting

Core named methods include:

- `problem.target.direct_heat_integration(...)`
- `problem.target.indirect_heat_integration(...)`
- `problem.target.total_site_heat_integration(...)`
- `problem.target.all_heat_integration(...)`
- `problem.target.heat_exchanger_area_and_cost(...)`

Advanced named methods include Carnot and vapour-compression heat pumps and
refrigeration, Brayton heat pump/refrigeration, MVR heat pump, cogeneration and
its named methods, exergy, and energy transfer. The corresponding supported
multiperiod operations are under `problem.target.all_periods` with an explicit
`workers` argument.

Arguments supplied directly to a method take precedence. When an argument is
omitted, the method resolves the corresponding current configuration value.
Configuration supplies defaults; it does not select which core analysis runs.

### Components

`problem.components.add_process_mvr(...)` adds a process MVR component and
invalidates affected results. `problem.components.inventory` is a read-only
mapping of current components.

### HEN design

Named methods under `problem.design` cover the base heat-exchanger-network
workflow, enhanced and multiperiod variants, OpenHENS, pinch design,
thermal-derivative design, and network evolution.

They return an explicit `HeatExchangerNetworkDesignView` with `result`,
`selected_network`, `top(n)`, `network(rank=...)`, `grid(...)`, heat/utility
totals, and `utility(...)`. Serialize through
`design.result.model_dump(mode="json")`.

### Reporting and presentation

- `summary_frame(...)`, `metrics(...)`, `report(...)`, and `compare_to(...)`
  observe current results without hidden execution.
- `problem.plot` exposes named graph operations and exports.
- `export_excel(...)` writes a report workbook.
- `show_dashboard(...)` renders the optional interactive presentation.

## `PinchWorkspace` Families

- `list_cases()`, `case(name)`, and `use_case(name)` select existing cases.
- `scenario(name, ...)` creates an unsolved named case from a base case.
- `cases(names)` returns an ordered batch surface mirroring supported target and
  design operations plus summaries, metrics, reports, and Excel export.
- `target`, `design`, `components`, `plot`, configuration, state, reporting, and
  export members delegate to the active `PinchProblem`.
- `compare_to(...)` and `compare_cases(...)` compare solved cases.
- `save_bundle(path)` and `load_bundle(path)` persist explicit
  `schema_version: "3"` case inputs without migrations.

Workspace case identifiers are non-empty, trimmed, portable path components.
They are rejected rather than normalized, and batch exports enforce resolved
destination containment.

## Contract and Runtime Owners

- `OpenPinch.contracts.input.TargetInput` owns stream, utility, zone, nonlinear
  segment, and serialized HEN input contracts.
- `OpenPinch.contracts.output.TargetOutput` owns primary serialized results.
- `OpenPinch.contracts.workspace.PinchWorkspaceBundle` owns workspace bundle
  schema version `3`.
- `OpenPinch.domain` owns runtime `Value`, `Stream`, `Zone`, target,
  `HeatExchanger`, and `HeatExchangerNetwork` behavior.

The supported HEN transport bridge is a mapping, not a JSON string:

```python
network_payload = network.model_dump(mode="json")
input_data = TargetInput.model_validate({"streams": [], "network": network_payload})
```

Private solver and source metadata excluded by the runtime dump are rejected by
the transport schemas.

## CLI Boundary

`openpinch notebook [--name NAME] [-o OUTPUT]` copies one packaged notebook or
the ordered notebook series. Analysis selection and export remain Python
workflow operations.

## Error Behavior

- Pydantic rejects unknown and invalid wire fields.
- Semantic validation is available through `ValidationReport`.
- Unloaded operations raise actionable runtime errors.
- Optional features raise targeted installation guidance.
- Batch operations isolate per-case exceptions in `CaseBatchResult`.
- Solver-backed design methods expose typed synthesis outcomes and do not hide
  missing solver capabilities.
