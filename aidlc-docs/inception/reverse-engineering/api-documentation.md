# API Documentation

## REST APIs

No REST or HTTP API is implemented. OpenPinch is an in-process Python library. Applications that need a service boundary use the typed `pinch_analysis_service` function and can wrap it in their own transport.

## Curated Package API

The package root exports the supported high-level surface:

- `PinchProblem(source=None, project_name="Site")` - load, validate, target, inspect, plot, and export one study.
- `PinchWorkspace(source=None, project_name="Site", baseline_name="baseline")` - manage named baseline and scenario cases.
- `pinch_analysis_service(input_data, project_name="Site") -> TargetOutput` - typed, stateless request-to-response boundary.
- `TargetInput`, `TargetOutput`, `StreamSchema`, `UtilitySchema`, and `ZoneTreeSchema` - typed I/O contracts.
- `Configuration` and `config_options()` - runtime options and option metadata.
- `list_sample_cases`, `read_sample_case`, `copy_sample_case`, and sample metadata functions - packaged case discovery.
- `list_notebooks`, `copy_notebook`, and notebook metadata functions - packaged notebook discovery.
- `get_piecewise_linearisation_for_streams` - nonlinear-stream preprocessing utility.
- Public enums and report models, including `TargetType`, `GraphType`, `HeatPumpAndRefrigerationCycle`, `ProblemReport`, and `ValidationReport`.

## `PinchProblem` API Families

### Lifecycle and validation

- `load(source) -> Zone | None` - load JSON, CSV, Excel, schema, mapping, tuple, or packaged sample input.
- `validation_report() -> ValidationReport` - validate source semantics without requiring a successful solve.
- `period_ids` and period-aware execution helpers - expose multiperiod study identifiers.

### Targeting

- `target()` - default direct heat-integration targeting.
- `target.direct_heat_integration(...)`
- `target.indirect_heat_integration(...)`
- `target.direct_heat_pump(...)` and `target.indirect_heat_pump(...)`
- `target.direct_refrigeration(...)` and `target.indirect_refrigeration(...)`
- `target.cogeneration(...)`, `target.exergy(...)`, and related advanced target accessors.
- `target_all_periods(...)` - replay a target workflow across operating periods, with serial, thread, or process execution backends.

### Components and design

- `add_component.process_mvr(...)` - add direct gas or vapour MVR modifications.
- `design.enhanced_synthesis_method(...)` - execute HEN synthesis and expose ranked candidates.

### Reporting and presentation

- `summary_frame(...)` - produce pandas summaries.
- `report(...)` and graph-data helpers - expose serializable reporting views.
- `plot.*` and `plot.export(...)` - build or export graph families.
- Excel and HTML export methods - persist summaries and visualizations.
- `show_dashboard()` - render the Streamlit result viewer.

## `PinchWorkspace` API Families

- `load`, `set_variant_input`, `get_variant_input`, and `input_view` - manage normalized case inputs.
- `list_variants`, `copy_case`, `scenario`, `case`, and active-case helpers - manage named variants.
- `validate_variant` and `validation_report` - produce structured validation results.
- `solve_variant(workflow=..., workflow_options=...)` - execute a configured case workflow and return a serializable view.
- `compare_cases` and `compare_to` calculate metric and problem-table deltas.
- `save_bundle` and `load_bundle` - persist scenario inputs, workflow metadata, and cached views as JSON.

## Internal Service APIs

`OpenPinch.services.services_entry` exposes zone-oriented functions used by the public accessors:

- `data_preprocessing_service(input_data, project_name)`
- `direct_heat_integration_service(zone, args=None)`
- `indirect_heat_integration_service(zone, args=None)`
- `direct_heat_pump_service(zone, args=None)`
- `indirect_heat_pump_service(zone, args=None)`
- `direct_refrigeration_service(zone, args=None)`
- `indirect_refrigeration_service(zone, args=None)`
- `power_cogeneration_service(zone, args=None)`
- `exergy_targeting_service(zone, args=None)`
- `area_cost_targeting_service(zone, args=None)`
- `energy_transfer_analysis_service(zone, args=None)`

These functions mutate or enrich a prepared `Zone` graph and return the zone. They are lower-level extension points rather than the preferred first-use API.

## CLI API

`openpinch notebook [--name NAME] [-o OUTPUT]` copies one packaged notebook or the ordered notebook series. `--debug` preserves tracebacks. Solving and exporting are intentionally Python-only.

## Data Models

### Input contracts

- `StreamSchema`: zone, name, supply and target states, heat flow, optional heat-capacity flow, contribution temperatures, heat-transfer coefficient, fluid metadata, and active flag.
- `UtilitySchema`: utility type, thermal states, optional duty, cost, contribution temperature, heat-transfer coefficient, fluid metadata, and active flag.
- `ZoneTreeSchema`: recursive name, type, optional contribution-temperature multiplier, and children.
- `TargetInput`: streams, utilities, options, and optional zone tree. Flat options are validated against the configuration catalog.

### Output contracts

- `TargetOutput`: study name, optional period identifier, target-result list, graph sets, and optional HEN design.
- `TargetResults`, `ReportMetric`, `ProblemReport`, and graph schemas: reporting-oriented thermal, economic, and presentation data.
- `HeatExchangerNetworkSynthesisResult` and related task, outcome, manifest, and export schemas: solver workflow inputs and design outputs.
- HPR schemas: targeting inputs, period cases, parsed optimizer state, thermodynamic artifacts, backend outcomes, and cost accounting.
- Workspace schemas: validation issues, tables, cards, graph catalogs, variant views, deltas, comparisons, and persisted bundles.

### Runtime domain models

- `Value`: numeric or period-valued data with Pint units.
- `Stream` and `StreamCollection`: process and utility stream behavior plus cached numeric views.
- `Zone`: hierarchical streams, utilities, targets, and subzones.
- `ProblemTable`: shifted-temperature intervals and heat balances.
- `HeatExchanger` and `HeatExchangerNetwork`: network structure and reporting models.
- Runtime target subclasses: utility-summary, exergy, cogeneration, HPR, and other specialized target families.

## Validation and Error Behavior

- Pydantic validates structural request and response contracts.
- Configuration metadata validates supported option keys and values.
- `ValidationReport` preserves semantic input issues for application-facing workflows.
- Optional dependency helpers raise targeted installation guidance when advanced surfaces are unavailable.
- `PinchWorkspace.solve_variant` converts expected workflow errors and unexpected exceptions into structured error views; direct `PinchProblem` calls generally raise exceptions.
