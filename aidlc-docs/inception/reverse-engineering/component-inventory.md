# Component Inventory

## Application Packages

- `OpenPinch.classes` - public orchestration and core domain objects.
- `OpenPinch.services.direct_heat_integration` - direct pinch targeting.
- `OpenPinch.services.indirect_heat_integration` - Total Site and indirect targeting.
- `OpenPinch.services.energy_transfer_analysis` - energy-transfer diagrams and surplus/deficit analysis.
- `OpenPinch.services.exergy_analysis` - exergy enrichment.
- `OpenPinch.services.power_cogeneration` - turbine and cogeneration targeting.
- `OpenPinch.services.components` - process-component and MVR transformations.
- `OpenPinch.services.heat_pump_integration` - HPR targeting and unit models.
- `OpenPinch.services.heat_exchanger_network_synthesis` - HEN task orchestration, models, execution, verification, and reporting.
- `OpenPinch.services.heat_exchanger_network_controllability` - controllability models and assessment.
- `OpenPinch.services.network_grid_diagram` - network diagram construction and rendering.
- `OpenPinch.streamlit_webviewer` - interactive result presentation.
- `OpenPinch.main` - typed service facade.
- `OpenPinch.__main__` - notebook-copying CLI.

## Infrastructure Packages

- None. No Terraform, CloudFormation, CDK, container, serverless, or database package is present.

## Shared Packages

- `OpenPinch.lib` - configuration, enums, units, schemas, and shared types.
- `OpenPinch.services.common` - reusable analysis operations and orchestration.
- `OpenPinch.services.input_data_processing` - canonicalization and domain construction.
- `OpenPinch.utils` - file adapters, exports, validation, costing, optimization utilities, and optional dependency checks.
- `OpenPinch.resources` and `OpenPinch.data` - packaged notebooks and sample cases.

## Test Packages

- `tests.e2e` - end-to-end service workflows.
- `tests.test_analysis` - targeting and service behavior.
- `tests.test_classes` - domain and orchestration behavior.
- `tests.heat_exchanger_network_synthesis` - HEN contracts, helpers, reporting, and sequence tests.
- `tests.test_lib` and `tests.test_utils` - schema, configuration, unit, adapter, and utility tests.
- Root-level test modules - packaging, CLI, docs, notebooks, optional boundaries, release, resource, and parity checks.
- `tests.test_streamlit_webviewer` - dashboard graphing tests.

## Repository Support Components

- `docs` - Sphinx documentation, guides, API reference, and developer material.
- `scripts` - docs/build/release helpers, performance benchmarks, fixture conversion, and OpenHENS comparisons.
- `examples` - JSON and legacy Excel example studies.
- `.github/workflows` - pull-request, develop-branch, and tag-publish automation.
- `Excel_Version` - legacy workbook implementation retained alongside the Python package.

## Total Count

- **Tracked repository files**: 774 at analysis time.
- **Tracked package files**: 213 under `OpenPinch`, including Python modules, notebooks, and sample cases.
- **Tracked tests and fixtures**: 271 paths under `tests`.
- **Python import packages**: 39 directories identified by `__init__.py`.
- **Application/service package families**: 14.
- **Shared package families**: 5.
- **Infrastructure packages**: 0.
- **Primary test areas**: 7 plus root-level repository contract tests.

