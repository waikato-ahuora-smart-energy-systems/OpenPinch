# Component Inventory

## Package Components

| Component | Current responsibility |
|---|---|
| `OpenPinch.application` | `PinchProblem`, `PinchWorkspace`, accessors, case batches, lifecycle, and workflow coordination |
| `OpenPinch.domain` | configuration, units-aware values, streams, zones, targets, problem tables, exchangers, and networks |
| `OpenPinch.contracts` | strict input/output, reporting, graph, HPR, workspace, turbine, controllability, and synthesis schemas |
| `OpenPinch.analysis.targeting` | direct, indirect, Total Site, area/cost, and shared targeting operations |
| `OpenPinch.analysis.heat_pumps` | Carnot, vapour-compression, Brayton, MVR, refrigeration, and multiperiod HPR workflows |
| `OpenPinch.analysis.power` | cogeneration and turbine analysis |
| `OpenPinch.analysis.exergy` | exergy targeting and enrichment |
| `OpenPinch.analysis.energy_transfer` | transfer opportunity analysis and diagrams |
| `OpenPinch.analysis.heat_exchanger_networks` | HEN targeting, synthesis models, execution, verification, ranking, and results |
| `OpenPinch.analysis.graphs` | graph metadata and calculation specifications |
| `OpenPinch.analysis.thermodynamics` | fluid-property and cycle thermodynamics |
| `OpenPinch.optimisation` | shared candidates, models, execution, services, errors, and backend adapters |
| `OpenPinch.adapters.io` | JSON, CSV, Excel, workspace bundle, and legacy input boundaries |
| `OpenPinch.adapters.optional_dependencies` | optional feature installation guards |
| `OpenPinch.presentation.reporting` | summary/report frames and workbook/file reporting |
| `OpenPinch.presentation.graphs` | graph construction, rendering, and gallery export |
| `OpenPinch.presentation.dashboard` | optional Streamlit presentation |
| `OpenPinch.presentation.network_grid` | HEN grid views and renderers |
| `OpenPinch.resources` and `OpenPinch.data` | packaged sample cases and tutorial notebooks |
| `OpenPinch.__main__` | notebook-copying command-line entry point |

## Repository Components

- `tests/architecture` enforces owner dependencies and the root API boundary.
- `tests/application`, `tests/domain`, `tests/contracts`, `tests/analysis`, and
  `tests/presentation` verify the corresponding owners.
- `tests/packaging` verifies docs, tutorials, resources, release artifacts,
  entry points, and distribution metadata.
- `docs` contains user, API, example, and developer Sphinx sources.
- `scripts` contains documentation/distribution builders and bounded comparison
  or benchmark utilities.
- `examples` contains study inputs retained for package and legacy workflows.
- `.github/workflows` contains test and publishing automation.

## Infrastructure Components

None. The package has no application server, container, database, cloud
resource definition, or infrastructure-as-code module.

## Public Boundary

The package root intentionally exposes only `PinchProblem` and
`PinchWorkspace`. The component inventory describes concrete internal owners;
it does not imply root-level stability for every class or function within them.
