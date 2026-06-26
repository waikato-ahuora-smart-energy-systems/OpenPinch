# OpenPinch

OpenPinch is an open-source toolkit for advanced Pinch Analysis and Total Site Integration. It supports direct and indirect heat integration targeting, multi-utility studies, graph generation, Excel-based workflows, and programmatic analysis in Python.

## Install

Install the published package from PyPI for core Python usage:

```bash
python -m pip install openpinch
```

If you plan to run the packaged Jupyter notebooks, graph rendering, or Excel
I/O, install the notebook extra:

```bash
python -m pip install "openpinch[notebook]"
```

If you plan to launch the Streamlit dashboard, install the dashboard extra:

```bash
python -m pip install "openpinch[dashboard]"
```

If you need TESPy-backed Brayton-cycle tooling, install the Brayton-cycle
extra:

```bash
python -m pip install "openpinch[brayton_cycle]"
```

If you plan to run solver-backed heat-exchanger-network synthesis, install the
synthesis extra and then download the IDAES solver extensions:

```bash
python -m pip install "openpinch[synthesis]"
idaes get-extensions
```

If you want the full optional surface in one install:

```bash
python -m pip install "openpinch[full]"
```

OpenPinch currently requires Python `>=3.14.2`.


## Packaged Resources

OpenPinch ships with sample cases and a notebook series for distinct outputs
and workflows. Discover them from Python:

```python
from OpenPinch import (
    list_notebooks,
    list_sample_cases,
    notebook_metadata,
    sample_case_metadata,
)

print(list_sample_cases())
print(sample_case_metadata("basic_pinch.json").description)
print(list_notebooks())
print(notebook_metadata("01_basic_pinch_and_dtcont_sensitivity.ipynb").title)
```

Copy notebooks into your working directory with `copy_notebook(...)` or the
reference notebook-copy command `openpinch notebook -o notebooks`. To run the
packaged notebooks in Jupyter, install the notebook extra first with
`python -m pip install "openpinch[notebook]"`.

The packaged notebook series currently includes:

- `01_basic_pinch_and_dtcont_sensitivity.ipynb`
- `02_total_site_targets_and_sugcc.ipynb`
- `03_carnot_hpr_comparison.ipynb`
- `04_multiperiod_targeting_and_period_comparison.ipynb`
- `05_schema_service_and_output_workflows.ipynb`
- `06_energy_transfer_analysis.ipynb`
- `07_vapour_compression_mvr_cascade_hpr.ipynb`
- `08_direct_gas_stream_mvr.ipynb`
- `09_hen_design_service_four_stream.ipynb`

These notebooks are intended to be the main learning path for new users. The
series now spans the single-case `PinchProblem` front door, named
`PinchWorkspace` studies, real multiperiod targeting, the typed/service plus
serialized-workspace boundaries, energy-transfer analysis, and the simulated
heat pump targeting backend, direct gas/vapour process-component MVR, and the
heat exchanger network design service on a compact four-stream problem.


## Python Workflow

For script and notebook usage, the main single-case front door is
`PinchProblem`.

```python
from OpenPinch import PinchProblem

problem = PinchProblem("basic_pinch.json", project_name="basic_pinch")

validation = problem.validation_report()
result = problem.target()
summary = problem.summary_frame()
plain_summary = problem.summary_frame(format="plain")
report = problem.report()
print(summary)

problem.export_excel("results")
problem.plot.export("graphs", graph_type="gcc")
problem.plot.export_gallery("graph_gallery")
```

When the PinchProblem data contains multiperiod values, the named
`problem.target.*` entry points also accept `period_id=...` so one cached solve
can be refreshed for a selected operating period without flattening the
in-memory model first:

```python
multiperiod_problem = PinchProblem(
    "crude_preheat_train_multiperiod.json",
    project_name="crude_multiperiod",
)
selected_period = multiperiod_problem.target.direct_heat_integration(period_id="peak")
period_summary = multiperiod_problem.summary_frame()
print(period_summary[["Target", "Period ID", "Hot Utility Target", "Cold Utility Target"]])
```

For named study cases and bundle save/load, use `PinchWorkspace`:

```python
from OpenPinch import PinchWorkspace

workspace = PinchWorkspace(
    source="crude_preheat_train.json",
    project_name="crude_preheat_train",
)
workspace.scenario("wide_dt", dt_cont_multiplier=0.5)
comparison = workspace.compare_cases("baseline", "wide_dt")
```

You can also build a payload directly from the validated schema models:

```python
from OpenPinch import pinch_analysis_service
from OpenPinch.lib.enums import StreamType
from OpenPinch.lib.schemas.io import StreamSchema, TargetInput, UtilitySchema

streams = [
    StreamSchema(
        zone="Process Unit",
        name="Reboiler Vapor",
        t_supply=200.0,
        t_target=120.0,
        heat_flow=8000.0,
        dt_cont=10.0,
        htc=1.5,
    ),
    StreamSchema(
        zone="Process Unit",
        name="Feed Preheat",
        t_supply=40.0,
        t_target=160.0,
        heat_flow=6000.0,
        dt_cont=10.0,
        htc=1.2,
    ),
]

utilities = [
    UtilitySchema(
        name="Cooling Water",
        type=StreamType.Cold,
        t_supply=25.0,
        t_target=35.0,
        heat_flow=120000.0,
        dt_cont=5.0,
        htc=0.8,
        price=12.0,
    )
]

payload = TargetInput(streams=streams, utilities=utilities)
result = pinch_analysis_service(payload, project_name="Example")
```

## Graphing and Dashboard

With the notebook or dashboard extra installed, graph generation in Python
looks like:

```python
figure = problem.plot.grand_composite_curve()
figure.show()
```

To launch the Streamlit dashboard after solving, install
`openpinch[dashboard]` and call:

```python
problem.show_dashboard()
```

## Highlights

- Multi-scale targeting for unit operation, process, site, community, and regional studies
- Direct heat integration and indirect integration through utility systems
- Multiple utility targeting, including non-isothermal utilities
- Composite Curve and Grand Composite Curve graph generation
- Excel workbook import and Excel summary export
- Packaged sample cases and notebook workflows
- Pydantic schema models for validated programmatic usage
- Direct process gas/vapour MVR components for workspace comparisons

## Documentation

Full documentation is available at:

https://openpinch.readthedocs.io/en/latest/

The documentation is organized around install, sample workflows, notebooks, graphing, and the public API.

## History

OpenPinch started in 2011 as an Excel workbook with macros. Since then it has expanded into Total Site Heat Integration, multiple utility targeting, retrofit targeting, cogeneration targeting, and related workflows. The Python implementation began in 2021 to bring those capabilities into a scriptable and testable package interface.

## Citation

In publications and forks, please cite and link the foundational article and this repository.

Timothy Gordon Walmsley, 2026. OpenPinch: An Open-Source Python Library for Advanced Pinch Analysis and Total Site Integration. Process Integration and Optimization for Sustainability. https://doi.org/10.1007/s41660-026-00729-6

## Testing

To run the test suite locally:

```bash
python -m pip install -e . pytest build "hatchling>=1.26"
pytest
```

## Contributors

Founder: Tim Walmsley, University of Waikato

Stephen Burroughs, Benjamin Lincoln, Alex Geary, Harrison Whiting, Khang Tran, Roger Padullés, Jasper Walden

## Contributing

Issues and pull requests are welcome. When submitting code, aim for:

- typed interfaces and clear docstrings
- small methods with singular purpose
- pytest coverage for new user-facing behaviour
- updated docs and notebooks where relevant

## License

OpenPinch is released under the MIT License. See `LICENSE` for details.
