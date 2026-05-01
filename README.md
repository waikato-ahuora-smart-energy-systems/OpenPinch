# OpenPinch

OpenPinch is an open-source toolkit for advanced Pinch Analysis and Total Site Integration. It supports direct and indirect heat integration targeting, multi-utility studies, graph generation, Excel-based workflows, and programmatic analysis in Python.

## Install

Install the published package from PyPI:

```bash
python -m pip install openpinch
```

OpenPinch currently requires Python `>=3.14`.

## First Run

Copy a known-good sample case and run it:

```bash
openpinch sample -o basic_pinch.json
openpinch run basic_pinch.json --graph-output graphs -o results
```

That command sequence will:

- print a compact terminal summary
- export an Excel workbook to `results/`
- export HTML graph files to `graphs/`

Validate an input file without running the full analysis:

```bash
openpinch validate basic_pinch.json
```

Export graphs only:

```bash
openpinch graph basic_pinch.json --graph-type gcc -o graphs
```

## Notebook Workflow

OpenPinch ships with a notebook series for distinct outputs and workflows. Copy them into your working directory with:

```bash
openpinch notebook -o notebooks
```

The packaged notebook series currently includes:

- `01_basic_pinch_analysis.ipynb`
- `02_graphs_and_interpretation.ipynb`
- `03_zonal_analysis.ipynb`
- `04_heat_pump_workflow.ipynb`
- `05_batch_comparison.ipynb`

These notebooks are intended to be the main learning path for new users.

## Interpreting Results

Start with the compact summary:

- `Hot Utility Target` is the minimum external heating demand for the case.
- `Cold Utility Target` is the minimum external cooling demand.
- `Heat Recovery` is the internal heat recovery achieved by the targeting result.
- `Hot Pinch` and `Cold Pinch` identify the constrained temperature region that limits further direct recovery.

For graph-based interpretation:

- composite curves show overall source and sink overlap
- shifted composite curves show the effect of the minimum approach temperature
- grand composite curves are the main view for utility selection and heat-pump opportunity identification
- total-site graphs are the right level for comparing zonal interactions and utility-system effects

The packaged `04_heat_pump_workflow.ipynb` notebook focuses on heat-pump targeting and integration. It compares a base case against an integrated heat-pump scenario and treats cycle performance as supporting context rather than the main result.

## Python Workflow

For script and notebook usage, the main front door is `PinchProblem`.

```python
from pathlib import Path

from OpenPinch import PinchProblem

problem = PinchProblem(problem_filepath=Path("basic_pinch.json"))
problem.run()

summary = problem.summary_frame()
print(summary)

problem.export_excel("results")
problem.export_graphs("graphs", graph_type="gcc")
```

You can also build a payload directly from the validated schema models:

```python
from OpenPinch import pinch_analysis_service
from OpenPinch.lib.enums import StreamType
from OpenPinch.lib.schema import StreamSchema, TargetInput, UtilitySchema

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

For graph generation in Python:

```python
figure = problem.plot_grand_composite_curve()
figure.show()
```

To launch the Streamlit dashboard after solving:

```python
problem.show_dashboard()
```

## Highlights

- Multi-scale targeting for unit operation, process, site, community, and regional studies
- Direct heat integration and indirect integration through utility systems
- Multiple utility targeting, including non-isothermal utilities
- Composite-curve and grand-composite-curve graph generation
- Excel workbook import and Excel summary export
- Packaged sample cases and notebook workflows
- Pydantic schema models for validated programmatic usage

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
python -m pip install -e .
pytest
```

## Contributors

Founder: Dr Tim Walmsley, University of Waikato

Stephen Burroughs, Benjamin Lincoln, Alex Geary, Harrison Whiting, Khang Tran, Roger Padullés, Jasper Walden

## Contributing

Issues and pull requests are welcome. When submitting code, aim for:

- typed interfaces and clear docstrings
- small methods with singular purpose
- pytest coverage for new user-facing behaviour
- updated docs and notebooks where relevant

## License

OpenPinch is released under the MIT License. See `LICENSE` for details.
