# OpenPinch

OpenPinch is an open-source toolkit for advanced Pinch Analysis and Total Site Integration. It supports direct and indirect heat integration targeting, multi-utility studies, graph generation, Excel-based workflows, and programmatic analysis in Python.

## Install

Install the published package from PyPI:

```bash
python -m pip install openpinch
```

OpenPinch currently requires Python `>=3.14`.


## Notebook Workflow

OpenPinch ships with a notebook series for distinct outputs and workflows. Copy them into your working directory with:

```bash
openpinch notebook -o notebooks
```

The packaged notebook series currently includes:

- `01_basic_pinch_and_dtcont_sensitivity.ipynb`
- `02_total_site_targets_and_sugcc.ipynb`
- `03_carnot_hpr_comparison.ipynb`

These notebooks are intended to be the main learning path for new users.


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
python -m pip install -e . pytest build "hatchling>=1.26"
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
