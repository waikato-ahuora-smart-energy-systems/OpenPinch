# OpenPinch

[![CI Develop](https://github.com/waikato-ahuora-smart-energy-systems/OpenPinch/actions/workflows/ci-develop.yml/badge.svg?branch=develop)](https://github.com/waikato-ahuora-smart-energy-systems/OpenPinch/actions/workflows/ci-develop.yml)
[![Documentation Status](https://readthedocs.org/projects/openpinch/badge/?version=latest)](https://openpinch.readthedocs.io/en/latest/)
[![PyPI version](https://img.shields.io/pypi/v/openpinch.svg)](https://pypi.org/project/OpenPinch/)
[![Python versions](https://img.shields.io/pypi/pyversions/openpinch.svg)](https://pypi.org/project/OpenPinch/)
[![License: MIT](https://img.shields.io/github/license/waikato-ahuora-smart-energy-systems/OpenPinch.svg)](LICENSE)

OpenPinch is an open-source Python toolkit for advanced Pinch Analysis and
Total Site Integration. It supports direct and indirect heat integration
targeting, graph interpretation, Heat Pump and refrigeration screening, exergy
and cogeneration post-processing, heat exchanger network synthesis,
multi-period analysis, stream piece-wise linearisation (for variable heat capacity
and phase change streams), and file-backed or schema-first workflows.

Full documentation is available at
https://openpinch.readthedocs.io/en/latest/.

## Install

Install the base package for validation, targeting, summaries, and schema-first
Python workflows:

```bash
python -m pip install openpinch
```

Install optional extras only for the workflows that need them:

```bash
python -m pip install "openpinch[notebook]"      # Jupyter, Plotly graphs, Excel I/O
python -m pip install "openpinch[dashboard]"     # Streamlit dashboard
python -m pip install "openpinch[synthesis]"     # HEN synthesis, then run: idaes get-extensions
python -m pip install "openpinch[brayton_cycle]" # TESPy-backed Brayton-cycle tooling
python -m pip install "openpinch[full]"          # all optional surfaces, including synthesis
```

OpenPinch currently requires Python `>=3.14.2`.

Both `synthesis` and `full` install the IDAES/Pyomo synthesis stack. Complete
the IDAES installation before running solver-backed workflows:

```bash
idaes get-extensions
```

## First Solve

OpenPinch exposes two package-root workflow classes. Use `PinchProblem` for one
case and `PinchWorkspace` for named cases and scenarios.

```python
from OpenPinch import PinchProblem

problem = PinchProblem(
    {
        "streams": [
            {
                "name": "Hot feed",
                "zone": "Process",
                "t_supply": 180.0,
                "t_target": 80.0,
                "heat_flow": 1000.0,
            },
            {
                "name": "Cold feed",
                "zone": "Process",
                "t_supply": 20.0,
                "t_target": 120.0,
                "heat_flow": 800.0,
            },
        ],
        "utilities": [],
    },
    project_name="First solve",
)
problem.validate()
problem.target.all_heat_integration()

print(problem.summary_frame())
```

Analysis is explicit: named methods execute work, while summaries, reports,
plots, and exports consume prepared or cached state.

## Packaged Resources

OpenPinch ships maintained sample cases and notebook workflows. The resource
helpers below are useful repository tooling, but are not compatibility
protected:

```python
from OpenPinch.resources import (
    list_notebooks,
    list_sample_cases,
    notebook_metadata,
    sample_case_metadata,
)

print(list_sample_cases())
print(sample_case_metadata("basic_pinch.json").description)
print(list_notebooks())
print(notebook_metadata("01_first_solve_and_core_curves.ipynb").title)
```

Copy the notebook series from the CLI:

```bash
openpinch notebook -o notebooks
```

The eighteen-notebook series progresses from first solve through multiperiod
HPR, cogeneration, HEN synthesis, and publication workflows.

The CLI intentionally copies notebooks only. Solves, validation, graph export,
Excel export, dashboards, and advanced targeting happen through Python.

## Documentation Map

- Getting started: https://openpinch.readthedocs.io/en/latest/getting-started.html
- Workflow choice: https://openpinch.readthedocs.io/en/latest/overview/workflow-map.html
- Guides: https://openpinch.readthedocs.io/en/latest/guides/index.html
- API reference: https://openpinch.readthedocs.io/en/latest/api/index.html

## Testing

Run the test suite locally:

```bash
uv sync --frozen --group dev
uv run --no-sync ruff check .
uv run --no-sync coverage run --branch --source=OpenPinch -m pytest --hypothesis-seed=20260715 -m "not solver"
uv run --no-sync coverage report --fail-under=95
uv run --no-sync python scripts/build_docs.py
uv run --no-sync python scripts/build_dist.py
```

Ubuntu runs the complete CI suite. Windows and macOS install the generated
wheel and verify the core import, CLI, and packaged resources. Tests marked
`solver` require external solver binaries; the release workflow installs the
IDAES extensions and runs this gate automatically. Run it locally with
`uv run pytest -m solver` when the required binaries are available.

## Release Process

1. In the pull request targeting `main`, set a strict, forward `X.Y.Z` version
   in `pyproject.toml` and keep the OpenPinch entry in `uv.lock` synchronized.
2. Merge only after the required read-only validation checks pass.
3. The main-branch workflow repeats the test, documentation, solver, build, and
   cross-platform artifact gates.
4. After those gates pass, it creates the annotated version tag and a draft
   GitHub release, then publishes the same artifacts to TestPyPI.
5. Approve the protected `pypi` environment to publish to production PyPI.
6. After PyPI succeeds, the workflow publishes the GitHub release.

Version bumping does not create a tag locally. The release workflow owns tags
and rejects malformed versions, mismatched lock metadata, or an existing tag
that points anywhere other than the main-branch release commit.

Build the documentation locally:

```bash
uv run scripts/build_docs.py
```

## History and Citation

OpenPinch started in 2011 as an Excel workbook with macros. The Python
implementation began in 2021 to make the workflows scriptable and testable.

In publications and forks, please cite and link the foundational article and
this repository:

Timothy Gordon Walmsley, 2026. OpenPinch: An Open-Source Python Library for
Advanced Pinch Analysis and Total Site Integration. Process Integration and
Optimization for Sustainability. https://doi.org/10.1007/s41660-026-00729-6

## Contributors

Founder: Tim Walmsley, University of Waikato

Stephen Burroughs, Benjamin Lincoln, Alex Geary, Harrison Whiting, Khang Tran,
Roger Padulles, Jasper Walden, Caleb Archer

## License

OpenPinch is released under the MIT License. See `LICENSE` for details.
