# OpenPinch

[![CI Develop](https://github.com/waikato-ahuora-smart-energy-systems/OpenPinch/actions/workflows/ci-develop.yml/badge.svg?branch=develop)](https://github.com/waikato-ahuora-smart-energy-systems/OpenPinch/actions/workflows/ci-develop.yml)
[![Documentation Status](https://readthedocs.org/projects/openpinch/badge/?version=latest)](https://openpinch.readthedocs.io/en/latest/)
[![PyPI version](https://img.shields.io/pypi/v/openpinch.svg)](https://pypi.org/project/OpenPinch/)
[![Python versions](https://img.shields.io/pypi/pyversions/openpinch.svg)](https://pypi.org/project/OpenPinch/)
[![License: MIT](https://img.shields.io/github/license/waikato-ahuora-smart-energy-systems/OpenPinch.svg)](LICENSE)

OpenPinch is an open-source Python toolkit for advanced Pinch Analysis and
Total Site Integration. It supports direct and indirect heat integration
targeting, graph interpretation, Heat Pump and refrigeration screening, exergy
and cogeneration post-processing, heat exchanger network synthesis, and
file-backed or schema-first workflows.

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

`PinchProblem` is the main single-case front door. If no local
`basic_pinch.json` exists, OpenPinch resolves the packaged sample case.

```python
from OpenPinch import PinchProblem

problem = PinchProblem("basic_pinch.json", project_name="basic_pinch")

validation = problem.validation_report()
result = problem.target()
summary = problem.summary_frame()

print(validation.valid)
print(summary[["Target", "Hot Utility Target", "Cold Utility Target"]])
```

For named baseline-versus-variant studies, use `PinchWorkspace`:

```python
from OpenPinch import PinchWorkspace

workspace = PinchWorkspace(
    source="crude_preheat_train.json",
    project_name="crude_preheat_train",
)
workspace.scenario("wide_dt", dt_cont_multiplier=0.5)
comparison = workspace.compare_cases("baseline", "wide_dt")
```

For application integration, build a typed `TargetInput` and call
`pinch_analysis_service(...)`.

## Packaged Resources

OpenPinch ships maintained sample cases and notebook workflows. Discover them
from Python:

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
print(notebook_metadata("01_first_solve_summary_graphs.ipynb").title)
```

Copy the notebook series from the CLI:

```bash
openpinch notebook -o notebooks
```

The series is ordered for three paths: solve cases with advanced methods,
understand the targeting methods, and integrate or extend OpenPinch through
typed service and export workflows.

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
python -m pip install -e .
python -m pip install --group dev
ruff check .
coverage run --source=OpenPinch -m pytest -m "not solver"
coverage report --fail-under=95
python scripts/build_docs.py
python scripts/build_dist.py
```

Ubuntu runs the complete CI suite. Windows and macOS install the generated
wheel and verify the core import, CLI, and packaged resources. Tests marked
`solver` require external solver binaries and remain a manual pre-release
check in a solver-enabled environment: `pytest -m solver`.

## Release Process

1. Merge a change only after the required CI checks pass.
2. Confirm `pyproject.toml` and `uv.lock` contain the intended release version.
3. Run the solver-marked tests in a solver-enabled environment.
4. Create a signed or annotated `vX.Y.Z` tag at the intended commit and push it.
5. Approve the protected `pypi` environment after TestPyPI publication succeeds.

The publish workflow rejects malformed tags and tags that do not exactly match
the project version. Pull-request automation updates versions with `--no-tag`;
maintainers always create release tags explicitly.

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
Roger Padulles, Jasper Walden

## License

OpenPinch is released under the MIT License. See `LICENSE` for details.
