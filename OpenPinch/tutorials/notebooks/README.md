# Packaged tutorial notebooks

## Overview
Jupyter notebooks (`01` to `19`) shipped inside the wheel; the `__init__.py` holds only a docstring so `importlib.resources` can find the data. `OpenPinch/resources.py` lists, describes and copies them. Each notebook's `metadata["openpinch"]` block gives its title and topics (the title falls back to the file stem) [resources.py:_notebook_metadata], and they run from first solve and core curves through heat pumps, cogeneration, heat exchanger network synthesis, and exports.

## Interface
- Data files: `01_first_solve_and_core_curves.ipynb` to `19_utility_placement_optimisation.ipynb`; the numbered sequence covers first solves, focused and total-site targeting, multi-segment streams, workspaces and persistence, multiperiod integration, area, cost and exergy, heat pumps and refrigeration, vapour compression, process MVR, cogeneration, energy transfer, HEN synthesis, results and exports, and utility placement.
- Loaders (outside this directory): `list_notebooks`, `notebook_metadata`, `copy_notebook` in `OpenPinch/resources.py`.
- Subdirectories: none.

## Invariants
- A file is listed as a notebook only if its name ends in `.ipynb`. [resources.py:list_notebooks]
- `copy_notebook` treats the destination as a directory when it exists as one or ends with a path separator. [resources.py:_names_directory]
- An unknown name raises `FileNotFoundError` listing the available names. [resources.py:_resolve_resource]

## Side Effects
- None in this directory. `copy_notebook` writes the file and creates the parent directory. [resources.py:copy_notebook]

## Verification
- Run: `uv run --no-sync pytest tests/packaging/test_notebooks.py`
