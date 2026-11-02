# Packaged sample cases

## Overview
JSON input files for example problems, shipped inside the wheel. The `__init__.py` is empty apart from a docstring; the package exists so `importlib.resources` can find the data. `OpenPinch/resources.py` lists, reads and copies the files, and `OpenPinch/adapters/io/problem_sources.py` reads them when a bare sample file name is given.

## Interface
- Data files, each a JSON object of problem inputs:
  - `basic_pinch.json`, `chocolate_factory.json`, `crude_preheat_train.json`, `crude_preheat_train_multiperiod.json`, `dairy_chilling_multiperiod.json`, `zonal_site.json`, `zonal_site_multiperiod.json`: top-level keys `streams`, `utilities`, `options`.
  - `pulp_mill.json`: keys `plant_profile_data`, `streams`, `utilities`, `options`.
  - `process_mvr.json`: keys `streams`, `utilities`, `zone_tree`.
  - `Four-stream-Yee-and-Grossmann-1990-1.json`: keys `options`, `streams`, `utilities`, `zone_tree`.
- Loaders (outside this directory): `list_sample_cases`, `read_sample_case`, `copy_sample_case`, `sample_case_metadata` in `OpenPinch/resources.py`.
- Subdirectories: none.

## Invariants
- A file is listed as a sample case only if its name ends in `.json`. [resources.py:list_sample_cases]
- Descriptive metadata is kept in a dictionary in `resources.py`, keyed by file name, separate from the files. [resources.py:_SAMPLE_CASE_METADATA]
- A name not in the directory raises `FileNotFoundError` listing the available names. [resources.py:_resolve_resource]

## Side Effects
- None in this directory. `copy_sample_case` writes a copy to the destination and creates its parent directory; unlike `copy_notebook`, it treats the destination as a directory only if it already exists as one, so a missing `out/` becomes a file named `out`. [resources.py:copy_sample_case]

## Verification
- Run: `uv run --no-sync pytest tests/packaging/test_resources.py`
