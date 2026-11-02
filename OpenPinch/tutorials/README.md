# Tutorials

## Overview
Packaged learning material: example notebooks and sample problem files. The package contains no logic; its `__init__.py` is a docstring, and the data is discovered through `importlib.resources` by `OpenPinch/resources.py`. Sample cases are also read by `OpenPinch/adapters/io/problem_sources.py`.

## Interface
- Subdirectories: [notebooks](./notebooks/README.md) holds the 19 tutorial notebooks; [sample_cases](./sample_cases/README.md) holds the JSON example problems.

## Invariants
- The directory names are resolved by package path (`OpenPinch.tutorials.notebooks`, `OpenPinch.tutorials.sample_cases`), so they must stay importable packages. [resources.py:_SAMPLE_CASE_ROOT] [resources.py:_NOTEBOOK_ROOT]

## Side Effects
- None.

## Verification
- Run: `uv run --no-sync pytest tests/packaging/test_resources.py` (no test targets this directory alone; this one covers the packaged resources).
