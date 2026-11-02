# Contract Resources

## Overview
Package-data container for files shipped with the contracts. Its `__init__.py`
holds a docstring only. Its single subdirectory holds the HPR performance-map
schema and golden fixtures, which `OpenPinch/resources.py` exposes through a
closed catalogue.

## Interface
- Subdirectories: [hpr_performance_map](./hpr_performance_map/README.md) - JSON Schema and golden maps for HPR performance-map schema 1.0.

## Invariants
- None.

## Side Effects
- None.

## Verification
- Run: `uv run --no-sync pytest tests/packaging/test_resources.py` (no focused test for the empty package; this covers the data below it).
