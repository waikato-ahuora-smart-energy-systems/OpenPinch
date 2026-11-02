# HPR Performance-Map Resources

## Overview
Packaged data for HPR performance-map schema 1.0: a JSON Schema and two golden
map fixtures, one heat pump and one refrigeration. The package `__init__.py` is a
docstring only. The files are read through `importlib.resources` by
`OpenPinch/resources.py`, and the maps they hold are the plain-data form of
`HprPerformanceMap` in `OpenPinch/contracts/hpr_performance_map.py`.

## Interface
- `schema-1.0.json`: JSON Schema (title `HprPerformanceMap`) with `additionalProperties: false`, `schema_version` fixed to `"1.0"`, and closed units (`degC`, `kW`).
- `heat-pump-1.0.json`: golden map with `map_id` `openpinch-golden-heat-pump-1.0`, `mode` `heat_pump`, `cop_convention` `heating`, `reference_capacity_basis` `q_sink`, three points on one curve.
- `refrigeration-1.0.json`: golden map with `map_id` `openpinch-golden-refrigeration-1.0`, `mode` `refrigeration`, `cop_convention` `cooling`.
- Loader: `list_hpr_performance_map_contract_resources`, `read_hpr_performance_map_contract_resource`, `load_hpr_performance_map_contract_resource` (`OpenPinch/resources.py`).
- Subdirectories: none.

## Invariants
- The catalogue is closed: the loader accepts only the names it lists and raises `FileNotFoundError` for any other [resources.py:_resolve_resource].
- Each resource must parse to a JSON object [resources.py:load_hpr_performance_map_contract_resource].
- The fixtures are intended to validate as `HprPerformanceMap`, which checks energy balance, useful duty and COP for each point [hpr_performance_map.py:HprPerformanceMap].
- Editing a file means editing the packaged contract; the catalogue and file set are asserted by tests/packaging/test_resources.py.

## Side Effects
- Package-data file reads only, UTF-8 text, performed by the loader [resources.py:read_hpr_performance_map_contract_resource]. Nothing here writes files.

## Verification
- Run: `uv run --no-sync pytest tests/packaging/test_resources.py`
