# OpenPinch Package Blueprint

## System Overview
The importable package. `PinchProblem` and `PinchWorkspace` are the only names exported at the package root, and everything else is reached through them. A case flows inward then back out: `adapters` read a source, `contracts` validate it, `application` builds the `domain` state and calls the `analysis` services, and `presentation` shapes the results into reports, graphs and exports. For how to use the package see the top-level `README.md` and `docs/`; for why the layers are arranged this way see `docs/developer/architecture.rst`.

## Local Interface & Invariants
- `PinchProblem`, `PinchWorkspace` (`__init__.py`): the public package root, listed in `__all__`.
- `main`, `build_parser` (`__main__.py`): the `openpinch` command line, with the `notebook` and `schema` subcommands. It does not run a solve.
- `list_notebooks`, `list_sample_cases`, `copy_notebook` and related functions (`resources.py`): lookup and copying of the packaged tutorial data.

## Subdirectories & System Boundaries
- [domain](./domain/README.md): unit-aware values, streams, the problem table, zones, targets and configuration. May import only itself.
- [contracts](./contracts/README.md): Pydantic input and output models at the application boundary. May import `contracts` and `domain`.
- [optimisation](./optimisation/README.md): solver-neutral bounded minimisation with several backends. May import only itself.
- [adapters](./adapters/README.md): file formats and optional dependency messages. May import `adapters`, `contracts`, `domain` and `resources`.
- [analysis](./analysis/README.md): deterministic engineering calculations. May import `adapters`, `analysis`, `contracts`, `domain` and `optimisation`.
- [presentation](./presentation/README.md): graphs, dashboards, reports and exports. May import `adapters`, `analysis`, `contracts`, `domain` and `presentation`.
- [application](./application/README.md): `PinchProblem`, `PinchWorkspace`, their accessors and the JSON commands. May import `adapters`, `analysis`, `application`, `contracts`, `domain` and `presentation` (not `optimisation`).
- [tutorials](./tutorials/README.md): packaged notebooks and sample cases, with no logic.

## Global Invariants
- The import rules above are the `LAYER_ALLOWED_ROOTS` table, enforced with no exceptions. [tests/architecture/test_dependency_rules.py:LAYER_ALLOWED_ROOTS]
- `import OpenPinch` does not import optional packages such as Plotly or openpyxl. [tests/architecture/test_cold_imports.py]
- Configuration keys are defined in one place. [domain/configuration_fields.py:CONFIG_FIELD_SPECS]

## Global Side Effects
- Importing the package builds the Pint registry and installs it as the Pint application registry [domain/value.py:ureg]. File IO happens in `adapters`, the `presentation` exporters, `resources.py`, the `notebook` and `schema` subcommands in `__main__.py` [__main__.py:_schema_command], the TESPy performance-map adapter (a private temporary design-state file), and `analysis/heat_exchanger_networks` (synthesis result exports and GEKKO run directories). GEKKO solves, which launch external solvers, are confined to `analysis/heat_exchanger_networks`.

## System Verification
- Run: `uv run --no-sync pytest tests/architecture`
- Run: `uv run --no-sync pytest -m "not solver and not tespy and not performance and not docs" --hypothesis-seed=20260715`
