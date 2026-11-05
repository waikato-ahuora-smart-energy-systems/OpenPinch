# Adapters

## Overview
Infrastructure adapters between OpenPinch and the outside world. The `io` subpackage reads and writes problem sources, workspace bundles and exported files; `optional_dependencies.py` builds the install messages that most presentation and analysis modules raise when an optional package is missing. The package may import only `adapters`, `contracts`, `domain` and `resources`, so analysis code and presentation code can both depend on it.

## Interface
- `optional_dependency_error` (`optional_dependencies.py`): returns a message naming the package, its purpose, and the `python -m pip install "openpinch[extra]"` commands, joined by " or "; takes `package`, `purpose`, `extras` and an optional `docs` text.
- Subdirectories: [io](./io/README.md) reads problem sources (JSON, Excel, CSV), persists workspace bundles and returns exported files as data.

## Invariants
- The function returns a string and does not raise; callers wrap it in their own `ImportError`. [optional_dependencies.py:optional_dependency_error]
- Layering: `adapters` may import only `adapters`, `contracts`, `domain` and `resources`. [tests/architecture/test_dependency_rules.py:LAYER_ALLOWED_ROOTS]

## Side Effects
- None.

## Verification
- Run: `uv run --no-sync pytest tests/adapters`
