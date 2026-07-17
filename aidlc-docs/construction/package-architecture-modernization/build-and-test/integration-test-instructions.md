# Package Architecture Modernization Integration Test Instructions

## Scenario 1 - External Contract Through the Full Use Case

Run `tests/e2e/test_main.py`. Caller mappings must cross contracts,
application orchestration, domain state, targeting analysis, and output
serialization without requiring optional dashboard or solver packages.

Expected result: 59 cases pass with exact signature, validation shape, field
ordering, representative numerical values, all shipped example solves, and
core-only cold import.

## Scenario 2 - Heat Pumps Through Shared Optimisation

```bash
uv run pytest -q tests/optimisation tests/analysis/heat_pumps --hypothesis-seed=20260715
```

Generic scalar objectives must run without HPR imports. HPR fixtures must cross
the single optimisation adapter and preserve objective, penalty, cost,
candidate-ordering, and multiperiod aggregation semantics.

## Scenario 3 - HEN Model Composition and Extraction

```bash
uv run pytest -q tests/analysis/heat_exchanger_networks -m 'not solver'
uv run pytest -q -m solver
```

Expected outcomes include preserved axes, equation order, warm starts, period
states, segment areas, later-period matches, topology, and solver result
classification.

## Scenario 4 - Optional Presentation and Infrastructure Leaves

```bash
uv run pytest -q tests/adapters tests/presentation tests/architecture/test_cold_imports.py
```

Core owners must remain importable when Plotly, Streamlit, workbook, and solver
libraries are absent. Installed optional leaves must preserve rendered data,
labels, coordinates, table ordering, and exported values.

## Scenario 5 - Installed Distribution

Install the isolated wheel in a fresh environment outside the checkout. Run the
artifact smoke and copied external suite with warnings as errors. The import
must resolve from `site-packages`; packaged notebooks/sample cases must be
present; retired modules and root aliases must not resolve.

No external service, database, or cleanup operation is required.
