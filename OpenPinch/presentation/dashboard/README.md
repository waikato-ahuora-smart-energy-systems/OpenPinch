# Streamlit dashboard

## Overview
Renders a solved zone as a Streamlit page: a sidebar of target metrics and utilities, graph cards and shifted and real problem tables with Excel download. It consumes graph data from `analysis.graphs.service`, figures from `presentation.graphs.plotly` and tables from `presentation.reporting`. Streamlit and openpyxl are optional dependencies.

## Interface
- `render_streamlit_dashboard` (`rendering.py`): renders the dashboard for a `Zone`; takes optional `graph_data`, `page_title` and `value_rounding`.
- `render_table_export` (`exports.py`): renders a path input and button that save a problem-table dataframe as an Excel file.
- `dashboard.css` (data): stylesheet injected into the page.
- Private helpers: `_require_streamlit`, `_require_openpyxl` (`dependencies.py`) and `_DashboardGraphSet` (`state.py`) group targets' graphs.
- Subdirectories: none.

## Invariants
- Only targets with `reportable` set are listed; a repeated target name gets the scope (or object id) appended so none is hidden. [rendering.py:_collect_targets]
- Metrics are shown in the zone's configured output units, converted from degC and kW; failure to convert falls back to the canonical unit; absent values show "n/a". [rendering.py:_display]
- Free text put into HTML is escaped. [rendering.py:render_streamlit_dashboard]
- The stylesheet is read from the package as UTF-8 beside `rendering.py`. [rendering.py:_dashboard_css]
- Exported Excel text cells are stored as plain strings, not formulas. [exports.py:render_table_export]
- A blank save path shows an error and writes nothing. [exports.py:render_table_export]

## Side Effects
- Calls Streamlit page configuration, markdown, sidebar and chart functions; imports Streamlit lazily, preferring an already loaded module. [rendering.py:render_streamlit_dashboard] [dependencies.py:_require_streamlit]
- Reads `dashboard.css` through `importlib.resources`. [rendering.py:_dashboard_css]
- The save button creates parent directories and writes the workbook to the user-typed path, defaulting to `results/<target>_shifted.xlsx` or `_real.xlsx`; an `OSError` is shown in the page. [exports.py:render_table_export]
- Imports openpyxl lazily. [dependencies.py:_require_openpyxl]

## Verification
- Run: `uv run --no-sync pytest tests/presentation/test_dashboard.py`
