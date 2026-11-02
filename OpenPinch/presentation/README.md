# Presentation

## Overview
Turns solved results into things people read: graphs, dashboards, Excel and CSV files, network grid diagrams and display metadata for configuration fields. It sits above `analysis`, `contracts` and `domain` and is used by the application layer. Rendering libraries (Plotly, Streamlit, openpyxl) are optional and imported lazily.

## Interface
- `configuration_field_metadata`, `configuration_options` (`configuration.py`): `ConfigurationFieldMetadata` for every editable configuration field; `configuration_options` is the report-friendly alias.
- `annotation_metadata` (`configuration.py`): maps a Python annotation to an input kind (`enum`, `boolean`, `number`, `number_list`, `string_list`, `object`, `string`) and a multiple flag.
- `CONFIGURATION_FIELD_TEXT`, `FieldText` (`configuration_text.py`): label, unit and one-sentence description for each field.
- Subdirectories:
  - [dashboard](./dashboard/README.md): Streamlit dashboard for a solved zone.
  - [graphs](./graphs/README.md): graph selection, Plotly figures and HTML export.
  - [network_grid](./network_grid/README.md): heat exchanger network grid diagrams and the HEN design view.
  - [reporting](./reporting/README.md): result adapters, Excel and CSV exports and summaries.

## Invariants
- Every field in `USER_CONFIG_FIELD_SPECS` needs an entry in `CONFIGURATION_FIELD_TEXT`; a missing one raises `KeyError`. [configuration.py:configuration_field_metadata]
- Fields with `positive` set report an exclusive numeric minimum. [configuration.py:configuration_field_metadata]
- Enum choices are the enum members' string values. [configuration.py:configuration_field_metadata]
- Optional annotations are unwrapped to their single member; an unrecognised annotation is `string`. [configuration.py:annotation_metadata]
- Layering: presentation may import only `adapters`, `analysis`, `contracts`, `domain` and `presentation`. [tests/architecture/test_dependency_rules.py:LAYER_ALLOWED_ROOTS]

## Side Effects
- None in this directory; the subdirectories write files and import optional packages.

## Verification
- Run: `uv run --no-sync pytest tests/presentation`
