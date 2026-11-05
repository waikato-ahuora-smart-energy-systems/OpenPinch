# Filesystem input and output adapters

## Overview
Readers and writers that turn external problem sources (JSON, Excel workbooks, CSV files, base64 file payloads, packaged sample cases) into plain input mappings, and persist workspace bundles and exported files. Workbook and CSV readers share the table parsing in `tabular.py` and the name clean-up in `records.py`. The application layer calls `problem_sources.py`; nothing here runs analysis.

## Interface
- `ExternalProblemSource`, `PathLike` (`problem_sources.py`): frozen dataclass holding `input_data`, `source_kind`, `problem_filepath` and `project_name`; and the `str | Path` alias.
- `load_external_problem_source` (`problem_sources.py`): reads a `.json`, workbook (`.xlsx`, `.xlsb`, `.xlsm`; a legacy `.xls` raises `ValueError`), a directory holding `streams.csv` and `utilities.csv`, or a `(streams, utilities)` tuple of CSV paths.
- `load_problem_source_from_files` (`problem_sources.py`): reads `{name, role, content_base64}` files with roles `problem`, or `streams` and `utilities`.
- `load_packaged_sample_case` (`problem_sources.py`): parses a packaged sample case by name.
- `get_problem_from_excel`, `get_results_from_excel` (`workbook.py`): read the "Stream Data" and "Utility Data" sheets and the optional "Options" sheet (names in column A, values in column B; a missing sheet gives `{}`), or the "Summary" sheet.
- `get_problem_from_csv`, `get_results_from_csv` (`csv.py`): CSV equivalents; accept a path or file object.
- `validate_stream_data`, `validate_utility_data` (`records.py`): normalise stream names and zones, drop unnamed utility rows.
- `get_column_names_and_units`, `problem_frame_from_sheet`, `summary_frame_from_sheet`, `problem_records_from_frame`, `target_records_from_frame` (`tabular.py`): shared table-to-record conversion.
- `parse_json`, `read_json`, `write_json` (`json.py`): UTF-8 JSON primitives.
- `load_workspace_bundle`, `save_workspace_bundle` (`workspace_bundles.py`): read and write a `PinchWorkspaceBundle` as JSON.
- `export_files` (`file_exports.py`): run a writer on a temporary directory and return `{name, content_base64}` entries.
- Subdirectories: none.

## Invariants
- Table values with a non-empty unit become `{"value": float | None, "unit": str}`; a non-numeric or boolean cell raises `ValueError`. [tabular.py:problem_records_from_frame]
- Columns absent from a stream or utility table stay absent, so the input contract applies its defaults. [csv.py:_parse_csv_with_units] [workbook.py:_parse_sheet_with_units]
- Columns are matched by header text through aliases; with no recognised header, the fixed column order is used and a `UserWarning` is issued. Two columns for one field raise `ValueError`. [tabular.py:_map_header_row] [tabular.py:problem_frame_from_sheet]
- Degrees signs in units are replaced with `deg`. [tabular.py:get_column_names_and_units]
- Table layout defaults: names row 0, units row 1, data from row 2 for problem tables; units row 2, data from row 4 for summaries. [csv.py:get_problem_from_csv] [csv.py:get_results_from_csv]
- `zone`, `name` and `type` columns stay text in CSV input, so leading zeros survive. [csv.py:_parse_csv_with_units]
- Stream names have dots replaced by dashes and numeric names prefixed (`S`, or `Z` for zones); two different names in one zone that normalise to the same name raise `ValueError`; exact duplicates are not rejected here; empty rows are dropped and a missing name becomes `S<n>`. Missing zones inherit the previous zone, defaulting to "Process Zone". [records.py:validate_stream_data]
- Summary `degree_of_integration` is multiplied by 100 and `pinch_temp` is split on `;` into cold and hot temperatures. [tabular.py:_summary_field_data] [tabular.py:_parse_temp_pinch]
- A bare `.json` file name that does not exist locally and matches a packaged sample case is read from the package. [problem_sources.py:_packaged_sample_case_name]
- `load_problem_source_from_files` accepts only role sets `["problem"]` or `["streams", "utilities"]`, and writes each file under its base name in a per-role folder. [problem_sources.py:load_problem_source_from_files]
- Layering: this package may import only `adapters`, `contracts`, `domain` and `resources`. [tests/architecture/test_dependency_rules.py:LAYER_ALLOWED_ROOTS]

## Side Effects
- Reads files given by path or file object, via pandas and `Path.read_text`. [csv.py:_parse_csv_with_units] [workbook.py:_parse_sheet_with_units] [json.py:read_json]
- Writes JSON when `output_json` is a string. [csv.py:get_problem_from_csv] [workbook.py:get_problem_from_excel]
- `save_workspace_bundle` creates parent directories and writes the file; `write_json` writes without creating directories. [workspace_bundles.py:save_workspace_bundle] [json.py:write_json]
- Emits `UserWarning` for renamed, unnamed or unrecognised columns, names and rows. [records.py:validate_stream_data] [tabular.py:_map_header_row]
- Writes uploaded and exported files to a `TemporaryDirectory` that is removed on exit. [problem_sources.py:load_problem_source_from_files] [file_exports.py:export_files]
- Reads packaged sample cases through `OpenPinch.resources`. [problem_sources.py:load_packaged_sample_case]
- Excel reading needs a pandas Excel engine installed. A missing engine raises `ImportError` naming the package and the `dashboard` and `notebook` extras, or asking for a `.xlsx` save for `.xls` content. [workbook.py:_open_workbook]

## Verification
- Run: `uv run --no-sync pytest tests/adapters`
