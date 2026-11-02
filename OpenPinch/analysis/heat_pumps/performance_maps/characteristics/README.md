# TESPy Compressor Characteristic Data

## Overview
Packaged data: `openpinch-single-stage-compressor-v1.json`, a canonical JSON compressor characteristic (relative mass flow against relative isentropic efficiency). The package `__init__.py` holds only a docstring. `../resources.py` loads it through `importlib.resources`, and `../adapters/tespy.py` uses it as the TESPy `eta_s_char` line.

## Interface
- `openpinch-single-stage-compressor-v1.json`: keys `schema_version`, `characteristic_set_id`, `abscissa`, `ordinate`, `points`.
- Loaders: `load_tespy_compressor_characteristic`, `read_tespy_compressor_characteristic_bytes` and `parse_tespy_compressor_characteristic` in [`../resources.py`](../resources.py).

## Invariants
- The file bytes must be canonical JSON (sorted keys, compact separators, trailing newline), with no duplicate keys, non-finite numbers or extra keys [../resources.py:parse_tespy_compressor_characteristic].
- Schema version `1.0`, identifier `openpinch-single-stage-compressor-v1`, abscissa `relative_mass_flow` and ordinate `relative_isentropic_efficiency` are fixed [../resources.py:parse_tespy_compressor_characteristic].
- At least two points, strictly ascending abscissae and positive ordinates [../resources.py:parse_tespy_compressor_characteristic].
- The SHA-256 of the bytes is recorded in provenance, so any edit changes recorded identity [../resources.py:TespyCompressorCharacteristic].

## Side Effects
- Reads the packaged resource from disk on each load [../resources.py:read_tespy_compressor_characteristic_bytes].

## Verification
- Run: `uv run --no-sync pytest tests/packaging/test_resources.py` (no directory-level test; this covers the packaged resource).
