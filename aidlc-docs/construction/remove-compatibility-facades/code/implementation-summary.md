# Remove Compatibility Facades Implementation Summary

The synthesis schema compatibility layer has been removed. The retired
`methods.py`, `tasks.py`, and `results.py` modules no longer exist, and the
synthesis package `__init__.py` no longer re-exports schema classes.

All production code, scripts, tests, and Sphinx documentation import the
concrete `common`, `topology`, `method`, `task`, or `result` owner. Intentional
public exports from `OpenPinch.lib` and `OpenPinch.lib.schemas` remain, but their
lazy mappings resolve directly to concrete modules. Old synthesis-barrel import
and pickle paths are intentionally unsupported.
