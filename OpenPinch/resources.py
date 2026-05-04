"""Helpers for accessing packaged OpenPinch sample cases and notebooks."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

_SAMPLE_CASE_ROOT = files("OpenPinch.data.sample_cases")
_NOTEBOOK_ROOT = files("OpenPinch.data.notebooks")


def list_sample_cases() -> list[str]:
    """Return the packaged sample-case filenames."""
    return sorted(
        item.name
        for item in _SAMPLE_CASE_ROOT.iterdir()
        if item.is_file() and item.name.endswith(".json")
    )


def list_notebooks() -> list[str]:
    """Return the packaged notebook filenames."""
    return sorted(
        item.name
        for item in _NOTEBOOK_ROOT.iterdir()
        if item.is_file() and item.name.endswith(".ipynb")
    )


def read_sample_case(name: str) -> str:
    """Return the text of a packaged sample case."""
    return _SAMPLE_CASE_ROOT.joinpath(name).read_text(encoding="utf-8")


def copy_sample_case(name: str, destination: str | Path) -> Path:
    """Copy a packaged sample case to ``destination``."""
    source = _SAMPLE_CASE_ROOT.joinpath(name)
    dest_path = Path(destination)
    if dest_path.is_dir():
        dest_path = dest_path / name
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return dest_path


def copy_notebook(name: str, destination: str | Path) -> Path:
    """Copy a packaged notebook to ``destination``."""
    source = _NOTEBOOK_ROOT.joinpath(name)
    dest_path = Path(destination)
    if dest_path.is_dir():
        dest_path = dest_path / name
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return dest_path
