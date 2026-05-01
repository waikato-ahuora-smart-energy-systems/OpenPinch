"""Smoke tests for the packaged OpenPinch notebook series."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from OpenPinch.resources import copy_notebook, list_notebooks


def _execute_notebook(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__"}
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        exec(compile(source, str(path), "exec"), namespace)


def test_packaged_notebook_series_is_present():
    assert list_notebooks() == [
        "01_basic_pinch_analysis.ipynb",
        "02_graphs_and_interpretation.ipynb",
        "03_zonal_analysis.ipynb",
        "04_heat_pump_workflow.ipynb",
        "05_batch_comparison.ipynb",
    ]


@pytest.mark.parametrize("notebook_name", list_notebooks())
def test_packaged_notebooks_execute_smoke(
    tmp_path: Path, monkeypatch, notebook_name: str
):
    notebook_path = copy_notebook(notebook_name, tmp_path / notebook_name)
    monkeypatch.chdir(tmp_path)
    _execute_notebook(notebook_path)
