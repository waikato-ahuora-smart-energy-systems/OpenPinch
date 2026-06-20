"""HENS synthesis dependency boundary tests."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from OpenPinch.services.heat_exchanger_network_synthesis.solver import (
    dependencies as _dependencies,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SYNTHESIS_ONLY_MODULES = [
    "gekko",
    "pyomo",
    "pyomo.environ",
    "pyomo.opt",
    "plotly",
    "plotly.graph_objects",
    "kaleido",
    "openpyxl",
    "wakepy",
    "idaes",
]


def test_import_openpinch_does_not_import_synthesis_only_packages():
    script = f"""
import importlib
import sys

import OpenPinch

forbidden = {SYNTHESIS_ONLY_MODULES!r}
loaded = [name for name in forbidden if name in sys.modules]
if loaded:
    raise SystemExit(f"import OpenPinch loaded optional modules: {{loaded}}")
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_missing_synthesis_dependency_error_names_extra_and_docs():
    with pytest.raises(
        _dependencies.MissingSynthesisDependencyError,
        match=r"openpinch\[synthesis\].*synthesis dependency policy",
    ):
        _dependencies.require_synthesis_dependency(
            "_openpinch_missing_synthesis_dependency_for_test",
            package="pyomo",
            purpose="heat exchanger network synthesis model construction",
        )


def test_missing_solver_binary_error_names_binary_and_solver_marker(monkeypatch):
    monkeypatch.setattr(_dependencies, "which", lambda _binary: None)
    monkeypatch.setattr(_dependencies, "_idaes_bin_directory", lambda: None)

    with pytest.raises(
        _dependencies.MissingSynthesisSolverError,
        match=r"couenne.*PATH.*synthesis dependency policy",
    ):
        _dependencies.require_solver_binary(
            "couenne",
            purpose="heat exchanger network synthesis solver regression tests",
        )


def test_solver_binary_falls_back_to_idaes_bin_directory(
    tmp_path,
    monkeypatch,
):
    solver = tmp_path / "ipopt"
    solver.write_text("#!/bin/sh\n", encoding="utf-8")
    solver.chmod(0o755)
    monkeypatch.setattr(_dependencies, "which", lambda _binary: None)
    monkeypatch.setattr(_dependencies, "_idaes_bin_directory", lambda: tmp_path)
    monkeypatch.setenv("PATH", "/usr/bin")

    path = _dependencies.require_solver_binary(
        "ipopt",
        purpose="heat exchanger network synthesis solver regression tests",
    )

    assert path == str(solver)
    assert os.environ["PATH"].split(os.pathsep)[0] == str(tmp_path)
