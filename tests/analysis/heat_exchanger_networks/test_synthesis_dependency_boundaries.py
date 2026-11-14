"""HENS synthesis dependency boundary tests."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from OpenPinch.analysis.heat_exchanger_networks.solver import (
    dependencies as _dependencies,
)
from tests.support.paths import REPOSITORY_ROOT

REPO_ROOT = REPOSITORY_ROOT
SYNTHESIS_ONLY_MODULES = [
    "pyomo",
    "pyomo.environ",
    "pyomo.opt",
    "plotly",
    "plotly.graph_objects",
    "kaleido",
    "openpyxl",
    "wakepy",
    "idaes",
    "highspy",
    "pyscipopt",
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
    monkeypatch.setattr(_dependencies, "_openpinch_solvers_bin_directory", lambda: None)
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
    monkeypatch.setattr(_dependencies, "_openpinch_solvers_bin_directory", lambda: None)
    monkeypatch.setattr(_dependencies, "_idaes_bin_directory", lambda: tmp_path)
    monkeypatch.setenv("PATH", "/usr/bin")

    path = _dependencies.require_solver_binary(
        "ipopt",
        purpose="heat exchanger network synthesis solver regression tests",
    )

    assert path == str(solver)
    assert os.environ["PATH"].split(os.pathsep)[0] == str(tmp_path)


def test_solver_binary_comes_from_openpinch_solvers_before_idaes(
    tmp_path,
    monkeypatch,
):
    # The synthesis extra installs Ipopt as the openpinch-solvers wheel;
    # IDAES's older download is only a fallback.
    package_bin = tmp_path / "openpinch_solvers"
    idaes_bin = tmp_path / "idaes"
    for directory in (package_bin, idaes_bin):
        directory.mkdir()
        solver = directory / "ipopt"
        solver.write_text("#!/bin/sh\n", encoding="utf-8")
        solver.chmod(0o755)
    monkeypatch.setattr(_dependencies, "which", lambda _binary: None)
    monkeypatch.setattr(
        _dependencies, "_openpinch_solvers_bin_directory", lambda: package_bin
    )
    monkeypatch.setattr(_dependencies, "_idaes_bin_directory", lambda: idaes_bin)
    monkeypatch.setenv("PATH", "/usr/bin")

    path = _dependencies.require_solver_binary("ipopt")

    assert path == str(package_bin / "ipopt")
    assert os.environ["PATH"].split(os.pathsep)[0] == str(package_bin)


def test_openpinch_solvers_bin_directory_reads_the_installed_package(monkeypatch):
    fake = type("Module", (), {"bin_directory": "/opt/openpinch_solvers/bin"})
    monkeypatch.setattr(
        _dependencies,
        "import_module",
        lambda name: fake if name == "openpinch_solvers" else None,
    )

    assert _dependencies._openpinch_solvers_bin_directory() == Path(
        "/opt/openpinch_solvers/bin"
    )
