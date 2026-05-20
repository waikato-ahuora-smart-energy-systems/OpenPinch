"""Checks for packaging metadata declared in ``pyproject.toml``."""

from __future__ import annotations

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
PYTHON_VERSION = REPO_ROOT / ".python-version"
UPDATE_TOOLCHAIN = REPO_ROOT / "scripts" / "update_toolchain.py"
WORKFLOWS = [
    REPO_ROOT / ".github" / "workflows" / "ci-develop.yml",
    REPO_ROOT / ".github" / "workflows" / "ci-pull-request.yml",
    REPO_ROOT / ".github" / "workflows" / "ci-publish.yml",
]


def _read_pyproject() -> dict:
    with PYPROJECT.open("rb") as handle:
        return tomllib.load(handle)


def _optional_deps() -> dict:
    return _read_pyproject()["project"]["optional-dependencies"]


def _dependency_groups() -> dict:
    return _read_pyproject()["dependency-groups"]


def _minimum_python_version() -> str:
    requires_python = _read_pyproject()["project"]["requires-python"]
    assert requires_python.startswith(">=")
    return requires_python.removeprefix(">=")


def test_notebook_extra_declares_jupyter_runtime_dependencies():
    assert _optional_deps()["notebook"] == [
        "ipykernel>=7.2.0",
        "nbformat>=5.10.4",
        "plotly",
        "openpyxl",
        "pyxlsb",
    ]


def test_dashboard_and_brayton_cycle_extras_are_declared():
    optional_deps = _optional_deps()
    assert optional_deps["dashboard"] == [
        "streamlit",
        "plotly",
        "openpyxl",
        "pyxlsb",
    ]
    assert optional_deps["brayton_cycle"] == [
        "tespy",
    ]


def test_full_extra_aggregates_optional_runtime_surfaces():
    assert _optional_deps()["full"] == [
        "streamlit",
        "plotly",
        "openpyxl",
        "pyxlsb",
        "tespy",
        "ipykernel>=7.2.0",
        "nbformat>=5.10.4",
    ]


def test_dev_dependency_group_retains_notebook_dependencies():
    dev_group = _dependency_groups()["dev"]

    assert "ipykernel>=7.2.0" in dev_group
    assert "nbformat>=5.10.4" in dev_group


def test_dev_dependency_group_has_one_ruff_entry():
    dev_group = _dependency_groups()["dev"]
    ruff_entries = [entry for entry in dev_group if entry.startswith("ruff")]

    assert ruff_entries == ["ruff>=0.15.8"]


def test_requires_python_matches_python_version_files_and_ci():
    minimum_version = _minimum_python_version()

    assert minimum_version == PYTHON_VERSION.read_text(encoding="utf-8").strip()

    update_toolchain = UPDATE_TOOLCHAIN.read_text(encoding="utf-8")
    assert "_read_python_minor" in update_toolchain
    assert "requires-python" in update_toolchain

    for workflow in WORKFLOWS:
        text = workflow.read_text(encoding="utf-8")
        assert 'PYTHON_VERSION: "3.14"' in text
        assert "python-version: ${{ env.PYTHON_VERSION }}" in text


def test_requires_python_classifier_matches_minimum_version():
    project = _read_pyproject()["project"]

    assert project["requires-python"] == ">=3.14"
    assert "Programming Language :: Python :: 3.14" in project["classifiers"]
