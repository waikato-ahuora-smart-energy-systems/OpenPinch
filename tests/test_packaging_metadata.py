"""Checks for packaging metadata declared in ``pyproject.toml``."""

from __future__ import annotations

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
UV_LOCK = REPO_ROOT / "uv.lock"
PYTHON_VERSION = REPO_ROOT / ".python-version"
PYTEST_INI = REPO_ROOT / "pytest.ini"
UPDATE_TOOLCHAIN = REPO_ROOT / "scripts" / "update_toolchain.py"
WORKFLOWS = [
    REPO_ROOT / ".github" / "workflows" / "ci-develop.yml",
    REPO_ROOT / ".github" / "workflows" / "ci-pull-request.yml",
    REPO_ROOT / ".github" / "workflows" / "ci-publish.yml",
]


def _read_pyproject() -> dict:
    with PYPROJECT.open("rb") as handle:
        return tomllib.load(handle)


def _read_uv_lock() -> dict:
    with UV_LOCK.open("rb") as handle:
        return tomllib.load(handle)


def _optional_deps() -> dict:
    return _read_pyproject()["project"]["optional-dependencies"]


def _dependency_groups() -> dict:
    return _read_pyproject()["dependency-groups"]


def _dependency_name(requirement: str) -> str:
    for separator in ("<", ">", "=", "!", "~", "[", ";"):
        requirement = requirement.split(separator, maxsplit=1)[0]
    return requirement.strip().lower().replace("_", "-")


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


def test_synthesis_extra_declares_optional_solver_stack_only():
    optional_deps = _optional_deps()

    assert optional_deps["synthesis"] == [
        "pyomo>=6.10.0",
        "gekko>=1.3.2",
        "matplotlib>=3.10.9",
        "plotly>=6.8.0",
        "kaleido>=1.3.0",
        "openpyxl>=3.1.5",
        "wakepy>=1.0.0",
    ]

    synthesis_only = {"pyomo", "gekko", "matplotlib", "kaleido", "wakepy"}
    core_deps = {
        _dependency_name(dep) for dep in _read_pyproject()["project"]["dependencies"]
    }
    full_deps = {_dependency_name(dep) for dep in optional_deps["full"]}
    unrelated_optional_deps = {
        _dependency_name(dep)
        for extra_name, deps in optional_deps.items()
        if extra_name not in {"synthesis", "full"}
        for dep in deps
    }

    assert synthesis_only.isdisjoint(core_deps)
    assert synthesis_only.isdisjoint(full_deps)
    assert synthesis_only.isdisjoint(unrelated_optional_deps)


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


def test_full_extra_does_not_include_solver_synthesis_stack():
    full_deps = {_dependency_name(dep) for dep in _optional_deps()["full"]}

    assert {"pyomo", "gekko", "matplotlib", "kaleido", "wakepy"}.isdisjoint(full_deps)


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


def test_pytest_marker_policy_declares_synthesis_and_solver_tiers():
    pytest_ini = PYTEST_INI.read_text(encoding="utf-8")

    assert (
        "synthesis: optional HEN synthesis tests that require the synthesis extra"
        in pytest_ini
    )
    assert (
        "solver: tests that require external solver binaries such as Couenne or IPOPT"
        in pytest_ini
    )


def test_lockfile_project_version_matches_pyproject():
    project_version = _read_pyproject()["project"]["version"]
    lock = _read_uv_lock()
    package = next(
        package
        for package in lock["package"]
        if package["name"] == "openpinch" and package["source"] == {"editable": "."}
    )

    assert package["version"] == project_version


def test_testpypi_publish_skips_existing_files_but_pypi_publish_does_not():
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci-publish.yml").read_text(
        encoding="utf-8"
    )
    testpypi_block = workflow.split("publish-testpypi:", 1)[1].split(
        "publish-pypi:", 1
    )[0]
    pypi_block = workflow.split("publish-pypi:", 1)[1]

    assert "skip-existing: true" in testpypi_block
    assert "skip-existing: true" not in pypi_block
