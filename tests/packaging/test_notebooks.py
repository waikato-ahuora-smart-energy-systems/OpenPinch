"""Executable contract checks for the packaged process-engineer tutorials."""

from __future__ import annotations

import ast
import csv
import json
import os
from pathlib import Path

import pytest

from OpenPinch.resources import (
    copy_notebook,
    list_notebooks,
    list_sample_cases,
    notebook_metadata,
    read_sample_case,
    sample_case_metadata,
)

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "docs" / "_data" / "tutorial-coverage.csv"
FORBIDDEN_IMPORT_PREFIXES = (
    "OpenPinch.analysis",
    "OpenPinch.application",
    "OpenPinch.contracts",
    "OpenPinch.domain",
    "OpenPinch.presentation",
)


def _manifest_rows() -> list[dict[str, str]]:
    with MANIFEST.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


EXPECTED_NOTEBOOKS = sorted({row["primary_tutorial"] for row in _manifest_rows()})


def _load_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _copied_notebook(tmp_path: Path, name: str) -> dict:
    return _load_notebook(copy_notebook(name, tmp_path / name))


def _code_sources(notebook: dict) -> list[str]:
    return [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]


def _combined_source(notebook: dict) -> str:
    return "\n".join(_code_sources(notebook))


def test_manifest_and_packaged_inventory_are_identical() -> None:
    rows = _manifest_rows()

    assert len(rows) >= 100
    assert list_notebooks() == EXPECTED_NOTEBOOKS
    assert len({row["operation"] for row in rows}) == len(rows)
    assert {row["execution_profile"] for row in rows} == {
        "base",
        "interactive",
        "slow-hpr",
        "solver",
    }
    assert all(row["operation"] for row in rows)


def test_notebooks_are_valid_source_only_nbformat_documents(tmp_path: Path) -> None:
    for name in EXPECTED_NOTEBOOKS:
        notebook = _copied_notebook(tmp_path, name)

        assert notebook["nbformat"] == 4, name
        assert notebook["nbformat_minor"] >= 5, name
        assert notebook["cells"], name
        assert notebook["cells"][0]["cell_type"] == "markdown", name
        introduction = "".join(notebook["cells"][0]["source"])
        for label in (
            "Learning outcome",
            "Level",
            "Execution profile",
            "Expected runtime",
            "Optional extras",
        ):
            assert label in introduction, (name, label)
        markdown_text = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "markdown"
        )
        assert (
            sum(cell["cell_type"] == "markdown" for cell in notebook["cells"]) >= 5
        ), name
        for heading in (
            "Study question and data",
            "Interpret the result",
            "Adapt this template",
        ):
            assert heading in markdown_text, (name, heading)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                assert cell["execution_count"] is None, name
                assert cell["outputs"] == [], name


def test_every_code_cell_compiles_and_uses_only_public_package_imports(
    tmp_path: Path,
) -> None:
    for name in EXPECTED_NOTEBOOKS:
        notebook = _copied_notebook(tmp_path, name)
        source = _combined_source(notebook)
        assert "from OpenPinch import " in source, name
        assert "pinch_analysis_service" not in source, name

        for index, code_source in enumerate(_code_sources(notebook), start=1):
            tree = ast.parse(code_source, filename=f"{name}:cell-{index}")
            compile(tree, f"{name}:cell-{index}", "exec")
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    assert not node.module.startswith(FORBIDDEN_IMPORT_PREFIXES), (
                        name,
                        node.module,
                    )
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        assert not alias.name.startswith(FORBIDDEN_IMPORT_PREFIXES), (
                            name,
                            alias.name,
                        )


def test_manifest_operations_are_demonstrated_in_notebook_source(
    tmp_path: Path,
) -> None:
    aliases = {
        "summary_frame.include_periods": "include_periods",
        "summary_frame.include_weighted_average": "include_weighted_average",
        "design.utility": "total_hot_utility",
        "design.selected_network": "network(rank=1)",
        "load": 'PinchProblem("basic_pinch.json"',
        "PinchProblem.__init__": "PinchProblem(",
        "PinchWorkspace.__init__": "PinchWorkspace(",
        "source": "basic_pinch.json",
        "show_dashboard": "show_dashboard",
    }
    for row in _manifest_rows():
        operation = row["operation"]
        notebook_name = row["primary_tutorial"]
        source = _combined_source(_copied_notebook(tmp_path, notebook_name))
        token = aliases.get(operation, operation.rsplit(".", maxsplit=1)[-1])
        assert token in source, (notebook_name, operation)
        if row["owner"].startswith("Batch"):
            secondary = row["secondary_tutorials"]
            assert secondary == "04_workspace_cases_and_scenarios.ipynb"
            secondary_source = _combined_source(_copied_notebook(tmp_path, secondary))
            assert "workspace.cases(" in secondary_source


@pytest.mark.parametrize(
    "name",
    sorted(
        {
            row["primary_tutorial"]
            for row in _manifest_rows()
            if row["execution_profile"] == "base"
        }
    ),
)
def test_base_profile_notebook_executes(name: str, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    namespace = {"__name__": "__main__"}
    notebook = _copied_notebook(tmp_path, name)

    for index, source in enumerate(_code_sources(notebook), start=1):
        exec(compile(source, f"{name}:cell-{index}", "exec"), namespace)


@pytest.mark.parametrize("profile", ["slow-hpr", "solver", "interactive"])
def test_optional_profile_notebooks_execute(
    profile: str, tmp_path: Path, monkeypatch
) -> None:
    enabled = set(filter(None, os.getenv("OPENPINCH_TUTORIAL_PROFILES", "").split(",")))
    if profile not in enabled and "all" not in enabled:
        pytest.skip(
            f"Set OPENPINCH_TUTORIAL_PROFILES={profile} with its declared extras."
        )

    if profile == "interactive":
        from OpenPinch import PinchProblem, PinchWorkspace

        monkeypatch.setattr(PinchProblem, "show_dashboard", lambda *_a, **_k: None)
        monkeypatch.setattr(PinchWorkspace, "show_dashboard", lambda *_a, **_k: None)

    monkeypatch.chdir(tmp_path)
    names = sorted(
        {
            row["primary_tutorial"]
            for row in _manifest_rows()
            if row["execution_profile"] == profile
        }
    )
    for name in names:
        namespace = {"__name__": "__main__"}
        notebook = _copied_notebook(tmp_path, name)
        for index, source in enumerate(_code_sources(notebook), start=1):
            exec(compile(source, f"{name}:cell-{index}", "exec"), namespace)


def test_packaged_resource_metadata_and_friendly_errors() -> None:
    sample_meta = sample_case_metadata("basic_pinch.json")
    notebook_meta = notebook_metadata(EXPECTED_NOTEBOOKS[0])

    assert sample_meta.title == "Basic Pinch"
    assert sample_case_metadata("process_mvr.json").title == "Process MVR"
    assert "quickstart" in sample_meta.topics
    assert notebook_meta.title == "First Solve and Core Curves"
    assert len(sample_case_metadata()) == len(list_sample_cases())
    assert [item.name for item in notebook_metadata()] == EXPECTED_NOTEBOOKS

    with pytest.raises(FileNotFoundError, match="Unknown OpenPinch sample case") as exc:
        read_sample_case("not-a-case.json")
    assert "basic_pinch.json" in str(exc.value)
