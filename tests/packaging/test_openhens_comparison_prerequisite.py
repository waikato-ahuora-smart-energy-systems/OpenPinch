"""Closed-contract tests for the OpenHENS comparison prerequisite."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import compare_openhens_openpinch_top5 as comparison


def _supported_modules() -> dict[str, SimpleNamespace]:
    return {
        "openhens": SimpleNamespace(OpenHENS=lambda: None),
        "openhens.main": SimpleNamespace(run_parallel_solutions=lambda: None),
        "openhens.classes.pinch_classes.process": SimpleNamespace(
            OrganiseArray=lambda: None
        ),
        "openhens.classes.pinch_classes.publicOperations": SimpleNamespace(
            OrganiseArray=lambda: None
        ),
    }


def test_supported_openhens_capabilities_are_read_without_mutation(tmp_path: Path):
    modules = _supported_modules()
    before = {name: deepcopy(vars(module)) for name, module in modules.items()}

    comparison._validate_openhens_capabilities(
        modules,
        openhens_root=tmp_path,
    )

    assert {name: vars(module) for name, module in modules.items()} == before


def test_missing_openhens_capability_has_actionable_error(tmp_path: Path):
    modules = _supported_modules()
    del modules["openhens.classes.pinch_classes.process"].OrganiseArray

    with pytest.raises(RuntimeError, match="missing required upstream capabilities"):
        comparison._validate_openhens_capabilities(
            modules,
            openhens_root=tmp_path,
        )


def test_unsupported_openhens_fails_before_output_creation(
    monkeypatch,
    tmp_path: Path,
):
    output_dir = tmp_path / "comparison-output"
    args = SimpleNamespace(
        repo_root=tmp_path,
        openhens_root=tmp_path / "missing-openhens",
        output_dir=output_dir,
        report_only=False,
        openpinch_diagnostics_only=False,
    )
    monkeypatch.setattr(comparison, "_parse_args", lambda: args)
    monkeypatch.setattr(comparison, "_configure_solver_path", lambda: None)

    with pytest.raises(RuntimeError, match="directory not found"):
        comparison.main()

    assert not output_dir.exists()
