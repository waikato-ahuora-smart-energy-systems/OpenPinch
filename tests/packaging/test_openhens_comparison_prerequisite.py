"""Closed-contract tests for the OpenHENS comparison prerequisite."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import compare_openhens_openpinch_top5 as comparison


def _supported_modules(root: Path | None = None) -> dict[str, SimpleNamespace]:
    modules = {
        "openhens": SimpleNamespace(OpenHENS=lambda: None),
        "openhens.main": SimpleNamespace(run_parallel_solutions=lambda: None),
        "openhens.classes.pinch_classes.process": SimpleNamespace(
            OrganiseArray=lambda: None
        ),
        "openhens.classes.pinch_classes.publicOperations": SimpleNamespace(
            OrganiseArray=lambda: None
        ),
    }
    if root is not None:
        module_paths = {
            "openhens": root / "openhens" / "__init__.py",
            "openhens.main": root / "openhens" / "main.py",
            "openhens.classes.pinch_classes.process": (
                root / "openhens" / "classes" / "pinch_classes" / "process.py"
            ),
            "openhens.classes.pinch_classes.publicOperations": (
                root / "openhens" / "classes" / "pinch_classes" / "publicOperations.py"
            ),
        }
        for name, module in modules.items():
            module.__file__ = str(module_paths[name])
    return modules


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


def test_exact_checkout_ignores_cached_foreign_modules_and_restores_state(
    monkeypatch,
    tmp_path: Path,
):
    requested_root = tmp_path / "requested"
    requested_root.mkdir()
    requested_modules = _supported_modules(requested_root)
    foreign_module = SimpleNamespace(__file__=str(tmp_path / "foreign.py"))
    monkeypatch.setitem(sys.modules, "openhens", foreign_module)
    original_path = list(sys.path)
    monkeypatch.setattr(
        comparison,
        "import_module",
        lambda name: requested_modules[name],
    )

    with comparison._supported_openhens_checkout(requested_root) as modules:
        assert modules == requested_modules
        assert sys.path[0] == str(requested_root.resolve())
        assert sys.modules.get("openhens") is not foreign_module

    assert sys.path == original_path
    assert sys.modules["openhens"] is foreign_module


def test_exact_checkout_rejects_foreign_module_origins_and_restores_state(
    monkeypatch,
    tmp_path: Path,
):
    requested_root = tmp_path / "requested"
    requested_root.mkdir()
    foreign_root = tmp_path / "foreign"
    foreign_modules = _supported_modules(foreign_root)
    original_path = list(sys.path)
    monkeypatch.setattr(
        comparison,
        "import_module",
        lambda name: foreign_modules[name],
    )

    with pytest.raises(RuntimeError, match="outside the requested checkout"):
        with comparison._supported_openhens_checkout(requested_root):
            pytest.fail("foreign modules must not be yielded")

    assert sys.path == original_path


def test_exact_checkout_restores_state_after_partial_import_failure(
    monkeypatch,
    tmp_path: Path,
):
    requested_root = tmp_path / "requested"
    requested_root.mkdir()
    original_path = list(sys.path)
    cached_module = SimpleNamespace(__file__=str(tmp_path / "cached.py"))
    monkeypatch.setitem(sys.modules, "openhens.cached", cached_module)

    def fail_import(name: str):
        if name == "openhens.main":
            raise ImportError("broken requested checkout")
        return _supported_modules(requested_root)[name]

    monkeypatch.setattr(comparison, "import_module", fail_import)

    with pytest.raises(RuntimeError, match="could not import"):
        with comparison._supported_openhens_checkout(requested_root):
            pytest.fail("partially imported modules must not be yielded")

    assert sys.path == original_path
    assert sys.modules["openhens.cached"] is cached_module


def test_source_runner_injects_factory_from_verified_checkout(
    monkeypatch,
    tmp_path: Path,
):
    verified_factory = object()
    captured: dict[str, Any] = {}

    @contextmanager
    def exact_checkout(_root: Path):
        yield {"openhens": SimpleNamespace(OpenHENS=verified_factory)}

    def execute_source(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return [], {"source_solution_count": 0, "source_esm_solution_count": 0}

    monkeypatch.setattr(comparison, "_supported_openhens_checkout", exact_checkout)
    monkeypatch.setattr(comparison, "_execute_source_openhens", execute_source)

    comparison._run_source_openhens(
        "case",
        openhens_root=tmp_path,
        d_tmin_grid=(10.0,),
        dqda_grid=(0.5,),
        top_n=5,
        max_parallel=1,
    )

    assert captured["kwargs"]["openhens_factory"] is verified_factory


def test_source_execution_uses_injected_factory(monkeypatch, tmp_path: Path):
    captured: dict[str, Any] = {}

    class FakeModel:
        solutions: list[Any] = []

        def solve(self):
            captured["solved"] = True

    def factory(**kwargs):
        captured["factory_kwargs"] = kwargs
        return FakeModel()

    monkeypatch.setattr(
        comparison, "_source_ranked_networks", lambda *args, **kwargs: []
    )

    rows, stats = comparison._execute_source_openhens(
        "case",
        openhens_root=tmp_path,
        d_tmin_grid=(10.0,),
        dqda_grid=(0.5,),
        top_n=5,
        max_parallel=2,
        openhens_factory=factory,
    )

    assert rows == []
    assert stats == {"source_solution_count": 0, "source_esm_solution_count": 0}
    assert captured["solved"] is True
    assert captured["factory_kwargs"]["max_parallel"] == 2
