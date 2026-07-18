"""Static guards preventing retired compatibility mechanisms from returning."""

from __future__ import annotations

import ast

from tests.support.paths import REPOSITORY_ROOT


def _source(relative_path: str) -> str:
    return (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")


def test_penalty_helper_has_no_string_selector_normalisation() -> None:
    source = _source("OpenPinch/analysis/numerics.py")
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "g_ineq_penalty"
    )
    string_constants = {
        node.value
        for node in ast.walk(function)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }

    assert "square" not in string_constants
    assert "square_root_smoothing" not in string_constants
    assert "square root smoothing" not in string_constants
    assert not any(
        isinstance(node, ast.Attribute) and node.attr == "lower"
        for node in ast.walk(function)
    )


def test_openhens_comparison_has_no_upstream_monkeypatch_layer() -> None:
    source = _source("scripts/compare_openhens_openpinch_top5.py")

    for retired in (
        "_install_openhens_compatibility",
        "_bounded_source_runner",
        "public_ops.OrganiseArray =",
        "process_module.OrganiseArray =",
        "source_main.run_parallel_solutions =",
    ):
        assert retired not in source


def test_solver_backend_has_no_pyomo_signature_retry() -> None:
    source = _source("OpenPinch/analysis/heat_exchanger_networks/solver/backend.py")

    assert "available(False)" not in source
    assert source.count("available(exception_flag=False)") == 1


def test_unit_policy_has_no_alias_constructor_or_attribute() -> None:
    source = _source("OpenPinch/contracts/units.py")
    tree = ast.parse(source)

    assert not any(
        isinstance(node, ast.Attribute) and node.attr == "aliases"
        for node in ast.walk(tree)
    )
    assert not any(
        isinstance(node, ast.keyword) and node.arg == "aliases"
        for node in ast.walk(tree)
    )
    assert not any(
        isinstance(node, ast.arg) and node.arg == "aliases" for node in ast.walk(tree)
    )


def test_removed_legacy_documentation_and_test_name_stay_absent() -> None:
    assert not (REPOSITORY_ROOT / "docs/reference/api-lib.rst").exists()
    assert "../reference/api-lib" not in _source("docs/api/generated-index.rst")
    assert "test_validate_utilities_data_alias_executes" not in _source(
        "tests/adapters/test_workbook.py"
    )
