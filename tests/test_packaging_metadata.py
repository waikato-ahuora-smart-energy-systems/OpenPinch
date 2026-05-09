"""Checks for packaging metadata declared in ``pyproject.toml``."""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"


def _read_pyproject() -> str:
    return PYPROJECT.read_text(encoding="utf-8")


def _extract_array(text: str, key: str) -> list[str]:
    pattern = rf"(?m)^{re.escape(key)}\s*=\s*\[(?P<body>.*?)^\]"
    match = re.search(pattern, text, re.DOTALL)
    assert match is not None, f"missing array for {key}"
    return re.findall(r'"([^"]+)"', match.group("body"))


def _extract_table(text: str, table_name: str) -> str:
    pattern = rf"(?ms)^\[{re.escape(table_name)}\]\n(?P<body>.*?)(?=^\[|\Z)"
    match = re.search(pattern, text)
    assert match is not None, f"missing table [{table_name}]"
    return match.group("body")


def test_notebook_extra_declares_jupyter_runtime_dependencies():
    optional_deps = _extract_table(_read_pyproject(), "project.optional-dependencies")
    assert _extract_array(optional_deps, "notebook") == [
        "ipykernel>=7.2.0",
        "nbformat>=5.10.4",
    ]


def test_dev_dependency_group_retains_notebook_dependencies():
    dependency_groups = _extract_table(_read_pyproject(), "dependency-groups")
    dev_group = _extract_array(dependency_groups, "dev")

    assert "ipykernel>=7.2.0" in dev_group
    assert "nbformat>=5.10.4" in dev_group
