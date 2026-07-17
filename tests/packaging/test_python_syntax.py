"""Repository-wide Python syntax gate."""

from __future__ import annotations

import ast

from tests.support.paths import REPOSITORY_ROOT

REPO_ROOT = REPOSITORY_ROOT
EXCLUDED_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "docs",
}


def test_tracked_python_files_parse():
    errors = []
    for path in REPO_ROOT.rglob("*.py"):
        if any(part in EXCLUDED_PARTS for part in path.relative_to(REPO_ROOT).parts):
            continue
        try:
            ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            errors.append(f"{path.relative_to(REPO_ROOT)}:{exc.lineno}: {exc.msg}")

    assert errors == []
