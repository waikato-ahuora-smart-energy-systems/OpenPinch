"""Repository-wide Python syntax gate."""

from __future__ import annotations

import ast
import subprocess

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


def _python_files() -> list:
    """Return git-tracked Python files, or every file when git is unavailable."""
    try:
        listed = subprocess.run(
            ["git", "ls-files", "-z", "--", "*.py"],
            cwd=REPO_ROOT,
            capture_output=True,
            check=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        # Not a checkout (e.g. an unpacked sdist): walk the tree instead.
        return list(REPO_ROOT.rglob("*.py"))
    # Untracked copies, such as agent worktrees under .claude/, are not ours.
    return [REPO_ROOT / name for name in listed.split("\0") if name]


def test_tracked_python_files_parse():
    errors = []
    for path in _python_files():
        if any(part in EXCLUDED_PARTS for part in path.relative_to(REPO_ROOT).parts):
            continue
        try:
            ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            errors.append(f"{path.relative_to(REPO_ROOT)}:{exc.lineno}: {exc.msg}")

    assert errors == []
