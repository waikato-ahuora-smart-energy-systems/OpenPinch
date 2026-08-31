"""Tests for main-branch release-version validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.check_release_version import (
    main,
    read_project_version,
    validate_version_advance,
)


def _write_pyproject(path: Path, version: str, *, name: str) -> Path:
    pyproject = path / name
    pyproject.write_text(f'[project]\nversion = "{version}"\n', encoding="utf-8")
    return pyproject


def test_release_version_accepts_strict_forward_semver(tmp_path: Path) -> None:
    current = _write_pyproject(tmp_path, "1.3.0", name="current.toml")
    base = _write_pyproject(tmp_path, "1.2.9", name="base.toml")

    assert read_project_version(current) == "1.3.0"
    assert validate_version_advance(current, base) == "1.3.0"
    assert main(["--pyproject", str(current), "--base-pyproject", str(base)]) == 0


@pytest.mark.parametrize(
    ("current_version", "base_version"),
    [("1.2.3", "1.2.3"), ("1.2.2", "1.2.3"), ("0.9.9", "1.0.0")],
)
def test_release_version_rejects_non_forward_versions(
    tmp_path: Path,
    current_version: str,
    base_version: str,
) -> None:
    current = _write_pyproject(tmp_path, current_version, name="current.toml")
    base = _write_pyproject(tmp_path, base_version, name="base.toml")

    with pytest.raises(ValueError, match="greater than"):
        validate_version_advance(current, base)
    assert main(["--pyproject", str(current), "--base-pyproject", str(base)]) == 1


@pytest.mark.parametrize("version", ["1.2", "v1.2.3", "1.2.3-rc1", "01.2.3"])
def test_release_version_rejects_noncanonical_versions(
    tmp_path: Path,
    version: str,
) -> None:
    current = _write_pyproject(tmp_path, version, name="current.toml")

    with pytest.raises(ValueError, match="X.Y.Z"):
        read_project_version(current)
    assert main(["--pyproject", str(current)]) == 1
