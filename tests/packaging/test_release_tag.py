"""Tests for release-tag validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.check_release_tag import expected_release_tag, main, validate_release_tag


def _write_pyproject(path: Path, version: str = "1.2.3") -> Path:
    pyproject = path / "pyproject.toml"
    pyproject.write_text(f'[project]\nversion = "{version}"\n', encoding="utf-8")
    return pyproject


def test_matching_release_tag_is_accepted(tmp_path: Path):
    pyproject = _write_pyproject(tmp_path)

    assert expected_release_tag(pyproject) == "v1.2.3"
    assert validate_release_tag("v1.2.3", pyproject) is None
    assert main(["v1.2.3", "--pyproject", str(pyproject)]) == 0


@pytest.mark.parametrize(
    "tag",
    ["1.2.3", "v1.2", "v1.2.3-rc1", "v01.2.3", "release-v1.2.3"],
)
def test_malformed_release_tag_is_rejected(tmp_path: Path, tag: str):
    pyproject = _write_pyproject(tmp_path)

    with pytest.raises(ValueError, match="exact form"):
        validate_release_tag(tag, pyproject)
    assert main([tag, "--pyproject", str(pyproject)]) == 1


def test_mismatched_release_tag_is_rejected(tmp_path: Path):
    pyproject = _write_pyproject(tmp_path)

    with pytest.raises(ValueError, match="expected 'v1.2.3'"):
        validate_release_tag("v1.2.4", pyproject)
    assert main(["v1.2.4", "--pyproject", str(pyproject)]) == 1
