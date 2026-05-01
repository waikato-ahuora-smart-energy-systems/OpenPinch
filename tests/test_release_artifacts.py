"""Release artifact checks for packaging boundaries."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

import pytest


def test_wheel_excludes_repo_only_assets(tmp_path):
    pytest.importorskip("build")

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = tmp_path / "dist"
    out_dir.mkdir()

    proc = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(out_dir)],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr

    wheel_path = next(out_dir.glob("*.whl"))

    with ZipFile(wheel_path) as wheel:
        names = wheel.namelist()

    forbidden_fragments = [
        "streamlit_app.py",
        "examples/",
        "tests/",
        "Excel_Version/",
        "__init__.py",
    ]

    for fragment in forbidden_fragments:
        if fragment == "__init__.py":
            assert "OpenPinch/__init__.py" in names
            assert "__init__.py" not in names
            continue
        assert not any(fragment in name for name in names), fragment
