"""Release artifact checks for packaging boundaries."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

import pytest


def test_wheel_excludes_repo_only_assets(tmp_path):
    pytest.importorskip("build")
    pytest.importorskip("hatchling")

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = tmp_path / "dist"
    out_dir.mkdir()

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(out_dir),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr

    wheel_paths = sorted(out_dir.glob("*.whl"))
    assert wheel_paths, (
        f"no wheel created in {out_dir}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    wheel_path = wheel_paths[0]

    with ZipFile(wheel_path) as wheel:
        names = wheel.namelist()

    assert "OpenPinch/__init__.py" in names
    assert "__init__.py" not in names

    forbidden_prefixes = [
        "examples/",
        "tests/",
        "Excel_Version/",
    ]
    forbidden_fragments = [
        "streamlit_app.py",
    ]

    for prefix in forbidden_prefixes:
        assert not any(name.startswith(prefix) for name in names), prefix
    for fragment in forbidden_fragments:
        assert not any(fragment in name for name in names), fragment

    assert any(
        name.endswith("OpenPinch/data/sample_cases/basic_pinch.json") for name in names
    )
    assert any(
        name.endswith("OpenPinch/data/sample_cases/heat_pump_targeting.json")
        for name in names
    )
    assert any(
        name.endswith("OpenPinch/data/notebooks/01_basic_pinch_analysis.ipynb")
        for name in names
    )
