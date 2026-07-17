"""Smoke test for the Sphinx documentation build."""

from __future__ import annotations

import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path
from shutil import which

import pytest

from tests.support.paths import REPOSITORY_ROOT


def test_sphinx_build_smoke(tmp_path: Path):
    if find_spec("sphinx") is None and which("uv") is None:
        pytest.skip("docs build requires sphinx or uv")

    repo_root = REPOSITORY_ROOT
    build_dir = tmp_path / "html"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/build_docs.py",
            "--output-dir",
            str(build_dir),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert (build_dir / "index.html").exists()
