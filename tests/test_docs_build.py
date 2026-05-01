"""Smoke test for the Sphinx documentation build."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest


def test_sphinx_build_smoke(tmp_path: Path):
    pytest.importorskip("sphinx")

    repo_root = Path(__file__).resolve().parents[1]
    build_dir = tmp_path / "html"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "sphinx",
            "-b",
            "html",
            "docs",
            str(build_dir),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert (build_dir / "index.html").exists()
