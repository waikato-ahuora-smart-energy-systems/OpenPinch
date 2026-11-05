"""Smoke test for the Sphinx documentation build."""

from __future__ import annotations

import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path
from shutil import which

import pytest

from tests.support.paths import REPOSITORY_ROOT


def _docs_toolchain_available() -> bool:
    return find_spec("sphinx") is not None or which("uv") is not None


@pytest.mark.docs
@pytest.mark.skipif(
    "not _docs_toolchain_available()", reason="docs build requires sphinx or uv"
)
def test_sphinx_build_smoke(tmp_path: Path):
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
