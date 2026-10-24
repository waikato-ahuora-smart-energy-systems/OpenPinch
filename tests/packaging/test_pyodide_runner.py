"""The Pyodide test runner reports a failure outside pytest as a Python traceback."""

import shutil
import subprocess
import sys

import pytest

from tests.support.paths import REPOSITORY_ROOT

RUNNER = REPOSITORY_ROOT / "scripts" / "pyodide" / "run_tests.mjs"
PYODIDE_PACKAGE = REPOSITORY_ROOT / "scripts" / "pyodide" / "node_modules" / "pyodide"
# GitHub cuts a log line at 64 KB and drops what follows it.
GITHUB_LINE_LIMIT = 64 * 1024


@pytest.mark.skipif(sys.platform == "emscripten", reason="starts a Node process")
@pytest.mark.skipif(shutil.which("node") is None, reason="Node is not installed")
@pytest.mark.skipif(
    not PYODIDE_PACKAGE.is_dir(),
    reason="the runner's packages are not installed (npm ci in scripts/pyodide)",
)
def test_a_failed_install_prints_the_traceback_and_exits_1(tmp_path):
    not_a_wheel = tmp_path / "openpinch-0.0.0-py3-none-any.whl"
    not_a_wheel.write_text("not a zip")

    run = subprocess.run(
        ["node", str(RUNNER), str(not_a_wheel), "-q"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )

    assert run.returncode == 1
    assert "Pyodide test run failed:" in run.stderr
    assert "zipfile.BadZipFile: File is not a zip file" in run.stderr
    # Before, Node printed pyodide.asm.mjs's 1.2 MB minified line first, and
    # GitHub dropped the traceback that followed it.
    longest = max(len(line) for line in (run.stdout + run.stderr).splitlines())
    assert longest < GITHUB_LINE_LIMIT
