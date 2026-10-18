"""The Pyodide lane installs the same test dependencies as uv.lock."""

import sys
import tomllib
from importlib.metadata import version

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from tests.support.paths import REPOSITORY_ROOT

PYODIDE_DIR = REPOSITORY_ROOT / "scripts" / "pyodide"


def pinned_requirements():
    lines = (PYODIDE_DIR / "requirements.txt").read_text().splitlines()
    return [
        Requirement(line)
        for line in (line.split("#")[0].strip() for line in lines)
        if line
    ]


def locked_versions():
    lock = tomllib.loads((REPOSITORY_ROOT / "uv.lock").read_text())
    return {canonicalize_name(p["name"]): p["version"] for p in lock["package"]}


def pinned_version(requirement):
    (specifier,) = requirement.specifier
    assert specifier.operator == "==", requirement
    return specifier.version


def test_every_pyodide_pin_is_exact_and_matches_uv_lock():
    locked = locked_versions()
    pins = pinned_requirements()
    names = {canonicalize_name(r.name) for r in pins}
    assert {"pytest", "hypothesis", "jsonschema", "openpyxl", "pyxlsb"} <= names
    assert {"plotly", "tespy", "gekko", "pint"} <= names
    for requirement in pins:
        name = canonicalize_name(requirement.name)
        assert name in locked, f"{name} is pinned for Pyodide but not in uv.lock"
        assert pinned_version(requirement) == locked[name], name


def test_the_pyodide_runner_installs_only_the_pinned_requirements():
    runner = (PYODIDE_DIR / "run_tests.mjs").read_text()
    assert "requirements.txt" in runner
    assert "constraints=" in runner
    # Anything else micropip fetched from PyPI would be an unpinned version.
    assert 'source == "pypi"' in runner


@pytest.mark.skipif(
    sys.platform != "emscripten", reason="only the Pyodide lane installs the pins"
)
def test_pyodide_lane_runs_the_pinned_versions():
    for requirement in pinned_requirements():
        assert version(requirement.name) == pinned_version(requirement)
