"""The Pyodide lane tests the built wheel, not the checkout it mounts."""

import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(
    sys.platform != "emscripten", reason="only the Pyodide lane installs the wheel"
)
def test_pyodide_lane_imports_the_installed_wheel():
    import OpenPinch

    assert "site-packages" in Path(OpenPinch.__file__).parts
