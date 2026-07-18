"""Structural regressions for marker packages and concrete synthesis schemas."""

from __future__ import annotations

import ast
import importlib.util
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

import OpenPinch
from OpenPinch.contracts.synthesis.common import (
    HeatExchangerNetworkSynthesisManifest,
)
from OpenPinch.contracts.synthesis.method import (
    HeatExchangerNetworkSynthesisMethodInput,
)
from OpenPinch.contracts.synthesis.result import (
    HeatExchangerNetworkSynthesisResult,
)
from OpenPinch.contracts.synthesis.task import HeatExchangerNetworkSynthesisTask
from OpenPinch.contracts.synthesis.topology import (
    HeatExchangerNetworkTopologyRestriction,
)


def test_owner_package_markers_have_no_runtime_imports() -> None:
    package_root = Path(OpenPinch.__file__).parent
    paths = (
        package_root / "contracts" / "__init__.py",
        package_root / "domain" / "__init__.py",
        package_root / "analysis" / "__init__.py",
        package_root / "analysis" / "heat_exchanger_networks" / "__init__.py",
        package_root
        / "analysis"
        / "heat_exchanger_networks"
        / "models"
        / "__init__.py",
    )

    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assert not any(
            isinstance(node, ast.Import | ast.ImportFrom) for node in tree.body
        ), path


def test_synthesis_definitions_have_concrete_owners_without_reverse_imports() -> None:
    package_dir = Path(OpenPinch.__file__).parent / "contracts" / "synthesis"
    expected_owners = {
        HeatExchangerNetworkSynthesisManifest: "common",
        HeatExchangerNetworkTopologyRestriction: "topology",
        HeatExchangerNetworkSynthesisMethodInput: "method",
        HeatExchangerNetworkSynthesisTask: "task",
        HeatExchangerNetworkSynthesisResult: "result",
    }
    for schema_type, owner in expected_owners.items():
        assert schema_type.__module__.endswith(f".synthesis.{owner}")

    for path in package_dir.glob("*.py"):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assert not any(
            isinstance(node, ast.ImportFrom) and node.level == 1 and node.module is None
            for node in ast.walk(tree)
        )


def test_synthesis_and_package_compatibility_facades_are_retired() -> None:
    synthesis_package = __import__("OpenPinch.contracts.synthesis", fromlist=["*"])
    assert not hasattr(
        synthesis_package,
        "HeatExchangerNetworkSynthesisManifest",
    )
    retired_modules = (
        "OpenPinch.classes",
        "OpenPinch.lib",
        "OpenPinch.services",
        "OpenPinch.utils",
        "OpenPinch.streamlit_webviewer",
        "OpenPinch.contracts.synthesis.methods",
        "OpenPinch.contracts.synthesis.tasks",
        "OpenPinch.contracts.synthesis.results",
    )
    assert all(importlib.util.find_spec(name) is None for name in retired_modules)


def test_old_synthesis_barrel_pickle_path_is_unsupported() -> None:
    payload = pickle.dumps(HeatExchangerNetworkSynthesisManifest, protocol=0)
    legacy_payload = payload.replace(
        b"OpenPinch.contracts.synthesis.common",
        b"OpenPinch.contracts.synthesis",
        1,
    )

    with pytest.raises(AttributeError):
        pickle.loads(legacy_payload)


def test_concrete_synthesis_leaves_cold_import_without_optional_packages() -> None:
    code = r"""
import builtins
real_import = builtins.__import__
blocked = ("streamlit", "plotly", "pyomo")
def guarded_import(name, *args, **kwargs):
    if name.split(".", 1)[0] in blocked:
        raise ModuleNotFoundError(name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import OpenPinch
import OpenPinch.contracts.input
import OpenPinch.contracts.synthesis.common
import OpenPinch.contracts.synthesis.topology
import OpenPinch.contracts.synthesis.method
import OpenPinch.contracts.synthesis.task
import OpenPinch.contracts.synthesis.result
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
