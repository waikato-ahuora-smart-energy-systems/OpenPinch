"""Structural regressions for lazy barrels and concrete synthesis schemas."""

from __future__ import annotations

import ast
import importlib.util
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

import OpenPinch
from OpenPinch import lib
from OpenPinch.lib import schemas
from OpenPinch.lib.schemas.synthesis.common import (
    HeatExchangerNetworkSynthesisManifest,
)
from OpenPinch.lib.schemas.synthesis.method import (
    HeatExchangerNetworkSynthesisMethodInput,
)
from OpenPinch.lib.schemas.synthesis.result import (
    HeatExchangerNetworkSynthesisResult,
)
from OpenPinch.lib.schemas.synthesis.task import HeatExchangerNetworkSynthesisTask
from OpenPinch.lib.schemas.synthesis.topology import (
    HeatExchangerNetworkTopologyRestriction,
)


def test_typed_lazy_barrels_have_no_eager_runtime_imports() -> None:
    package_root = Path(OpenPinch.__file__).parent
    paths = (
        package_root / "classes" / "__init__.py",
        package_root / "lib" / "__init__.py",
        package_root / "lib" / "schemas" / "__init__.py",
    )

    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assert any(
            isinstance(node, ast.FunctionDef) and node.name == "__getattr__"
            for node in tree.body
        )
        assert any(
            isinstance(node, ast.FunctionDef) and node.name == "__dir__"
            for node in tree.body
        )
        runtime_relative_imports = [
            node
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.level > 0
        ]
        assert runtime_relative_imports == []


def test_synthesis_definitions_have_concrete_owners_without_reverse_barrel_imports():
    package_dir = Path(OpenPinch.__file__).parent / "lib" / "schemas" / "synthesis"
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


def test_synthesis_compatibility_facades_are_retired() -> None:
    package = __import__("OpenPinch.lib.schemas.synthesis", fromlist=["*"])
    assert not hasattr(package, "HeatExchangerNetworkSynthesisManifest")
    for module_name in ("methods", "tasks", "results"):
        assert (
            importlib.util.find_spec(f"OpenPinch.lib.schemas.synthesis.{module_name}")
            is None
        )

    schema_names = {
        "HeatExchangerNetworkSynthesisManifest",
        "HeatExchangerNetworkSynthesisMethodInput",
        "HeatExchangerNetworkSynthesisResult",
        "HeatExchangerNetworkSynthesisTask",
        "HeatExchangerNetworkTopologyRestriction",
    }
    for barrel in (schemas, lib):
        for name in schema_names:
            module_name, _ = barrel._EXPORTS[name]
            assert module_name != "OpenPinch.lib.schemas.synthesis"
            assert module_name.startswith("OpenPinch.lib.schemas.synthesis.")


def test_old_synthesis_barrel_pickle_path_is_unsupported() -> None:
    payload = pickle.dumps(HeatExchangerNetworkSynthesisManifest, protocol=0)
    legacy_payload = payload.replace(
        b"OpenPinch.lib.schemas.synthesis.common",
        b"OpenPinch.lib.schemas.synthesis",
        1,
    )

    with pytest.raises(AttributeError):
        pickle.loads(legacy_payload)


def test_public_barrels_and_synthesis_leaves_cold_import_without_optional_packages():
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
import OpenPinch.classes
import OpenPinch.lib
import OpenPinch.lib.schemas
import OpenPinch.lib.schemas.synthesis.common
import OpenPinch.lib.schemas.synthesis.topology
import OpenPinch.lib.schemas.synthesis.method
import OpenPinch.lib.schemas.synthesis.task
import OpenPinch.lib.schemas.synthesis.result
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
