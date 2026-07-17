"""API-surface tests for the marker root and concrete owner modules."""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

import OpenPinch
import OpenPinch.domain.heat_exchanger as heat_exchanger
import OpenPinch.domain.stream as stream
from OpenPinch.contracts.hpr import HPRBackendResult, HPRParsedState
from OpenPinch.contracts.input import (
    HeatExchangerAreaSliceSchema,
    HeatExchangerNetworkSchema,
    HeatExchangerPeriodStateSchema,
    HeatExchangerSchema,
    StreamSegmentSchema,
    TargetInput,
)

PACKAGE_DIR = Path(OpenPinch.__file__).parent
RETIRED_PACKAGES = (
    "classes",
    "lib",
    "services",
    "utils",
    "streamlit_webviewer",
)


def test_root_package_is_an_import_free_marker() -> None:
    tree = ast.parse(Path(OpenPinch.__file__).read_text(encoding="utf-8"))

    assert not hasattr(OpenPinch, "__all__")
    assert not any(isinstance(node, ast.Import | ast.ImportFrom) for node in tree.body)
    assert not hasattr(OpenPinch, "PinchProblem")
    assert not hasattr(OpenPinch, "PinchWorkspace")
    assert not hasattr(OpenPinch, "TargetInput")
    assert not hasattr(OpenPinch, "pinch_analysis_service")


@pytest.mark.parametrize("package_name", RETIRED_PACKAGES)
def test_retired_package_imports_fail(package_name: str) -> None:
    qualified_name = f"OpenPinch.{package_name}"
    assert importlib.util.find_spec(qualified_name) is None

    completed = subprocess.run(
        [sys.executable, "-c", f"import {qualified_name}"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "ModuleNotFoundError" in completed.stderr


def test_parent_owned_runtime_records_are_not_public() -> None:
    private_records = {
        "StreamSegment",
        "HeatExchangerPeriodState",
        "HeatExchangerAreaSlice",
    }
    assert all(not hasattr(stream, name) for name in private_records)
    assert all(not hasattr(heat_exchanger, name) for name in private_records)


def test_stream_segment_schema_remains_on_the_main_input_contract() -> None:
    assert StreamSegmentSchema.__module__ == "OpenPinch.contracts.input"
    schema = TargetInput.model_json_schema()
    assert "StreamSegmentSchema" in schema["$defs"]


def test_serialized_hen_schemas_are_owned_by_the_main_input_contract() -> None:
    schemas = (
        HeatExchangerAreaSliceSchema,
        HeatExchangerPeriodStateSchema,
        HeatExchangerSchema,
        HeatExchangerNetworkSchema,
    )
    assert all(schema.__module__ == "OpenPinch.contracts.input" for schema in schemas)
    definitions = TargetInput.model_json_schema()["$defs"]
    assert {schema.__name__ for schema in schemas} <= definitions.keys()
    assert not hasattr(OpenPinch, "HeatExchangerNetworkSchema")


def test_service_runtime_records_and_graph_specs_are_private() -> None:
    import OpenPinch.analysis.heat_pumps._multiperiod.state as multiperiod_state
    import OpenPinch.analysis.heat_pumps.process_mvr as process_mvr
    from OpenPinch.analysis.graphs import metadata, specifications
    from OpenPinch.presentation.dashboard import rendering

    assert not hasattr(process_mvr, "ProcessMVRStreamRecord")
    assert not hasattr(specifications, "GraphBuildSpec")
    assert not hasattr(metadata, "GraphSeriesMeta")
    assert not hasattr(multiperiod_state, "PreparedHPRPeriodCase")
    assert not hasattr(rendering, "StreamlitGraphSet")


def test_typed_hpr_records_do_not_emulate_dictionaries() -> None:
    assert {"__getitem__", "get"}.isdisjoint(HPRParsedState.__dict__)
    assert {"__getitem__", "get", "__contains__"}.isdisjoint(HPRBackendResult.__dict__)


def test_hen_model_package_is_a_marker_without_runtime_exports() -> None:
    from OpenPinch.analysis.heat_exchanger_networks import models

    forbidden = {
        "InternalHeatExchangerNetworkProblem",
        "ModelSliceUnavailableError",
        "SolverRun",
    }
    assert not hasattr(models, "__all__")
    assert all(not hasattr(models, name) for name in forbidden)
