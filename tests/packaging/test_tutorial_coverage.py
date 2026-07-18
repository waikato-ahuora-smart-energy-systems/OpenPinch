"""Drift guards for the public workflow and tutorial coverage manifest."""

from __future__ import annotations

import csv
import inspect
from pathlib import Path

from OpenPinch import PinchProblem, PinchWorkspace
from OpenPinch.analysis.heat_pumps.process_mvr import ProcessMVRComponent
from OpenPinch.application._problem.accessors.component import _ComponentAccessor
from OpenPinch.application._problem.accessors.design import (
    HeatExchangerNetworkDesignView,
    _DesignAccessor,
)
from OpenPinch.application._problem.accessors.target import (
    _AllPeriodsTargetAccessor,
    _TargetAccessor,
)
from OpenPinch.application.workspace import (
    CaseBatchResult,
    _CaseBatch,
    _CaseBatchAllPeriodsTargetAccessor,
    _CaseBatchDesignAccessor,
    _CaseBatchTargetAccessor,
)
from OpenPinch.presentation.graphs.problem import _PlotAccessor

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "docs" / "_data" / "tutorial-coverage.csv"
OWNERS = {
    "PinchProblem": (PinchProblem, "PinchProblem"),
    "PinchWorkspace": (PinchWorkspace, "PinchWorkspace"),
    "Target": (_TargetAccessor, "problem.target"),
    "All-period target": (
        _AllPeriodsTargetAccessor,
        "problem.target.all_periods",
    ),
    "Components": (_ComponentAccessor, "problem.components"),
    "Design": (_DesignAccessor, "problem.design"),
    "Design result": (HeatExchangerNetworkDesignView, "design_result"),
    "Plot": (_PlotAccessor, "problem.plot"),
    "Process MVR result": (ProcessMVRComponent, "mvr"),
    "Case batch": (_CaseBatch, "batch"),
    "Batch target": (_CaseBatchTargetAccessor, "batch.target"),
    "Batch all-period target": (
        _CaseBatchAllPeriodsTargetAccessor,
        "batch.target.all_periods",
    ),
    "Batch design": (_CaseBatchDesignAccessor, "batch.design"),
    "Batch result": (CaseBatchResult, "batch_result"),
}


def _rows() -> list[dict[str, str]]:
    with MANIFEST.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _live_operations() -> dict[str, str]:
    operations = {
        "PinchProblem.__init__": "constructor",
        "PinchWorkspace.__init__": "constructor",
    }
    for _owner_name, (owner, prefix) in OWNERS.items():
        for name, value in inspect.getmembers(owner):
            if name.startswith("_"):
                continue
            classification = (
                "property"
                if isinstance(value, property)
                else "method"
                if callable(value)
                else "accessor"
            )
            operations[f"{prefix}.{name}"] = classification
    return operations


def test_manifest_exactly_matches_live_public_inventory() -> None:
    rows = _rows()
    manifest = {row["operation"]: row["classification"] for row in rows}

    assert manifest == _live_operations()
    assert all(row["owner"] in OWNERS for row in rows)
    assert {row["coverage_status"] for row in rows} == {
        "mapped and executable",
        "mapped; runtime unsupported",
    }
    unsupported = {
        row["operation"]
        for row in rows
        if row["coverage_status"] == "mapped; runtime unsupported"
    }
    assert unsupported == {
        "problem.target.brayton_heat_pump",
        "problem.target.brayton_refrigeration",
        "batch.target.brayton_heat_pump",
        "batch.target.brayton_refrigeration",
    }


def test_manifest_has_complete_tutorial_and_profile_ownership() -> None:
    notebooks = {
        path.name
        for path in (ROOT / "OpenPinch" / "data" / "notebooks").glob("*.ipynb")
    }
    rows = _rows()

    assert {row["primary_tutorial"] for row in rows} == notebooks
    assert all(row["primary_tutorial"] in notebooks for row in rows)
    assert all(
        row["execution_profile"] in {"base", "slow-hpr", "solver", "interactive"}
        for row in rows
    )


def test_manifest_tracks_required_semantic_dimensions_and_evidence() -> None:
    rows = _rows()
    dimensions = {
        "source_type",
        "zone_scope",
        "config_precedence",
        "placement",
        "period_scope",
        "aggregation",
        "workspace_selection",
        "hen_method",
        "plot_behavior",
        "execution_evidence",
    }

    assert dimensions <= set(rows[0])
    for dimension in dimensions:
        assert all(row[dimension] for row in rows), dimension
    combined = "\n".join(";".join(row.values()) for row in rows)
    for required_mode in (
        "packaged sample;mapping",
        "focused zone;site;total site",
        "keyword > options > stored config > default",
        "process;utility",
        "all periods",
        "per-period;weighted average",
        "named case;active case;ordered batch",
        "multiperiod_heat_exchanger_network",
        "return figure;explicit export",
    ):
        assert required_mode in combined


def test_rtd_coverage_summary_matches_manifest_denominator() -> None:
    summary = (ROOT / "docs" / "examples" / "tutorial-coverage-map.rst").read_text(
        encoding="utf-8"
    )
    count = len(_rows())

    assert f"**{count} operations**" in summary
    assert f"**{count}/{count}, or 100 percent mapping" in summary
