"""Drift guards for the public workflow and tutorial coverage manifest."""

from __future__ import annotations

import csv
import inspect
import tomllib
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


UNDER_DEVELOPMENT = "unmapped; under development"


def _rows() -> list[dict[str, str]]:
    with MANIFEST.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _mapped_rows() -> list[dict[str, str]]:
    return [row for row in _rows() if row["coverage_status"] != UNDER_DEVELOPMENT]


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
        "mapped; documented independent replay",
        UNDER_DEVELOPMENT,
    }
    # Methods still under development stay public but no notebook shows them.
    under_development = {
        row["operation"] for row in rows if row["coverage_status"] == UNDER_DEVELOPMENT
    }
    assert under_development == {
        "problem.target.brayton_heat_pump",
        "problem.target.brayton_refrigeration",
        "batch.target.brayton_heat_pump",
        "batch.target.brayton_refrigeration",
        "problem.target.carnot_orc",
        "problem.target.all_periods.carnot_orc",
        "batch.target.carnot_orc",
        "batch.target.all_periods.carnot_orc",
        "problem.target.organic_rankine_cycle",
        "problem.target.all_periods.organic_rankine_cycle",
        "batch.target.organic_rankine_cycle",
        "batch.target.all_periods.organic_rankine_cycle",
        "problem.plot.grand_composite_curve_with_orc",
        "problem.plot.net_load_profiles_with_orc",
    }
    documented_replays = {
        row["operation"]
        for row in rows
        if row["coverage_status"] == "mapped; documented independent replay"
    }
    assert documented_replays == {
        "problem.target.all_periods.carnot_heat_pump",
        "problem.target.all_periods.carnot_refrigeration",
        "problem.target.all_periods.mvr_heat_pump",
        "problem.target.all_periods.vapour_compression_heat_pump",
        "problem.target.all_periods.vapour_compression_refrigeration",
    }


def test_manifest_has_complete_tutorial_and_profile_ownership() -> None:
    notebooks = {
        path.name
        for path in (ROOT / "OpenPinch" / "tutorials" / "notebooks").glob("*.ipynb")
    }
    rows = _mapped_rows()

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
    mapped = len(_mapped_rows())

    assert f"**{count} operations**" in summary
    assert f"**{mapped}/{count} operations are mapped**" in summary


def test_committed_manifest_matches_generator_output() -> None:
    from scripts.generate_tutorial_coverage import render_manifest

    assert MANIFEST.read_text(encoding="utf-8") == render_manifest()


def test_observation_operations_are_not_explicit_execution() -> None:
    modes = {row["operation"]: row["semantic_mode"] for row in _rows()}

    observations = {
        "PinchProblem.validate",
        "PinchProblem.compare_to",
        "PinchProblem.to_problem_json",
        "PinchWorkspace.compare_cases",
        "PinchWorkspace.list_cases",
        "PinchWorkspace.to_bundle",
        "batch.reports",
        "batch.summary_frames",
        "design_result.views",
        "mvr.work_for_zone",
    }
    assert {modes[name] for name in observations} == {"cached observation"}
    plots = {
        row["semantic_mode"]
        for row in _rows()
        if row["owner"] == "Plot"
        and row["operation"]
        not in {"problem.plot.export", "problem.plot.export_gallery"}
    }
    assert plots == {"cached observation"}
    assert modes["PinchWorkspace.save_bundle"] == "explicit side effect"
    assert modes["problem.target.direct_heat_integration"] == "explicit execution"
    assert modes["problem.design.heat_exchanger_network"] == "explicit execution"


def test_optional_dependencies_name_real_extras() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = set(pyproject["project"]["optional-dependencies"])
    rows = _rows()

    for row in rows:
        value = row["optional_dependency"].split(" (")[0]
        assert value in extras | {"none", "n/a"}, row["operation"]
    by_operation = {row["operation"]: row["optional_dependency"] for row in rows}
    assert by_operation["problem.plot.composite_curve"] == "notebook"
    assert by_operation["problem.plot.export"] == "notebook"
    assert by_operation["problem.plot.catalog"] == "none"
    assert by_operation["problem.design.heat_exchanger_network"] == "synthesis"
    assert by_operation["problem.target.vapour_compression_heat_pump"].startswith(
        "tespy"
    )
    assert by_operation["problem.target.hpr_performance_map"].startswith("tespy")
    assert by_operation["problem.components.add_process_mvr"] == "none"
    assert by_operation["problem.target.carnot_heat_pump"] == "none"


def test_execution_evidence_matches_ci_gating() -> None:
    evidence = {row["execution_profile"]: set() for row in _rows()}
    for row in _rows():
        evidence[row["execution_profile"]].add(row["execution_evidence"])
    text = "\n".join(row["execution_evidence"] for row in _rows())

    assert "opt-in executable profile: interactive" not in text
    assert "opt-in executable profile: slow-hpr" not in text
    assert "opt-in slow-hpr" not in text
    assert "routine pytest execution" in evidence["interactive"]
    assert "main-bound solver CI execution" in evidence["solver"]
