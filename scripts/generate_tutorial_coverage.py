"""Generate the operation-level tutorial coverage manifest."""

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

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "docs" / "_data" / "tutorial-coverage.csv"

NOTEBOOK_PROFILES = {
    **{index: "base" for index in range(1, 8)},
    8: "slow-hpr",
    9: "slow-hpr",
    10: "slow-hpr",
    11: "slow-hpr",
    12: "base",
    13: "base",
    14: "base",
    15: "solver",
    16: "solver",
    17: "solver",
    18: "interactive",
}
NOTEBOOKS = {
    1: "01_first_solve_and_core_curves.ipynb",
    2: "02_focused_direct_and_total_site.ipynb",
    3: "03_multisegment_streams.ipynb",
    4: "04_workspace_cases_and_scenarios.ipynb",
    5: "05_workspace_persistence.ipynb",
    6: "06_multiperiod_heat_integration.ipynb",
    7: "07_area_cost_and_exergy.ipynb",
    8: "08_carnot_heat_pump_and_refrigeration.ipynb",
    9: "09_vapour_compression_and_brayton.ipynb",
    10: "10_multiperiod_heat_pumps.ipynb",
    11: "11_process_mvr_and_cascade.ipynb",
    12: "12_cogeneration.ipynb",
    13: "13_multiperiod_cogeneration.ipynb",
    14: "14_energy_transfer.ipynb",
    15: "15_hen_synthesis_and_selection.ipynb",
    16: "16_advanced_hen_methods.ipynb",
    17: "17_multiperiod_hen_synthesis.ipynb",
    18: "18_results_plots_reports_exports.ipynb",
}

TARGET_TUTORIALS = {
    "all_heat_integration": 1,
    "direct_heat_integration": 2,
    "indirect_heat_integration": 2,
    "total_site_heat_integration": 2,
    "heat_exchanger_area_and_cost": 7,
    "exergy": 7,
    "carnot_heat_pump": 8,
    "carnot_refrigeration": 8,
    "vapour_compression_heat_pump": 9,
    "vapour_compression_refrigeration": 9,
    "brayton_heat_pump": 9,
    "brayton_refrigeration": 9,
    "mvr_heat_pump": 11,
    "cogeneration": 12,
    "sun_smith_cogeneration": 12,
    "varbanov_cogeneration": 12,
    "isentropic_cogeneration": 12,
    "energy_transfer": 14,
    "all_periods": 6,
}
ALL_PERIOD_TUTORIALS = {
    "all_heat_integration": 6,
    "direct_heat_integration": 6,
    "indirect_heat_integration": 6,
    "total_site_heat_integration": 6,
    "heat_exchanger_area_and_cost": 7,
    "exergy": 7,
    "energy_transfer": 14,
    "carnot_heat_pump": 10,
    "carnot_refrigeration": 10,
    "vapour_compression_heat_pump": 10,
    "vapour_compression_refrigeration": 10,
    "mvr_heat_pump": 10,
    "cogeneration": 13,
    "sun_smith_cogeneration": 13,
    "varbanov_cogeneration": 13,
    "isentropic_cogeneration": 13,
}
DESIGN_TUTORIALS = {
    "heat_exchanger_network": 15,
    "multiperiod_heat_exchanger_network": 17,
    "enhanced_heat_exchanger_network": 16,
    "open_hens": 16,
    "pinch_design": 16,
    "thermal_derivative": 16,
    "network_evolution": 16,
}
PLOT_TUTORIALS = {
    "catalog": 1,
    "data": 1,
    "composite_curve": 1,
    "shifted_composite_curve": 1,
    "balanced_composite_curve": 1,
    "grand_composite_curve": 1,
    "real_grand_composite_curve": 1,
    "total_site_profiles": 2,
    "site_utility_grand_composite_curve": 2,
    "exergetic_grand_composite_curve": 7,
    "exergetic_net_load_profiles": 7,
    "grand_composite_curve_with_heat_pump": 8,
    "net_load_profiles_with_heat_pump": 8,
    "energy_transfer_diagram": 14,
    "net_load_profiles": 18,
    "export": 18,
    "export_gallery": 18,
}


def _public_members(owner: type) -> list[tuple[str, object]]:
    return [
        (name, value)
        for name, value in inspect.getmembers(owner)
        if not name.startswith("_")
    ]


def _classification(value: object) -> str:
    if isinstance(value, property):
        return "property"
    if callable(value):
        return "method"
    return "accessor"


def _mode(name: str, classification: str) -> str:
    if name in {"export", "export_gallery", "export_excel", "show_dashboard"}:
        return "explicit side effect"
    if classification in {"property", "accessor"} or name in {
        "catalog",
        "data",
        "metrics",
        "report",
        "summary_frame",
        "validation_report",
    }:
        return "cached observation"
    if name in {"load", "load_bundle", "scenario", "update_options"}:
        return "preparation"
    return "explicit execution"


def _dimensions(
    owner_name: str,
    name: str,
    tutorial_number: int,
    profile: str,
) -> dict[str, str]:
    operation = f"{owner_name}.{name}".lower()
    return {
        "source_type": (
            "packaged sample;mapping"
            if name in {"__init__", "load"}
            else "workspace bundle"
            if name in {"load_bundle", "save_bundle"}
            else "n/a"
        ),
        "zone_scope": (
            "focused zone;site;total site"
            if tutorial_number in {2, 14}
            else "site"
            if "target" in operation
            else "n/a"
        ),
        "config_precedence": (
            "keyword > options > stored config > default"
            if owner_name
            in {
                "Target",
                "All-period target",
                "Batch target",
                "Batch all-period target",
                "Components",
                "Batch design",
                "Design",
            }
            else "stored fallback"
            if name in {"config", "update_options", "set_dt_cont_multiplier"}
            else "n/a"
        ),
        "placement": (
            "process;utility"
            if "carnot" in name
            else "process"
            if name in {"add_process_mvr", "mvr_heat_pump"}
            else "n/a"
        ),
        "period_scope": "all periods"
        if "period" in owner_name.lower()
        else "selected period",
        "aggregation": (
            "per-period;weighted average"
            if tutorial_number in {6, 10, 13, 17}
            else "single result"
        ),
        "workspace_selection": (
            "named case;active case;ordered batch"
            if owner_name.startswith("Batch") or owner_name == "Case batch"
            else "active case"
            if owner_name == "PinchWorkspace"
            else "n/a"
        ),
        "hen_method": (
            name
            if owner_name in {"Design", "Batch design"}
            else "ranked result inspection"
            if owner_name == "Design result"
            else "n/a"
        ),
        "plot_behavior": (
            "return figure;explicit export"
            if owner_name == "Plot" and name in {"export", "export_gallery"}
            else "return figure"
            if owner_name == "Plot"
            else "n/a"
        ),
        "execution_evidence": (
            "batch delegation contract; "
            + (
                "routine pytest execution"
                if profile == "base"
                else f"opt-in executable profile: {profile}"
            )
            if owner_name.startswith("Batch") or owner_name == "Case batch"
            else "routine pytest execution"
            if profile == "base"
            else f"opt-in executable profile: {profile}"
        ),
    }


def _problem_tutorial(name: str) -> int:
    if name in {"period_ids", "period_results"}:
        return 6
    if name in {"components", "process_components"}:
        return 11
    if name == "design":
        return 15
    if name in {"compare_to", "config", "set_dt_cont_multiplier", "update_options"}:
        return 4
    if name in {
        "hot_streams",
        "cold_streams",
        "hot_utilities",
        "cold_utilities",
        "master_zone",
    }:
        return 3
    if name in {"target", "validate", "validation_report", "project_name"}:
        return 1
    return 18


def _workspace_tutorial(name: str) -> int:
    if name in {
        "active_case_name",
        "case",
        "cases",
        "compare_cases",
        "compare_to",
        "config",
        "list_cases",
        "scenario",
        "set_dt_cont_multiplier",
        "target",
        "update_options",
        "use_case",
    }:
        return 4
    if name in {
        "load",
        "load_bundle",
        "master_zone",
        "metrics",
        "problem_data",
        "problem_filepath",
        "report",
        "results",
        "save_bundle",
        "summary_frame",
        "to_problem_json",
        "validate",
        "validation_report",
    }:
        return 5
    return 18


def _rows() -> list[dict[str, str]]:
    specifications = (
        ("PinchProblem", PinchProblem, "PinchProblem", _problem_tutorial),
        ("PinchWorkspace", PinchWorkspace, "PinchWorkspace", _workspace_tutorial),
        ("Target", _TargetAccessor, "problem.target", TARGET_TUTORIALS.__getitem__),
        (
            "All-period target",
            _AllPeriodsTargetAccessor,
            "problem.target.all_periods",
            ALL_PERIOD_TUTORIALS.__getitem__,
        ),
        ("Components", _ComponentAccessor, "problem.components", lambda _name: 11),
        ("Design", _DesignAccessor, "problem.design", DESIGN_TUTORIALS.__getitem__),
        (
            "Design result",
            HeatExchangerNetworkDesignView,
            "design_result",
            lambda _name: 15,
        ),
        ("Plot", _PlotAccessor, "problem.plot", PLOT_TUTORIALS.__getitem__),
        (
            "Process MVR result",
            ProcessMVRComponent,
            "mvr",
            lambda _name: 11,
        ),
        (
            "Case batch",
            _CaseBatch,
            "batch",
            lambda name: 18 if name == "export_excel" else 4,
        ),
        (
            "Batch target",
            _CaseBatchTargetAccessor,
            "batch.target",
            TARGET_TUTORIALS.__getitem__,
        ),
        (
            "Batch all-period target",
            _CaseBatchAllPeriodsTargetAccessor,
            "batch.target.all_periods",
            ALL_PERIOD_TUTORIALS.__getitem__,
        ),
        (
            "Batch design",
            _CaseBatchDesignAccessor,
            "batch.design",
            DESIGN_TUTORIALS.__getitem__,
        ),
        ("Batch result", CaseBatchResult, "batch_result", lambda _name: 4),
    )
    rows = []
    constructor_rows = (
        ("PinchProblem", "PinchProblem.__init__", 1),
        ("PinchWorkspace", "PinchWorkspace.__init__", 4),
    )
    for owner_name, operation, tutorial_number in constructor_rows:
        profile = NOTEBOOK_PROFILES[tutorial_number]
        rows.append(
            {
                "owner": owner_name,
                "operation": operation,
                "classification": "constructor",
                "semantic_mode": "preparation",
                "primary_tutorial": NOTEBOOKS[tutorial_number],
                "secondary_tutorials": "",
                "execution_profile": profile,
                "optional_dependency": "none",
                "coverage_status": "mapped and executable",
                **_dimensions(owner_name, "__init__", tutorial_number, profile),
            }
        )
    for owner_name, owner, prefix, tutorial_for in specifications:
        for name, value in _public_members(owner):
            tutorial_number = tutorial_for(name)
            profile = NOTEBOOK_PROFILES[tutorial_number]
            classification = _classification(value)
            rows.append(
                {
                    "owner": owner_name,
                    "operation": f"{prefix}.{name}",
                    "classification": classification,
                    "semantic_mode": _mode(name, classification),
                    "primary_tutorial": NOTEBOOKS[tutorial_number],
                    "secondary_tutorials": (
                        NOTEBOOKS[4]
                        if owner_name.startswith("Batch") and tutorial_number != 4
                        else ""
                    ),
                    "execution_profile": profile,
                    "optional_dependency": {
                        "base": "none",
                        "slow-hpr": "hpr",
                        "solver": "hen",
                        "interactive": "plot;excel;dashboard",
                    }[profile],
                    "coverage_status": (
                        "mapped; runtime unsupported"
                        if name.startswith("brayton_")
                        else "mapped and executable"
                    ),
                    **_dimensions(owner_name, name, tutorial_number, profile),
                }
            )
    return rows


def main() -> None:
    rows = _rows()
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    with DESTINATION.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
