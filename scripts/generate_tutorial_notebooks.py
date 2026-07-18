"""Generate the supported process-engineer tutorial notebook sources."""

# Long prose literals remain readable as complete tutorial sentences.
# ruff: noqa: E501

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "OpenPinch" / "data" / "notebooks"


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(True),
    }


def tutorial(
    title: str,
    *,
    level: str,
    profile: str,
    runtime: str,
    extras: str,
    cells: list[dict],
) -> dict:
    introduction = markdown(
        f"# {title}\n\n"
        f"**Learning outcome:** Apply {title.lower()} through the public "
        "`PinchProblem` or `PinchWorkspace` workflow.\n\n"
        f"**Level:** {level}  \n**Execution profile:** `{profile}`  \n"
        f"**Expected runtime:** {runtime}  \n**Optional extras:** {extras}\n\n"
        "The lifecycle is explicit: prepare the study, run the named method, "
        "then inspect cached results. Observation cells do not launch analysis."
    )
    notebook_cells = [introduction, *cells]
    for index, cell in enumerate(notebook_cells, start=1):
        cell["id"] = f"cell-{index:02d}"
    return {
        "cells": notebook_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
            "openpinch": {"profile": profile},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


NOTEBOOKS = {
    "01_first_solve_and_core_curves.ipynb": tutorial(
        "First Solve and Core Curves",
        level="Core",
        profile="base",
        runtime="under 1 minute",
        extras="plot",
        cells=[
            code(
                "from OpenPinch import PinchProblem\n\n"
                'problem = PinchProblem("basic_pinch.json", project_name="Site")\n'
                "validated = problem.validate()\n"
                "validation = problem.validation_report()\n"
                "target = problem.target.all_heat_integration()\n"
                "cached_results = problem.results\n"
                "summary = problem.summary_frame()\n"
                "summary"
            ),
            code(
                "metrics = problem.metrics()\n"
                "report = problem.report()\n"
                "available_plots = problem.plot.catalog()\n"
                "graph_data = problem.plot.data()\n"
                "available_plots"
            ),
            code(
                "composite = problem.plot.composite_curve()\n"
                "shifted = problem.plot.shifted_composite_curve()\n"
                "balanced = problem.plot.balanced_composite_curve()\n"
                "grand = problem.plot.grand_composite_curve()\n"
                "real_grand = problem.plot.real_grand_composite_curve()"
            ),
        ],
    ),
    "02_focused_direct_and_total_site.ipynb": tutorial(
        "Focused Direct and Total Site",
        level="Core",
        profile="base",
        runtime="under 2 minutes",
        extras="plot",
        cells=[
            code(
                "from OpenPinch import PinchProblem\n\n"
                'problem = PinchProblem("pulp_mill.json", project_name="Site")\n'
                "complete = problem.target.all_heat_integration()\n"
                "summary = problem.summary_frame()\n"
                "summary"
            ),
            code(
                'direct = problem.target.direct_heat_integration(zone="Bleaching")\n'
                "indirect = problem.target.indirect_heat_integration()\n"
                "total_site = problem.target.total_site_heat_integration()\n"
                "total_site_profiles = problem.plot.total_site_profiles()\n"
                "site_utility_curve = problem.plot.site_utility_grand_composite_curve()"
            ),
        ],
    ),
    "03_multisegment_streams.ipynb": tutorial(
        "Multi-Segment Streams",
        level="Intermediate",
        profile="base",
        runtime="under 1 minute",
        extras="none",
        cells=[
            code(
                "from OpenPinch import PinchProblem\n\n"
                "segmented_input = {\n"
                '    "streams": [\n'
                '        {"zone": "Site", "name": "Variable CP hot", '
                '"segments": [\n'
                '            {"name": "hot-1", "t_supply": 200.0, '
                '"t_target": 150.0, "heat_flow": 50.0},\n'
                '            {"name": "hot-2", "t_supply": 150.0, '
                '"t_target": 100.0, "heat_flow": 100.0},\n'
                "        ]},\n"
                "        {\n"
                '            "zone": "Site",\n'
                '            "name": "Cold demand",\n'
                '            "t_supply": 40.0,\n'
                '            "t_target": 170.0,\n'
                '            "heat_flow": 130.0,\n'
                "        },\n"
                "    ],\n"
                '    "utilities": [],\n'
                "}\n"
                'problem = PinchProblem(segmented_input, project_name="Site")\n'
                "validated = problem.validate()\n"
                "target = problem.target.direct_heat_integration()\n"
                "site_zone = problem.master_zone\n"
                "problem.summary_frame()"
            ),
            code(
                "prepared_hot = list(problem.hot_streams)[0]\n"
                "prepared_cold = list(problem.cold_streams)[0]\n"
                "hot_utilities = list(problem.hot_utilities)\n"
                "cold_utilities = list(problem.cold_utilities)\n"
                "segment_table = [\n"
                '    {"name": segment.name, "supply": float(segment.supply_temperature), '
                '"target": float(segment.target_temperature), '
                '"duty": float(segment.heat_flow)}\n'
                "    for segment in prepared_hot.segments\n"
                "]\n"
                "assert all(\n"
                "    first.target_temperature == second.supply_temperature\n"
                "    for first, second in zip(\n"
                "        prepared_hot.segments, prepared_hot.segments[1:]\n"
                "    )\n"
                ")\n"
                "segment_table"
            ),
        ],
    ),
    "04_workspace_cases_and_scenarios.ipynb": tutorial(
        "Workspace Cases and Scenarios",
        level="Intermediate",
        profile="base",
        runtime="under 2 minutes",
        extras="none",
        cells=[
            code(
                "from OpenPinch import PinchWorkspace\n\n"
                'workspace = PinchWorkspace("crude_preheat_train.json", '
                'project_name="Crude Site")\n'
                'baseline = workspace.case("baseline")\n'
                "baseline.target.direct_heat_integration()\n"
                'tight_dt = workspace.scenario("tight_dt", '
                "dt_cont_multiplier=0.75)\n"
                "tight_dt.target.direct_heat_integration()\n"
                'case_comparison = workspace.compare_cases("baseline", "tight_dt")\n'
                "baseline_comparison = baseline.compare_to(tight_dt)"
            ),
            code(
                'batch = workspace.cases(["baseline", "tight_dt"])\n'
                "batch_results = batch.target.direct_heat_integration()\n"
                "batch_summaries = batch.summary_frames()\n"
                "batch_metrics = batch.metrics()\n"
                "batch_reports = batch.reports()\n"
                "batch_errors = batch_results.errors\n"
                'workspace.use_case("tight_dt")\n'
                'baseline.update_options({"THERMAL_DT_CONT": 10.0})\n'
                "baseline.set_dt_cont_multiplier(1.0)\n"
                "baseline.config\n"
                'workspace.update_options({"THERMAL_DT_CONT": 12.0})\n'
                "workspace.set_dt_cont_multiplier(1.0)\n"
                "workspace.config\n"
                "workspace.project_name\n"
                "workspace.baseline_name\n"
                "workspace.active_case_name\n"
                "workspace.list_cases()"
            ),
        ],
    ),
    "05_workspace_persistence.ipynb": tutorial(
        "Workspace Data and Persistence",
        level="Intermediate",
        profile="base",
        runtime="under 1 minute",
        extras="none",
        cells=[
            code(
                "from pathlib import Path\n\n"
                "from OpenPinch import PinchWorkspace\n\n"
                'workspace = PinchWorkspace("basic_pinch.json", project_name="Site")\n'
                "workspace.target.direct_heat_integration()\n"
                "active_problem = workspace.case()\n"
                "problem_data = active_problem.problem_data\n"
                "problem_filepath = active_problem.problem_filepath\n"
                "master_zone = active_problem.master_zone\n"
                "cached_results = workspace.results\n"
                "summary = workspace.summary_frame()\n"
                "metrics = workspace.metrics()\n"
                "report = workspace.report()\n"
                "summary"
            ),
            code(
                'bundle_path = Path("openpinch-workspace.json")\n'
                "saved = workspace.save_bundle(bundle_path)\n"
                "restored = PinchWorkspace.load_bundle(saved)\n"
                'loaded = PinchWorkspace(project_name="Site")\n'
                'case_input = restored.to_problem_json(case_name="baseline")\n'
                'validated = restored.validate("baseline")\n'
                'validation = restored.validation_report("baseline")\n'
                "case_input"
            ),
        ],
    ),
    "06_multiperiod_heat_integration.ipynb": tutorial(
        "Multiperiod Heat Integration",
        level="Intermediate",
        profile="base",
        runtime="under 2 minutes",
        extras="none",
        cells=[
            code(
                "from OpenPinch import PinchProblem\n\n"
                "base_problem = PinchProblem(\n"
                '    "Four-stream-Yee-and-Grossmann-1990-1.json"\n'
                ")\n"
                "multiperiod_input = base_problem.to_problem_json()\n"
                'multiperiod_input["options"]["PROBLEM_PERIOD_IDS"] = [\n'
                '    "base", "turndown"\n'
                "]\n"
                'multiperiod_input["options"]["PROBLEM_PERIOD_WEIGHTS"] = [\n'
                "    0.7, 0.3\n"
                "]\n"
                'for stream in multiperiod_input["streams"]:\n'
                '    duty = stream["heat_flow"]\n'
                '    stream["heat_flow"] = {\n'
                '        "values": [duty["value"], 0.8 * duty["value"]],\n'
                '        "unit": duty["unit"],\n'
                "    }\n"
                '    stream["heat_capacity_flowrate"] = None\n'
                "problem = PinchProblem(\n"
                '    multiperiod_input, project_name="Four Stream Periods"\n'
                ")\n"
                "period_ids = list(problem.period_ids)\n"
                "period_outputs = problem.target.all_periods.all_heat_integration()\n"
                "direct_periods = "
                "problem.target.all_periods.direct_heat_integration()\n"
                "indirect_periods = "
                "problem.target.all_periods.indirect_heat_integration()\n"
                "site_periods = "
                "problem.target.all_periods.total_site_heat_integration()\n"
                "all_periods = problem.summary_frame(include_periods=True)\n"
                "assert list(period_outputs) == period_ids\n"
                "assert len(period_ids) >= 2\n"
                "all_periods"
            ),
            code(
                "weighted = problem.summary_frame(include_weighted_average=True)\n"
                "combined = problem.summary_frame(\n"
                "    include_periods=True, include_weighted_average=True\n"
                ")\n"
                "list(problem.period_results)"
            ),
        ],
    ),
    "07_area_cost_and_exergy.ipynb": tutorial(
        "Area Cost and Exergy",
        level="Intermediate",
        profile="base",
        runtime="under 2 minutes",
        extras="plot",
        cells=[
            code(
                "from OpenPinch import PinchProblem\n\n"
                'problem = PinchProblem("basic_pinch.json", project_name="Site")\n'
                "area_cost = problem.target.heat_exchanger_area_and_cost()\n"
                "area_cost_summary = problem.summary_frame()\n"
                "area_cost_summary"
            ),
            code(
                "base_target = problem.target.direct_heat_integration()\n"
                "exergy = problem.target.exergy(base_target=base_target)\n"
                "period_area_cost = "
                "problem.target.all_periods.heat_exchanger_area_and_cost()\n"
                "period_exergy = problem.target.all_periods.exergy()\n"
                "exergy_curve = problem.plot.exergetic_grand_composite_curve()\n"
                "exergy_loads = problem.plot.exergetic_net_load_profiles()"
            ),
        ],
    ),
    "08_carnot_heat_pump_and_refrigeration.ipynb": tutorial(
        "Carnot Heat Pump and Refrigeration",
        level="Intermediate",
        profile="slow-hpr",
        runtime="2 to 10 minutes",
        extras="hpr",
        cells=[
            code(
                "from OpenPinch import PinchProblem\n\n"
                'problem = PinchProblem("heat_pump_targeting.json", '
                'project_name="Heat Pump Study")\n'
                "heat_pump = problem.target.carnot_heat_pump(\n"
                "    is_utility_heat_pump=False,\n"
                "    is_cascade_cycle=True,\n"
                "    load_fraction=0.25,\n"
                "    condensers=1,\n"
                "    evaporators=1,\n"
                "    maximum_restarts=1,\n"
                ")\n"
                "hpr_summary = problem.summary_frame()"
            ),
            code(
                "refrigeration = problem.target.carnot_refrigeration(\n"
                "    is_utility_refrigeration=True,\n"
                "    load_fraction=0.25,\n"
                "    maximum_restarts=1,\n"
                ")\n"
                "load_plot = problem.plot.net_load_profiles_with_heat_pump()\n"
                "gcc_plot = problem.plot.grand_composite_curve_with_heat_pump()"
            ),
        ],
    ),
    "09_vapour_compression_and_brayton.ipynb": tutorial(
        "Vapour Compression and Brayton HPR",
        level="Advanced",
        profile="slow-hpr",
        runtime="5 to 30 minutes",
        extras="hpr",
        cells=[
            code(
                "from OpenPinch import PinchProblem\n\n"
                'problem = PinchProblem("heat_pump_targeting.json", '
                'project_name="HPR Models")\n'
                "def screen_cycle(method, **arguments):\n"
                "    try:\n"
                '        return {"status": "feasible", '
                '"result": method(**arguments)}\n'
                "    except (ValueError, NotImplementedError) as error:\n"
                '        return {"status": "no feasible solution", '
                '"reason": str(error)}\n\n'
                "vapour_compression = screen_cycle(\n"
                "    problem.target.vapour_compression_heat_pump,\n"
                '    refrigerants=["water", "ammonia"],\n'
                "    load_fraction=0.25,\n"
                "    maximum_restarts=1,\n"
                ")\n"
                "vapour_compression"
            ),
            code(
                "vc_refrigeration = screen_cycle(\n"
                "    problem.target.vapour_compression_refrigeration,\n"
                '    refrigerants=["ammonia"],\n'
                "    load_fraction=0.25,\n"
                "    maximum_restarts=1,\n"
                ")\n"
                "brayton = screen_cycle(\n"
                "    problem.target.brayton_heat_pump,\n"
                "    load_fraction=0.25, maximum_restarts=1\n"
                ")\n"
                "brayton_refrigeration = screen_cycle(\n"
                "    problem.target.brayton_refrigeration,\n"
                "    load_fraction=0.25, maximum_restarts=1\n"
                ")"
            ),
        ],
    ),
    "10_multiperiod_heat_pumps.ipynb": tutorial(
        "Multiperiod Heat Pumps",
        level="Advanced",
        profile="slow-hpr",
        runtime="5 to 30 minutes",
        extras="hpr",
        cells=[
            code(
                "from OpenPinch import PinchProblem\n\n"
                'problem = PinchProblem("crude_preheat_train_multiperiod.json", '
                'project_name="Crude HPR")\n'
                "def screen_periods(method, **arguments):\n"
                "    try:\n"
                '        return {"status": "feasible", '
                '"results": method(**arguments)}\n'
                "    except ValueError as error:\n"
                '        return {"status": "no shared feasible solution", '
                '"reason": str(error)}\n\n'
                "period_heat_pumps = problem.target.all_periods.carnot_heat_pump(\n"
                "    load_fraction=0.25, maximum_restarts=1\n"
                ")\n"
                "weighted = problem.summary_frame(include_weighted_average=True)\n"
                "weighted"
            ),
            code(
                "period_refrigeration = screen_periods(\n"
                "    problem.target.all_periods.carnot_refrigeration,\n"
                "    load_fraction=0.25, maximum_restarts=1\n"
                ")\n"
                "period_vc_heat_pumps = screen_periods(\n"
                "    problem.target.all_periods.vapour_compression_heat_pump,\n"
                '    refrigerants=["water"], load_fraction=0.25,\n'
                "    maximum_restarts=1,\n"
                ")\n"
                "period_vc_refrigeration = screen_periods(\n"
                "    problem.target.all_periods.vapour_compression_refrigeration,\n"
                '    refrigerants=["ammonia"], load_fraction=0.25,\n'
                "    maximum_restarts=1,\n"
                ")\n"
                "period_mvr = screen_periods(\n"
                "    problem.target.all_periods.mvr_heat_pump,\n"
                "    load_fraction=0.25, maximum_restarts=1\n"
                ")\n"
                "list(problem.period_results)"
            ),
        ],
    ),
    "11_process_mvr_and_cascade.ipynb": tutorial(
        "Process MVR and VC Cascade",
        level="Advanced",
        profile="slow-hpr",
        runtime="5 to 30 minutes",
        extras="hpr",
        cells=[
            code(
                "from OpenPinch import PinchProblem\n\n"
                'problem = PinchProblem("process_mvr.json", project_name="Site")\n'
                "mvr = problem.components.add_process_mvr(\n"
                '    "Evaporator vapour",\n'
                "    liquid_injection=False,\n"
                "    compressor_efficiency=0.72,\n"
                "    motor_efficiency=0.96,\n"
                ")\n"
                "target = problem.target.direct_heat_integration(\n"
                '    zone="Evaporation Train"\n'
                ")\n"
                "mvr_summary = problem.summary_frame()"
            ),
            code(
                "component_inventory = problem.components.inventory\n"
                "process_components = problem.process_components\n"
                "component_is_active = mvr.active\n"
                "component_type = mvr.component_type\n"
                "original_streams = mvr.original_streams\n"
                "replacement_streams = mvr.replacement_streams\n"
                "stage_results = mvr.stage_results_by_period\n"
                "affected_zones = mvr.affected_zone_paths\n"
                "compressor_work = mvr.work_for_zone(problem.master_zone)\n"
                "try:\n"
                "    cascade = problem.target.mvr_heat_pump(\n"
                "        load_fraction=0.25, maximum_restarts=1\n"
                "    )\n"
                "except (ValueError, RuntimeError, NotImplementedError) as error:\n"
                '    cascade = {"status": "no feasible solution", '
                '"reason": str(error)}\n'
                "mvr.deactivate()\n"
                "mvr.activate()\n"
                "component_inventory"
            ),
        ],
    ),
    "12_cogeneration.ipynb": tutorial(
        "Cogeneration",
        level="Intermediate",
        profile="base",
        runtime="under 3 minutes",
        extras="none",
        cells=[
            code(
                "from OpenPinch import PinchProblem\n\n"
                'problem = PinchProblem("pulp_mill.json", project_name="Site")\n'
                "problem.target.all_heat_integration()\n"
                "base = problem.target.total_site_heat_integration()\n"
                "default = problem.target.cogeneration(base_target=base)\n"
                "cogeneration_summary = problem.summary_frame()"
            ),
            code(
                "sun_smith = problem.target.sun_smith_cogeneration(base_target=base)\n"
                "varbanov = problem.target.varbanov_cogeneration(base_target=base)\n"
                "isentropic = problem.target.isentropic_cogeneration(\n"
                "    efficiency=0.8, base_target=base\n"
                ")"
            ),
        ],
    ),
    "13_multiperiod_cogeneration.ipynb": tutorial(
        "Multiperiod Cogeneration",
        level="Advanced",
        profile="base",
        runtime="under 5 minutes",
        extras="none",
        cells=[
            code(
                "from OpenPinch import PinchProblem\n\n"
                'problem = PinchProblem("zonal_site_multiperiod.json", '
                'project_name="Site")\n'
                "problem.target.all_periods.all_heat_integration()\n"
                "period_cogeneration = problem.target.all_periods.cogeneration()\n"
                "weighted = problem.summary_frame(include_weighted_average=True)\n"
                "weighted"
            ),
            code(
                "sun_smith = problem.target.all_periods.sun_smith_cogeneration()\n"
                "varbanov = problem.target.all_periods.varbanov_cogeneration()\n"
                "isentropic = "
                "problem.target.all_periods.isentropic_cogeneration(efficiency=0.8)\n"
                "list(problem.period_results)"
            ),
        ],
    ),
    "14_energy_transfer.ipynb": tutorial(
        "Energy Transfer",
        level="Intermediate",
        profile="base",
        runtime="under 3 minutes",
        extras="plot",
        cells=[
            code(
                "from OpenPinch import PinchProblem\n\n"
                'site = PinchProblem("pulp_mill.json", project_name="Site")\n'
                "site.target.all_heat_integration()\n"
                "base = site.target.total_site_heat_integration()\n"
                "energy_transfer = site.target.energy_transfer(base_target=base)\n"
                "transfer_summary = site.summary_frame()"
            ),
            code(
                "diagram = site.plot.energy_transfer_diagram()\n"
                "period_transfers = site.target.all_periods.energy_transfer()\n"
                'bleaching = PinchProblem("pulp_mill.json", project_name="Site")\n'
                'bleaching.target.direct_heat_integration(zone="Bleaching")\n'
                "comparison = site.compare_to(bleaching)"
            ),
        ],
    ),
    "15_hen_synthesis_and_selection.ipynb": tutorial(
        "HEN Synthesis and Selection",
        level="Advanced",
        profile="solver",
        runtime="5 to 60 minutes",
        extras="hen",
        cells=[
            code(
                "from OpenPinch import PinchProblem\n\n"
                "problem = PinchProblem(\n"
                '    "Four-stream-Yee-and-Grossmann-1990-1.json", '
                'project_name="Four Stream"\n'
                ")\n"
                "design = problem.design.heat_exchanger_network(\n"
                "    approach_temperatures=[14.0], stages=[1], best_solutions=1\n"
                ")\n"
                "ranked_networks = design.top(1)"
            ),
            code(
                "selected = design.network(rank=1)\n"
                "selected_property = design.selected_network\n"
                "grid = design.grid(rank=1)\n"
                'high_pressure_steam = design.utility("HP Steam")\n'
                "metrics = {\n"
                '    "recovery": design.total_heat_recovery,\n'
                '    "hot utility": design.total_hot_utility,\n'
                '    "cold utility": design.total_cold_utility,\n'
                "}\n"
                'serialized = design.result.model_dump(mode="json")\n'
                "metrics"
            ),
        ],
    ),
    "16_advanced_hen_methods.ipynb": tutorial(
        "Advanced HEN Methods",
        level="Advanced",
        profile="solver",
        runtime="10 to 90 minutes",
        extras="hen",
        cells=[
            code(
                "from OpenPinch import PinchProblem\n\n"
                "problem = PinchProblem(\n"
                '    "Four-stream-Yee-and-Grossmann-1990-1.json", '
                'project_name="Four Stream"\n'
                ")\n"
                "enhanced = problem.design.enhanced_heat_exchanger_network(\n"
                "    quality_tier=1\n"
                ")\n"
                "enhanced_ranking = enhanced.top(1)"
            ),
            code(
                "open_hens = problem.design.open_hens()\n"
                "open_hens_ranking = open_hens.top(1)\n"
                "pinch_design = problem.design.pinch_design()\n"
                "pinch_ranking = pinch_design.top(1)\n"
                "thermal = problem.design.thermal_derivative(\n"
                "    (pinch_design.selected_network,)\n"
                ")\n"
                "thermal_ranking = thermal.top(1)\n"
                "evolution = problem.design.network_evolution(\n"
                "    (thermal.selected_network,)\n"
                ")\n"
                "evolution_ranking = evolution.top(1)"
            ),
        ],
    ),
    "17_multiperiod_hen_synthesis.ipynb": tutorial(
        "Multiperiod HEN Synthesis",
        level="Advanced",
        profile="solver",
        runtime="10 to 90 minutes",
        extras="hen",
        cells=[
            code(
                "from OpenPinch import PinchProblem\n\n"
                "base_problem = PinchProblem(\n"
                '    "Four-stream-Yee-and-Grossmann-1990-1.json"\n'
                ")\n"
                "multiperiod_input = base_problem.to_problem_json()\n"
                'multiperiod_input["options"]["PROBLEM_PERIOD_IDS"] = [\n'
                '    "base", "turndown"\n'
                "]\n"
                'multiperiod_input["options"]["PROBLEM_PERIOD_WEIGHTS"] = [\n'
                "    0.7, 0.3\n"
                "]\n"
                'for stream in multiperiod_input["streams"]:\n'
                '    duty = stream["heat_flow"]\n'
                '    stream["heat_flow"] = {\n'
                '        "values": [duty["value"], 0.8 * duty["value"]],\n'
                '        "unit": duty["unit"],\n'
                "    }\n"
                '    stream["heat_capacity_flowrate"] = None\n'
                "problem = PinchProblem(\n"
                '    multiperiod_input, project_name="Four Stream Periods"\n'
                ")\n"
                "period_targets = "
                "problem.target.all_periods.all_heat_integration()\n"
                "design = problem.design.multiperiod_heat_exchanger_network(\n"
                "    stages=[3], best_solutions=1\n"
                ")\n"
                "ranked_networks = design.top(1)"
            ),
            code(
                "shared_network = design.network(rank=1)\n"
                "period_ids = list(problem.period_results)\n"
                "shared_grid = design.grid(rank=1, period_id=period_ids[0])\n"
                "shared_network.summary_metrics"
            ),
        ],
    ),
    "18_results_plots_reports_exports.ipynb": tutorial(
        "Results Plots Reports and Exports",
        level="Intermediate",
        profile="interactive",
        runtime="under 5 minutes",
        extras="plot, excel, dashboard",
        cells=[
            code(
                "from pathlib import Path\n\n"
                "from OpenPinch import PinchProblem, PinchWorkspace\n\n"
                'problem = PinchProblem("basic_pinch.json", project_name="Site")\n'
                'problem.update_options({"THERMAL_DT_CONT": 10.0})\n'
                "problem.set_dt_cont_multiplier(1.0)\n"
                "problem.target.all_heat_integration()\n"
                "problem_data = problem.problem_data\n"
                "problem_filepath = problem.problem_filepath\n"
                "cached_results = problem.results\n"
                "metrics = problem.metrics()\n"
                "summary = problem.summary_frame()\n"
                "report = problem.report()\n"
                "summary"
            ),
            code(
                'output_dir = Path("openpinch-results")\n'
                "input_payload = problem.to_problem_json()\n"
                'reloaded = PinchProblem(project_name="Site")\n'
                'reloaded.load("basic_pinch.json")\n'
                "net_loads = problem.plot.net_load_profiles()\n"
                "graph_data = problem.plot.data()\n"
                "catalog = problem.plot.catalog()\n"
                "plot_paths = problem.plot.export(\n"
                '    output_dir / "plots", plot=problem.plot.grand_composite_curve\n'
                ")\n"
                'gallery = problem.plot.export_gallery(output_dir / "gallery")\n'
                'workbook = problem.export_excel(output_dir / "tables")'
            ),
            code(
                "workspace = PinchWorkspace(\n"
                '    "basic_pinch.json", project_name="Site"\n'
                ")\n"
                "workspace.target.all_heat_integration()\n"
                "workspace.components.inventory\n"
                "workspace.design\n"
                "workspace.plot.catalog()\n"
                'workspace.export_excel(output_dir / "workspace-tables")\n'
                "workspace.show_dashboard()\n"
                "# Explicit interactive side effect; run after targeting.\n"
                "dashboard = problem.show_dashboard()"
            ),
        ],
    ),
}


GUIDANCE = {
    "01_first_solve_and_core_curves.ipynb": (
        "What are the minimum heating and cooling duties, and which curves explain them?",
        "Check the summary duties first, then use the composite and grand-composite curves to locate the pinch and utility opportunities.",
        "Replace `basic_pinch.json` with a plant case; keep the validate-target-inspect sequence unchanged.",
        (
            "Prepare and solve one case",
            "Inspect cached engineering outputs",
            "Build the core diagnostic curves",
        ),
    ),
    "02_focused_direct_and_total_site.ipynb": (
        "How do local process targets differ from indirect and Total Site opportunities?",
        "Compare like-for-like duties and retain the zone name with every focused result; Total Site curves describe utility-system opportunities, not individual exchanger matches.",
        'Change `zone="Bleaching"` to a path from your zone tree and compare the same scope across scenarios.',
        ("Establish the site-wide reference", "Run explicit integration scopes"),
    ),
    "03_multisegment_streams.ipynb": (
        "How should a stream with changing heat-capacity flow be represented without hiding its temperature intervals?",
        "Confirm adjacent segments are continuous and inspect each segment's temperature range and duty before trusting the target.",
        "Add one segment per thermodynamic interval; do not average heat-capacity flow across phase or property changes.",
        ("Define and target a segmented stream", "Audit the prepared segments"),
    ),
    "04_workspace_cases_and_scenarios.ipynb": (
        "How does a process engineer compare named design assumptions without mixing their state?",
        "Use the baseline as the reference, inspect batch errors explicitly, and compare the same method and reporting scope for every case.",
        "Create one scenario per engineering assumption, then use `workspace.cases(...)` for repeatable batch targeting and reporting.",
        ("Create and compare two cases", "Batch, inspect, and select cases"),
    ),
    "05_workspace_persistence.ipynb": (
        "How can a prepared case set be saved, restored, and validated without rerunning analysis implicitly?",
        "A restored bundle contains case inputs, while analysis results remain explicit runtime state; validate before launching a new workflow.",
        "Choose a project-specific bundle path and commit only stable input bundles, not transient result folders.",
        ("Inspect the active case", "Persist and restore the workspace"),
    ),
    "06_multiperiod_heat_integration.ipynb": (
        "How do heat-integration targets change by operating period, and what is the weighted annual view?",
        "Review every period before the weighted average; an annual average can conceal a limiting or infeasible operating state.",
        "Replace the sample with period-tagged plant data and verify period order and weights before comparing annual totals.",
        ("Run aligned period analyses", "Compare period and weighted summaries"),
    ),
    "07_area_cost_and_exergy.ipynb": (
        "How do energy targets translate into exchanger area, cost, and thermodynamic quality?",
        "Treat area and exergy as complementary screens: area exposes transfer difficulty, while exergy exposes quality loss.",
        "Supply your economic and film-coefficient assumptions through explicit arguments or stored fallback configuration.",
        ("Estimate area and cost", "Add exergy and period views"),
    ),
    "08_carnot_heat_pump_and_refrigeration.ipynb": (
        "Where can idealized heat pumping or refrigeration reduce utility demand before detailed equipment selection?",
        "Use Carnot results as screening bounds; compare delivered duty, lift, and utility displacement rather than COP alone.",
        "Vary load fraction and utility placement deliberately, then carry promising duties into a simulated-cycle study.",
        ("Screen a process heat pump", "Screen refrigeration and inspect curves"),
    ),
    "09_vapour_compression_and_brayton.ipynb": (
        "Which simulated heat-pump family and working fluid best fit the required temperature lift?",
        "Reject candidates on feasibility before ranking energy performance; Brayton and vapour-compression results are not interchangeable design models.",
        "Use a short, defensible refrigerant list and record why each cycle family is in scope.",
        (
            "Evaluate vapour-compression candidates",
            "Compare refrigeration and Brayton variants",
        ),
    ),
    "10_multiperiod_heat_pumps.ipynb": (
        "Does one heat-pump concept remain useful across all operating periods?",
        "Inspect period feasibility and load before annual weighting; a shared concept must work at the operating extremes.",
        "Use period-specific loads when plant availability or heat-source duty changes materially.",
        ("Run period Carnot screening", "Compare simulated and MVR period results"),
    ),
    "11_process_mvr_and_cascade.ipynb": (
        "How does direct process-vapour recompression change streams and the resulting heat-integration target?",
        "Audit original and replacement streams, compressor work, affected zones, and active state before comparing targets.",
        "Select vapour streams by stable names, expose efficiency assumptions, and keep one component id per physical proposal.",
        (
            "Add and target a process MVR component",
            "Inspect lifecycle and compare cascade targeting",
        ),
    ),
    "12_cogeneration.ipynb": (
        "How sensitive is cogeneration potential to the turbine model used for screening?",
        "Compare models against the same Total Site base target and state efficiency assumptions beside the reported power potential.",
        "Choose the model that matches available steam data; do not mix model outputs in one economic baseline.",
        ("Establish the cogeneration reference", "Compare named turbine models"),
    ),
    "13_multiperiod_cogeneration.ipynb": (
        "How does cogeneration potential vary with the site's operating periods?",
        "Check the period with the lowest viable steam flow as well as the weighted annual opportunity.",
        "Use actual period durations and compare identical turbine assumptions in every period.",
        ("Run the period reference and default model", "Compare period turbine models"),
    ),
    "14_energy_transfer.ipynb": (
        "Where does heat cross zone boundaries, and which transfers merit utility-system or project attention?",
        "Interpret transfers with their source and sink zones; a large transfer is an opportunity only if the physical connection is credible.",
        "Retain stable zone paths and compare focused cases to test proposed site boundaries.",
        (
            "Quantify site energy transfer",
            "Visualize periods and compare a focused zone",
        ),
    ),
    "15_hen_synthesis_and_selection.ipynb": (
        "Which ranked heat-exchanger network best balances recovery, utilities, area, and cost?",
        "Review several feasible networks and the exchanger grid; never select rank one from objective value alone.",
        "Narrow approach-temperature and stage ranges after a coarse screen, then serialize the chosen network for traceability.",
        (
            "Generate ranked network candidates",
            "Inspect and serialize a selected network",
        ),
    ),
    "16_advanced_hen_methods.ipynb": (
        "How do alternative synthesis and network-improvement methods change the candidate design?",
        "Compare every method on the same streams, utility assumptions, and economic basis; record solver status with each result.",
        "Start with one method and small search bounds, then pass only reviewed networks into derivative or evolution workflows.",
        ("Run enhanced synthesis", "Compare advanced design methods"),
    ),
    "17_multiperiod_hen_synthesis.ipynb": (
        "Can one exchanger network serve every operating period without hiding period constraints?",
        "Inspect the shared network grid in each period and identify the period controlling area and utility demand.",
        "Use representative periods and verified weights; increase stages only after a smaller shared-design model is feasible.",
        (
            "Target periods and synthesize a shared network",
            "Inspect the period-specific shared design",
        ),
    ),
    "18_results_plots_reports_exports.ipynb": (
        "How are validated results turned into reviewable tables, figures, workbooks, and dashboards?",
        "Export only after checking cached method, scope, units, and case name; output calls do not rerun engineering analysis.",
        "Use a project output directory and keep data preparation separate from optional publication side effects.",
        (
            "Prepare reportable cached results",
            "Export problem outputs",
            "Publish active-workspace outputs",
        ),
    ),
}


PRESENTATIONS: dict[str, tuple[str, str]] = {
    "01_first_solve_and_core_curves.ipynb": (
        "Review the target table first, then use the composite and grand composite curves to connect the utility targets to the remaining heat surplus and deficit.",
        "from IPython.display import display\n\n"
        "display(summary)\n"
        "display(composite)\n"
        "display(grand)",
    ),
    "02_focused_direct_and_total_site.ipynb": (
        "Compare the process summary with the Total Site profiles and utility grand composite curve; the site views reveal opportunities that are not visible within one process zone.",
        "from IPython.display import display\n\n"
        "display(summary)\n"
        "display(total_site_profiles)\n"
        "display(site_utility_curve)",
    ),
    "03_multisegment_streams.ipynb": (
        "Inspect the prepared segment boundaries and duties beside the target summary before replacing the example with plant-specific variable-heat-capacity data.",
        "from IPython.display import display\n\n"
        "display(segment_table)\n"
        "display(problem.summary_frame())",
    ),
    "04_workspace_cases_and_scenarios.ipynb": (
        "Use the case comparison for the engineering change and the batch summaries for a consistent review across every selected scenario.",
        "from IPython.display import display\n\n"
        "display(case_comparison)\n"
        "display(baseline_comparison)\n"
        "display(batch_summaries)",
    ),
    "05_workspace_persistence.ipynb": (
        "Compare the original summary with the restored validation and case payload to confirm that the saved workspace contains a reusable study rather than only a file reference.",
        "from IPython.display import display\n\n"
        "display(summary)\n"
        "display(validation)\n"
        "display(case_input)",
    ),
    "06_multiperiod_heat_integration.ipynb": (
        "Review period results beside the weighted aggregate; the largest single-period target does not necessarily dominate the weighted study.",
        "from IPython.display import display\n\n"
        "display(all_periods)\n"
        "display(weighted)\n"
        "display(combined)",
    ),
    "07_area_cost_and_exergy.ipynb": (
        "Read the area and cost summary together with the exergy views to avoid selecting a design from energy targets alone.",
        "from IPython.display import display\n\n"
        "display(area_cost_summary)\n"
        "display(exergy_curve)\n"
        "display(exergy_loads)",
    ),
    "08_carnot_heat_pump_and_refrigeration.ipynb": (
        "Compare cycle performance with the modified load and grand composite profiles to see whether the selected lift and placement reduce the intended utility demand.",
        "from IPython.display import display\n\n"
        "display(hpr_summary)\n"
        "display(load_plot)\n"
        "display(gcc_plot)",
    ),
    "09_vapour_compression_and_brayton.ipynb": (
        "Compare feasibility and returned cycle results on the same process basis before selecting a thermodynamic model.",
        "from IPython.display import display\n\n"
        "cycle_comparison = {\n"
        '    "vapour compression heat pump": vapour_compression,\n'
        '    "vapour compression refrigeration": vc_refrigeration,\n'
        '    "Brayton heat pump": brayton,\n'
        '    "Brayton refrigeration": brayton_refrigeration,\n'
        "}\n"
        "display(cycle_comparison)",
    ),
    "10_multiperiod_heat_pumps.ipynb": (
        "Use the weighted summary and period-screen outcomes to identify which operating condition controls capacity, lift, and feasibility.",
        "from IPython.display import display\n\n"
        "period_screen = {\n"
        '    "Carnot heat pump": period_heat_pumps,\n'
        '    "Carnot refrigeration": period_refrigeration,\n'
        '    "vapour compression heat pump": period_vc_heat_pumps,\n'
        '    "vapour compression refrigeration": period_vc_refrigeration,\n'
        '    "MVR": period_mvr,\n'
        "}\n"
        "display(weighted)\n"
        "display(period_screen)",
    ),
    "11_process_mvr_and_cascade.ipynb": (
        "Review the targeted study, component inventory, stage results, and cascade outcome together to check that compression work and replacement streams serve the intended process zone.",
        "from IPython.display import display\n\n"
        "display(mvr_summary)\n"
        "display(component_inventory)\n"
        "display(stage_results)\n"
        "display(cascade)",
    ),
    "12_cogeneration.ipynb": (
        "Compare the common process summary with the alternative cogeneration results to see how utility conditions and turbine assumptions change recoverable power.",
        "from IPython.display import display\n\n"
        "cogeneration_methods = {\n"
        '    "default": default,\n'
        '    "Sun-Smith": sun_smith,\n'
        '    "Varbanov": varbanov,\n'
        '    "isentropic": isentropic,\n'
        "}\n"
        "display(cogeneration_summary)\n"
        "display(cogeneration_methods)",
    ),
    "13_multiperiod_cogeneration.ipynb": (
        "Review the weighted table with all period method results to identify which operating conditions drive annual cogeneration value.",
        "from IPython.display import display\n\n"
        "period_cogeneration_methods = {\n"
        '    "default": period_cogeneration,\n'
        '    "Sun-Smith": sun_smith,\n'
        '    "Varbanov": varbanov,\n'
        '    "isentropic": isentropic,\n'
        "}\n"
        "display(weighted)\n"
        "display(period_cogeneration_methods)",
    ),
    "14_energy_transfer.ipynb": (
        "Use the transfer summary and diagram to screen magnitude and temperature, then use the focused comparison to challenge whether the connection is physically credible.",
        "from IPython.display import display\n\n"
        "display(transfer_summary)\n"
        "display(diagram)\n"
        "display(comparison)",
    ),
    "15_hen_synthesis_and_selection.ipynb": (
        "Review ranked candidates, recovery and utility metrics, and the selected grid together; objective rank alone is not a complete network-selection basis.",
        "from IPython.display import display\n\n"
        "display(ranked_networks)\n"
        "display(metrics)\n"
        "display(grid)",
    ),
    "16_advanced_hen_methods.ipynb": (
        "Compare the leading result from every method on the same stream and economic basis before attributing design differences to the search method.",
        "from IPython.display import display\n\n"
        "display(enhanced_ranking)\n"
        "display(open_hens_ranking)\n"
        "display(pinch_ranking)\n"
        "display(thermal_ranking)\n"
        "display(evolution_ranking)",
    ),
    "17_multiperiod_hen_synthesis.ipynb": (
        "Review the shared network metrics and one period-specific grid with the ranking to identify the period that controls the common design.",
        "from IPython.display import display\n\n"
        "display(ranked_networks)\n"
        "display(shared_network.summary_metrics)\n"
        "display(shared_grid)",
    ),
    "18_results_plots_reports_exports.ipynb": (
        "Inspect the cached summary and net-load view before reviewing publication outputs; verify method, scope, units, and case identity before sharing files.",
        "from IPython.display import display\n\n"
        "publication_outputs = {\n"
        '    "plot paths": plot_paths,\n'
        '    "gallery": gallery,\n'
        '    "problem workbook": workbook,\n'
        "}\n"
        "display(summary)\n"
        "display(net_loads)\n"
        "display(catalog)\n"
        "display(publication_outputs)",
    ),
}


def enrich(name: str, notebook: dict) -> dict:
    question, interpretation, adaptation, step_titles = GUIDANCE[name]
    review, presentation = PRESENTATIONS[name]
    original = notebook["cells"]
    code_cells = [cell for cell in original if cell["cell_type"] == "code"]
    if len(code_cells) != len(step_titles):
        raise ValueError(f"Guidance step count does not match {name}.")
    cells = [
        original[0],
        markdown(
            "## Study question and data\n\n"
            f"**Study question:** {question}\n\n"
            "The sample data is packaged with OpenPinch, so the notebook runs "
            "without path setup. Read the named inputs and assumptions before "
            "substituting plant data."
        ),
    ]
    for index, (title, code_cell) in enumerate(zip(step_titles, code_cells), start=1):
        cells.extend(
            [
                markdown(
                    f"## Step {index}: {title}\n\n"
                    "Run this cell once, then inspect its named outputs. Arguments "
                    "on the method call apply to this analysis; stored configuration "
                    "is only the fallback when an argument is omitted."
                ),
                code_cell,
            ]
        )
    cells.extend(
        [
            markdown(f"## Review the result\n\n{review}"),
            code(presentation),
            markdown(f"## Interpret the result\n\n{interpretation}"),
            markdown(
                "## Adapt this template\n\n"
                f"{adaptation}\n\n"
                "Keep the workflow explicit: prepare input, call one named "
                "engineering method, inspect cached results, then export."
            ),
        ]
    )
    for index, cell in enumerate(cells, start=1):
        cell["id"] = f"cell-{index:02d}"
    notebook["cells"] = cells
    return notebook


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    for path in NOTEBOOK_DIR.glob("*.ipynb"):
        path.unlink()
    for name, notebook in NOTEBOOKS.items():
        destination = NOTEBOOK_DIR / name
        destination.write_text(
            json.dumps(enrich(name, deepcopy(notebook)), indent=1, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
