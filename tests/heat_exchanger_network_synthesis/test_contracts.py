"""Fast validation tests for OpenPinch heat exchanger network synthesis schemas."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import OpenPinch
import OpenPinch.classes
import OpenPinch.lib
import OpenPinch.lib.schemas as schemas
from OpenPinch.classes import (
    HeatExchanger,
    HeatExchangerKind,
    HeatExchangerNetwork,
    HeatExchangerStreamRole,
)
from OpenPinch.classes._heat_exchanger.period_state import HeatExchangerPeriodState
from OpenPinch.classes.pinch_problem import PinchProblem
from OpenPinch.lib.config import Configuration
from OpenPinch.lib.config_metadata import CONFIG_FIELD_SPECS
from OpenPinch.lib.enums import (
    HeatExchangerNetworkDesignMethod,
    HeatExchangerNetworkLabel,
)
from OpenPinch.lib.schemas.io import TargetInput, TargetOutput
from OpenPinch.lib.schemas.synthesis.common import (
    HeatExchangerNetworkSynthesisExportRecord,
    HeatExchangerNetworkSynthesisManifest,
)
from OpenPinch.lib.schemas.synthesis.method import (
    HeatExchangerNetworkSynthesisMethodInput,
    HeatExchangerNetworkSynthesisMethodOutput,
)
from OpenPinch.lib.schemas.synthesis.result import HeatExchangerNetworkSynthesisResult
from OpenPinch.lib.schemas.synthesis.task import (
    HeatExchangerNetworkSynthesisTask,
    HeatExchangerNetworkSynthesisTaskOutcome,
)
from OpenPinch.services.heat_exchanger_network_synthesis.common.errors import (
    WorkflowContractError,
)
from OpenPinch.services.heat_exchanger_network_synthesis.common.execution import (
    task_builders,
)
from OpenPinch.services.heat_exchanger_network_synthesis.common.execution.settings import (
    SynthesisWorkflowSettings,
    workflow_settings_from_problem,
)
from OpenPinch.services.heat_exchanger_network_synthesis.common.execution.task_builders import (
    approach_temperature_from_network,
    stage_count_from_network,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "openhens"


def _network() -> HeatExchangerNetwork:
    return HeatExchangerNetwork(
        exchangers=(
            HeatExchanger(
                kind=HeatExchangerKind.RECOVERY,
                source_stream="H1",
                sink_stream="C1",
                source_stream_role=HeatExchangerStreamRole.PROCESS,
                sink_stream_role=HeatExchangerStreamRole.PROCESS,
                stage=1,
                period_states=(
                    HeatExchangerPeriodState(
                        period_id="0",
                        period_idx=0,
                        duty=100.0,
                    ),
                ),
                area=20.0,
            ),
        ),
        run_id="run-1",
        task_id="task-1",
        stage_count=1,
    )


def _workflow_settings(**updates) -> SynthesisWorkflowSettings:
    params = {
        "run_id": "contract-test",
        "approach_temperatures": (10.0,),
        "derivative_thresholds": (0.5,),
        "stage_selection": (1,),
        "method_sequence": ("pinch_design_method",),
        "output_formats": (),
        "solve_tolerance": 1e-3,
        "best_solutions_to_save": 2,
        "max_parallel": 1,
        "pdm_solver": "couenne",
        "tdm_solver": "baron",
        "evm_solver": "ipopt-pyomo",
        "pdm_solver_options": {"time_limit": 10},
        "tdm_solver_options": {"time_limit": 20},
        "evm_solver_options": {"time_limit": 30},
    }
    params.update(updates)
    return SynthesisWorkflowSettings(**params)


def test_synthesis_task_generates_deterministic_task_id():
    first = HeatExchangerNetworkSynthesisTask(
        run_id="run-1",
        method="pinch_design_method",
        approach_temperature=14.0,
        derivative_threshold=0.5,
        stage_count=3,
        problem_id="problem-a",
    )
    second = HeatExchangerNetworkSynthesisTask(
        run_id="run-1",
        method="pinch_design_method",
        approach_temperature=14.0,
        derivative_threshold=0.5,
        stage_count=3,
        problem_id="problem-a",
    )

    assert first.task_id == second.task_id
    assert first.task_id.startswith("hens-task-")


def test_hen_design_method_stringifies_to_canonical_method_identifier():
    assert str(HeatExchangerNetworkDesignMethod.OpenHENS) == ("open_hens_method")
    assert str(HeatExchangerNetworkDesignMethod.NetworkEvolution) == (
        "network_evolution_method"
    )
    assert (
        f"{HeatExchangerNetworkDesignMethod.ThermalDerivative}"
        == "thermal_derivative_method"
    )


def test_method_input_is_consistent_task_contract_for_all_synthesis_methods():
    pdm = HeatExchangerNetworkSynthesisMethodInput(
        run_id="run-1",
        method="pinch_design_method",
        approach_temperature=14.0,
    )
    tdm = HeatExchangerNetworkSynthesisMethodInput(
        run_id="run-1",
        method="thermal_derivative_method",
        approach_temperature=14.0,
        derivative_threshold=0.5,
        stage_count=3,
        seed_network_index=0,
    )
    evolution = HeatExchangerNetworkSynthesisMethodInput(
        run_id="run-1",
        method="network_evolution_method",
        approach_temperature=14.0,
        stage_count=3,
        seed_network_index=1,
    )

    assert pdm.task_id.startswith("hens-task-")
    assert tdm.task_id != evolution.task_id
    assert [item.method for item in (pdm, tdm, evolution)] == [
        "pinch_design_method",
        "thermal_derivative_method",
        "network_evolution_method",
    ]


def test_method_output_is_consistent_outcome_contract_for_all_synthesis_methods():
    method_input = HeatExchangerNetworkSynthesisMethodInput(
        run_id="run-1",
        method="network_evolution_method",
        approach_temperature=14.0,
        stage_count=3,
    )
    output = HeatExchangerNetworkSynthesisMethodOutput(
        task=method_input,
        status="success",
        network=_network(),
        objective_value=1000.0,
        solver_status="optimal",
    )

    round_tripped = HeatExchangerNetworkSynthesisMethodOutput.model_validate_json(
        output.model_dump_json(),
    )

    assert round_tripped == output
    assert round_tripped.task.method == "network_evolution_method"


def test_synthesis_schema_serialization_round_trips():
    task = HeatExchangerNetworkSynthesisTask(
        run_id="run-1",
        method="thermal_derivative_method",
        approach_temperature=14.0,
        derivative_threshold=0.5,
        stage_count=3,
    )
    outcome = HeatExchangerNetworkSynthesisTaskOutcome(
        task=task,
        status="success",
        network=_network(),
        objective_value=154853.8518602861,
        solver_status="optimal",
    )
    export_record = HeatExchangerNetworkSynthesisExportRecord(
        run_id="run-1",
        format="json",
        path="exports/run-1/network.json",
    )
    manifest = HeatExchangerNetworkSynthesisManifest(
        run_id="run-1",
        approach_temperatures=(10.0, 14.0),
        derivative_thresholds=(0.5, 0.9),
        stage_selection=(2, 3),
        export_formats=("json", "csv"),
        task_ids=(task.task_id,),
        export_records=(export_record,),
    )
    result = HeatExchangerNetworkSynthesisResult(
        network=_network(),
        run_id="run-1",
        task_id=task.task_id,
        solver_name="pyomo",
        solver_status="optimal",
        method="thermal_derivative_method",
        stage_count=3,
        objective_values={
            "total_annual_cost": 154853.8518602861,
            "utility_cost": 123.0,
        },
        ranked_networks=(outcome,),
        manifest=manifest,
    )

    for model in (task, outcome, export_record, manifest, result):
        assert type(model).model_validate_json(model.model_dump_json()) == model


def test_target_output_accepts_design_result_payload():
    output = TargetOutput(
        name="Site",
        targets=[],
        design=HeatExchangerNetworkSynthesisResult(
            network=_network(),
            run_id="run-1",
            objective_values={"total_annual_cost": 1000.0},
        ),
    )

    round_tripped = TargetOutput.model_validate_json(output.model_dump_json())

    assert round_tripped.design is not None
    assert round_tripped.design.network.total_duty(stream="H1") == pytest.approx(100.0)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {
                "run_id": "run-1",
                "method": "pinch_design_method",
                "approach_temperature": 0.0,
            },
            "finite and positive",
        ),
        (
            {
                "run_id": "bad run id",
                "method": "pinch_design_method",
                "approach_temperature": 14.0,
            },
            "run_id",
        ),
        (
            {
                "run_id": "run-1",
                "method": "pinch_design_method",
                "approach_temperature": 14.0,
                "stage_count": 0,
            },
            "stage_count",
        ),
    ],
)
def test_synthesis_task_rejects_invalid_values(payload, message):
    with pytest.raises(ValidationError, match=message):
        HeatExchangerNetworkSynthesisTask.model_validate(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {
            "run_id": "run-1",
            "approach_temperatures": [14.0],
            "derivative_thresholds": [0.5],
            "stage_selection": [1],
            "export_formats": ["xml"],
        },
        {
            "run_id": "run-1",
            "approach_temperatures": [14.0],
            "derivative_thresholds": [0.5],
            "stage_selection": [1, 1],
        },
        {
            "run_id": "run-1",
            "approach_temperatures": [float("nan")],
            "derivative_thresholds": [0.5],
            "stage_selection": [1],
        },
        {
            "run_id": "run-1",
            "approach_temperatures": [14.0],
            "derivative_thresholds": [0.5],
            "stage_selection": [1],
            "solve_tolerance": -1.0,
        },
        {
            "run_id": "run-1",
            "approach_temperatures": [14.0],
            "derivative_thresholds": [0.5],
            "stage_selection": [1],
            "evm_n_ad_branches": 0,
        },
        {
            "run_id": "run-1",
            "approach_temperatures": [14.0],
            "derivative_thresholds": [0.5],
            "stage_selection": [1],
            "evm_n_rm_branches": 0,
        },
    ],
)
def test_manifest_rejects_invalid_grids_formats_stages_and_tolerances(payload):
    with pytest.raises(ValidationError):
        HeatExchangerNetworkSynthesisManifest.model_validate(payload)


@pytest.mark.parametrize(
    "legacy_key",
    ["min_dT_values", "min_dqda_values", "output_folder", "output_formats"],
)
def test_openhens_legacy_aliases_are_not_accepted(legacy_key):
    payload = {
        "run_id": "run-1",
        "approach_temperatures": [14.0],
        "derivative_thresholds": [0.5],
        "stage_selection": [1],
        legacy_key: [1.0],
    }

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        HeatExchangerNetworkSynthesisManifest.model_validate(payload)

    with pytest.raises(ValueError, match="Unknown configuration option"):
        Configuration(options={legacy_key: [1.0]})


def test_hen_config_field_specs_use_openpinch_names():
    expected = {
        "HENS_APPROACH_TEMPERATURES",
        "HENS_DERIVATIVE_THRESHOLDS",
        "HENS_STAGE_SELECTION",
        "HENS_METHOD_SEQUENCE",
        "HENS_SOLVER_PDM",
        "HENS_SOLVER_TDM",
        "HENS_SOLVER_EVM",
        "HENS_SOLVER_OPTIONS_PDM",
        "HENS_SOLVER_OPTIONS_TDM",
        "HENS_SOLVER_OPTIONS_EVM",
        "HENS_SOLVE_TOLERANCE",
        "HENS_MAX_PARALLEL",
        "HENS_SYNTHESIS_QUALITY_TIER",
        "HENS_PDM_STAGE_PAIR_LIMIT",
        "HENS_TDM_PARENT_LIMIT",
        "HENS_STAGE_PACKING",
        "HENS_EVM_N_AD_BRANCHES",
        "HENS_EVM_N_RM_BRANCHES",
        "HENS_LOG_LEVEL",
        "HENS_OUTPUT_FOLDER",
        "HENS_OUTPUT_FORMATS",
        "HENS_RUN_ID",
        "HENS_BEST_SOLUTIONS_TO_SAVE",
    }

    assert expected <= set(CONFIG_FIELD_SPECS)
    assert {CONFIG_FIELD_SPECS[key].group for key in expected} == {"hens"}
    assert CONFIG_FIELD_SPECS["HENS_SOLVER_EVM"].config_path == (
        "hens",
        "solver_evm",
    )
    assert CONFIG_FIELD_SPECS["HENS_SYNTHESIS_QUALITY_TIER"].default == 1
    assert CONFIG_FIELD_SPECS["HENS_EVM_N_AD_BRANCHES"].default is None
    assert CONFIG_FIELD_SPECS["HENS_EVM_N_RM_BRANCHES"].default is None
    assert not any(key.startswith("OPENHENS_") for key in CONFIG_FIELD_SPECS)
    assert not any("OPTIMIZED" in key for key in CONFIG_FIELD_SPECS)
    assert not any("ENHANCED" in key for key in CONFIG_FIELD_SPECS)


@pytest.mark.parametrize(
    "legacy_key",
    [
        "HENS_PDM_SOLVER",
        "HENS_TDM_SOLVER",
        "HENS_ESM_SOLVER",
        "HENS_PDM_SOLVER_OPTIONS",
        "HENS_TDM_SOLVER_OPTIONS",
        "HENS_ESM_SOLVER_OPTIONS",
    ],
)
def test_hen_config_rejects_retired_solver_option_names(legacy_key):
    with pytest.raises(ValueError, match="Unknown configuration option"):
        Configuration(options={legacy_key: "couenne"})


def test_hen_config_options_accept_valid_values_on_canonical_paths():
    options = {
        "HENS_APPROACH_TEMPERATURES": [10, 14.0],
        "HENS_DT_CONT_MULTIPLIERS": [1.0, 1.5],
        "HENS_DERIVATIVE_THRESHOLDS": [0.5, 0.9],
        "HENS_STAGE_SELECTION": [2, 3],
        "HENS_METHOD_SEQUENCE": [
            "pinch_design_method",
            "thermal_derivative_method",
            "network_evolution_method",
        ],
        "HENS_OUTPUT_FORMATS": ["json", "csv"],
        "HENS_SOLVE_TOLERANCE": 1e-3,
        "HENS_MAX_PARALLEL": 2,
        "HENS_SYNTHESIS_QUALITY_TIER": 3,
        "HENS_PDM_STAGE_PAIR_LIMIT": 6,
        "HENS_TDM_PARENT_LIMIT": 4,
        "HENS_STAGE_PACKING": "pdm",
        "HENS_EVM_N_AD_BRANCHES": 2,
        "HENS_EVM_N_RM_BRANCHES": 3,
        "HENS_RUN_ID": "run-1",
        "HENS_BEST_SOLUTIONS_TO_SAVE": 3,
        "HENS_SOLVER_OPTIONS_PDM": {"node_limit": 50},
        "HENS_SOLVER_OPTIONS_TDM": {"feas_tolerance": 0.02},
        "HENS_SOLVER_OPTIONS_EVM": {"max_iter": 500},
    }

    cfg = Configuration(options=options)
    target_input = TargetInput(streams=[], options=options)

    assert cfg.hens.approach_temperatures == [10.0, 14.0]
    assert cfg.hens.dt_cont_multipliers == [1.0, 1.5]
    assert cfg.hens.output_formats == ["json", "csv"]
    assert target_input.options["HENS_DT_CONT_MULTIPLIERS"] == [1.0, 1.5]
    assert target_input.options["HENS_DERIVATIVE_THRESHOLDS"] == [0.5, 0.9]
    assert target_input.options["HENS_STAGE_SELECTION"] == [2, 3]
    assert cfg.hens.synthesis_quality_tier == 3
    assert cfg.hens.pdm_stage_pair_limit == 6
    assert cfg.hens.tdm_parent_limit == 4
    assert cfg.hens.stage_packing == "pdm"
    assert target_input.options["HENS_SYNTHESIS_QUALITY_TIER"] == 3
    assert target_input.options["HENS_STAGE_PACKING"] == "pdm"
    assert cfg.hens.evm_n_ad_branches == 2
    assert cfg.hens.evm_n_rm_branches == 3
    assert target_input.options["HENS_EVM_N_AD_BRANCHES"] == 2
    assert target_input.options["HENS_EVM_N_RM_BRANCHES"] == 3
    assert cfg.hens.solver_options_pdm == {"node_limit": 50}
    assert target_input.options["HENS_SOLVER_OPTIONS_EVM"] == {"max_iter": 500}


def test_hen_dt_cont_multipliers_feed_quality_pdm_multiplier_grid():
    problem = PinchProblem(
        source=FIXTURE_ROOT / "Four-stream-Yee-and-Grossmann-1990-1.json"
    )
    problem.master_zone.config = Configuration(
        options={
            "HENS_APPROACH_TEMPERATURES": [10.0, 20.0],
            "HENS_DT_CONT_MULTIPLIERS": [1.0, 1.5],
        }
    )

    settings = workflow_settings_from_problem(problem)

    assert settings.approach_temperatures == (10.0, 20.0)
    assert settings.dt_cont_multipliers == (1.0, 1.5)
    assert settings.user_dt_cont_multipliers is True
    assert settings.quality_dt_cont_multipliers == ()

    tier_problem = PinchProblem(
        source=FIXTURE_ROOT / "Four-stream-Yee-and-Grossmann-1990-1.json"
    )
    tier_problem.master_zone.config = Configuration(
        options={
            "HENS_APPROACH_TEMPERATURES": [10.0, 20.0],
            "HENS_DT_CONT_MULTIPLIERS": [1.0, 1.5],
            "HENS_SYNTHESIS_QUALITY_TIER": 3,
        }
    )
    tier_settings = workflow_settings_from_problem(tier_problem)

    assert tier_settings.quality_dt_cont_multipliers == (1.0, 1.5)
    assert tier_settings.quality_pdm_approach_temperatures == (10.0, 15.0)


def test_workflow_settings_solver_options_and_quality_edges() -> None:
    standard = _workflow_settings()
    expanded = _workflow_settings(
        synthesis_quality_tier=7,
        tdm_parent_limit=0,
    )

    assert standard.solver_for("not-a-method") is None
    assert standard.solver_options_for("thermal_derivative_method") == {
        "time_limit": 20
    }
    assert standard.solver_options_for("network_evolution_method") == {"time_limit": 30}
    assert standard.solver_options_for("not-a-method") == {}
    assert standard.quality_fraction == 0.0
    assert standard.is_standard_quality_tier is True
    assert standard.quality_pdm_approach_temperatures == ()
    assert standard.quality_tdm_parent_limit == 2
    assert expanded.quality_fraction == 1.0
    assert expanded.quality_tdm_parent_limit == 1


def test_workflow_settings_require_loaded_problem() -> None:
    with pytest.raises(RuntimeError, match="requires a loaded PinchProblem"):
        workflow_settings_from_problem(SimpleNamespace(master_zone=None))


def test_task_builder_helpers_fall_back_to_seed_network_metadata() -> None:
    staged_network = HeatExchangerNetwork(
        exchangers=(
            HeatExchanger(
                kind=HeatExchangerKind.RECOVERY,
                source_stream="H1",
                sink_stream="C1",
                source_stream_role=HeatExchangerStreamRole.PROCESS,
                sink_stream_role=HeatExchangerStreamRole.PROCESS,
                stage=3,
                period_states=(
                    HeatExchangerPeriodState(
                        period_id="0",
                        period_idx=0,
                        duty=100.0,
                        approach_temperatures=(12.0,),
                    ),
                ),
            ),
        ),
        source_metadata={"solver_dTmin": 9.0},
    )
    no_stage_network = HeatExchangerNetwork(
        exchangers=(
            HeatExchanger(
                kind=HeatExchangerKind.HOT_UTILITY,
                source_stream="Steam",
                sink_stream="C1",
                source_stream_role=HeatExchangerStreamRole.UTILITY,
                sink_stream_role=HeatExchangerStreamRole.PROCESS,
                period_states=(
                    HeatExchangerPeriodState(
                        period_id="0",
                        period_idx=0,
                        duty=10.0,
                    ),
                ),
            ),
        )
    )
    approach_from_exchanger = HeatExchangerNetwork(
        exchangers=(
            HeatExchanger(
                kind=HeatExchangerKind.RECOVERY,
                source_stream="H1",
                sink_stream="C1",
                source_stream_role=HeatExchangerStreamRole.PROCESS,
                sink_stream_role=HeatExchangerStreamRole.PROCESS,
                stage=1,
                period_states=(
                    HeatExchangerPeriodState(
                        period_id="0",
                        period_idx=0,
                        duty=100.0,
                        approach_temperatures=(7.0,),
                    ),
                ),
            ),
        )
    )
    approach_from_settings = HeatExchangerNetwork(exchangers=())
    settings = _workflow_settings(approach_temperatures=(11.0,))

    assert (
        stage_count_from_network(
            staged_network,
            downstream_method="thermal_derivative_method",
        )
        == 3
    )
    with pytest.raises(WorkflowContractError, match="stage_count"):
        stage_count_from_network(
            no_stage_network,
            downstream_method="thermal_derivative_method",
        )
    assert approach_temperature_from_network(staged_network, settings) == pytest.approx(
        9.0
    )
    assert approach_temperature_from_network(
        approach_from_exchanger,
        settings,
    ) == pytest.approx(7.0)
    assert approach_temperature_from_network(
        approach_from_settings,
        settings,
    ) == pytest.approx(11.0)


def test_required_stage_count_reports_missing_stage_metadata() -> None:
    task = HeatExchangerNetworkSynthesisTask(
        run_id="run-1",
        method="pinch_design_method",
        approach_temperature=10.0,
        task_id="seed",
    )
    outcome = HeatExchangerNetworkSynthesisTaskOutcome(
        task=task,
        status="success",
    )

    with pytest.raises(WorkflowContractError, match="without a stage count"):
        task_builders._required_stage_count(
            outcome,
            downstream_method="thermal_derivative_method",
        )


def test_evm_branch_options_round_trip_to_workflow_settings():
    problem = PinchProblem(
        source=FIXTURE_ROOT / "Four-stream-Yee-and-Grossmann-1990-1.json"
    )
    problem.master_zone.config = Configuration(
        options={
            "HENS_EVM_N_AD_BRANCHES": 2,
            "HENS_EVM_N_RM_BRANCHES": 3,
        }
    )

    settings = workflow_settings_from_problem(problem)

    assert settings.evm_n_ad_branches == 2
    assert settings.evm_n_rm_branches == 3
    assert settings.effective_evm_n_ad_branches == 2
    assert settings.effective_evm_n_rm_branches == 3


def test_synthesis_quality_tier_derives_evm_branch_widths() -> None:
    problem = PinchProblem(
        source=FIXTURE_ROOT / "Four-stream-Yee-and-Grossmann-1990-1.json"
    )
    problem.master_zone.config = Configuration(
        options={"HENS_SYNTHESIS_QUALITY_TIER": 5}
    )

    settings = workflow_settings_from_problem(problem)

    assert settings.evm_n_ad_branches is None
    assert settings.evm_n_rm_branches is None
    assert settings.effective_evm_n_ad_branches == 2
    assert settings.effective_evm_n_rm_branches == 2


@pytest.mark.parametrize(
    "options",
    [
        {"HENS_APPROACH_TEMPERATURES": [0.0]},
        {"HENS_DT_CONT_MULTIPLIERS": [0.0]},
        {"HENS_DERIVATIVE_THRESHOLDS": [float("inf")]},
        {"HENS_STAGE_SELECTION": [0]},
        {"HENS_STAGE_SELECTION": [1, 1]},
        {"HENS_OUTPUT_FORMATS": ["xml"]},
        {"HENS_SOLVE_TOLERANCE": 0.0},
        {"HENS_SYNTHESIS_QUALITY_TIER": -1},
        {"HENS_SYNTHESIS_QUALITY_TIER": 6},
        {"HENS_PDM_STAGE_PAIR_LIMIT": -1},
        {"HENS_PDM_STAGE_PAIR_LIMIT": 13},
        {"HENS_TDM_PARENT_LIMIT": 0},
        {"HENS_STAGE_PACKING": "everywhere"},
        {"HENS_EVM_N_AD_BRANCHES": 0},
        {"HENS_EVM_N_RM_BRANCHES": -1},
        {"HENS_RUN_ID": "bad run id"},
        {"HENS_SOLVER_OPTIONS_PDM": ["node_limit 50"]},
        {"HENS_SOLVER_OPTIONS_TDM": {"": 50}},
    ],
)
def test_hen_config_options_reject_invalid_values_on_canonical_paths(options):
    with pytest.raises(ValueError):
        Configuration(options=options)

    with pytest.raises(ValidationError):
        TargetInput(streams=[], options=options)


@pytest.mark.parametrize(
    "removed_key",
    [
        "HENS_OPTIMIZED_ENABLE_STAGE_PACKING",
        "HENS_OPTIMIZED_STAGE_PACKING",
        "HENS_OPTIMIZED_PDM_QUALITY",
        "HENS_ENHANCED_QUALITY",
    ],
)
def test_removed_prototype_hen_options_are_not_accepted(removed_key):
    with pytest.raises(ValueError, match="Unknown configuration option"):
        Configuration(options={removed_key: True})

    with pytest.raises(ValidationError):
        TargetInput(streams=[], options={removed_key: True})


def test_public_synthesis_exports_are_openpinch_native():
    synthesis_package = __import__(
        "OpenPinch.lib.schemas.synthesis",
        fromlist=["*"],
    )

    expected_class_exports = {
        "HeatExchanger",
        "HeatExchangerKind",
        "HeatExchangerNetwork",
        "HeatExchangerStreamRole",
    }
    expected_schema_exports = {
        "HeatExchangerNetworkSynthesisExportRecord",
        "HeatExchangerNetworkSynthesisManifest",
        "HeatExchangerNetworkSynthesisMethodInput",
        "HeatExchangerNetworkSynthesisMethodOutput",
        "HeatExchangerNetworkSynthesisResult",
        "HeatExchangerNetworkSynthesisTask",
        "HeatExchangerNetworkSynthesisTaskOutcome",
        "SynthesisDesignMethod",
    }

    assert expected_class_exports <= set(OpenPinch.classes.__all__)
    assert expected_schema_exports <= set(schemas.__all__)
    assert "HeatExchangerNetworkSynthesisResult" in OpenPinch.lib.__all__
    assert "HeatExchangerKind" in OpenPinch.lib.__all__
    assert "HeatExchangerNetworkLabel" in OpenPinch.lib.__all__
    assert "HeatExchangerStreamRole" in OpenPinch.lib.__all__
    assert OpenPinch.lib.HeatExchangerKind is HeatExchangerKind
    assert OpenPinch.lib.HeatExchangerStreamRole is HeatExchangerStreamRole
    assert not hasattr(synthesis_package, "__all__")
    assert expected_schema_exports.isdisjoint(vars(synthesis_package))
    for compatibility_module in ("methods", "results", "tasks"):
        assert (
            importlib.util.find_spec(
                f"OpenPinch.lib.schemas.synthesis.{compatibility_module}"
            )
            is None
        )
    assert isinstance(
        HeatExchangerNetworkLabel.RECOVERY_DUTY.value,
        str,
    )


def test_openhens_compatibility_surfaces_are_absent():
    forbidden_exports = {
        "OpenHENS",
        "CaseStudy",
        "SynthesisStudy",
        "run_synthesis_workflow",
        "run_heat_exchanger_network_synthesis",
        "HeatExchangerNetworkDesignSpace",
        "HeatExchangerNetworkMethodSequence",
        "HeatExchangerNetworkSolveSetup",
        "HeatExchangerNetworkOutputs",
    }

    modules = [OpenPinch, OpenPinch.classes, OpenPinch.lib, schemas]
    for module in modules:
        assert forbidden_exports.isdisjoint(set(getattr(module, "__all__", ())))
        for name in forbidden_exports:
            assert not hasattr(module, name)

    assert importlib.util.find_spec("OpenPinch.openhens") is None
    assert importlib.util.find_spec("OpenPinch.OpenHENS") is None
