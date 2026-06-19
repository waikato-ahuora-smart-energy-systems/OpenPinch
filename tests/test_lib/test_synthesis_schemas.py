"""Fast validation tests for OpenPinch HEN synthesis schemas."""

from __future__ import annotations

import importlib.util

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
from OpenPinch.lib.config import Configuration
from OpenPinch.lib.config_metadata import CONFIG_FIELD_SPECS
from OpenPinch.lib.enums import HeatExchangerNetworkLabel
from OpenPinch.lib.schemas.io import TargetInput, TargetOutput
from OpenPinch.lib.schemas.synthesis import (
    HeatExchangerNetworkSynthesisExportRecord,
    HeatExchangerNetworkSynthesisManifest,
    HeatExchangerNetworkSynthesisResult,
    HeatExchangerNetworkSynthesisTask,
    HeatExchangerNetworkSynthesisTaskOutcome,
)


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
                duty=100.0,
                area=20.0,
            ),
        ),
        run_id="run-1",
        task_id="task-1",
        stage_count=1,
    )


def test_synthesis_task_generates_deterministic_task_id():
    first = HeatExchangerNetworkSynthesisTask(
        run_id="run-1",
        method="pinch_decomposition",
        approach_temperature=14.0,
        derivative_threshold=0.5,
        stage_count=3,
        problem_id="problem-a",
    )
    second = HeatExchangerNetworkSynthesisTask(
        run_id="run-1",
        method="pinch_decomposition",
        approach_temperature=14.0,
        derivative_threshold=0.5,
        stage_count=3,
        problem_id="problem-a",
    )

    assert first.task_id == second.task_id
    assert first.task_id.startswith("hens-task-")


def test_synthesis_schema_serialization_round_trips():
    task = HeatExchangerNetworkSynthesisTask(
        run_id="run-1",
        method="topology_design",
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
        method="topology_design",
        stage_count=3,
        objective_values={
            "total_annual_cost": 154853.8518602861,
            "utility_cost": 123.0,
        },
        task_outcomes=(outcome,),
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
    assert round_tripped.design.network.total_duty(stream="H1") == pytest.approx(
        100.0
    )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {
                "run_id": "run-1",
                "method": "pinch_decomposition",
                "approach_temperature": 0.0,
            },
            "finite and positive",
        ),
        (
            {
                "run_id": "bad run id",
                "method": "pinch_decomposition",
                "approach_temperature": 14.0,
            },
            "run_id",
        ),
        (
            {
                "run_id": "run-1",
                "method": "pinch_decomposition",
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
        "HENS_PDM_SOLVER",
        "HENS_TDM_SOLVER",
        "HENS_ESM_SOLVER",
        "HENS_PDM_SOLVER_OPTIONS",
        "HENS_TDM_SOLVER_OPTIONS",
        "HENS_ESM_SOLVER_OPTIONS",
        "HENS_SOLVE_TOLERANCE",
        "HENS_MAX_PARALLEL",
        "HENS_LOG_LEVEL",
        "HENS_OUTPUT_FOLDER",
        "HENS_OUTPUT_FORMATS",
        "HENS_RUN_ID",
        "HENS_BEST_SOLUTIONS_TO_SAVE",
    }

    assert expected <= set(CONFIG_FIELD_SPECS)
    assert {CONFIG_FIELD_SPECS[key].group for key in expected} == {"synthesis"}
    assert not any(key.startswith("OPENHENS_") for key in CONFIG_FIELD_SPECS)


def test_hen_config_options_accept_valid_values_on_canonical_paths():
    options = {
        "HENS_APPROACH_TEMPERATURES": [10, 14.0],
        "HENS_DERIVATIVE_THRESHOLDS": [0.5, 0.9],
        "HENS_STAGE_SELECTION": [2, 3],
        "HENS_METHOD_SEQUENCE": [
            "pinch_decomposition",
            "topology_design",
            "energy_stage_refinement",
        ],
        "HENS_OUTPUT_FORMATS": ["json", "csv"],
        "HENS_SOLVE_TOLERANCE": 1e-3,
        "HENS_MAX_PARALLEL": 2,
        "HENS_RUN_ID": "run-1",
        "HENS_BEST_SOLUTIONS_TO_SAVE": 3,
        "HENS_PDM_SOLVER_OPTIONS": {"node_limit": 50},
        "HENS_TDM_SOLVER_OPTIONS": {"feas_tolerance": 0.02},
        "HENS_ESM_SOLVER_OPTIONS": {"max_iter": 500},
    }

    cfg = Configuration(options=options)
    target_input = TargetInput(streams=[], options=options)

    assert cfg.HENS_APPROACH_TEMPERATURES == [10.0, 14.0]
    assert cfg.HENS_OUTPUT_FORMATS == ["json", "csv"]
    assert target_input.options["HENS_DERIVATIVE_THRESHOLDS"] == [0.5, 0.9]
    assert target_input.options["HENS_STAGE_SELECTION"] == [2, 3]
    assert cfg.HENS_PDM_SOLVER_OPTIONS == {"node_limit": 50}
    assert target_input.options["HENS_ESM_SOLVER_OPTIONS"] == {"max_iter": 500}


@pytest.mark.parametrize(
    "options",
    [
        {"HENS_APPROACH_TEMPERATURES": [0.0]},
        {"HENS_DERIVATIVE_THRESHOLDS": [float("inf")]},
        {"HENS_STAGE_SELECTION": [0]},
        {"HENS_STAGE_SELECTION": [1, 1]},
        {"HENS_OUTPUT_FORMATS": ["xml"]},
        {"HENS_SOLVE_TOLERANCE": 0.0},
        {"HENS_RUN_ID": "bad run id"},
        {"HENS_PDM_SOLVER_OPTIONS": ["node_limit 50"]},
        {"HENS_TDM_SOLVER_OPTIONS": {"": 50}},
    ],
)
def test_hen_config_options_reject_invalid_values_on_canonical_paths(options):
    with pytest.raises(ValueError):
        Configuration(options=options)

    with pytest.raises(ValidationError):
        TargetInput(streams=[], options=options)


def test_public_synthesis_exports_are_openpinch_native():
    expected_class_exports = {
        "HeatExchanger",
        "HeatExchangerKind",
        "HeatExchangerNetwork",
        "HeatExchangerStreamRole",
    }
    expected_schema_exports = {
        "HeatExchangerNetworkSynthesisExportRecord",
        "HeatExchangerNetworkSynthesisManifest",
        "HeatExchangerNetworkSynthesisResult",
        "HeatExchangerNetworkSynthesisTask",
        "HeatExchangerNetworkSynthesisTaskOutcome",
    }

    assert expected_class_exports <= set(OpenPinch.classes.__all__)
    assert expected_schema_exports <= set(schemas.__all__)
    assert "HeatExchangerNetworkSynthesisResult" in OpenPinch.lib.__all__
    assert "HeatExchangerNetworkLabel" in OpenPinch.lib.__all__
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
