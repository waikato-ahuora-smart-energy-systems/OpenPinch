"""Schema validator tests for heat exchanger network synthesis records."""

from __future__ import annotations

import json
import math

import pytest
from pydantic import ValidationError

import OpenPinch.analysis.heat_exchanger_networks.results.selection as result_operations
from OpenPinch.analysis.heat_exchanger_networks.results.selection import select_network
from OpenPinch.contracts.synthesis.common import (
    HeatExchangerNetworkSynthesisManifest,
)
from OpenPinch.contracts.synthesis.method import (
    HeatExchangerNetworkSynthesisMethodOutput,
)
from OpenPinch.contracts.synthesis.result import (
    HeatExchangerNetworkSynthesisResult,
)
from OpenPinch.contracts.synthesis.task import (
    HeatExchangerNetworkSynthesisTask,
    HeatExchangerNetworkSynthesisTaskOutcome,
)
from OpenPinch.contracts.synthesis.topology import (
    HeatExchangerNetworkTopologyRestriction,
)
from OpenPinch.domain.enums import HeatExchangerNetworkDesignMethod
from OpenPinch.domain.heat_exchanger_network import HeatExchangerNetwork
from OpenPinch.presentation.network_grid import service as grid_service
from OpenPinch.presentation.network_grid.results import build_result_grid_diagram
from tests.support.paths import FIXTURES_ROOT

FIXTURE_PATH = FIXTURES_ROOT / "synthesis_schema_cases.json"


def _fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _task(**updates) -> HeatExchangerNetworkSynthesisTask:
    data = {**_fixture()["task"], **updates}
    return HeatExchangerNetworkSynthesisTask.model_validate(data)


def _manifest(**updates) -> HeatExchangerNetworkSynthesisManifest:
    data = {**_fixture()["manifest"], **updates}
    return HeatExchangerNetworkSynthesisManifest.model_validate(data)


def _network(**updates) -> HeatExchangerNetwork:
    return HeatExchangerNetwork(run_id="schema-run-1", **updates)


def test_synthesis_task_generates_stable_id_from_static_fixture():
    first = _task()
    second = _task()

    assert first.task_id == second.task_id
    assert first.problem_id == "problem-a"
    assert first.workspace_variant == "variant-a"
    assert first.period_id == "peak"
    assert first.parent_task_id == "parent-1"
    assert first.seed_network_index == 0
    assert first.topology_restrictions[0].duty == pytest.approx(12.5)

    without_threshold = _task(derivative_threshold=None, seed_network_index=None)
    assert without_threshold.derivative_threshold is None
    assert without_threshold.seed_network_index is None


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"run_id": "-bad"}, "run_id must start"),
        ({"problem_id": " "}, "identity fields"),
        ({"approach_temperature": 0.0}, "finite and positive"),
        ({"stage_count": 0}, "positive integer"),
        ({"seed_network_index": -1}, "non-negative"),
    ],
)
def test_synthesis_task_rejects_invalid_identity_and_numeric_values(
    updates: dict,
    message: str,
):
    with pytest.raises(ValidationError, match=message):
        _task(**updates)


def test_topology_restriction_validates_stream_stage_and_duty_edges():
    valid = HeatExchangerNetworkTopologyRestriction(
        source_stream=" H1 ",
        sink_stream=" C1 ",
        stage=1,
        duty=0.0,
    )
    assert valid.source_stream == "H1"
    assert valid.sink_stream == "C1"

    with pytest.raises(ValueError, match="stream identities"):
        HeatExchangerNetworkTopologyRestriction._validate_stream_identity(None)
    with pytest.raises(ValidationError, match="positive integer"):
        HeatExchangerNetworkTopologyRestriction(
            source_stream="H1",
            sink_stream="C1",
            stage=0,
            duty=1.0,
        )
    with pytest.raises(ValidationError, match="non-negative"):
        HeatExchangerNetworkTopologyRestriction(
            source_stream="H1",
            sink_stream="C1",
            stage=1,
            duty=-1.0,
        )


def test_export_record_and_manifest_accept_static_fixture_values():
    manifest = _manifest()

    assert manifest.task_ids == ("task-a", "task-b")
    assert manifest.problem_id == "problem-a"
    assert manifest.selected_pathway_kind == "protected"
    assert manifest.task_count_by_method == {"pinch_design_method": 2}
    assert manifest.export_records[0].record_id == "record-1"
    assert manifest.export_records[0].content_type == "application/json"


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"approach_temperatures": []}, "grid values"),
        ({"derivative_thresholds": [math.inf]}, "finite and positive"),
        ({"stage_selection": []}, "at least one stage"),
        ({"stage_selection": [0]}, "positive integers"),
        ({"stage_selection": [1, 1]}, "unique"),
        ({"solve_tolerance": 0.0}, "finite and positive"),
        ({"best_solutions_to_save": 0}, "positive integer"),
        ({"synthesis_quality_tier": 6}, "between 0 and 5"),
        ({"pdm_stage_pair_limit": 13}, "between 0 and 12"),
        ({"tdm_parent_limit": 0}, "positive when supplied"),
        ({"stage_packing": "packed"}, "stage_packing must be one of"),
        ({"task_ids": [" "]}, "identity fields"),
        ({"selected_tier_origin": -1}, "between 0 and 5"),
        ({"task_count_by_method": {"method": -1}}, "non-negative"),
    ],
)
def test_synthesis_manifest_rejects_invalid_static_edges(
    updates: dict,
    message: str,
):
    with pytest.raises(ValidationError, match=message):
        _manifest(**updates)


def test_manifest_task_count_validator_rejects_none_method_key():
    with pytest.raises(ValueError, match="keys must be non-empty"):
        HeatExchangerNetworkSynthesisManifest._validate_task_count_by_method({None: 1})


def test_method_output_fills_method_from_task_and_validates_optional_fields():
    task = _task()
    output = HeatExchangerNetworkSynthesisMethodOutput(
        task=task,
        status="success",
        objective_value=0.0,
        solver_status=" solved ",
        error=None,
        diagnostic_references=(" log-1 ",),
    )

    assert output.method == HeatExchangerNetworkDesignMethod.PinchDesign
    assert output.solver_status == "solved"
    assert output.diagnostic_references == ("log-1",)

    empty_objective = HeatExchangerNetworkSynthesisMethodOutput(
        status="skipped",
        objective_value=None,
    )
    assert empty_objective.objective_value is None

    with pytest.raises(ValidationError, match="non-negative"):
        HeatExchangerNetworkSynthesisMethodOutput(
            status="failed",
            objective_value=-1.0,
        )
    with pytest.raises(ValidationError, match="identity fields"):
        HeatExchangerNetworkSynthesisMethodOutput(
            status="failed",
            solver_status=" ",
        )
    with pytest.raises(ValidationError, match="identity fields"):
        HeatExchangerNetworkSynthesisMethodOutput(
            status="failed",
            diagnostic_references=(" ",),
        )


def test_synthesis_result_validates_objectives_and_empty_rank_selection():
    result = HeatExchangerNetworkSynthesisResult(
        network=_network(stage_count=1),
        run_id="schema-run-1",
        task_id=" task-1 ",
        problem_id=" problem-a ",
        workspace_variant=" variant-a ",
        period_id=" peak ",
        solver_name=" fake-solver ",
        solver_status=" solved ",
        stage_count=1,
        objective_values={"total_annual_cost": 12},
        diagnostic_references=(" result-log ",),
    )

    assert result.task_id == "task-1"
    assert result.objective_values == {"total_annual_cost": 12.0}
    assert result.diagnostic_references == ("result-log",)
    assert select_network(result, 1) is result

    with pytest.raises(IndexError, match="1-based"):
        select_network(result, 0)
    with pytest.raises(IndexError, match="only 1 network"):
        select_network(result, 2)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"run_id": "-bad"}, "run_id must start"),
        ({"task_id": " "}, "identity fields"),
        ({"stage_count": 0}, "positive integer"),
        ({"objective_values": {"": 1.0}}, "identity fields"),
        ({"objective_values": {"cost": -1.0}}, "non-negative"),
        ({"diagnostic_references": (" ",)}, "identity fields"),
    ],
)
def test_synthesis_result_rejects_invalid_static_edges(
    updates: dict,
    message: str,
):
    data = {
        "network": _network(),
        "run_id": "schema-run-1",
    } | updates

    with pytest.raises(ValidationError, match=message):
        HeatExchangerNetworkSynthesisResult(**data)


def test_synthesis_result_grid_diagram_uses_base_network_when_unranked(monkeypatch):
    result = HeatExchangerNetworkSynthesisResult(
        network=_network(),
        run_id="schema-run-1",
    )
    calls = {}

    def fake_build_grid_diagram(
        network,
        *,
        period_id: str | None,
        temperature_scaled: bool,
    ):
        calls["network"] = network
        calls["period_id"] = period_id
        calls["temperature_scaled"] = temperature_scaled
        return {"diagram": "ok"}

    monkeypatch.setattr(grid_service, "build_grid_diagram", fake_build_grid_diagram)

    assert build_result_grid_diagram(
        result,
        temperature_scaled=True,
    ) == {"diagram": "ok"}
    assert calls == {
        "network": result.network,
        "period_id": None,
        "temperature_scaled": True,
    }
    with pytest.raises(IndexError, match="1-based"):
        build_result_grid_diagram(result, 0)
    with pytest.raises(IndexError, match="only 1 network"):
        build_result_grid_diagram(result, 2)


def test_synthesis_result_selects_ranked_network_and_copies_metrics(monkeypatch):
    task = _task()
    selected_network = _network(
        stage_count=3,
        total_annual_cost=10.0,
        utility_cost=4.0,
        capital_cost=6.0,
    )
    outcome = HeatExchangerNetworkSynthesisTaskOutcome(
        task=task,
        status="success",
        network=selected_network,
        objective_value=10.0,
        solver_status="optimal",
    )
    result = HeatExchangerNetworkSynthesisResult(
        network=_network(stage_count=1),
        run_id="schema-run-1",
    )

    monkeypatch.setattr(
        result_operations,
        "ranked_networks",
        lambda _result, n=None: (outcome,),
    )

    assert select_network(result, 1) is result
    assert result.network is selected_network
    assert result.task_id == task.task_id
    assert result.solver_status == "optimal"
    assert result.method == task.method
    assert result.stage_count == 3
    assert result.objective_values == {
        "total_annual_cost": 10.0,
        "utility_cost": 4.0,
        "capital_cost": 6.0,
    }

    with pytest.raises(IndexError, match="only 1 network"):
        select_network(result, 2)


def test_synthesis_result_grid_diagram_uses_ranked_network(monkeypatch):
    task = _task()
    selected_network = _network(stage_count=2)
    outcome = HeatExchangerNetworkSynthesisTaskOutcome(
        task=task,
        status="success",
        network=selected_network,
    )
    result = HeatExchangerNetworkSynthesisResult(
        network=_network(stage_count=1),
        run_id="schema-run-1",
    )
    calls = {}

    def fake_build_grid_diagram(network, **kwargs):
        calls["network"] = network
        calls["kwargs"] = kwargs
        return network

    monkeypatch.setattr(
        result_operations,
        "ranked_networks",
        lambda _result, n=None: (outcome,),
    )
    monkeypatch.setattr(grid_service, "build_grid_diagram", fake_build_grid_diagram)

    assert build_result_grid_diagram(result) is selected_network
    assert calls["network"] is selected_network
    assert calls["kwargs"] == {"period_id": None, "temperature_scaled": False}
    with pytest.raises(IndexError, match="only 1 network"):
        build_result_grid_diagram(result, 2)


def test_synthesis_result_ranked_selection_rejects_missing_network(monkeypatch):
    missing_network = HeatExchangerNetworkSynthesisTaskOutcome(
        task=_task(),
        status="success",
        network=None,
    )
    result = HeatExchangerNetworkSynthesisResult(
        network=_network(),
        run_id="schema-run-1",
    )

    monkeypatch.setattr(
        result_operations,
        "ranked_networks",
        lambda _result, n=None: (missing_network,),
    )

    with pytest.raises(ValueError, match="missing network output"):
        select_network(result, 1)
    with pytest.raises(ValueError, match="missing network output"):
        build_result_grid_diagram(result, 1)
