"""Focused tests for heat-pump schema helper behavior."""

from __future__ import annotations

import json

import numpy as np
import pytest

from OpenPinch.contracts.hpr import (
    HPRBackendResult,
    HPRParsedState,
    HPRThermoArtifacts,
)
from OpenPinch.domain.stream import Stream
from OpenPinch.domain.stream_collection import StreamCollection
from tests.support.paths import FIXTURES_ROOT

FIXTURE_PATH = FIXTURES_ROOT / "hpr_schema_cases.json"


def _fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _stream_collection() -> StreamCollection:
    collection = StreamCollection()
    for spec in _fixture()["hpr_streams"]:
        collection.add(
            Stream(
                name=spec["name"],
                supply_temperature=spec["t_supply"],
                target_temperature=spec["t_target"],
                heat_flow=spec["heat_flow"],
                is_process_stream=False,
            )
        )
    return collection


def _backend_payload(**updates) -> dict:
    payload = dict(_fixture()["backend_result"])
    for key in (
        "T_cond",
        "T_evap",
        "Q_cond",
        "Q_evap",
        "Q_cond_he",
        "Q_evap_he",
        "dT_subcool",
        "dT_superheat",
        "T_comp_out",
        "dT_gc",
        "dT_comp",
        "Q_heat",
        "Q_cool",
    ):
        payload[key] = np.array(payload[key], dtype=float)
    payload.update(updates)
    return payload


def test_hpr_parsed_state_supports_attributes_and_model_serialization():
    state = HPRParsedState(Q_amb_hot=1.5)

    assert state.Q_amb_hot == pytest.approx(1.5)
    assert state.model_dump()["Q_amb_hot"] == pytest.approx(1.5)


def test_hpr_backend_result_projects_empty_artifacts_and_failure_state():
    result = HPRBackendResult(**_backend_payload(artifacts=None, amb_streams=None))

    assert result.Q_ext == pytest.approx(7.0)
    assert result.hpr_streams is None
    assert result.hpr_hot_streams is None
    assert result.hpr_cold_streams is None
    assert result.model is None
    serialized = result.model_dump()
    assert serialized["Q_ext_heat"] == pytest.approx(3.0)
    assert serialized["Q_ext_cold"] == pytest.approx(4.0)
    assert serialized["artifacts"] is None

    updated = result.with_updates(utility_tot=30.0)
    assert updated.utility_tot == pytest.approx(30.0)
    assert result.utility_tot == pytest.approx(20.0)

    output_fields = result.to_output_fields()
    assert output_fields["Q_ext"] == pytest.approx(7.0)
    assert "hpr_hot_streams" not in output_fields
    assert "model" not in output_fields

    failure = HPRBackendResult.failure(
        reason="no feasible point",
        Q_amb_hot=1.0,
        Q_amb_cold=2.0,
    )
    assert failure.success is False
    assert failure.failure_reason == "no feasible point"
    assert np.isinf(failure.obj)
    assert failure.Q_amb_hot == pytest.approx(1.0)
    assert failure.Q_amb_cold == pytest.approx(2.0)


def test_hpr_backend_result_projects_artifacts_and_output_fields():
    hpr_streams = _stream_collection()
    artifacts = HPRThermoArtifacts(
        hpr_streams=hpr_streams,
        model={"name": "model"},
        debug_figure="figure",
    )
    result = HPRBackendResult(
        **_backend_payload(
            artifacts=artifacts,
            amb_streams=StreamCollection(),
        )
    )

    assert result.hpr_streams is hpr_streams
    assert len(result.hpr_hot_streams) == 1
    assert result.hpr_hot_streams[0].name == "HPR hot utility"
    assert len(result.hpr_cold_streams) == 1
    assert result.hpr_cold_streams[0].name == "HPR cold utility"
    assert result.model == {"name": "model"}

    output_fields = result.to_output_fields()

    assert output_fields["hpr_hot_streams"][0].name == "HPR hot utility"
    assert output_fields["hpr_cold_streams"][0].name == "HPR cold utility"
    assert output_fields["amb_streams"] == StreamCollection()
    assert output_fields["model"] == {"name": "model"}
    assert output_fields["T_cond"].tolist() == [90.0]
    assert output_fields["Q_cool"].tolist() == [10.0]
