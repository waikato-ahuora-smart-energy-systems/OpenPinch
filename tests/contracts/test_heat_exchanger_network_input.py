"""Contract tests for serialized heat exchanger networks on ``TargetInput``."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest
from pydantic import ValidationError

from OpenPinch.application._problem.input.canonicalization import (
    canonical_problem_inputs,
)
from OpenPinch.contracts.input import (
    HeatExchangerAreaSliceSchema,
    HeatExchangerNetworkSchema,
    HeatExchangerPeriodStateSchema,
    HeatExchangerSchema,
    TargetInput,
)
from OpenPinch.domain._heat_exchanger.area import HeatExchangerAreaSlice
from OpenPinch.domain._heat_exchanger.period_state import HeatExchangerPeriodState
from OpenPinch.domain.enums import HeatExchangerKind, StreamID
from OpenPinch.domain.heat_exchanger import HeatExchanger
from OpenPinch.domain.heat_exchanger_network import HeatExchangerNetwork


def _period_states(count: int = 1) -> tuple[HeatExchangerPeriodState, ...]:
    return tuple(
        HeatExchangerPeriodState(
            period_id=("base", "peak")[period_idx],
            period_idx=period_idx,
            duty=20.0 + period_idx,
            active=period_idx == 0,
            approach_temperatures=(10.0 + period_idx, 8.0),
            source_split_fraction=0.6,
            sink_split_fraction=0.4,
            source_inlet_temperature=180.0 + period_idx,
            source_outlet_temperature=120.0,
            sink_inlet_temperature=60.0,
            sink_outlet_temperature=110.0 + period_idx,
        )
        for period_idx in range(count)
    )


def _exchanger(
    kind: HeatExchangerKind = HeatExchangerKind.RECOVERY,
    *,
    period_count: int = 1,
    with_area_slices: bool = False,
) -> HeatExchanger:
    identities = {
        HeatExchangerKind.RECOVERY: (
            "H1",
            "C1",
            StreamID.Process,
            StreamID.Process,
            1,
        ),
        HeatExchangerKind.HOT_UTILITY: (
            "Steam",
            "C1",
            StreamID.Utility,
            StreamID.Process,
            None,
        ),
        HeatExchangerKind.COLD_UTILITY: (
            "H1",
            "CoolingWater",
            StreamID.Process,
            StreamID.Utility,
            None,
        ),
    }
    source, sink, source_role, sink_role, stage = identities[kind]
    contributions = (
        tuple(
            HeatExchangerAreaSlice(
                period=("base", "peak")[period_idx],
                hot_segment_identity=f"{source}.S1",
                cold_segment_identity=f"{sink}.S1",
                duty=10.0 + period_idx,
                hot_inlet_temperature=180.0,
                hot_outlet_temperature=120.0,
                cold_inlet_temperature=60.0,
                cold_outlet_temperature=110.0,
                hot_htc=2.0,
                cold_htc=3.0,
                overall_htc=1.2,
                lmtd=25.0,
                area=2.0 + period_idx,
            )
            for period_idx in range(period_count)
        )
        if with_area_slices
        else ()
    )
    return HeatExchanger(
        exchanger_id=f"{kind.value}-1",
        kind=kind,
        source_stream=source,
        sink_stream=sink,
        source_stream_role=source_role,
        sink_stream_role=sink_role,
        stage=stage,
        period_states=_period_states(period_count),
        area=None if with_area_slices else 4.0,
        match_allowed=False,
        capital_cost=5000.0,
        segment_area_contributions=contributions,
        solver_metadata={"private": "solver"},
        source_metadata={"private": "source"},
    )


def _fully_populated_network() -> HeatExchangerNetwork:
    return HeatExchangerNetwork(
        exchangers=(_exchanger(period_count=2, with_area_slices=True),),
        run_id="run-1",
        task_id="task-1",
        period_id="base",
        method="open_hens_method",
        stage_count=2,
        objective_value=100.0,
        total_annual_cost=90.0,
        utility_cost=40.0,
        capital_cost=50.0,
        summary_metrics={
            "count": 1,
            "score": 0.75,
            "accepted": True,
            "label": "candidate",
            "optional": None,
        },
        solver_axis_metadata={"private": "axes"},
        source_metadata={"private": "source"},
    )


@pytest.mark.parametrize(
    "network",
    [
        pytest.param(HeatExchangerNetwork(), id="empty"),
        pytest.param(
            HeatExchangerNetwork(exchangers=(_exchanger(),)),
            id="recovery-single-period",
        ),
        pytest.param(
            HeatExchangerNetwork(
                exchangers=(_exchanger(HeatExchangerKind.HOT_UTILITY),)
            ),
            id="hot-utility",
        ),
        pytest.param(
            HeatExchangerNetwork(
                exchangers=(_exchanger(HeatExchangerKind.COLD_UTILITY),)
            ),
            id="cold-utility",
        ),
        pytest.param(_fully_populated_network(), id="multiperiod-area-fully-populated"),
    ],
)
def test_runtime_network_json_dump_has_exact_target_input_parity(
    network: HeatExchangerNetwork,
) -> None:
    payload = network.model_dump(mode="json")

    validated = TargetInput.model_validate({"streams": [], "network": payload})

    assert isinstance(validated.network, HeatExchangerNetworkSchema)
    assert validated.model_dump(mode="json")["network"] == payload
    restored = TargetInput.model_validate_json(validated.model_dump_json())
    assert restored.model_dump(mode="json")["network"] == payload


def test_network_input_accepts_null_and_noncanonical_runtime_dump_variants() -> None:
    assert TargetInput.model_validate({"streams": []}).network is None
    assert TargetInput.model_validate({"streams": [], "network": None}).network is None
    network = _fully_populated_network()

    for options in (
        {"exclude_none": True},
        {"exclude_defaults": True},
        {"exclude_unset": True},
    ):
        payload = network.model_dump(mode="json", **options)
        assert TargetInput.model_validate({"streams": [], "network": payload}).network


def test_canonical_problem_inputs_retain_the_transport_network_without_consuming_it() -> (
    None
):
    input_data = TargetInput.model_validate(
        {
            "streams": [],
            "network": _fully_populated_network().model_dump(mode="json"),
        }
    )

    canonical = canonical_problem_inputs(input_data, project_name="Site")

    assert canonical["network"] == input_data.model_dump(mode="python")["network"]
    assert (
        json.loads(json.dumps(canonical))["network"]
        == input_data.model_dump(mode="json")["network"]
    )


def test_network_input_rejects_an_encoded_json_string() -> None:
    with pytest.raises(ValidationError, match="valid dictionary"):
        TargetInput.model_validate(
            {"streams": [], "network": _fully_populated_network().model_dump_json()}
        )


@pytest.mark.parametrize(
    ("metadata_name", "nested"),
    [
        ("solver_axis_metadata", False),
        ("source_metadata", False),
        ("solver_metadata", True),
        ("source_metadata", True),
    ],
)
def test_private_runtime_metadata_is_excluded_and_rejected(
    metadata_name: str,
    nested: bool,
) -> None:
    payload = _fully_populated_network().model_dump(mode="json")
    target = payload["exchangers"][0] if nested else payload
    assert metadata_name not in target
    target[metadata_name] = {"private": True}

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TargetInput.model_validate({"streams": [], "network": payload})


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_stream_role", "process", "Input should be"),
        ("source_stream_role", "Unassigned", "recovery exchangers"),
        ("source_stream_role", "Utility", "recovery exchangers"),
        ("sink_stream_role", "Unassigned", "recovery exchangers"),
        ("stage", None, "must include a synthesis stage"),
    ],
)
def test_network_input_rejects_noncanonical_endpoint_contracts(
    field: str,
    value: object,
    message: str,
) -> None:
    payload = HeatExchangerNetwork(exchangers=(_exchanger(),)).model_dump(mode="json")
    payload["exchangers"][0][field] = value

    with pytest.raises(ValidationError, match=message):
        TargetInput.model_validate({"streams": [], "network": payload})


def test_network_input_mirrors_nested_order_area_and_extra_field_validation() -> None:
    payload = _fully_populated_network().model_dump(mode="json")

    invalid_idx = deepcopy(payload)
    invalid_idx["exchangers"][0]["period_states"][1]["period_idx"] = 3
    with pytest.raises(ValidationError, match="contiguous period_idx"):
        TargetInput.model_validate({"streams": [], "network": invalid_idx})

    invalid_area = deepcopy(payload)
    invalid_area["exchangers"][0]["area"] = 99.0
    with pytest.raises(ValidationError, match="maximum period-total segment area"):
        TargetInput.model_validate({"streams": [], "network": invalid_area})

    unknown_nested = deepcopy(payload)
    unknown_nested["exchangers"][0]["period_states"][0]["private"] = True
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TargetInput.model_validate({"streams": [], "network": unknown_nested})


def test_runtime_and_transport_dump_keys_are_guarded_against_drift() -> None:
    runtime_network = _fully_populated_network()
    runtime_dump = runtime_network.model_dump(mode="json")
    schema_dump = HeatExchangerNetworkSchema.model_validate(runtime_dump).model_dump(
        mode="json"
    )

    assert set(schema_dump) == set(runtime_dump)
    assert set(schema_dump["exchangers"][0]) == set(runtime_dump["exchangers"][0])
    assert set(HeatExchangerNetwork.model_fields) - set(runtime_dump) == {
        "solver_axis_metadata",
        "source_metadata",
    }
    assert set(HeatExchanger.model_fields) - set(runtime_dump["exchangers"][0]) == {
        "solver_metadata",
        "source_metadata",
    }
    assert set(HeatExchangerNetworkSchema.model_fields) == set(runtime_dump)
    assert set(HeatExchangerSchema.model_fields) == set(runtime_dump["exchangers"][0])
    assert set(HeatExchangerPeriodStateSchema.model_fields) == set(
        runtime_dump["exchangers"][0]["period_states"][0]
    )
    assert set(HeatExchangerAreaSliceSchema.model_fields) == set(
        runtime_dump["exchangers"][0]["segment_area_contributions"][0]
    )
