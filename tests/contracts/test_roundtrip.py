"""Generated round-trip properties for external contracts."""

from __future__ import annotations

from hypothesis import given, seed, settings
from hypothesis import strategies as st

from OpenPinch.contracts.input import TargetInput
from OpenPinch.contracts.output import TargetOutput
from tests.strategies.contracts import target_input_payloads
from tests.strategies.heat_exchangers import aligned_heat_exchanger_networks


@seed(20260715)
@given(payload=target_input_payloads())
@settings(max_examples=40)
def test_target_input_json_round_trip(payload) -> None:
    validated = TargetInput.model_validate(payload)

    restored = TargetInput.model_validate_json(validated.model_dump_json())

    assert restored == validated
    assert restored.model_dump(mode="json") == validated.model_dump(mode="json")


@seed(20260715)
@given(
    name=st.text(min_size=1, max_size=40),
    period_id=st.one_of(
        st.none(),
        st.text(min_size=1, max_size=20),
    ),
)
@settings(max_examples=30)
def test_target_output_json_round_trip(name: str, period_id: str | None) -> None:
    output = TargetOutput(name=name, period_id=period_id, targets=[])

    restored = TargetOutput.model_validate_json(output.model_dump_json())

    assert restored == output


@seed(20260717)
@given(network=aligned_heat_exchanger_networks())
@settings(max_examples=30)
def test_serialized_hen_target_input_json_round_trip(network) -> None:
    payload = network.model_dump(mode="json")
    validated = TargetInput.model_validate({"streams": [], "network": payload})

    restored = TargetInput.model_validate_json(validated.model_dump_json())
    restored_payload = restored.model_dump(mode="json")["network"]

    assert validated.model_dump(mode="json")["network"] == payload
    assert restored_payload == payload
    assert [item["exchanger_id"] for item in restored_payload["exchangers"]] == [
        item["exchanger_id"] for item in payload["exchangers"]
    ]
    assert [
        [state["period_id"] for state in item["period_states"]]
        for item in restored_payload["exchangers"]
    ] == [
        [state["period_id"] for state in item["period_states"]]
        for item in payload["exchangers"]
    ]
    assert [
        [
            (area["hot_segment_identity"], area["cold_segment_identity"])
            for area in item["segment_area_contributions"]
        ]
        for item in restored_payload["exchangers"]
    ] == [
        [
            (area["hot_segment_identity"], area["cold_segment_identity"])
            for area in item["segment_area_contributions"]
        ]
        for item in payload["exchangers"]
    ]
