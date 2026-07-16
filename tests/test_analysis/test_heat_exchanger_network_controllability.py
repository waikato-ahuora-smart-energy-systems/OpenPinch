"""Tests for heat exchanger network controllability screening."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from OpenPinch.classes import (
    HeatExchanger,
    HeatExchangerKind,
    HeatExchangerNetwork,
    HeatExchangerStreamRole,
)
from OpenPinch.classes._heat_exchanger.period_state import HeatExchangerPeriodState
from OpenPinch.services.heat_exchanger_network_controllability import (
    HeatExchangerNetworkControllabilityActuator,
    HeatExchangerNetworkControllabilityEndpoint,
    HeatExchangerNetworkControllabilityPairing,
    HeatExchangerNetworkControllabilityResult,
    quantify_heat_exchanger_network_controllability,
)
from OpenPinch.services.heat_exchanger_network_controllability import (
    service as controllability_service,
)


def _recovery_exchanger(
    exchanger_id: str,
    source: str,
    sink: str,
    *,
    duty: float,
    stage: int = 1,
    approach: float = 8.0,
) -> HeatExchanger:
    return HeatExchanger(
        exchanger_id=exchanger_id,
        kind=HeatExchangerKind.RECOVERY,
        source_stream=source,
        sink_stream=sink,
        source_stream_role=HeatExchangerStreamRole.PROCESS,
        sink_stream_role=HeatExchangerStreamRole.PROCESS,
        stage=stage,
        period_states=(
            HeatExchangerPeriodState(
                period_id="0",
                period_idx=0,
                duty=duty,
                approach_temperatures=(approach,),
            ),
        ),
    )


def _hot_utility_exchanger(
    exchanger_id: str = "hu-C1",
    sink: str = "C1",
    *,
    duty: float = 20.0,
) -> HeatExchanger:
    return HeatExchanger(
        exchanger_id=exchanger_id,
        kind=HeatExchangerKind.HOT_UTILITY,
        source_stream="Steam",
        sink_stream=sink,
        source_stream_role=HeatExchangerStreamRole.UTILITY,
        sink_stream_role=HeatExchangerStreamRole.PROCESS,
        period_states=(
            HeatExchangerPeriodState(
                period_id="0",
                period_idx=0,
                duty=duty,
                approach_temperatures=(7.0,),
            ),
        ),
    )


def _cold_utility_exchanger(
    exchanger_id: str = "cu-H2",
    source: str = "H2",
    *,
    duty: float = 15.0,
) -> HeatExchanger:
    return HeatExchanger(
        exchanger_id=exchanger_id,
        kind=HeatExchangerKind.COLD_UTILITY,
        source_stream=source,
        sink_stream="CoolingWater",
        source_stream_role=HeatExchangerStreamRole.PROCESS,
        sink_stream_role=HeatExchangerStreamRole.UTILITY,
        period_states=(
            HeatExchangerPeriodState(
                period_id="0",
                period_idx=0,
                duty=duty,
                approach_temperatures=(6.0,),
            ),
        ),
    )


def _screening_network() -> HeatExchangerNetwork:
    return HeatExchangerNetwork(
        exchangers=(
            _recovery_exchanger("r-H1-C1", "H1", "C1", duty=70.0),
            _recovery_exchanger("r-H1-C2", "H1", "C2", duty=30.0, stage=2),
            _recovery_exchanger("r-H2-C2", "H2", "C2", duty=90.0),
            _hot_utility_exchanger(),
            _cold_utility_exchanger(),
        )
    )


def test_controllability_service_builds_interaction_matrix_and_pairings() -> None:
    result = quantify_heat_exchanger_network_controllability(_screening_network())

    assert [output.output_id for output in result.outputs] == [
        "source:H1",
        "sink:C1",
        "sink:C2",
        "source:H2",
    ]
    assert [actuator.actuator_id for actuator in result.actuators] == [
        "r-H1-C1",
        "r-H1-C2",
        "r-H2-C2",
        "hu-C1",
        "cu-H2",
    ]
    assert result.interaction_matrix[0] == pytest.approx((0.7, 0.3, 0.0, 0.0, 0.0))
    assert result.interaction_matrix[1] == pytest.approx(
        (70.0 / 90.0, 0.0, 0.0, 20.0 / 90.0, 0.0)
    )
    assert result.matrix_rank == 4
    assert result.score > 0.0
    assert result.rating in {"moderate", "strong"}
    assert result.components.rank == pytest.approx(1.0)
    assert result.components.conditioning > 0.0
    assert result.components.thermal_margin == pytest.approx(1.0)
    assert {pairing.output_id for pairing in result.pairings} == {
        "source:H1",
        "sink:C1",
        "sink:C2",
        "source:H2",
    }


def test_controllability_service_can_exclude_utility_actuators() -> None:
    with_utilities = quantify_heat_exchanger_network_controllability(
        _screening_network(),
    )
    without_utilities = quantify_heat_exchanger_network_controllability(
        _screening_network(),
        include_utility_actuators=False,
    )

    assert len(without_utilities.actuators) == 3
    assert len(with_utilities.actuators) == 5
    assert without_utilities.matrix_rank < with_utilities.matrix_rank
    assert (
        without_utilities.components.redundancy < with_utilities.components.redundancy
    )
    assert without_utilities.score < with_utilities.score
    assert "utility flow actuators were excluded from the analysis" in (
        without_utilities.diagnostics
    )


def test_controllability_service_reports_low_thermal_margin() -> None:
    network = HeatExchangerNetwork(
        exchangers=(
            _recovery_exchanger(
                "tight-match",
                "H1",
                "C1",
                duty=50.0,
                approach=2.0,
            ),
        )
    )

    result = network.quantify_controllability(minimum_approach_temperature=5.0)

    assert result.components.thermal_margin == pytest.approx(0.4)
    assert result.minimum_approach_temperature == pytest.approx(2.0)
    assert (
        "minimum approach temperature is below the requested margin (2 < 5)"
        in result.diagnostics
    )


def test_controllability_result_serializes_and_round_trips() -> None:
    result = _screening_network().quantify_controllability()

    round_tripped = HeatExchangerNetworkControllabilityResult.model_validate_json(
        result.model_dump_json()
    )

    assert round_tripped == result


def test_empty_network_returns_poor_zero_score() -> None:
    result = quantify_heat_exchanger_network_controllability(
        HeatExchangerNetwork(period_id="0")
    )

    assert result.score == pytest.approx(0.0)
    assert result.rating == "poor"
    assert result.outputs == ()
    assert result.actuators == ()
    assert result.interaction_matrix == ()
    assert "no process-stream outlet temperatures were found" in result.diagnostics
    assert "no manipulated variables were found" in result.diagnostics


def test_controllability_service_rejects_invalid_options() -> None:
    with pytest.raises(ValueError, match="desired_redundancy"):
        quantify_heat_exchanger_network_controllability(
            _screening_network(),
            desired_redundancy=0,
        )

    with pytest.raises(ValueError, match="minimum_interaction"):
        quantify_heat_exchanger_network_controllability(
            _screening_network(),
            minimum_interaction=-1.0,
        )


def test_partial_controllability_case_distinguishes_projected_outputs() -> None:
    monotubular_like = HeatExchangerNetwork(
        exchangers=(_hot_utility_exchanger(sink="TubeOutlet", duty=25.0),)
    )
    two_stream_like = HeatExchangerNetwork(
        exchangers=(
            _recovery_exchanger(
                "single-boundary-control",
                "H1",
                "C1",
                duty=25.0,
            ),
        )
    )

    single_projection = monotubular_like.quantify_controllability()
    coupled_projection = two_stream_like.quantify_controllability()

    assert [output.output_id for output in single_projection.outputs] == [
        "sink:TubeOutlet"
    ]
    assert single_projection.matrix_rank == 1
    assert single_projection.components.rank == pytest.approx(1.0)
    assert not any(
        "rank deficient" in diagnostic for diagnostic in single_projection.diagnostics
    )

    assert [output.output_id for output in coupled_projection.outputs] == [
        "source:H1",
        "sink:C1",
    ]
    assert coupled_projection.matrix_rank == 1
    assert coupled_projection.components.rank == pytest.approx(0.5)
    assert (
        "interaction matrix is rank deficient for full outlet-temperature control"
        in coupled_projection.diagnostics
    )


def test_multistage_utility_streams_create_independent_actuators() -> None:
    network = HeatExchangerNetwork(
        exchangers=(
            _recovery_exchanger("r-H1-C1", "H1", "C1", duty=50.0),
            _hot_utility_exchanger("thermal-oil-hot", "C1", duty=25.0),
            _hot_utility_exchanger("thermal-oil-warm", "C1", duty=25.0),
        )
    )

    with_utilities = network.quantify_controllability(desired_redundancy=3)
    without_utilities = network.quantify_controllability(
        desired_redundancy=3,
        include_utility_actuators=False,
    )

    assert [actuator.actuator_id for actuator in with_utilities.actuators] == [
        "r-H1-C1",
        "thermal-oil-hot",
        "thermal-oil-warm",
    ]
    assert [actuator.manipulated_variable for actuator in with_utilities.actuators] == [
        "recovery_bypass_fraction",
        "hot_utility_flow",
        "hot_utility_flow",
    ]
    assert with_utilities.interaction_matrix[1] == pytest.approx((0.5, 0.25, 0.25))
    assert with_utilities.matrix_rank == 2
    assert without_utilities.matrix_rank == 1
    assert (
        with_utilities.components.redundancy > without_utilities.components.redundancy
    )


def test_output_feedback_case_does_not_require_temperature_measurements() -> None:
    exchanger = _hot_utility_exchanger(sink="ControlledOutlet", duty=30.0)
    assert exchanger.state().sink_outlet_temperature is None

    result = HeatExchangerNetwork(exchangers=(exchanger,)).quantify_controllability()

    assert result.outputs[0].output_id == "sink:ControlledOutlet"
    assert result.actuators[0].manipulated_variable == "hot_utility_flow"
    assert result.interaction_matrix[0] == pytest.approx((1.0,))
    assert result.matrix_rank == 1
    assert result.components.authority == pytest.approx(1.0)


def test_ill_conditioned_partial_gramian_proxy_is_reported() -> None:
    network = HeatExchangerNetwork(
        exchangers=(
            _recovery_exchanger("dominant-recovery", "H1", "C1", duty=100.0),
            _hot_utility_exchanger("trim-heat", "C1", duty=1.0),
            _cold_utility_exchanger("trim-cool", "H1", duty=1.0),
        )
    )

    result = network.quantify_controllability(condition_warning_threshold=10.0)

    assert result.matrix_rank == 2
    assert result.condition_number is not None
    assert result.condition_number > 10.0
    assert result.components.conditioning < 0.1
    assert any("poorly conditioned" in item for item in result.diagnostics)


def test_controllability_model_validators_reject_invalid_fields() -> None:
    with pytest.raises(ValidationError, match="endpoint identities"):
        HeatExchangerNetworkControllabilityEndpoint(
            output_id=" ",
            stream_id="H1",
            side="source",
            exchanger_count=1,
            total_duty=1.0,
        )
    with pytest.raises(ValidationError, match="exchanger_count"):
        HeatExchangerNetworkControllabilityEndpoint(
            output_id="source:H1",
            stream_id="H1",
            side="source",
            exchanger_count=-1,
            total_duty=1.0,
        )

    actuator = HeatExchangerNetworkControllabilityActuator(
        actuator_id=" trim ",
        exchanger_id=None,
        kind=HeatExchangerKind.RECOVERY,
        source_stream=" H1 ",
        sink_stream=" C1 ",
        manipulated_variable="recovery_bypass_fraction",
        duty=1.0,
    )
    assert actuator.actuator_id == "trim"
    assert actuator.exchanger_id is None
    assert actuator.source_stream == "H1"
    assert actuator.sink_stream == "C1"

    with pytest.raises(ValidationError, match="actuator identities"):
        HeatExchangerNetworkControllabilityActuator(
            actuator_id="",
            kind=HeatExchangerKind.RECOVERY,
            source_stream="H1",
            sink_stream="C1",
            manipulated_variable="recovery_bypass_fraction",
            duty=1.0,
        )
    with pytest.raises(ValidationError, match="exchanger_id"):
        HeatExchangerNetworkControllabilityActuator(
            actuator_id="a",
            exchanger_id=" ",
            kind=HeatExchangerKind.RECOVERY,
            source_stream="H1",
            sink_stream="C1",
            manipulated_variable="recovery_bypass_fraction",
            duty=1.0,
        )
    with pytest.raises(ValidationError, match="stage"):
        HeatExchangerNetworkControllabilityActuator(
            actuator_id="a",
            kind=HeatExchangerKind.RECOVERY,
            source_stream="H1",
            sink_stream="C1",
            stage=0,
            manipulated_variable="recovery_bypass_fraction",
            duty=1.0,
        )
    with pytest.raises(ValidationError, match="pairing identities"):
        HeatExchangerNetworkControllabilityPairing(
            output_id="",
            actuator_id="a",
            interaction=0.5,
        )

    result_kwargs = {
        "score": 0.5,
        "rating": "moderate",
        "components": {
            "rank": 1.0,
            "pairing": 1.0,
            "authority": 1.0,
            "conditioning": 1.0,
            "redundancy": 1.0,
        },
    }
    with pytest.raises(ValidationError, match="equal width"):
        HeatExchangerNetworkControllabilityResult(
            **result_kwargs,
            interaction_matrix=((0.5,), (0.5, 0.2)),
        )
    with pytest.raises(ValidationError, match="matrix_rank"):
        HeatExchangerNetworkControllabilityResult(**result_kwargs, matrix_rank=-1)
    with pytest.raises(ValidationError, match="condition_number"):
        HeatExchangerNetworkControllabilityResult(
            **result_kwargs,
            condition_number=float("inf"),
        )
    with pytest.raises(ValidationError, match="between 0 and 1"):
        HeatExchangerNetworkControllabilityResult(**{**result_kwargs, "score": 1.5})
    with pytest.raises(ValidationError, match="singular value"):
        HeatExchangerNetworkControllabilityResult(
            **result_kwargs,
            singular_values=(-1.0,),
        )


def test_controllability_service_option_validation_and_helper_edges() -> None:
    with pytest.raises(ValueError, match="minimum_approach_temperature"):
        quantify_heat_exchanger_network_controllability(
            _screening_network(),
            minimum_approach_temperature=-1.0,
        )
    with pytest.raises(ValueError, match="rank_tolerance"):
        quantify_heat_exchanger_network_controllability(
            _screening_network(),
            rank_tolerance=-1.0,
        )
    with pytest.raises(ValueError, match="condition_warning_threshold"):
        quantify_heat_exchanger_network_controllability(
            _screening_network(),
            condition_warning_threshold=0.0,
        )

    zero_duty = _recovery_exchanger("zero", "H0", "C0", duty=0.0)
    assert controllability_service._build_outputs((zero_duty,), period_id="0") == ()
    assert (
        controllability_service._build_actuators(
            (zero_duty,),
            period_id="0",
            include_utility_actuators=True,
        )
        == ()
    )

    duplicate_actuators = controllability_service._build_actuators(
        (
            _recovery_exchanger("dup", "H1", "C1", duty=1.0),
            _recovery_exchanger("dup", "H1", "C2", duty=1.0),
        ),
        period_id="0",
        include_utility_actuators=True,
    )
    assert [actuator.actuator_id for actuator in duplicate_actuators] == [
        "dup",
        "dup#2",
    ]

    zero_duty_output = HeatExchangerNetworkControllabilityEndpoint(
        output_id="source:H0",
        stream_id="H0",
        side="source",
        exchanger_count=1,
        total_duty=0.0,
    )
    process_actuator = HeatExchangerNetworkControllabilityActuator(
        actuator_id="a",
        kind=HeatExchangerKind.RECOVERY,
        source_stream="H0",
        sink_stream="C0",
        manipulated_variable="recovery_bypass_fraction",
        duty=1.0,
    )
    np.testing.assert_allclose(
        controllability_service._build_interaction_matrix(
            (zero_duty_output,),
            (process_actuator,),
        ),
        [[0.0]],
    )

    rank, singular_values, condition_number, conditioning_score = (
        controllability_service._matrix_diagnostics(np.zeros((1, 1)), None)
    )
    assert rank == 0
    assert singular_values == (0.0,)
    assert condition_number is None
    assert conditioning_score == 0.0

    assert controllability_service._matrix_diagnostics(
        np.eye(1),
        rank_tolerance=2.0,
    ) == (0, (1.0,), None, 0.0)
    assert (
        controllability_service._redundancy_score(
            np.array([[1.0]]),
            desired_redundancy=1,
            minimum_interaction=0.1,
        )
        == 1.0
    )
    assert controllability_service._thermal_margin_score(
        (_recovery_exchanger("margin", "H1", "C1", duty=1.0),),
        period_id="0",
        minimum_approach_temperature=0.0,
    ) == (1.0, 8.0)
    assert controllability_service._rating(0.3) == "weak"


def test_controllability_diagnostics_reports_zero_conditioning_direction() -> None:
    outputs = (
        HeatExchangerNetworkControllabilityEndpoint(
            output_id="source:H1",
            stream_id="H1",
            side="source",
            exchanger_count=1,
            total_duty=1.0,
        ),
    )
    actuators = (
        HeatExchangerNetworkControllabilityActuator(
            actuator_id="a",
            kind=HeatExchangerKind.RECOVERY,
            source_stream="H1",
            sink_stream="C1",
            manipulated_variable="recovery_bypass_fraction",
            duty=1.0,
        ),
    )

    diagnostics = controllability_service._diagnostics(
        outputs=outputs,
        actuators=actuators,
        matrix=np.zeros((1, 1)),
        matrix_rank=1,
        condition_number=None,
        condition_warning_threshold=10.0,
        conditioning_score=0.0,
        include_utility_actuators=True,
        minimum_approach_temperature=0.0,
        minimum_observed_approach=8.0,
        thermal_margin_score=1.0,
        minimum_interaction=0.1,
    )

    assert "interaction matrix has no usable singular direction" in diagnostics
