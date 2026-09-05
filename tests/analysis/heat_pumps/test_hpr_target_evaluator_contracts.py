"""Engine-neutral HPR targeting evaluator model and lifecycle contracts."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, replace

import pytest
from hypothesis import given, seed, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from OpenPinch.analysis.heat_pumps.performance_maps.fluids import (
    parse_hpr_working_fluid,
)
from OpenPinch.analysis.heat_pumps.performance_maps.targeting import (
    HprTargetEvaluatorCoordinator,
    validate_hpr_target_result,
)
from OpenPinch.analysis.heat_pumps.performance_maps.targeting_models import (
    HprTargetEvaluationFailure,
    HprTargetEvaluatorError,
    HprTargetEvaluatorMetadata,
    HprTargetThermodynamicRequest,
    HprTargetThermodynamicResult,
    HprThermalProfilePoint,
)
from tests.analysis.heat_pumps.hpr_targeting_fakes import FakeHprTargetEvaluator


def _request(**overrides) -> HprTargetThermodynamicRequest:
    values = {
        "mode": "heat_pump",
        "cycle_id": "single_stage_vapour_compression",
        "model_id": "openpinch-tespy-single-stage-v1",
        "working_fluid": parse_hpr_working_fluid("R134a"),
        "evaporating_temperature": 5.0,
        "condensing_temperature": 55.0,
        "useful_duty": 100.0,
        "source_approach_temperature": 5.0,
        "sink_approach_temperature": 5.0,
        "compressor_isentropic_efficiency": 0.75,
        "superheat": 5.0,
        "subcooling": 2.0,
        "internal_hx_gas_temperature_change": 0.0,
        "candidate_id": "candidate-1",
    }
    values.update(overrides)
    return HprTargetThermodynamicRequest(**values)


def _result(request=None, **overrides) -> HprTargetThermodynamicResult:
    request = request or _request()
    values = {
        "backend": "tespy",
        "model_id": request.model_id,
        "working_fluid": request.working_fluid,
        "converged": True,
        "q_source": 75.0,
        "q_sink": 100.0,
        "compressor_power": 25.0,
        "cop": 4.0,
        "source_profile": (
            HprThermalProfilePoint(temperature=10.0, enthalpy=0.0),
            HprThermalProfilePoint(temperature=5.0, enthalpy=75.0),
        ),
        "sink_profile": (
            HprThermalProfilePoint(temperature=55.0, enthalpy=0.0),
            HprThermalProfilePoint(temperature=60.0, enthalpy=100.0),
        ),
        "engine_version": "fake-1.0",
        "design_details": {"solver": "fake", "iterations": 4},
    }
    values.update(overrides)
    return HprTargetThermodynamicResult(**values)


def test_request_and_profile_are_frozen_hashable_detached_values() -> None:
    request = _request()
    same_physics = replace(request, candidate_id="diagnostic-only")

    assert request == same_physics
    assert hash(request) == hash(same_physics)
    with pytest.raises(FrozenInstanceError):
        request.useful_duty = 50.0
    with pytest.raises(FrozenInstanceError):
        request.working_fluid = parse_hpr_working_fluid("R32")


@pytest.mark.parametrize(
    "overrides",
    [
        {"mode": "invalid"},
        {"cycle_id": "invalid"},
        {"working_fluid": object()},
        {"candidate_id": ""},
    ],
)
def test_request_rejects_invalid_identity_fields(overrides) -> None:
    with pytest.raises((TypeError, ValueError)):
        _request(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"evaporating_temperature": math.nan},
        {"condensing_temperature": 5.0},
        {"useful_duty": 0.0},
        {"source_approach_temperature": -1.0},
        {"sink_approach_temperature": math.inf},
        {"compressor_isentropic_efficiency": 0.0},
        {"compressor_isentropic_efficiency": 1.01},
        {"superheat": -1.0},
        {"subcooling": -1.0},
        {"internal_hx_gas_temperature_change": -1.0},
        {"model_id": ""},
    ],
)
def test_request_rejects_invalid_physics(overrides) -> None:
    with pytest.raises((TypeError, ValueError)):
        _request(**overrides)


@pytest.mark.parametrize(
    "values",
    [(math.nan, 1.0), (1.0, math.inf)],
)
def test_profile_point_rejects_nonfinite_coordinates(values) -> None:
    with pytest.raises(ValueError, match="finite"):
        HprThermalProfilePoint(temperature=values[0], enthalpy=values[1])


def test_validator_accepts_physical_success() -> None:
    request = _request()
    result = _result(request)

    assert validate_hpr_target_result(request, result) is result


@pytest.mark.parametrize(
    "overrides",
    [
        {"backend": "invalid"},
        {"model_id": ""},
        {"working_fluid": object()},
        {"engine_version": ""},
    ],
)
def test_result_rejects_invalid_identity_fields(overrides) -> None:
    with pytest.raises((TypeError, ValueError)):
        _result(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"backend": "invalid"},
        {"engine_version": ""},
        {"model_id": ""},
    ],
)
def test_metadata_rejects_invalid_identity_fields(overrides) -> None:
    values = {
        "backend": "tespy",
        "engine_version": "1.0",
        "model_id": "model",
        "assumptions": {"nested": {"items": [1, 2]}},
    }
    values.update(overrides)
    with pytest.raises(ValueError):
        HprTargetEvaluatorMetadata(**values)


def test_metadata_freezes_nested_assumptions() -> None:
    metadata = HprTargetEvaluatorMetadata(
        backend="tespy",
        engine_version="1.0",
        model_id="model",
        assumptions={"nested": {"items": [1, 2]}},
    )

    assert metadata.assumptions == (("nested", (("items", (1, 2)),)),)


@pytest.mark.parametrize(
    ("updates", "code"),
    [
        ({"backend": "coolprop"}, "backend_mismatch"),
        ({"model_id": "wrong"}, "model_mismatch"),
        ({"working_fluid": parse_hpr_working_fluid("R32")}, "working_fluid_mismatch"),
        ({"converged": False}, "not_converged"),
        ({"cop": math.nan}, "nonfinite_result"),
        ({"q_source": -1.0}, "invalid_duty"),
        ({"compressor_power": 0.0}, "invalid_power"),
        ({"q_sink": 101.0}, "energy_balance_error"),
        ({"cop": 5.0}, "cop_mismatch"),
        ({"q_sink": 99.0, "q_source": 74.0}, "useful_duty_mismatch"),
        ({"source_profile": ()}, "invalid_source_profile"),
        (
            {
                "source_profile": (
                    HprThermalProfilePoint(temperature=10.0, enthalpy=1.0),
                    HprThermalProfilePoint(temperature=5.0, enthalpy=0.0),
                )
            },
            "invalid_source_profile",
        ),
        (
            {
                "source_profile": (
                    HprThermalProfilePoint(temperature=10.0, enthalpy=0.0),
                    HprThermalProfilePoint(temperature=5.0, enthalpy=70.0),
                )
            },
            "invalid_source_profile",
        ),
        (
            {
                "sink_profile": (
                    HprThermalProfilePoint(temperature=60.0, enthalpy=0.0),
                    HprThermalProfilePoint(temperature=55.0, enthalpy=100.0),
                )
            },
            "invalid_sink_profile",
        ),
    ],
)
def test_validator_translates_corrupt_results_to_local_failures(updates, code) -> None:
    request = _request()
    failure = validate_hpr_target_result(request, _result(request, **updates))

    assert isinstance(failure, HprTargetEvaluationFailure)
    assert failure.code == code
    assert failure.session_fatal is False


def test_local_failure_is_sanitized_and_session_remains_ready() -> None:
    evaluator = FakeHprTargetEvaluator(("local_failure", "success"))
    coordinator = HprTargetEvaluatorCoordinator(evaluator)
    coordinator.open()

    first = coordinator.evaluate(_request(candidate_id="secret-candidate"))
    second = coordinator.evaluate(_request(candidate_id="next", useful_duty=120.0))
    coordinator.close()

    assert isinstance(first, HprTargetEvaluationFailure)
    assert first.code == "candidate_state_failure"
    assert dict(first.details) == {"candidate": "secret-candidate"}
    assert isinstance(second, HprTargetThermodynamicResult)
    assert evaluator.state == "closed"
    assert evaluator.close_calls == 1


@pytest.mark.parametrize("behavior", ["fatal_failure", "restoration_failure"])
def test_fatal_evaluator_failure_aborts_session(behavior: str) -> None:
    evaluator = FakeHprTargetEvaluator((behavior,))
    coordinator = HprTargetEvaluatorCoordinator(evaluator)
    coordinator.open()

    with pytest.raises(HprTargetEvaluatorError) as captured:
        coordinator.evaluate(_request())
    with pytest.raises(RuntimeError, match="fatal"):
        coordinator.evaluate(_request(candidate_id="later"))
    coordinator.close()

    assert captured.value.session_fatal is True
    assert evaluator.close_calls == 1


def test_cleanup_failure_is_fatal_and_close_is_exactly_once() -> None:
    evaluator = FakeHprTargetEvaluator(cleanup_failure=True)
    coordinator = HprTargetEvaluatorCoordinator(evaluator)
    coordinator.open()

    with pytest.raises(HprTargetEvaluatorError) as captured:
        coordinator.close()
    coordinator.close()

    assert captured.value.code == "cleanup_failure"
    assert evaluator.close_calls == 1


def test_validator_rejects_non_result_object() -> None:
    failure = validate_hpr_target_result(_request(), object())

    assert failure.code == "invalid_result_type"


class _ExceptionalEvaluator:
    def __init__(
        self, *, open_value=None, open_error=None, evaluate_error=None, close_error=None
    ):
        self.open_value = open_value
        self.open_error = open_error
        self.evaluate_error = evaluate_error
        self.close_error = close_error

    def open(self):
        if self.open_error:
            raise self.open_error
        return self.open_value

    def evaluate(self, _request):
        if self.evaluate_error:
            raise self.evaluate_error
        return object()

    def close(self):
        if self.close_error:
            raise self.close_error


def test_coordinator_rejects_reopen_and_invalid_metadata() -> None:
    coordinator = HprTargetEvaluatorCoordinator(
        _ExceptionalEvaluator(open_value=object())
    )
    with pytest.raises(HprTargetEvaluatorError, match="invalid metadata"):
        coordinator.open()

    ready = HprTargetEvaluatorCoordinator(FakeHprTargetEvaluator())
    ready.open()
    with pytest.raises(RuntimeError, match="only from created"):
        ready.open()
    ready.close()


@pytest.mark.parametrize(
    "error",
    [
        HprTargetEvaluatorError("open", "classified", session_fatal=True),
        RuntimeError("unexpected"),
    ],
)
def test_coordinator_translates_open_failures(error) -> None:
    coordinator = HprTargetEvaluatorCoordinator(_ExceptionalEvaluator(open_error=error))
    with pytest.raises(HprTargetEvaluatorError):
        coordinator.open()
    assert coordinator.state == "fatal"


def test_coordinator_translates_unexpected_evaluate_and_close_failures() -> None:
    metadata = HprTargetEvaluatorMetadata(
        backend="tespy", engine_version="1", model_id="model", assumptions={}
    )
    evaluating = HprTargetEvaluatorCoordinator(
        _ExceptionalEvaluator(open_value=metadata, evaluate_error=RuntimeError("boom"))
    )
    evaluating.open()
    with pytest.raises(HprTargetEvaluatorError) as captured:
        evaluating.evaluate(_request())
    assert captured.value.code == "unexpected_evaluator_failure"

    closing = HprTargetEvaluatorCoordinator(
        _ExceptionalEvaluator(open_value=metadata, close_error=RuntimeError("boom"))
    )
    closing.open()
    with pytest.raises(HprTargetEvaluatorError) as captured:
        closing.close()
    assert captured.value.code == "cleanup_failure"


@seed(20260715)
@given(
    useful_duty=st.floats(
        min_value=1.0,
        max_value=1e6,
        allow_nan=False,
        allow_infinity=False,
    ),
    cop=st.floats(
        min_value=1.01,
        max_value=15.0,
        allow_nan=False,
        allow_infinity=False,
    ),
)
def test_generated_results_preserve_energy_and_cop(
    useful_duty: float, cop: float
) -> None:
    request = _request(useful_duty=useful_duty)
    power = useful_duty / cop
    result = _result(
        request,
        q_sink=useful_duty,
        q_source=useful_duty - power,
        compressor_power=power,
        cop=cop,
        source_profile=(
            HprThermalProfilePoint(temperature=10.0, enthalpy=0.0),
            HprThermalProfilePoint(
                temperature=5.0,
                enthalpy=useful_duty - power,
            ),
        ),
        sink_profile=(
            HprThermalProfilePoint(temperature=55.0, enthalpy=0.0),
            HprThermalProfilePoint(temperature=60.0, enthalpy=useful_duty),
        ),
    )

    assert validate_hpr_target_result(request, result) is result


@settings(max_examples=20, stateful_step_count=12, derandomize=True)
class HprEvaluatorLifecycleMachine(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.fake = FakeHprTargetEvaluator()
        self.coordinator = HprTargetEvaluatorCoordinator(self.fake)
        self.coordinator.open()
        self.closed = False

    @precondition(lambda self: not self.closed)
    @rule()
    def evaluate_success(self) -> None:
        assert isinstance(
            self.coordinator.evaluate(_request()),
            HprTargetThermodynamicResult,
        )

    @precondition(lambda self: not self.closed)
    @rule()
    def close(self) -> None:
        self.coordinator.close()
        self.closed = True

    @precondition(lambda self: self.closed)
    @rule()
    def closed_is_idempotent(self) -> None:
        self.coordinator.close()

    @invariant()
    def lifecycle_counts_are_bounded(self) -> None:
        assert self.fake.open_calls == 1
        assert self.fake.close_calls <= 1

    def teardown(self) -> None:
        self.coordinator.close()
        assert self.fake.close_calls == 1


TestHprEvaluatorLifecycle = HprEvaluatorLifecycleMachine.TestCase
