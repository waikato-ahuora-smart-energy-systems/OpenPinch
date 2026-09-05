"""Exact bounded call-local cache contracts for TESPy target candidates."""

from __future__ import annotations

import gc
import tracemalloc
import weakref
from dataclasses import replace

import pytest
from hypothesis import given, seed
from hypothesis import strategies as st

from OpenPinch.analysis.heat_pumps.performance_maps.fluids import (
    parse_hpr_working_fluid,
)
from OpenPinch.analysis.heat_pumps.performance_maps.targeting import (
    HprTargetEvaluatorCoordinator,
)
from OpenPinch.analysis.heat_pumps.performance_maps.targeting_models import (
    HprTargetEvaluationFailure,
    HprTargetEvaluatorError,
    HprTargetThermodynamicRequest,
    HprTargetThermodynamicResult,
    HprThermalProfilePoint,
)
from tests.analysis.heat_pumps.hpr_targeting_fakes import FakeHprTargetEvaluator
from tests.analysis.heat_pumps.test_hpr_target_evaluator_contracts import _request

_TARGET_CACHE_MEMORY_LIMIT_BYTES = 64 * 1024 * 1024
_MAXIMUM_FAKE_PROFILE_POINTS = 20


class _MaximumSizeFakeHprTargetEvaluator(FakeHprTargetEvaluator):
    """Return the largest bounded detail/profile shape used by target tests."""

    def evaluate(
        self,
        request: HprTargetThermodynamicRequest,
    ) -> HprTargetThermodynamicResult:
        base = super().evaluate(request)
        points = _MAXIMUM_FAKE_PROFILE_POINTS
        source_profile = tuple(
            HprThermalProfilePoint(
                temperature=request.evaporating_temperature + 5.0 * (1.0 - i / 19),
                enthalpy=base.q_source * i / 19,
            )
            for i in range(points)
        )
        sink_profile = tuple(
            HprThermalProfilePoint(
                temperature=request.condensing_temperature + 5.0 * i / 19,
                enthalpy=base.q_sink * i / 19,
            )
            for i in range(points)
        )
        maximum_details = {
            f"detail-{outer}": [
                f"{request.candidate_id}-{outer}-{inner}-" + "x" * 160
                for inner in range(20)
            ]
            for outer in range(20)
        }
        return replace(
            base,
            source_profile=source_profile,
            sink_profile=sink_profile,
            design_details=maximum_details,
        )


def _opened(behaviors=("success",), *, cleanup_failure=False):
    fake = FakeHprTargetEvaluator(behaviors, cleanup_failure=cleanup_failure)
    coordinator = HprTargetEvaluatorCoordinator(fake)
    coordinator.open()
    return fake, coordinator


def test_exact_duplicate_ignores_only_diagnostic_candidate_id() -> None:
    fake, coordinator = _opened()
    first = coordinator.evaluate(_request(candidate_id="first"))
    second = coordinator.evaluate(_request(candidate_id="second"))

    assert second == first
    assert fake.evaluate_calls == 1
    assert coordinator.cache_stats.callbacks == 2
    assert coordinator.cache_stats.hits == 1
    assert coordinator.cache_stats.misses == 1
    assert coordinator.cache_stats.solves == 1
    assert coordinator.cache_stats.insertions == 1
    assert coordinator.cache_stats.evictions == 0
    coordinator.close()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mode", "refrigeration"),
        ("model_id", "openpinch-tespy-single-stage-v2"),
        ("working_fluid", parse_hpr_working_fluid("R32")),
        ("evaporating_temperature", 6.0),
        ("condensing_temperature", 56.0),
        ("useful_duty", 101.0),
        ("source_approach_temperature", 6.0),
        ("sink_approach_temperature", 6.0),
        ("compressor_isentropic_efficiency", 0.8),
        ("superheat", 6.0),
        ("subcooling", 3.0),
        ("internal_hx_gas_temperature_change", 1.0),
    ],
)
def test_every_variable_physical_field_changes_the_exact_key(field, value) -> None:
    fake, coordinator = _opened()
    base = _request()

    coordinator.evaluate(base)
    coordinator.evaluate(replace(base, **{field: value}, candidate_id="changed"))

    assert fake.evaluate_calls == 2
    assert coordinator.cache_stats.misses == 2
    assert coordinator.cache_stats.hits == 0
    coordinator.close()


def test_candidate_local_failure_is_cached_but_fatal_failure_is_not() -> None:
    local_fake, local = _opened(("local_failure", "success"))
    first = local.evaluate(_request())
    second = local.evaluate(_request(candidate_id="duplicate"))

    assert isinstance(first, HprTargetEvaluationFailure)
    assert second == first
    assert local_fake.evaluate_calls == 1
    assert local.cache_size == 1
    local.close()

    fatal_fake, fatal = _opened(("fatal_failure",))
    with pytest.raises(HprTargetEvaluatorError):
        fatal.evaluate(_request())
    assert fatal_fake.evaluate_calls == 1
    assert fatal.cache_size == 0
    assert fatal.cache_stats.insertions == 0
    fatal.close()


def test_lru_moves_hits_and_evicts_exactly_at_513th_unique_key() -> None:
    fake, coordinator = _opened()
    requests = [
        _request(useful_duty=float(index + 1), candidate_id=str(index))
        for index in range(513)
    ]
    for request in requests[:512]:
        coordinator.evaluate(request)

    coordinator.evaluate(requests[0])
    coordinator.evaluate(requests[512])

    assert coordinator.cache_size == 512
    assert coordinator.cache_stats.evictions == 1
    assert requests[0] in coordinator.cached_requests
    assert requests[1] not in coordinator.cached_requests

    before = fake.evaluate_calls
    coordinator.evaluate(requests[1])
    assert fake.evaluate_calls == before + 1
    assert coordinator.cache_stats.evictions == 2
    coordinator.close()


def test_cache_is_call_local_and_cleared_on_close() -> None:
    first_fake, first = _opened()
    second_fake, second = _opened()
    request = _request()

    first.evaluate(request)
    second.evaluate(request)

    assert first_fake.evaluate_calls == second_fake.evaluate_calls == 1
    assert first.cache_size == second.cache_size == 1
    first.close()
    second.close()
    assert first.cache_size == second.cache_size == 0


@pytest.mark.parametrize("callbacks", [1, 8, 64, 512])
def test_callback_and_solve_counts_scale_linearly_with_resident_unique_keys(
    callbacks: int,
) -> None:
    fake, coordinator = _opened()
    unique_count = max(1, callbacks // 2)

    for ordinal in range(callbacks):
        coordinator.evaluate(
            _request(
                useful_duty=float(ordinal % unique_count + 1),
                candidate_id=str(ordinal),
            )
        )

    stats = coordinator.cache_stats
    assert stats.callbacks == callbacks
    assert stats.solves == unique_count
    assert stats.solves <= len(coordinator.cached_requests) <= callbacks
    assert fake.evaluate_calls == stats.solves
    coordinator.close()


def test_512_maximum_size_fake_results_stay_below_python_cache_memory_limit() -> None:
    fake = _MaximumSizeFakeHprTargetEvaluator()
    coordinator = HprTargetEvaluatorCoordinator(fake)
    coordinator.open()
    tracemalloc.start()
    baseline_current, _ = tracemalloc.get_traced_memory()

    for ordinal in range(512):
        coordinator.evaluate(
            _request(
                useful_duty=float(ordinal + 1),
                candidate_id=f"maximum-{ordinal}",
            )
        )

    _current, peak = tracemalloc.get_traced_memory()
    traced_growth = peak - baseline_current
    tracemalloc.stop()
    assert coordinator.cache_size == 512
    assert fake.evaluate_calls == 512
    assert traced_growth < _TARGET_CACHE_MEMORY_LIMIT_BYTES
    coordinator.close()


def test_ten_fake_calls_release_coordinators_evaluators_and_cache_entries() -> None:
    coordinator_refs = []
    evaluator_refs = []

    for ordinal in range(10):
        fake, coordinator = _opened()
        coordinator.evaluate(_request(candidate_id=f"call-{ordinal}"))
        coordinator.close()
        assert coordinator.state == "closed"
        assert coordinator.cache_size == 0
        assert fake.state == "closed"
        coordinator_refs.append(weakref.ref(coordinator))
        evaluator_refs.append(weakref.ref(fake))
        del coordinator, fake

    gc.collect()
    assert all(reference() is None for reference in coordinator_refs)
    assert all(reference() is None for reference in evaluator_refs)


def test_candidate_order_does_not_change_physical_results() -> None:
    requests = [
        _request(useful_duty=value, candidate_id=str(value)) for value in (80.0, 120.0)
    ]
    first_fake, first = _opened()
    second_fake, second = _opened()

    forward = {request.useful_duty: first.evaluate(request) for request in requests}
    reverse = {
        request.useful_duty: second.evaluate(request) for request in reversed(requests)
    }

    assert forward == reverse
    assert first_fake.evaluate_calls == second_fake.evaluate_calls == 2
    first.close()
    second.close()


@seed(20260715)
@given(
    duties=st.lists(
        st.integers(min_value=1, max_value=500),
        min_size=1,
        max_size=40,
        unique=True,
    )
)
def test_generated_candidate_order_does_not_change_physical_results(duties) -> None:
    requests = [
        _request(useful_duty=float(duty), candidate_id=str(ordinal))
        for ordinal, duty in enumerate(duties)
    ]
    first_fake, first = _opened()
    second_fake, second = _opened()

    forward = {request.useful_duty: first.evaluate(request) for request in requests}
    reverse = {
        request.useful_duty: second.evaluate(request) for request in reversed(requests)
    }

    assert forward == reverse
    assert first_fake.evaluate_calls == second_fake.evaluate_calls == len(requests)
    first.close()
    second.close()


@seed(20260715)
@given(
    duties=st.lists(
        st.integers(min_value=1, max_value=50),
        min_size=1,
        max_size=100,
    )
)
def test_generated_duplicate_sequences_solve_once_per_exact_key(duties) -> None:
    fake, coordinator = _opened()
    for ordinal, duty in enumerate(duties):
        coordinator.evaluate(
            _request(useful_duty=float(duty), candidate_id=str(ordinal))
        )

    assert fake.evaluate_calls == len(set(duties))
    assert coordinator.cache_stats.callbacks == len(duties)
    assert coordinator.cache_stats.hits == len(duties) - len(set(duties))
    assert coordinator.cache_stats.misses == len(set(duties))
    coordinator.close()
