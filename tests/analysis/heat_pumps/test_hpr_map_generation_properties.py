"""Property tests for HPR generation context and canonical grids."""

from __future__ import annotations

import math
import time
import tracemalloc

import pytest
from hypothesis import given, seed
from hypothesis import strategies as st

from OpenPinch.analysis.heat_pumps.performance_maps.context import (
    build_hpr_map_generation_context,
)
from OpenPinch.analysis.heat_pumps.performance_maps.errors import (
    HprMapGenerationError,
    HprSimulatorFailure,
)
from OpenPinch.analysis.heat_pumps.performance_maps.fluids import (
    parse_hpr_working_fluid,
)
from OpenPinch.analysis.heat_pumps.performance_maps.generation import (
    generate_hpr_performance_map,
)
from OpenPinch.analysis.heat_pumps.performance_maps.points import (
    iter_hpr_operating_points,
)
from OpenPinch.contracts.hpr_performance_map import HprPerformanceMap
from tests.analysis.heat_pumps.hpr_map_fakes import FakeHprPointSimulator
from tests.strategies.hpr_map_generation import (
    explicit_molar_mixture_text,
    hpr_map_requests,
    hpr_target_map_bases,
)

_TEN_THOUSAND_POINT_TIME_LIMIT_SECONDS = 5.0
_TEN_THOUSAND_POINT_MEMORY_LIMIT_BYTES = 256 * 1024 * 1024


@seed(20260715)
@given(hpr_target_map_bases(), hpr_map_requests())
def test_cartesian_grid_has_exact_size_order_and_unique_identity(basis, request):
    context = build_hpr_map_generation_context(basis, request)

    points = list(iter_hpr_operating_points(context))

    expected_size = (
        len(request.source_temperatures)
        * len(request.sink_temperatures)
        * len(request.load_fractions)
    )
    assert len(points) == expected_size
    assert [point.ordinal for point in points] == list(range(expected_size))
    coordinates = [
        (point.source_temperature, point.sink_temperature, point.load_fraction)
        for point in points
    ]
    assert coordinates == sorted(coordinates)
    assert len({point.name for point in points}) == expected_size
    assert all(
        point.requested_useful_duty
        == pytest.approx(point.load_fraction * context.reference_capacity)
        for point in points
    )


@seed(20260715)
@given(explicit_molar_mixture_text())
def test_explicit_mixture_normalization_is_deterministic_and_preserves_order(data):
    source, components, raw_fractions = data

    first = parse_hpr_working_fluid(source)
    second = parse_hpr_working_fluid(source)

    assert first == second
    assert first.source_spec == source
    assert first.components == components
    assert len(first.mole_fractions) == len(components)
    assert sum(first.mole_fractions) == pytest.approx(1.0)
    expected = tuple(value / sum(raw_fractions) for value in raw_fractions)
    assert first.mole_fractions == pytest.approx(expected)
    assert all(math.isfinite(value) and value >= 0.0 for value in first.mole_fractions)


@seed(20260715)
@given(hpr_target_map_bases(), hpr_map_requests())
def test_fake_generated_map_is_complete_physical_repeatable_and_round_trips(
    basis,
    request,
):
    first = generate_hpr_performance_map(
        basis,
        request,
        simulator_factory=lambda backend: FakeHprPointSimulator(),
    )
    second = generate_hpr_performance_map(
        basis,
        request,
        simulator_factory=lambda backend: FakeHprPointSimulator(),
    )

    assert first == second
    assert len(first.points) == (
        len(request.source_temperatures)
        * len(request.sink_temperatures)
        * len(request.load_fractions)
    )
    assert HprPerformanceMap.model_validate_json(first.model_dump_json()) == first
    for point in first.points:
        assert point.q_sink == pytest.approx(
            point.q_source + point.electric_power,
            abs=first.energy_balance_tolerance,
        )
        useful = point.q_sink if basis.mode == "heat_pump" else point.q_source
        assert point.cop == pytest.approx(useful / point.electric_power)


@seed(20260715)
@given(
    failure_ordinals=st.sets(
        st.integers(min_value=0, max_value=7),
        min_size=1,
    )
)
def test_generated_point_failures_are_ordered_atomic_and_called_once(
    failure_ordinals,
):
    fake = FakeHprPointSimulator(
        point_failures={
            ordinal: HprSimulatorFailure(
                "non_converged",
                "generated local failure",
                session_fatal=False,
            )
            for ordinal in failure_ordinals
        }
    )
    from tests.analysis.heat_pumps.test_hpr_map_generation import _basis, _request

    with pytest.raises(HprMapGenerationError) as raised:
        generate_hpr_performance_map(
            _basis(),
            _request(),
            simulator_factory=lambda backend: fake,
        )

    assert [
        diagnostic.point_ordinal for diagnostic in raised.value.diagnostics
    ] == sorted(failure_ordinals)
    assert [event for event in fake.events if event[0] == "simulate"] == [
        ("simulate", ordinal) for ordinal in range(8)
    ]


def test_ten_thousand_point_fake_generation_stays_within_coarse_unit_budget():
    from tests.analysis.heat_pumps.test_hpr_map_generation import _basis, _request

    fake = FakeHprPointSimulator()
    request = _request(
        source_temperatures=[index / 10.0 for index in range(100)],
        sink_temperatures=[40.0 + index / 10.0 for index in range(100)],
        load_fractions=[1.0],
    )
    tracemalloc.start()
    baseline_current, _ = tracemalloc.get_traced_memory()
    started = time.perf_counter()

    performance_map = generate_hpr_performance_map(
        _basis(),
        request,
        simulator_factory=lambda backend: fake,
    )

    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert len(performance_map.points) == 10_000
    assert len([event for event in fake.events if event[0] == "simulate"]) == 10_000
    assert elapsed < _TEN_THOUSAND_POINT_TIME_LIMIT_SECONDS
    assert peak - baseline_current < _TEN_THOUSAND_POINT_MEMORY_LIMIT_BYTES


@pytest.mark.parametrize("axis_size", [1, 2, 3, 4])
def test_fake_simulator_call_count_scales_with_cartesian_grid(axis_size):
    from tests.analysis.heat_pumps.test_hpr_map_generation import _basis, _request

    fake = FakeHprPointSimulator()
    request = _request(
        source_temperatures=[float(index) for index in range(axis_size)],
        sink_temperatures=[40.0 + index for index in range(axis_size)],
        load_fractions=[(index + 1) / axis_size for index in range(axis_size)],
    )

    performance_map = generate_hpr_performance_map(
        _basis(),
        request,
        simulator_factory=lambda backend: fake,
    )

    expected_count = axis_size**3
    assert len(performance_map.points) == expected_count
    assert len([event for event in fake.events if event[0] == "simulate"]) == (
        expected_count
    )
