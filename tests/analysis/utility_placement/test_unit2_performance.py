"""Approved tiered performance gates for utility-placement evaluation."""

from __future__ import annotations

import time

from OpenPinch.analysis.utility_placement.evaluation import PlacementEvaluationSession
from OpenPinch.analysis.utility_placement.thermodynamics import stream_entropy_change
from tests.analysis.utility_placement.test_evaluation import _Adapter, _case


def _p95(samples: list[float]) -> float:
    return sorted(samples)[int(0.95 * (len(samples) - 1))]


def test_pure_entropy_kernel_batch_p95_is_at_most_50_ms() -> None:
    def run_batch():
        for period in range(100):
            for level in range(40):
                stream_entropy_change(
                    10.0 + level,
                    300.0 + period + level,
                    300.01 + period + level,
                )

    for _ in range(3):
        run_batch()
    samples = []
    for _ in range(10):
        started = time.perf_counter()
        run_batch()
        samples.append(time.perf_counter() - started)
    assert _p95(samples) <= 0.050


def test_cold_replay_and_memo_hit_performance_gates() -> None:
    request, context, model = _case()
    cold_samples = []
    for _ in range(3):
        PlacementEvaluationSession(
            request=request,
            context=context,
            model=model,
            allocation_adapter=_Adapter(),
        ).evaluate(model.initial_points[0])
    for _ in range(10):
        session = PlacementEvaluationSession(
            request=request,
            context=context,
            model=model,
            allocation_adapter=_Adapter(),
        )
        started = time.perf_counter()
        session.evaluate(model.initial_points[0])
        cold_samples.append(time.perf_counter() - started)
    assert _p95(cold_samples) <= 1.0

    session = PlacementEvaluationSession(
        request=request,
        context=context,
        model=model,
        allocation_adapter=_Adapter(),
    )
    session.evaluate(model.initial_points[0])
    memo_samples = []
    for _ in range(10):
        started = time.perf_counter()
        session.evaluate(model.initial_points[0])
        memo_samples.append(time.perf_counter() - started)
    assert _p95(memo_samples) <= 0.001
