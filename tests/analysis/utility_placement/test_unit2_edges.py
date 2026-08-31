"""Focused validation and operational failure coverage for Unit 2."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from OpenPinch.analysis.utility_placement.allocation import (
    AllocationAdapterResult,
    ExistingTargetingAllocationAdapter,
    allocate_placement_period,
)
from OpenPinch.analysis.utility_placement.context import (
    PlacementTargetSnapshot,
    ProcessEntropySlice,
    build_utility_placement_context,
)
from OpenPinch.analysis.utility_placement.errors import (
    PlacementOptimisationError,
    PlacementTargetingError,
    PlacementThermodynamicError,
)
from OpenPinch.analysis.utility_placement.evaluation import PlacementEvaluationSession
from OpenPinch.analysis.utility_placement.normalization import (
    prepare_template_blueprints,
)
from OpenPinch.analysis.utility_placement.optimisation import coordinate_optimisation
from OpenPinch.analysis.utility_placement.penalties import (
    aggregate_weighted_objective,
    feasible_objective_scalar,
)
from OpenPinch.analysis.utility_placement.thermodynamics import (
    stream_entropy_change,
)
from OpenPinch.contracts.utility_placement import (
    UtilityPlacementBaseTarget,
    UtilityPlacementRequest,
    UtilitySide,
)
from tests.analysis.utility_placement.test_allocation import _period, _placement
from tests.analysis.utility_placement.test_evaluation import _Adapter, _case


@pytest.mark.parametrize(
    "kwargs",
    [
        {"interval_index": True},
        {"interval_index": -1},
        {"temperature_in_kelvin": 0.0},
        {"temperature_out_kelvin": 0.0},
        {"available_duty": -1.0},
        {"heat_capacity_flow": -1.0},
    ],
)
def test_entropy_slice_validation_edges(kwargs) -> None:
    values = {
        "interval_index": 0,
        "side": UtilitySide.HOT,
        "temperature_in_kelvin": 300.0,
        "temperature_out_kelvin": 400.0,
        "available_duty": 100.0,
        "heat_capacity_flow": 1.0,
        **kwargs,
    }
    with pytest.raises(ValidationError):
        ProcessEntropySlice(**values)


@pytest.mark.parametrize(
    "update",
    [
        {"real_temperatures": (1.0,)},
        {"hot_pinch_index": 3},
        {"cold_pinch_index": -1},
        {"hot_load_profile": (1.0, -1.0)},
    ],
)
def test_target_snapshot_validation_edges(update) -> None:
    values = {
        "shifted_temperatures": (2.0, 1.0),
        "real_temperatures": (2.0, 1.0),
        "hot_load_profile": (1.0, 0.0),
        "cold_load_profile": (0.0, 1.0),
        "real_hot_composite": (1.0, 0.0),
        "real_cold_composite": (1.0, 0.0),
        "hot_pinch_index": 1,
        "cold_pinch_index": 0,
        "entropy_slices": (),
        **update,
    }
    with pytest.raises(ValidationError):
        PlacementTargetSnapshot(**values)


def test_context_rejects_empty_target_periods_duplicates_and_zero_weights() -> None:
    request, context, _ = _case()
    blueprints = prepare_template_blueprints(request)
    period = context.periods[0]
    calls = [
        {"base_target_id": "", "periods": (period,)},
        {"base_target_id": "x", "periods": ()},
        {"base_target_id": "x", "periods": (period, period)},
        {
            "base_target_id": "x",
            "periods": (period.model_copy(update={"weight": 0.0}),),
        },
    ]
    for values in calls:
        with pytest.raises(Exception):
            build_utility_placement_context(
                request=request,
                blueprints=blueprints,
                scope=UtilityPlacementBaseTarget.DIRECT,
                **values,
            )


def test_default_targeting_adapter_and_operational_failures() -> None:
    raw = ExistingTargetingAllocationAdapter().allocate(_period(), _placement())
    assert len(raw.hot_duties) == 2
    assert len(raw.cold_duties) == 2

    class Broken:
        def allocate(self, period, placement):
            raise RuntimeError("boom")

    with pytest.raises(PlacementTargetingError):
        allocate_placement_period(
            request=UtilityPlacementRequest(isothermal_level_count=2),
            period=_period(),
            placement=_placement(),
            adapter=Broken(),
        )
    with pytest.raises(ValidationError):
        AllocationAdapterResult(hot_duties=(-1.0,), cold_duties=())


def test_penalty_thermodynamic_and_optimizer_failures() -> None:
    for args in [((1.0,), ()), ((1.0,), (-1.0,)), ((math.inf,), (1.0,))]:
        with pytest.raises(ValueError):
            aggregate_weighted_objective(*args)
    with pytest.raises(ValueError):
        feasible_objective_scalar(1.0, scale=0.0)
    with pytest.raises(PlacementThermodynamicError):
        stream_entropy_change(-1.0, 300.0, 400.0)
    with pytest.raises(PlacementThermodynamicError):
        stream_entropy_change(math.inf, 300.0, 400.0)

    request, context, model = _case()
    session = PlacementEvaluationSession(
        request=request,
        context=context,
        model=model,
        allocation_adapter=_Adapter(),
    )

    class BrokenRunner:
        def __call__(self, problem, *, method, options):
            raise RuntimeError("boom")

    with pytest.raises(PlacementOptimisationError):
        coordinate_optimisation(session=session, runner=BrokenRunner())


def test_evaluation_budget_is_bounded() -> None:
    request, context, model = _case()
    request = request.model_copy(
        update={"options": request.options.model_copy(update={"evaluation_limit": 1})}
    )
    session = PlacementEvaluationSession(
        request=request,
        context=context,
        model=model,
        allocation_adapter=_Adapter(),
    )
    session.evaluate(model.initial_points[0])
    second = session.evaluate(
        tuple(coordinate.bounds.lower for coordinate in model.coordinates)
    )
    assert second.diagnostics[0].code == "evaluation_budget_exhausted"
