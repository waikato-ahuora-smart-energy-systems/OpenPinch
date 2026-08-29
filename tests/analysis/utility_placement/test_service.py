"""End-to-end detached utility-placement service result tests."""

from __future__ import annotations

from OpenPinch.analysis.utility_placement.normalization import (
    prepare_template_blueprints,
)
from OpenPinch.analysis.utility_placement.service import optimise_utility_placement
from tests.analysis.utility_placement.test_evaluation import _Adapter, _case
from tests.analysis.utility_placement.test_optimisation import _Runner


def test_service_returns_detached_default_thermodynamic_result() -> None:
    request, context, _ = _case()
    result = optimise_utility_placement(
        request=request,
        blueprints=prepare_template_blueprints(request),
        context=context,
        allocation_adapter=_Adapter(),
        runner=_Runner(),
    )

    assert result.scope.value == "direct"
    assert result.period_ids == ("p1",)
    assert result.period_weights == (2.0,)
    assert result.best.feasible
    assert result.best.thermodynamic_total is not None
    assert len(result.alternatives) == request.options.candidate_limit - 1
    assert all(candidate.feasible for candidate in result.alternatives)
    candidates = (result.best, *result.alternatives)
    assert len({candidate.coordinates for candidate in candidates}) == len(candidates)
    assert result == result.model_validate_json(result.model_dump_json())
