"""Optimizer-option and Unit 2 error boundary tests."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from OpenPinch.analysis.utility_placement.errors import (
    NoFeasiblePlacementError,
    PlacementContextError,
    PlacementOptimisationError,
    PlacementTargetingError,
    PlacementThermodynamicError,
    UtilityPlacementError,
)
from OpenPinch.analysis.utility_placement.optimisation import map_optimisation_options
from OpenPinch.contracts.utility_placement import (
    QuantityValue,
    UtilityPlacementOptimisationMethod,
    UtilityPlacementOptions,
)
from OpenPinch.optimisation.models import OptimisationMethod


def test_options_use_cmaes_optimizer_defaults() -> None:
    options = UtilityPlacementOptions()

    assert options.method is UtilityPlacementOptimisationMethod.CMA_ES
    assert options.run_count == 1
    assert options.cluster_tolerance == pytest.approx(0.01)
    assert options.local_method == "SLSQP"
    assert options.backend_options == ()
    assert options.candidate_limit == 5
    assert options.seed == 20260715
    assert options.minimum_separation == QuantityValue(value=1.0, unit="delta_degC")
    assert options.minimum_sensible_span == QuantityValue(value=0.01, unit="delta_degC")
    assert options.default_isothermal_span == QuantityValue(
        value=0.01, unit="delta_degC"
    )


def test_options_round_trip_sorted_json_safe_overrides() -> None:
    options = UtilityPlacementOptions(
        method="bo",
        run_count=3,
        cluster_tolerance=0.2,
        local_method="Powell",
        backend_options=(("xi", 0.1), ("n_init", 8)),
    )

    restored = UtilityPlacementOptions.model_validate_json(options.model_dump_json())

    assert restored == options
    assert restored.backend_options == (("n_init", 8), ("xi", 0.1))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"method": "missing"}, "method"),
        ({"run_count": 0}, "positive"),
        ({"run_count": True}, "integer"),
        ({"cluster_tolerance": -1.0}, "non-negative"),
        ({"cluster_tolerance": math.inf}, "finite"),
        ({"local_method": "  "}, "empty"),
        ({"backend_options": (("", 1),)}, "name"),
        ({"backend_options": (("xi", math.nan),)}, "finite"),
        ({"backend_options": (("xi", 1), ("xi", 2))}, "unique"),
        (
            {"minimum_separation": QuantityValue(value=0.0, unit="delta_degC")},
            "positive",
        ),
        (
            {"minimum_sensible_span": QuantityValue(value=-0.1, unit="delta_degC")},
            "positive",
        ),
        (
            {"default_isothermal_span": QuantityValue(value=0.0, unit="delta_degC")},
            "positive",
        ),
    ],
)
def test_options_reject_invalid_optimizer_values(kwargs, message) -> None:
    with pytest.raises(ValidationError, match=message):
        UtilityPlacementOptions(**kwargs)


def test_map_options_uses_existing_solver_neutral_contract() -> None:
    method, mapped = map_optimisation_options(
        UtilityPlacementOptions(
            method="rbf_surrogate",
            run_count=2,
            iteration_limit=40,
            evaluation_limit=120,
            candidate_limit=3,
            seed=17,
            cluster_tolerance=0.05,
            local_method="SLSQP",
            backend_options=(("n_init", 10),),
        )
    )

    assert method is OptimisationMethod.RBF
    assert mapped.n_runs == 2
    assert mapped.maxiter == 40
    assert mapped.maxfun == 120
    assert mapped.max_minima == 3
    assert mapped.seed == 17
    assert mapped.cluster_tol == pytest.approx(0.05)
    assert mapped.backend_options == (("n_init", 10),)


@pytest.mark.parametrize(
    "error_type",
    [
        PlacementContextError,
        PlacementTargetingError,
        PlacementThermodynamicError,
        PlacementOptimisationError,
        NoFeasiblePlacementError,
    ],
)
def test_unit2_operational_errors_share_specialist_runtime_root(error_type) -> None:
    error = error_type(code="unit2_failure", message="Unit 2 failed.")

    assert isinstance(error, UtilityPlacementError)
    assert not isinstance(error, ValueError)
    assert error.context == {
        "code": "unit2_failure",
        "message": "Unit 2 failed.",
    }
