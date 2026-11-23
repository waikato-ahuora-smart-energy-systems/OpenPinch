"""The Carnot heat pump lift limit must be a usable number.

A NaN compares false with every lift and an infinite limit is above every
lift, so either would switch the limit off without saying so; only ``None``
is documented to. A negative limit would leave no heat pump to offer.
"""

from __future__ import annotations

import pytest

from tests.analysis.heat_exchanger_networks.test_retrofit import (
    _case_problem,
    _existing_network,
)


@pytest.mark.parametrize("lift", [-1.0, float("nan"), float("inf")])
def test_retrofit_search_rejects_a_lift_limit_that_is_not_finite_and_non_negative(
    lift,
):
    from OpenPinch.analysis.heat_exchanger_networks.retrofit.search import (
        RetrofitSettings,
        run_retrofit_search,
    )

    with pytest.raises(ValueError, match="heat_pump_max_lift"):
        run_retrofit_search(
            _case_problem(),
            _existing_network(),
            RetrofitSettings(heat_pump_cycle="carnot", heat_pump_max_lift=lift),
        )
