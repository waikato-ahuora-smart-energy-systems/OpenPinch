"""The non-isothermal stage-wise model keeps the approach inside a match.

The isothermal model checks each solution at the segment boundaries inside
its matches and cuts the pairs that cross. The non-isothermal one, whose
matches each have their own branch outlet temperatures, held the approach
at the match's ends alone, so a match on a condensing stream could cross
at the kink. On the condensing stream V (desuperheating, then condensing
at 174.75 degC) matched in one stage against C (160 to 190 degC), each
segment pair the match spans must keep the pair's 2 K at both ends.
"""

from __future__ import annotations

import pytest

from OpenPinch.analysis.heat_exchanger_networks.models.stagewise import (
    StageWiseModel,
)
from OpenPinch.analysis.heat_exchanger_networks.solver.arrays import (
    problem_to_solver_arrays,
)
from tests.analysis.heat_exchanger_networks.test_stagewise_nlp_interior_approach import (
    _couenne_available,
    _kink_problem,
)


@pytest.mark.solver
@pytest.mark.skipif(not _couenne_available(), reason="needs couenne")
def test_a_non_isothermal_model_holds_the_approach_at_the_kink():
    model = StageWiseModel(
        name="kink-non-isothermal",
        framework="ESM",
        solver="couenne",
        solver_arrays=problem_to_solver_arrays(_kink_problem(), 2.0),
        stages=1,
        dTmin=2.0,
        z_restriction=None,
        min_dqda=0.0,
        minimisation_goal="hot utility",
        non_isothermal_model=True,
        integers=False,
        tol=1e-6,
    )

    model.optimise(False)

    slices = model.segment_area_contributions_by_period[0][0][0][0]
    assert sum(item.duty for item in slices) > 0.0
    for item in slices:
        assert item.hot_inlet_temperature - item.cold_outlet_temperature >= 2.0 - 1e-3
        assert item.hot_outlet_temperature - item.cold_inlet_temperature >= 2.0 - 1e-3
