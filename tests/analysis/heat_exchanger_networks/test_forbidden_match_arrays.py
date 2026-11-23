"""Forbidden matches reach every model through the solver arrays.

``HENS_FORBIDDEN_MATCHES`` names pairs; the arrays hold them as masks by
index, so the pinch sides, the stage-wise model and its moves read one
answer. A problem with none carries no mask.
"""

from __future__ import annotations

import pytest

from OpenPinch.analysis.heat_exchanger_networks.solver.arrays import (
    problem_to_solver_arrays,
)
from tests.analysis.heat_exchanger_networks.test_staged_utilities import (
    D_TMIN,
    _heating_problem,
)

MASKS = ("forbidden_recovery", "forbidden_hu", "forbidden_cu")


def _arrays(pairs):
    problem = _heating_problem()
    problem.update_options({"HENS_FORBIDDEN_MATCHES": pairs})
    return problem_to_solver_arrays(problem, D_TMIN)


def test_named_pairs_become_masks_over_streams_and_utilities():
    # Hot streams H1, H2; cold stream C1; hot utilities LPS, HPS; cold CW.
    prepared = _arrays([["H1", "C1"], ["Hot Utility.HPS", "Plant.C1"], ["H2", "CW"]])
    axes = prepared.axis_maps
    hot = {
        name.rsplit(".", 1)[-1]: i for name, i in axes["hot_process_streams"].items()
    }
    hot_utility = {
        name.rsplit(".", 1)[-1]: u for name, u in axes["hot_utilities"].items()
    }

    recovery = prepared.arrays["forbidden_recovery"]
    assert recovery.shape == (2, 1)
    assert recovery[hot["H1"]][0] == 1 and recovery[hot["H2"]][0] == 0
    # The utility masks run utility first (hot utility by cold stream, cold
    # utility by hot stream), which lets a sliced utility's slices take its
    # row (test_forbidden_match_slices.py); they were stream first.
    heaters = prepared.arrays["forbidden_hu"]
    assert heaters.shape == (2, 1)
    assert heaters[hot_utility["HPS"]][0] == 1 and heaters[hot_utility["LPS"]][0] == 0
    coolers = prepared.arrays["forbidden_cu"]
    assert coolers.shape == (1, 2)
    assert coolers[0][hot["H2"]] == 1 and coolers[0][hot["H1"]] == 0

    assert not set(MASKS) & set(_arrays([]).arrays)


@pytest.mark.parametrize(
    ("pair", "message"),
    [
        (["H9", "C1"], "H9"),
        (["C1", "H1"], "C1"),
        (["HPS", "CW"], "two utilities"),
    ],
)
def test_a_pair_that_names_no_match_is_an_error(pair, message):
    with pytest.raises(ValueError, match=message):
        _arrays([pair])
