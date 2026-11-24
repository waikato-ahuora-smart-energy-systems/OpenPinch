"""Pinch design keeps the approach inside a match on a segmented stream.

A segmented stream stays one stream in design, so a match along it must
hold the minimum approach at every segment boundary it spans, not only at
its two ends. Both pinch design methods take their duties from one side
model; on the kink case it sized a match across the condensing stream's
kink by its ends alone, and the network then failed when its areas were
taken ("Segmented exchanger slice has a non-positive terminal temperature
difference"), so neither method realised a network.
"""

from __future__ import annotations

import pytest

from OpenPinch.analysis.heat_exchanger_networks.solver import backend
from OpenPinch.analysis.heat_exchanger_networks.solver.dependencies import (
    MissingSynthesisDependencyError,
    MissingSynthesisSolverError,
)
from tests.analysis.heat_exchanger_networks.kink_case import kink_problem


def _highs_available() -> bool:
    try:
        backend.require_solver_backend("highs")
    except (MissingSynthesisDependencyError, MissingSynthesisSolverError):
        return False
    return True


@pytest.mark.synthesis
@pytest.mark.skipif("not _highs_available()", reason="needs highspy")
def test_pinch_design_designs_a_stream_that_kinks_inside_a_match():
    view = kink_problem({"HENS_DT_CONT_MULTIPLIERS": [1.0]}).design.pinch_design()

    assert [outcome.network for outcome in view.result.ranked_networks]
