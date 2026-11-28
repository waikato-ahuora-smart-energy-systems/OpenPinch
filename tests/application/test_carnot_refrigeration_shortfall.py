"""A Carnot refrigeration target says when it serves part of its duty.

A Carnot target serves what its model can reach: on the basic pinch case,
whose cooling is all above ambient, 487 of the 1,000 kW asked for at a full
search. It reported success with no word of the rest.
"""

from __future__ import annotations

import json

import pytest

from OpenPinch import PinchProblem
from OpenPinch.resources import read_sample_case


def test_a_carnot_refrigeration_target_warns_of_cooling_it_does_not_serve():
    problem = PinchProblem(
        json.loads(read_sample_case("basic_pinch.json")), project_name="basic"
    )

    # One evaluation, which serves 889.1 kW.
    with pytest.warns(UserWarning, match=r"serves 889\.1 of the 1000 kW"):
        target = problem.target.carnot_refrigeration(
            load_fraction=1.0, maximum_restarts=0, maximum_evaluations=1
        )

    assert target.hpr_load.achieved == pytest.approx(889.09, abs=0.01)
