"""The retrofit evaluator knows each process stream's contribution.

A heat pump step places its matches at each process stream's own
contribution plus ``HPR_DT_CONT``, so the evaluator maps each stream, by
the name the existing network uses, to its contribution with the zone's
dt_cont multiplier: the levels case's 5 K streams at a multiplier of 2
contribute 10 K.
"""

from __future__ import annotations

import pytest

from OpenPinch.analysis.heat_exchanger_networks.context import ensure_target_results
from OpenPinch.analysis.heat_exchanger_networks.retrofit.search import (
    RetrofitSettings,
    _Evaluator,
)
from tests.analysis.heat_exchanger_networks.heat_pump_levels_case import (
    EXISTING,
    levels_problem,
)


def test_the_evaluator_maps_each_stream_to_its_contribution():
    problem = levels_problem()
    problem.set_dt_cont_multiplier(2.0)
    ensure_target_results(problem, {}, None)
    existing = problem.design._network_from_structure(EXISTING)

    evaluator = _Evaluator(problem, existing, RetrofitSettings(approach=None))

    # Hot and cold streams apart: a zone may name one of each alike.
    assert evaluator.hot_contributions == {"H1": pytest.approx(10.0)}
    assert evaluator.cold_contributions == {"C1": pytest.approx(10.0)}
