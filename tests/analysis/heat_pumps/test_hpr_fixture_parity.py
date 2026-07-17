"""Exact numerical parity checks captured before retiring the old HPR paths."""

from __future__ import annotations

import json

import numpy as np
import pytest

import OpenPinch.analysis.heat_pumps.optimisation_adapter as adapter
from OpenPinch.contracts.hpr import HPRBackendResult, HPRThermoArtifacts
from OpenPinch.domain.stream_collection import StreamCollection
from OpenPinch.domain.value import Value
from OpenPinch.optimisation.models import OptimisationCandidate
from tests.support.paths import FIXTURES_ROOT

from .helpers import _base_args

_FIXTURE_PATH = FIXTURES_ROOT / "hpr_optimisation_parity.json"


@pytest.fixture(scope="module")
def parity_cases() -> dict:
    return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))


def _result_from_fixture(values: dict) -> HPRBackendResult:
    return HPRBackendResult(
        obj=values["objective"],
        utility_tot=values["utility_total"],
        w_net=values["utility_total"] / 10.0,
        Q_ext_heat=0.0,
        Q_ext_cold=0.0,
        hpr_operating_cost=Value(values["operating_cost"], "$/y"),
        hpr_capital_cost=Value(values["capital_cost"], "$"),
        hpr_annualized_capital_cost=Value(
            values["annualized_capital_cost"],
            "$/y",
        ),
        feasibility_penalty=values["feasibility_penalty"],
        Q_amb_hot=0.0,
        Q_amb_cold=0.0,
        artifacts=HPRThermoArtifacts(hpr_streams=StreamCollection()),
    )


def test_single_period_candidate_selection_matches_pre_move_fixture(
    monkeypatch,
    parity_cases,
):
    case = parity_cases["single_period"]
    monkeypatch.setattr(
        adapter,
        "run_hpr_candidate_search",
        lambda **_kwargs: tuple(
            OptimisationCandidate(objective=value, point=(value,))
            for value in sorted(case["candidate_objectives"])
        ),
    )
    result = adapter.solve_hpr_placement(
        f_obj=lambda point, _args, debug=False: HPRBackendResult(
            obj=float(point[0]),
            utility_tot=float(point[0]),
            w_net=float(point[0]),
            Q_ext_heat=0.0,
            Q_ext_cold=0.0,
            Q_amb_hot=0.0,
            Q_amb_cold=0.0,
            artifacts=HPRThermoArtifacts(hpr_streams=StreamCollection()),
        ),
        x0_ls=None,
        bnds=[(0.0, 1.0)],
        args=_base_args(),
    )

    assert result.obj == case["selected_objective"]


def test_multiperiod_cost_policy_matches_pre_move_fixture(parity_cases):
    case = parity_cases["multiperiod"]
    weighted, objective = adapter.aggregate_hpr_period_results(
        {
            period_id: _result_from_fixture(values)
            for period_id, values in case["periods"].items()
        },
        np.asarray(case["weights"], dtype=float),
    )
    expected = case["expected"]

    assert objective == expected["objective"]
    assert weighted.utility_tot == expected["utility_total"]
    assert weighted.hpr_operating_cost.value == expected["operating_cost"]
    assert weighted.hpr_capital_cost.value == expected["capital_cost"]
    assert (
        weighted.hpr_annualized_capital_cost.value
        == expected["annualized_capital_cost"]
    )
    assert weighted.hpr_total_annualized_cost.value == expected["annualized_total_cost"]
