"""A shared multiperiod heat pump design reports the machine installed.

The target of a shared design is its selected period's result. Its capital
was that period's own sizing: on the reviewer's three-period vapour
compression case the base period reported 238,470 $ where the machine
installed for the peak costs 374,537 $ (review finding 19). The period's
operation stays its own; its capital, and so its total annualised cost,
are the installed machine's.
"""

from __future__ import annotations

import pytest

import OpenPinch.analysis.heat_pumps._multiperiod.aggregation as hp_aggregation
from OpenPinch.contracts.hpr import (
    HPRBackendResult,
    HPRPeriodCase,
    HPRThermoArtifacts,
    MultiPeriodHPRTargetInputs,
)
from OpenPinch.domain.enums import HeatPumpAndRefrigerationCycle
from OpenPinch.domain.stream_collection import StreamCollection
from OpenPinch.domain.value import Value

from .test_multiperiod_hpr import _input_args


def _period_result(args) -> HPRBackendResult:
    sized = 1.0 + args.period_idx
    return HPRBackendResult(
        obj=1.0,
        utility_tot=10.0,
        w_net=1.0,
        Q_ext_heat=0.0,
        Q_ext_cold=0.0,
        hpr_operating_cost=Value(100.0 * sized, "$/y"),
        hpr_capital_cost=Value(1000.0 * sized, "$"),
        hpr_annualized_capital_cost=Value(200.0 * sized, "$/y"),
        hpr_total_annualized_cost=Value(300.0 * sized, "$/y"),
        Q_amb_hot=0.0,
        Q_amb_cold=0.0,
        cop_h=3.0,
        amb_streams=StreamCollection(),
        artifacts=HPRThermoArtifacts(hpr_streams=StreamCollection()),
    )


def test_a_shared_design_reports_the_installed_machines_capital():
    cases = [
        HPRPeriodCase(
            period_id=period,
            period_idx=idx,
            weight=0.5,
            args=_input_args(period_idx=idx),
        )
        for idx, period in enumerate(("base", "peak"))
    ]
    args = MultiPeriodHPRTargetInputs(
        period_cases=cases,
        selected_period_id="base",
        selected_period_idx=0,
        hpr_type=HeatPumpAndRefrigerationCycle.CascadeCarnot.value,
        max_multi_start=1,
        bb_minimiser="rbf_surrogate",
    )

    result = hp_aggregation.evaluate_multiperiod_candidate(
        [0.5],
        args,
        period_objective=lambda _x, period_args, debug=False: _period_result(
            period_args
        ),
    )

    assert result.hpr_operating_cost.value == pytest.approx(100.0)
    assert result.hpr_capital_cost.value == pytest.approx(2000.0)
    assert result.hpr_annualized_capital_cost.value == pytest.approx(400.0)
    assert result.hpr_total_annualized_cost.value == pytest.approx(500.0)
