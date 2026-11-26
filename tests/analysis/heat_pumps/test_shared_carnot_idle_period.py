"""A Carnot machine shared by several periods may be off in some.

A Carnot design that serves no duty is no target. A shared design is
judged period by period, and a machine that runs in one period and is off
in another is a heat pump: it is "no heat pump" only when every period is
idle (``_multiperiod/README.md``).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from OpenPinch.analysis.heat_pumps._multiperiod.execution import (
    reject_penalised_periods,
)
from OpenPinch.contracts.hpr import HPRBackendResult

from .helpers import _base_args


def _period(condenser: float, evaporator: float) -> HPRBackendResult:
    return HPRBackendResult(
        obj=0.5,
        utility_tot=0.5,
        w_net=condenser - evaporator,
        Q_ext_heat=0.0,
        Q_ext_cold=0.0,
        Q_amb_hot=0.0,
        Q_amb_cold=0.0,
        Q_cond=np.array([condenser]),
        Q_evap=np.array([evaporator]),
    )


def _shared(**periods):
    args = SimpleNamespace(
        period_cases=[
            SimpleNamespace(period_id=period, args=_base_args()) for period in periods
        ]
    )
    return _period(150.0, 90.0).with_updates(period_outputs=periods), args


def test_a_shared_carnot_machine_off_in_one_period_is_still_a_target():
    shared, args = _shared(base=_period(150.0, 90.0), off=_period(0.0, 0.0))

    assert reject_penalised_periods(shared, args) is shared
