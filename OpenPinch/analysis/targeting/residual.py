"""Utility allocation on a frozen HPR residual, without process reconstruction."""

import numpy as np

from ...domain.enums import GraphType, TargetType
from ...domain.enums import ProblemTableLabel as L
from ...domain.problem_table import ProblemTable
from ...domain.targets import ResidualUtilityTarget
from ..heat_pumps.common.postprocessing import (
    _compute_utility_cost,
    _retarget_hpr_residual_utilities,
)
from .cascade import get_utility_heat_cascade


def allocate_residual_utilities(data, hot_utilities, cold_utilities, *, period_idx=0):
    p = data.profile
    return _retarget_hpr_residual_utilities(
        T_vals=np.asarray(p.temperatures),
        residual_net=np.asarray(p.net),
        hot_profile=np.asarray(p.heating),
        cold_profile=-np.asarray(p.cooling),
        hot_utilities=hot_utilities.copy(deep=True),
        cold_utilities=cold_utilities.copy(deep=True),
        is_real_temperatures=p.temperature_basis == "real",
        period_idx=period_idx,
    )


def compute_residual_utility_target(zone, data):
    hot, cold = allocate_residual_utilities(
        data, zone.hot_utilities, zone.cold_utilities
    )
    p = data.profile
    pt = ProblemTable(
        {
            L.T: p.temperatures,
            L.H_NET: p.net,
            L.H_NET_A: p.net,
            L.H_NET_COLD: p.heating,
            L.H_NET_HOT: -np.asarray(p.cooling),
        }
    )
    pt.update(
        **get_utility_heat_cascade(
            T_int_vals=np.asarray(p.temperatures),
            hot_utilities=hot,
            cold_utilities=cold,
            is_shifted=p.temperature_basis == "shifted",
            period_idx=0,
        )
    )
    physical = ProblemTable(
        {
            L.T: data.physical_temperatures,
            L.H_HOT: data.physical_hot_composite,
            L.H_COLD: data.physical_cold_composite,
        }
    )
    hp, cp = pt.pinch_temperatures()
    return ResidualUtilityTarget(
        zone_name=zone.name,
        scope=zone.address,
        zone_type=zone.type,
        type=TargetType.DI.value,
        config=zone.config,
        pt=pt,
        pt_real=physical,
        period_id=data.period_id,
        period_idx=0,
        hot_utilities=hot,
        cold_utilities=cold,
        hot_utility_target=float(hot.sum_stream_attribute("heat_flow", idx=0)),
        cold_utility_target=float(cold.sum_stream_attribute("heat_flow", idx=0)),
        heat_recovery_target=0.0,
        utility_cost=_compute_utility_cost(hot, cold, period_idx=0),
        hot_pinch=hp,
        cold_pinch=cp,
        graphs={GraphType.GCC.value: pt.copy, GraphType.NLP.value: pt.copy},
    )
