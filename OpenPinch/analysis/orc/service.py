"""Public entry point for organic Rankine cycle (ORC) targeting.

The ORC is a heat engine on a zone's process surplus below the pinch: its
evaporators take heat the process would otherwise reject to cold utility, and
its condensers reject what is not turned into power to cooling. It shares no
code or settings with heat pump and refrigeration targeting.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from ...domain.configuration import Configuration, tol
from ...domain.enums import ProblemTableLabel, TargetType
from ...domain.problem_table import ProblemTable
from ...domain.targets import DirectOrcTarget
from ...domain.zone import Zone
from ..economics import compute_capital_recovery_factor
from ..numerics import get_period_index
from ..targeting.grand_composite import (
    get_GCC_without_pockets,
    get_seperated_gcc_heat_load_profiles,
)
from ..targeting.utilities import (
    _apply_utility_duties,
    target_utilities_for_load_profiles,
)
from .carnot import OrcDesign
from .inputs import OrcTargetInputs
from .profile import orc_heat_source_from_gcc
from .search import OrcSearchResult, OrcTargetingError, optimise_carnot_orc

__all__ = [
    "OrcTargetingError",
    "compute_direct_orc_target",
    "orc_target_inputs",
]


def orc_target_inputs(
    config: Configuration,
    args: dict | None = None,
) -> OrcTargetInputs:
    """Collect the ORC settings of one zone's configuration."""
    orc = config.orc
    costing = config.costing
    crf = (
        compute_capital_recovery_factor(costing.discount_rate, costing.service_life)
        if costing.orc_capital_recovery_enabled
        else 0.0
    )
    runtime = args or {}
    return OrcTargetInputs(
        n_stages=int(orc.n_stages),
        eta_ii=float(orc.eta_ii_carnot),
        dt_cont=float(orc.dt_cont),
        T_cond=float(orc.t_cond),
        min_lift=float(orc.min_lift),
        dt_phase_change=float(config.thermal.dt_phase_change),
        load_fraction=float(orc.load_fraction),
        ele_price=float(costing.orc_ele_price),
        cooling_price=float(costing.orc_cooling_price),
        annual_hours=float(costing.annual_op_time),
        equipment_cost=float(costing.orc_equipment_cost),
        installation_factor=float(costing.orc_installation_factor),
        cost_exp=float(costing.orc_cost_exp),
        capital_recovery_factor=float(crf),
        max_multistart=int(orc.max_multistart),
        bb_minimiser=str(orc.bb_minimiser),
        maximum_iterations=int(runtime.get("maximum_iterations", 300)),
    )


def compute_direct_orc_target(
    zone: Zone,
    args: dict | None = None,
) -> DirectOrcTarget | None:
    """Target a Carnot ORC on one zone's direct-integration surplus.

    Returns ``None`` when the zone has no surplus below the pinch. Raises
    ``OrcTargetingError`` when the surplus is too cold or no ORC pays.
    """
    idx, period_id = get_period_index(period_ids=zone.period_ids, args=args)
    base_target = zone.targets[TargetType.DI.value]
    pt = deepcopy(base_target.pt)
    source = orc_heat_source_from_gcc(
        np.asarray(pt[ProblemTableLabel.T], dtype=float),
        np.asarray(pt[ProblemTableLabel.H_NET_A], dtype=float),
        tol=tol,
    )
    if source is None:
        return None
    inputs = orc_target_inputs(zone.config, args)
    result = optimise_carnot_orc(source, inputs)
    utilities = _residual_utility_summary(
        pt=pt,
        base_target=base_target,
        design=result.design,
        dt_phase_change=inputs.dt_phase_change,
        period_idx=idx,
    )
    return DirectOrcTarget.model_validate(
        {
            "zone_name": zone.name,
            "scope": zone.address,
            "zone_type": zone.type,
            "type": TargetType.DORC.value,
            "parent_zone": zone.parent_zone,
            "config": zone.config,
            "pt": pt,
            "period_id": period_id,
            "period_idx": idx,
            **utilities,
            **_orc_summary(result),
        }
    )


def _orc_summary(result: OrcSearchResult) -> dict:
    design, costs = result.design, result.costs
    return {
        "orc_model": "carnot",
        "orc_n_stages": len(design.Q_in),
        "orc_evaporating_temperatures": design.T_evap,
        "orc_condensing_temperature": design.T_cond,
        "orc_stage_heat_in": design.Q_in,
        "orc_stage_net_power": design.W_net,
        "orc_heat_in": design.Q_in_total,
        "orc_condenser_duty": design.Q_out_total,
        "orc_net_power": design.W_net_total,
        "orc_thermal_efficiency": design.eta_thermal,
        "orc_machine_capital_costs": costs.machine_capital_costs,
        "orc_capital_cost": costs.capital_cost,
        "orc_annualized_capital_cost": costs.annualized_capital_cost,
        "orc_power_value": costs.power_value,
        "orc_cooling_cost_change": costs.cooling_cost_change,
        "orc_total_annualized_cost_change": costs.total_annualized_cost_change,
    }


def orc_residual_gcc(
    T_vals: np.ndarray,
    H_net: np.ndarray,
    design: OrcDesign,
    dt_phase_change: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the GCC with the ORC evaporators as sinks, on shifted T.

    Each evaporator takes its heat linearly over ``dt_phase_change`` below
    its shifted temperature, so the cascade falls by that heat at every
    temperature below it.
    """
    T = np.asarray(T_vals, dtype=float)
    H = np.asarray(H_net, dtype=float)
    extra = []
    for T_s in design.T_evap_shifted:
        extra.extend((T_s, T_s - dt_phase_change))
    new_T = np.array(
        [t for t in extra if not np.any(np.abs(T - t) <= tol)], dtype=float
    )
    grid = np.concatenate([T, new_T])
    order = np.argsort(-grid, kind="stable")
    grid = grid[order]
    # Original rows keep their values; inserted rows are interpolated.
    ascending = np.argsort(T, kind="stable")
    values = np.concatenate([H, np.interp(new_T, T[ascending], H[ascending])])[order]
    for T_s, Q in zip(design.T_evap_shifted, design.Q_in, strict=True):
        if dt_phase_change > 0.0:
            share = np.clip((T_s - grid) / dt_phase_change, 0.0, 1.0)
        else:
            share = (grid < T_s).astype(float)
        values = values - Q * share
    return grid, values


def _residual_utility_summary(
    *,
    pt: ProblemTable,
    base_target,
    design: OrcDesign,
    dt_phase_change: float,
    period_idx: int,
) -> dict:
    """Retarget the zone's utilities against the GCC with the ORC in place."""
    T_vals, residual = orc_residual_gcc(
        pt[ProblemTableLabel.T],
        pt[ProblemTableLabel.H_NET_A],
        design,
        dt_phase_change,
    )
    residual = residual - min(float(residual.min()), 0.0)
    residual[np.abs(residual) <= tol] = 0.0
    residual_pt = ProblemTable(
        {ProblemTableLabel.T: T_vals, ProblemTableLabel.H_NET: residual}
    )
    get_GCC_without_pockets(residual_pt)
    T_vals = np.asarray(residual_pt[ProblemTableLabel.T], dtype=float)
    net = np.asarray(residual_pt[ProblemTableLabel.H_NET_NP], dtype=float)
    updates = get_seperated_gcc_heat_load_profiles(
        T_col=T_vals,
        H_net=net,
        is_process_stream=True,
    )["updates"]
    hot_utilities = base_target.hot_utilities.copy(deep=True)
    cold_utilities = base_target.cold_utilities.copy(deep=True)
    _apply_utility_duties(hot_utilities, (0.0,) * len(hot_utilities), idx=period_idx)
    _apply_utility_duties(cold_utilities, (0.0,) * len(cold_utilities), idx=period_idx)
    net_pt = ProblemTable({ProblemTableLabel.T: T_vals, ProblemTableLabel.H_NET: net})
    hot_utilities, cold_utilities = target_utilities_for_load_profiles(
        hot_utilities=hot_utilities,
        cold_utilities=cold_utilities,
        T_vals=T_vals,
        H_net_cold=np.asarray(updates[ProblemTableLabel.H_NET_COLD], dtype=float),
        H_net_hot=np.asarray(updates[ProblemTableLabel.H_NET_HOT], dtype=float),
        pinch_idx=net_pt.pinch_idx(ProblemTableLabel.H_NET),
        is_real_temperatures=False,
        idx=period_idx,
    )
    hot_pinch, cold_pinch = net_pt.pinch_temperatures(col_H=ProblemTableLabel.H_NET)
    utility_cost = sum(
        float(utility.utility_cost[period_idx])
        for utility in hot_utilities + cold_utilities
        if utility.utility_cost is not None
    )
    return {
        "hot_utilities": hot_utilities,
        "cold_utilities": cold_utilities,
        "hot_utility_target": float(
            hot_utilities.sum_stream_attribute("heat_flow", idx=period_idx)
        ),
        "cold_utility_target": float(
            cold_utilities.sum_stream_attribute("heat_flow", idx=period_idx)
        ),
        "heat_recovery_target": float(base_target.heat_recovery_target),
        "heat_recovery_limit": base_target.heat_recovery_limit,
        "degree_of_int": base_target.degree_of_int,
        "utility_cost": utility_cost,
        "hot_pinch": hot_pinch,
        "cold_pinch": cold_pinch,
    }
