"""Indirect heat-integration entry point for total site style targeting.

The routines in this module aggregate process-level direct-integration outputs
from subzones, construct site process/utility cascades, and calculate net
utility balances after feasible inter-zone heat recovery.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Tuple

import numpy as np

from ...domain._problem_table.types import ProblemTableUpdateKwargs
from ...domain.enums import GraphType, ProblemTableLabel, TargetType
from ...domain.problem_table import ProblemTable
from ...domain.stream_collection import StreamCollection
from ...domain.targets import IndirectIntegrationTarget, SubzoneAggregateTarget
from ...domain.zone import Zone
from ..numerics import get_period_index
from .cascade import get_process_heat_cascade
from .direct import _create_net_hot_and_cold_stream_collections_for_site_analysis

__all__ = [
    "compute_subzone_aggregate_target",
    "compute_indirect_integration_targets",
]

_DIRECT_TARGET_GRAPH_DECIMALS = 4


################################################################################
# Public API
################################################################################


def compute_subzone_aggregate_target(
    zone: Zone,
    args: dict | None = None,
) -> SubzoneAggregateTarget:
    """Sums and records zonal targets."""
    # Sum targets from subzones
    idx, sid = get_period_index(period_ids=zone.period_ids, args=args)
    hot_utility_target = cold_utility_target = heat_recovery_target = 0.0
    utility_cost = 0.0

    hot_utilities = deepcopy(zone.hot_utilities).set_common_stream_attribute(
        attr_name="heat_flow", value=0.0, idx=idx
    )
    cold_utilities = deepcopy(zone.cold_utilities).set_common_stream_attribute(
        attr_name="heat_flow", value=0.0, idx=idx
    )
    for subzone in zone.subzones.values():
        t = subzone.targets[TargetType.DI.value]
        hot_utility_target += t.hot_utility_target
        cold_utility_target += t.cold_utility_target
        heat_recovery_target += t.heat_recovery_target
        utility_cost += t.utility_cost

        for j in range(len(hot_utilities)):
            hot_utilities[j].set_value_attr_at_idx(
                attr_name="heat_flow",
                value=hot_utilities[j].heat_flow[idx]
                + t.hot_utilities[j].heat_flow[idx],
                idx=idx,
            )

        for j in range(len(cold_utilities)):
            cold_utilities[j].set_value_attr_at_idx(
                attr_name="heat_flow",
                value=cold_utilities[j].heat_flow[idx]
                + t.cold_utilities[j].heat_flow[idx],
                idx=idx,
            )

    heat_recovery_limit = zone.targets[TargetType.DI.value].heat_recovery_limit
    output = {
        "zone_name": zone.name,
        "scope": zone.address,
        "zone_type": zone.type,
        "type": TargetType.SA.value,
        "parent_zone": zone.parent_zone,
        "config": zone.config,
        "hot_utilities": hot_utilities,
        "cold_utilities": cold_utilities,
        "hot_utility_target": hot_utility_target,
        "cold_utility_target": cold_utility_target,
        "heat_recovery_target": heat_recovery_target,
        "heat_recovery_limit": heat_recovery_limit,
        "degree_of_int": (
            (heat_recovery_target / heat_recovery_limit)
            if heat_recovery_limit > 0
            else 1.0
        ),
        "utility_cost": utility_cost,
        "period_id": sid,
        "period_idx": idx,
    }
    return SubzoneAggregateTarget.model_validate(output)


def compute_indirect_integration_targets(
    zone: Zone,
    args: dict | None = None,
) -> IndirectIntegrationTarget | None:
    """Compute indirect integration targets for an aggregated zone.

    The routine assumes the relevant child zones have already been solved for
    direct integration. It then sums subzone targets, builds site-level net
    stream cascades, performs utility-to-utility balancing, and records the
    resulting Total Site target on ``zone`` before returning it.
    """
    idx, sid = get_period_index(period_ids=zone.period_ids, args=args)
    s_tzt = zone.targets[TargetType.SA.value]
    net_hot_streams, net_cold_streams = _reconstruct_subzone_direct_profiles(
        zone,
        args,
    )
    if len(net_hot_streams) == 0 and len(net_cold_streams) == 0:
        return None

    # Total site profiles - process side
    pt = get_process_heat_cascade(
        hot_streams=net_hot_streams,
        cold_streams=net_cold_streams,
        is_shifted=True,  # Align a second shift with the real utility scale.
        period_idx=idx,
    )
    pt.update(
        **_shift_site_process_profiles(
            T_col=pt[ProblemTableLabel.T],
            H_hot=pt[ProblemTableLabel.H_HOT],
            H_cold=pt[ProblemTableLabel.H_COLD],
        )
    )
    # Apply the problem table algorithm to subzone utility use
    pt.update(
        **_build_site_utility_profile(
            hot_utilities=s_tzt.hot_utilities,
            cold_utilities=s_tzt.cold_utilities,
            is_shifted=False,
            idx=idx,
        )
    )

    # Extract overall heat integration targets
    hot_utility_target = pt.loc[0, ProblemTableLabel.H_NET_UT]
    cold_utility_target = pt.loc[-1, ProblemTableLabel.H_NET_UT]
    heat_recovery_target = s_tzt.heat_recovery_target + (
        s_tzt.hot_utility_target - hot_utility_target
    )
    hot_pinch, cold_pinch = pt.pinch_temperatures(col_H=ProblemTableLabel.H_NET_UT)

    # Apply the utility targeting method to determine the net utility use and generation
    hot_utilities, cold_utilities = _match_utility_gen_and_use_at_same_level(
        hot_utilities=deepcopy(s_tzt.hot_utilities),
        cold_utilities=deepcopy(s_tzt.cold_utilities),
        period_idx=idx,
    )

    output = {
        "zone_name": zone.name,
        "scope": zone.address,
        "zone_type": zone.type,
        "type": TargetType.II.value,
        "parent_zone": zone.parent_zone,
        "config": zone.config,
        "pt": pt,
        "graphs": _save_graph_data(pt),
        "hot_pinch": hot_pinch,
        "cold_pinch": cold_pinch,
        "hot_utilities": hot_utilities,
        "cold_utilities": cold_utilities,
        "hot_utility_target": hot_utility_target,
        "cold_utility_target": cold_utility_target,
        "heat_recovery_target": heat_recovery_target,
        "heat_recovery_limit": s_tzt.heat_recovery_limit,
        "degree_of_int": (
            (heat_recovery_target / s_tzt.heat_recovery_limit)
            if s_tzt.heat_recovery_limit > 0
            else 1.0
        ),
        "utility_cost": _compute_utility_cost(
            hot_utilities + cold_utilities,
            idx=idx,
        ),
        "period_id": sid,
        "period_idx": idx,
    }
    return IndirectIntegrationTarget.model_validate(output)


################################################################################
# Helper Functions
################################################################################


def _reconstruct_subzone_direct_profiles(
    zone: Zone,
    args: dict | None = None,
) -> tuple[StreamCollection, StreamCollection]:
    """Build one Total Site layer from immediate-child Direct targets.

    A zone's ``net_hot_streams`` and ``net_cold_streams`` belong to its own
    Direct Integration target. Indirect targeting therefore reconstructs fresh
    child profiles from the child targets and stores them in the zone's
    dedicated immediate-subzone collections. The zone's own Direct Integration
    profiles remain independent.
    """
    if not zone.subzones:
        zone.subzone_net_hot_streams = StreamCollection()
        zone.subzone_net_cold_streams = StreamCollection()
        return zone.net_hot_streams, zone.net_cold_streams

    period_idx, _period_id = get_period_index(period_ids=zone.period_ids, args=args)
    zone.subzone_net_hot_streams = StreamCollection()
    zone.subzone_net_cold_streams = StreamCollection()
    net_hot_streams = zone.subzone_net_hot_streams
    net_cold_streams = zone.subzone_net_cold_streams

    for subzone in zone.subzones.values():
        direct_target = subzone.targets[TargetType.DI.value]
        target_period_idx = (
            direct_target.period_idx
            if direct_target.period_idx is not None
            else period_idx
        )
        child_hot_streams, child_cold_streams = (
            _create_net_hot_and_cold_stream_collections_for_site_analysis(
                T_vals=direct_target.pt[ProblemTableLabel.T],
                H_vals=direct_target.pt[ProblemTableLabel.H_NET_A],
                hot_utilities=direct_target.hot_utilities,
                cold_utilities=direct_target.cold_utilities,
                idx=target_period_idx,
            )
        )
        for stream in child_hot_streams + child_cold_streams:
            stream.supply_temperature = np.round(
                np.asarray(stream.supply_temperature, dtype=float),
                decimals=_DIRECT_TARGET_GRAPH_DECIMALS,
            )
            stream.target_temperature = np.round(
                np.asarray(stream.target_temperature, dtype=float),
                decimals=_DIRECT_TARGET_GRAPH_DECIMALS,
            )
        for key, stream in child_hot_streams.items():
            net_hot_streams.add(stream, key=f"{subzone.name}.{key}")
        for key, stream in child_cold_streams.items():
            net_cold_streams.add(stream, key=f"{subzone.name}.{key}")

    return net_hot_streams, net_cold_streams


def _match_utility_gen_and_use_at_same_level(
    hot_utilities: StreamCollection,
    cold_utilities: StreamCollection,
    period_idx: int | None = None,
) -> Tuple[StreamCollection, StreamCollection]:
    for u_h in hot_utilities:
        for u_c in cold_utilities:
            if (
                abs(
                    (
                        u_h.supply_temperature[period_idx]
                        - u_c.target_temperature[period_idx]
                    )
                )
                < 1
                and abs(
                    (
                        u_h.target_temperature[period_idx]
                        - u_c.supply_temperature[period_idx]
                    )
                )
                < 1
            ):
                Q = min(u_h.heat_flow[period_idx], u_c.heat_flow[period_idx])
                u_h.set_value_attr_at_idx(
                    "heat_flow",
                    u_h.heat_flow[period_idx] - Q,
                    idx=period_idx,
                )
                u_c.set_value_attr_at_idx(
                    "heat_flow",
                    u_c.heat_flow[period_idx] - Q,
                    idx=period_idx,
                )
    return hot_utilities, cold_utilities


def _compute_utility_cost(
    utilities: StreamCollection,
    idx: int | None = None,
) -> np.float64:
    return np.sum([u.utility_cost[idx] for u in utilities])


def _shift_site_process_profiles(
    T_col: np.ndarray,
    H_hot: np.ndarray,
    H_cold: np.ndarray,
) -> ProblemTableUpdateKwargs:
    return {
        "T_col": T_col,
        "updates": {
            ProblemTableLabel.H_HOT: H_hot - H_hot[0],
            ProblemTableLabel.H_COLD: H_cold - H_cold[-1],
        },
    }


def _build_site_utility_profile(
    hot_utilities: StreamCollection,
    cold_utilities: StreamCollection,
    is_shifted: bool = False,
    idx: int | None = None,
) -> ProblemTableUpdateKwargs:
    pt_ut = get_process_heat_cascade(
        hot_streams=hot_utilities,
        cold_streams=cold_utilities,
        is_shifted=is_shifted,
        period_idx=idx,
    )
    h_net_ut = pt_ut[ProblemTableLabel.H_HOT] - pt_ut[ProblemTableLabel.H_COLD]
    return {
        "T_col": pt_ut[ProblemTableLabel.T],
        "updates": {
            ProblemTableLabel.H_NET_UT: h_net_ut - h_net_ut.min(),
            ProblemTableLabel.H_HOT_UT: pt_ut[ProblemTableLabel.H_HOT],
            ProblemTableLabel.H_COLD_UT: pt_ut[ProblemTableLabel.H_COLD]
            - pt_ut[ProblemTableLabel.H_COLD].max(),
        },
    }


def _save_graph_data(
    pt: ProblemTable,
) -> Zone:
    """Prepare graph-ready tables capturing site-level utility composite curves."""
    pt.round(decimals=4)
    return {
        GraphType.TSP.value: pt.slice(
            [
                ProblemTableLabel.T,
                ProblemTableLabel.H_HOT,
                ProblemTableLabel.H_COLD,
                ProblemTableLabel.H_HOT_UT,
                ProblemTableLabel.H_COLD_UT,
            ]
        ),
        GraphType.SUGCC.value: pt.slice(
            [
                ProblemTableLabel.T,
                ProblemTableLabel.H_NET_UT,
                ProblemTableLabel.H_NET_HP,
            ]
        ),
    }
