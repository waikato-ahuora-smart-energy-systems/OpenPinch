"""Prepare aligned period cases for one shared HPR design solve."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from ....analysis.targeting.direct import compute_direct_integration_targets
from ....analysis.targeting.total_site import (
    compute_indirect_integration_targets,
    compute_total_subzone_utility_targets,
)
from ....contracts.hpr import HPRPeriodCase
from ....domain._stream.value_state import resolve_period_weights
from ....domain.configuration import tol
from ....domain.enums import PT
from ....domain.problem_table import ProblemTable
from ....domain.zone import Zone
from ...targeting.cascade import get_process_heat_cascade
from ..common.load_selection import resolve_hpr_target_load
from ..common.preprocessing import construct_HPRTargetInputs
from .state import _PreparedHPRPeriodCase


def build_multiperiod_hpr_cases(
    *,
    zone: Zone,
    is_heat_pumping: bool,
    is_direct: bool,
    args: dict | None = None,
) -> list[_PreparedHPRPeriodCase]:
    """Prepare aligned single-period HPR inputs for one shared design vector."""
    raw_cases = []
    weights = _canonical_period_weights(zone)
    for period_id, period_idx in _canonical_period_items(zone):
        period_args = _period_args(args, period_id=period_id, period_idx=period_idx)
        base_target = _compute_hpr_base_target_for_period(
            zone=zone,
            period_args=period_args,
            is_direct=is_direct,
        )
        if base_target is None:
            raise ValueError(
                "Multi-period HPR optimisation requires a valid base target for "
                f"period {period_id!r}."
            )
        optimizer_pt = _optimizer_problem_table_for_hpr(
            zone=zone,
            base_target=base_target,
            is_direct=is_direct,
            period_idx=period_idx,
        )
        raw_cases.append(
            {
                "period_id": period_id,
                "period_idx": period_idx,
                "weight": weights[period_id],
                "base_target": base_target,
                "optimizer_pt": optimizer_pt,
            }
        )

    _align_hpr_problem_tables([case["optimizer_pt"] for case in raw_cases])

    period_cases = []
    for case in raw_cases:
        pt = case["optimizer_pt"]
        period_id = case["period_id"]
        period_idx = case["period_idx"]
        target_load = resolve_hpr_target_load(
            H_net_cold=pt[PT.H_NET_COLD],
            H_net_hot=pt[PT.H_NET_HOT],
            is_heat_pumping=is_heat_pumping,
            is_refrigeration=not is_heat_pumping,
            config=zone.config,
            period_id=period_id,
            period_idx=period_idx,
        )
        if target_load < tol:
            raise ValueError(
                "Multi-period HPR optimisation requires a non-zero HPR load for "
                f"period {period_id!r}."
            )
        solver_case = HPRPeriodCase(
            period_id=period_id,
            period_idx=period_idx,
            weight=case["weight"],
            args=construct_HPRTargetInputs(
                Q_hpr_target=target_load,
                T_vals=pt[PT.T],
                H_hot=np.abs(pt[PT.H_NET_HOT]) * -1,
                H_cold=np.abs(pt[PT.H_NET_COLD]),
                is_heat_pumping=is_heat_pumping,
                config=zone.config,
                period_idx=period_idx,
                debug=False,
            ),
        )
        period_cases.append(
            _PreparedHPRPeriodCase(
                period_id=period_id,
                period_idx=period_idx,
                weight=case["weight"],
                solver_case=solver_case,
                base_target=case["base_target"],
                optimizer_pt=pt,
            )
        )
    return period_cases


def period_id_for_index(zone: Zone, period_idx: int) -> str:
    for period_id, idx in (zone.period_ids or {"0": 0}).items():
        if int(idx) == int(period_idx):
            return str(period_id)
    raise ValueError(f"period_idx {period_idx!r} was not found on this zone.")


def period_case_by_id(
    period_cases: list[_PreparedHPRPeriodCase],
    period_id: str,
) -> _PreparedHPRPeriodCase:
    for case in period_cases:
        if str(case.period_id) == str(period_id):
            return case
    raise ValueError(f"period_id {period_id!r} was not prepared for HPR targeting.")


def _compute_hpr_base_target_for_period(
    *,
    zone: Zone,
    period_args: dict,
    is_direct: bool,
):
    if is_direct:
        return compute_direct_integration_targets(zone, period_args)
    _refresh_direct_targets_for_subtree(zone, period_args)
    zone.import_hot_and_cold_streams_from_sub_zones(
        get_net_streams=True,
        is_n_zone_depth=False,
        is_new_stream_collection=True,
    )
    zone.add_target(compute_total_subzone_utility_targets(zone, period_args))
    return compute_indirect_integration_targets(zone, period_args)


def _refresh_direct_targets_for_subtree(zone: Zone, period_args: dict) -> None:
    for subzone in zone.subzones.values():
        _refresh_direct_targets_for_subtree(subzone, period_args)
    zone.add_target(compute_direct_integration_targets(zone, period_args))


def _optimizer_problem_table_for_hpr(
    *,
    zone: Zone,
    base_target,
    is_direct: bool,
    period_idx: int,
) -> ProblemTable:
    if is_direct:
        return deepcopy(base_target.pt)
    return get_process_heat_cascade(
        hot_streams=zone.cold_utilities.get_hot_streams(invert_utility=True),
        cold_streams=zone.hot_utilities.get_cold_streams(invert_utility=True),
        is_shifted=True,
        is_full_analysis=True,
        period_idx=period_idx,
    )


def _align_hpr_problem_tables(tables: list[ProblemTable]) -> None:
    if not tables:
        raise ValueError("At least one HPR problem table is required.")
    for i, table in enumerate(tables):
        for other in tables[i + 1 :]:
            table.share_temperature_intervals(other)

    reference = tables[0][PT.T]
    for table in tables[1:]:
        if len(table[PT.T]) != len(reference) or not np.allclose(
            table[PT.T],
            reference,
            rtol=0.0,
            atol=tol,
        ):
            raise ValueError(
                "Multi-period HPR optimisation requires aligned PT temperature grids."
            )


def _canonical_period_items(zone: Zone) -> list[tuple[str, int]]:
    period_ids = zone.period_ids or {"0": 0}
    return [(str(period_id), int(idx)) for period_id, idx in period_ids.items()]


def _canonical_period_weights(zone: Zone) -> dict[str, float]:
    items = _canonical_period_items(zone)
    flat_weights = resolve_period_weights(
        [period_id for period_id, _idx in items],
        getattr(zone, "weights", None),
    )
    return {
        period_id: float(flat_weights[period_idx]) for period_id, period_idx in items
    }


def _period_args(
    args: dict | None,
    *,
    period_id: str,
    period_idx: int,
) -> dict:
    period_args = dict(args or {})
    period_args["period_id"] = period_id
    period_args["period_idx"] = period_idx
    return period_args


__all__ = [
    "build_multiperiod_hpr_cases",
    "period_case_by_id",
    "period_id_for_index",
]
