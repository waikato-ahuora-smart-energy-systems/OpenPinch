"""Run vapour compression targets on every fluid set a refrigerant screen proposes."""

from __future__ import annotations

from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from typing import Any

from ..contracts.hpr import HPRTopologyIdentifier
from ..contracts.hpr_fluid_set_runs import HprFluidSetRun, HprFluidSetRunsResult
from ..contracts.hpr_refrigerant_screen import (
    HprRefrigerantScreenResult,
    RefrigerantScreenLayout,
)


def is_screened(refrigerants: Any) -> bool:
    """Whether ``refrigerants`` is a screen result or one of its layouts."""
    return isinstance(
        refrigerants, (HprRefrigerantScreenResult, RefrigerantScreenLayout)
    )


@dataclass
class _PlannedRun:
    method: HPRTopologyIdentifier
    condensers: int
    evaporators: int
    mvr_stages: int | None
    fluids: tuple[str, ...]
    layouts: list[RefrigerantScreenLayout] = field(default_factory=list)


def _stage_counts(layout: RefrigerantScreenLayout) -> tuple[int, int, int | None]:
    """The condensers, evaporators and MVR stages a target needs for ``layout``.

    A VC+MVR target counts the condensers of its vapour compression stages
    only; the MVR stages sit above them.
    """
    if layout.method is not HPRTopologyIdentifier.VAPOUR_COMPRESSION_MVR:
        return len(layout.T_cond_levels), len(layout.T_evap_levels), None
    vc = [stage for stage in layout.stages if stage.kind == "vapour_compression"]
    return (
        len({stage.T_cond.value for stage in vc}),
        len(layout.T_evap_levels),
        len(layout.stages) - len(vc),
    )


def plan_fluid_set_runs(
    source: HprRefrigerantScreenResult | RefrigerantScreenLayout,
    *,
    methods: Collection[HPRTopologyIdentifier],
    is_heat_pumping: bool,
    condensers: int | None = None,
    evaporators: int | None = None,
    mvr_stages: int | None = None,
) -> list[_PlannedRun]:
    """List the distinct runs, in screening order, for one target method.

    Layouts of another method, or with other stage counts than those given,
    are left out. Layouts that ask for the same stage counts and fluids
    share one run.
    """
    if isinstance(source, HprRefrigerantScreenResult):
        if source.is_heat_pumping != is_heat_pumping:
            raise ValueError(
                "The refrigerant screen is for a "
                f"{'heat pump' if source.is_heat_pumping else 'refrigeration'} "
                "target; screen the matching Carnot target instead."
            )
        layouts = source.layouts
    else:
        layouts = (source,)
    wanted = (condensers, evaporators, mvr_stages)
    planned: dict[tuple, _PlannedRun] = {}
    for layout in layouts:
        if layout.method not in methods:
            continue
        counts = _stage_counts(layout)
        if any(w is not None and w != c for w, c in zip(wanted, counts)):
            continue
        for fluids in layout.fluid_sets():
            key = (layout.method, *counts, fluids)
            run = planned.setdefault(key, _PlannedRun(layout.method, *counts, fluids))
            run.layouts.append(layout)
    if not planned:
        names = ", ".join(sorted(method.value for method in methods))
        raise ValueError(
            f"The refrigerant screen proposes no fluid set for {names}"
            + (
                ""
                if wanted == (None, None, None)
                else " with the requested stage counts"
            )
            + "."
        )
    return list(planned.values())


def run_fluid_sets(
    plan: list[_PlannedRun],
    solve: Callable[[_PlannedRun], Any],
) -> HprFluidSetRunsResult:
    """Solve each planned run; a run that raises is recorded with its error."""
    runs = []
    for item in plan:
        record = {
            "method": item.method,
            "condensers": item.condensers,
            "evaporators": item.evaporators,
            "mvr_stages": item.mvr_stages,
            "fluids": item.fluids,
            "layouts": tuple(item.layouts),
        }
        try:
            target = solve(item)
        except Exception as error:  # each fluid set's outcome is the contract
            runs.append(
                HprFluidSetRun(
                    **record, success=False, error=f"{type(error).__name__}: {error}"
                )
            )
            continue
        cost = target.hpr_total_annualized_cost
        runs.append(
            HprFluidSetRun(
                **record,
                success=bool(target.hpr_success),
                total_annualized_cost=None
                if cost is None
                else float(cost.to("$/y").value),
                target=target,
            )
        )
    return HprFluidSetRunsResult(runs=tuple(runs))
