"""The approach inside a match on a segmented stream, as lazy cuts.

A segmented stream stays one stream in design, so a match along it holds
the approach wherever its profile kinks, not only at its two ends. Checking
that exactly is bilinear in the branch's share of each stream's stage heat,
so a model with integers holds the ends and checks each solution here,
cutting the pairs that cross; the solve loop solves again while it cuts
(``execution._solve_and_post_process``). The model needs the isothermal
stage temperatures (``T_h_by_period``, ``T_h_mid_by_period``,
``T_c_by_period``, ``T_c_mid_by_period``) and match binaries ``z``.
"""

from __future__ import annotations

import numpy as np

from ...solver.arrays import pair_minimum_approach
from . import piecewise as _piecewise

_CUT_TOLERANCE = 1e-6
"""Kelvin a solved match may fall short of its approach inside by rounding."""


def install_interior_cuts(owner, M_ij_period) -> None:
    """Check each solution of a model with integers and a segmented stream.

    ``M_ij_period[n][i][j]`` is the pair's big-M on its approach.
    """

    if owner.integers and (
        any(_piecewise._hot_parent_segmented(owner, i) for i in range(owner.I))
        or any(_piecewise._cold_parent_segmented(owner, j) for j in range(owner.J))
    ):
        # Ordered, so a model of the same superstructure can replay them.
        owner._interior_cuts = {}
        owner._interior_big_m = M_ij_period
        owner._add_lazy_cuts = lambda: _add_interior_approach_cuts(owner, M_ij_period)


def _add_interior_approach_cuts(owner, M_ij_period) -> bool:
    """Cut a solution whose matches cross inside, at a segment boundary.

    Each solved match is checked at every segment boundary strictly inside
    it: the partner's temperature is the one at the same fraction of its own
    stage heat, counter-current. Where the approach fails, the match gets a
    linear cut for that boundary and stage: with the boundary inside the hot
    branch, the boundary less the cold branch outlet must be at least the
    approach (inside a heating branch the cold side never exceeds its
    outlet); a cold boundary likewise against the hot branch outlet. The cut
    is conservative, so the model may serve the match with a stage boundary
    at the kink instead. Returns whether a cut was added, so the caller
    re-solves.
    """

    from ...solver.piecewise import profile_from_solver_arrays
    from .execution import _solver_value

    def value(variable) -> float:
        return _solver_value(owner, variable)

    def profile(side: str, index: int, n: int):
        ends = (
            (owner.T_h_in_period[n][index], owner.T_h_out_period[n][index])
            if side == "hot"
            else (owner.T_c_in_period[n][index], owner.T_c_out_period[n][index])
        )
        return profile_from_solver_arrays(
            owner.solver_arrays, side=side, parent_index=index, period_index=n
        ).clipped(*ends)

    added = False
    for n in range(owner.N_periods):
        for k in range(owner.S):
            for i in range(owner.I):
                for j in range(owner.J):
                    # A match with duty has both streams active.
                    if not (
                        owner.z_allowed[i][j][k] > 0
                        and value(owner.Q_r_by_period[n][i][j][k]) > _CUT_TOLERANCE
                    ):
                        continue
                    hot, cold = profile("hot", i, n), profile("cold", j, n)
                    h_in = value(owner.T_h_by_period[n][i][k])
                    h_out = value(owner.T_h_mid_by_period[n][i][k + 1])
                    c_in = value(owner.T_c_by_period[n][j][k + 1])
                    c_out = value(owner.T_c_mid_by_period[n][j][k])
                    hot_span = hot.heat_at_temperature(h_out) - hot.heat_at_temperature(
                        h_in
                    )
                    cold_span = cold.heat_at_temperature(
                        c_out
                    ) - cold.heat_at_temperature(c_in)
                    for boundary in (float(t) for t in hot.temperatures_in[1:]):
                        if not h_out < boundary < h_in or hot_span <= 0.0:
                            continue
                        heat = hot.heat_at_temperature(boundary)
                        share = (heat - hot.heat_at_temperature(h_in)) / hot_span
                        partner_heat = (
                            cold.heat_at_temperature(c_out) - share * cold_span
                        )
                        partner = cold.temperature_at_heat(partner_heat)
                        # The segments meeting here set the approach, not
                        # the parent streams.
                        local = pair_minimum_approach(
                            hot.temperature_contribution_at_heat(heat),
                            cold.temperature_contribution_at_heat(partner_heat),
                        )
                        if boundary - partner < local - _CUT_TOLERANCE:
                            # The cut cannot know where the partner will
                            # sit, so it holds the partner's largest.
                            approach = pair_minimum_approach(
                                hot.temperature_contribution_at_heat(heat),
                                float(np.max(cold.temperature_contributions)),
                            )
                            added |= _cut_pair(
                                owner, M_ij_period, n, i, j, "hot", boundary, approach
                            )
                    for boundary in (float(t) for t in cold.temperatures_in[1:]):
                        if not c_in < boundary < c_out or cold_span <= 0.0:
                            continue
                        heat = cold.heat_at_temperature(boundary)
                        share = (heat - cold.heat_at_temperature(c_in)) / cold_span
                        partner_heat = hot.heat_at_temperature(h_out) - share * hot_span
                        partner = hot.temperature_at_heat(partner_heat)
                        local = pair_minimum_approach(
                            hot.temperature_contribution_at_heat(partner_heat),
                            cold.temperature_contribution_at_heat(heat),
                        )
                        if partner - boundary < local - _CUT_TOLERANCE:
                            approach = pair_minimum_approach(
                                float(np.max(hot.temperature_contributions)),
                                cold.temperature_contribution_at_heat(heat),
                            )
                            added |= _cut_pair(
                                owner, M_ij_period, n, i, j, "cold", boundary, approach
                            )
    return added


def _cut_pair(
    owner, M_ij_period, n, i, j, side: str, boundary: float, approach: float
) -> bool:
    """Cut the pair at ``boundary`` in every stage it may use: cut in one
    stage only, the match moves to the next and crosses there."""

    added = False
    for k in range(owner.S):
        if owner.z_allowed[i][j][k] > 0:
            added |= _cut(owner, M_ij_period, n, i, j, k, side, boundary, approach)
    return added


def replay_interior_cuts(target, source) -> None:
    """Add ``source``'s interior approach cuts to ``target``, a model of the
    same superstructure, in the order they were made, so their variables
    line up."""

    for (n, i, j, k, side, _key), (boundary, approach) in getattr(
        source, "_interior_cuts", {}
    ).items():
        _cut(target, target._interior_big_m, n, i, j, k, side, boundary, approach)


def _cut(
    owner, M_ij_period, n, i, j, k, side: str, boundary: float, approach: float
) -> bool:
    """Add the linear cut for one match, stage and segment boundary, once."""

    key = (n, i, j, k, side, round(boundary, 9))
    if key in owner._interior_cuts:
        return False
    owner._interior_cuts[key] = (boundary, approach)
    m = owner.m
    if side == "hot":
        ends = (owner.T_h_by_period[n][i][k], owner.T_h_mid_by_period[n][i][k + 1])
        span = abs(owner.T_h_in_period[n][i] - owner.T_h_out_period[n][i]) + 1.0
    else:
        ends = (owner.T_c_mid_by_period[n][j][k], owner.T_c_by_period[n][j][k + 1])
        span = abs(owner.T_c_out_period[n][j] - owner.T_c_in_period[n][j]) + 1.0
    # 1 when the branch end is at or above the boundary: the boundary is
    # inside the branch when its upper end is above and its lower end not.
    flags = []
    for temperature in ends:
        flag = m.Var(value=0, lb=0, ub=1, integer=True)
        m.Equation(temperature - boundary <= span * flag)
        m.Equation(temperature - boundary >= -span * (1 - flag))
        flags.append(flag)
    upper, lower = flags
    big = M_ij_period[n][i][j]
    gap = (
        boundary - owner.T_c_mid_by_period[n][j][k]
        if side == "hot"
        else owner.T_h_mid_by_period[n][i][k + 1] - boundary
    )
    m.Equation(
        gap >= approach - big * (1 - owner.z[i][j][k]) - big * (1 - upper) - big * lower
    )
    return True
