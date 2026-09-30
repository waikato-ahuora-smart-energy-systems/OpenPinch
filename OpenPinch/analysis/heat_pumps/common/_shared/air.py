"""Ambient air as a free, unlimited utility inside an HPR heat cascade.

Air can take heat from the cascade at or above its sink temperature and
give heat at or below its source temperature. How much it takes or gives is
an outcome of the cascade, not a decision: air that is not needed is simply
not used, so it never creates a utility demand.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "CascadeWithAir",
    "ambient_sink_temperature",
    "ambient_source_temperature",
    "cascade_with_air",
]


def ambient_source_temperature(args) -> float:
    """Shifted temperature at or below which air can supply heat."""
    return (
        float(args.T_env)
        - float(getattr(args, "dt_env_cont", 0.0))
        + float(getattr(args, "dtcont_hp", 0.0))
    )


def ambient_sink_temperature(args) -> float:
    """Shifted temperature at or above which air can take heat."""
    return (
        float(args.T_env)
        + float(getattr(args, "dt_env_cont", 0.0))
        - float(getattr(args, "dtcont_hp", 0.0))
    )


@dataclass(frozen=True)
class CascadeWithAir:
    """External duties of a heat cascade after free ambient air is used.

    ``Q_ext_top`` is the heat still needed at the top of the cascade and
    ``Q_ext_bottom`` the heat still to be removed at the bottom. ``Q_air_source``
    is the heat air supplies and ``Q_air_sink`` the heat air takes.
    """

    Q_ext_top: float
    Q_ext_bottom: float
    Q_air_source: float
    Q_air_sink: float


def cascade_with_air(
    T: np.ndarray,
    H_net: np.ndarray,
    *,
    T_air_source: float,
    T_air_sink: float,
) -> CascadeWithAir:
    """Place free ambient air in a feasible cascade and return external duties.

    ``T`` descends and ``H_net`` is the heat flowing down through each
    temperature, with ``H_net[0]`` the heat needed at the top and
    ``H_net[-1]`` the heat left at the bottom (both non-negative). Supplying
    heat at a level lowers every flow above it, and rejecting heat at a level
    lowers every flow below it, so each is limited by the smallest flow on
    its side. Air heat can only meet deficits below its level, and air can
    only take heat released above its level, so heat merely passing through
    the cascade is never swapped for air. Air first supplies heat, which
    saves the dearer hot utility, then takes heat from what flows on.
    """
    T = np.asarray(T, dtype=float)
    H_net = np.maximum(np.asarray(H_net, dtype=float), 0.0)
    if T.size == 0:
        return CascadeWithAir(0.0, 0.0, 0.0, 0.0)

    T_grid, H_grid = _with_levels(T, H_net, (T_air_source, T_air_sink))

    # Air heat enters at the source level and can only meet deficits below it;
    # it also lowers every flow above that level, which must stay non-negative.
    Q_air_source = 0.0
    if T_air_source > T_grid[-1]:
        above = T_grid >= T_air_source
        at = float(np.interp(T_air_source, T_grid[::-1], H_grid[::-1]))
        absorbed_below = at - float(H_grid[T_grid <= T_air_source].min())
        flow_limit = float(H_grid[above].min()) if above.any() else float(H_grid[0])
        Q_air_source = max(min(flow_limit, absorbed_below), 0.0)

    # Air takes heat at the sink level and can only take heat released above
    # it; it lowers every flow below that level, which must stay non-negative.
    remaining = H_grid - np.where(T_grid >= T_air_source, Q_air_source, 0.0)
    Q_air_sink = 0.0
    if T_air_sink < T_grid[0]:
        below = T_grid <= T_air_sink
        at = float(np.interp(T_air_sink, T_grid[::-1], remaining[::-1]))
        released_above = at - float(remaining[T_grid >= T_air_sink].min())
        flow_limit = (
            float(remaining[below].min()) if below.any() else float(remaining[-1])
        )
        Q_air_sink = max(min(flow_limit, released_above), 0.0)

    return CascadeWithAir(
        Q_ext_top=float(H_net[0]) - Q_air_source,
        Q_ext_bottom=float(H_net[-1]) - Q_air_sink,
        Q_air_source=Q_air_source,
        Q_air_sink=Q_air_sink,
    )


def _with_levels(
    T: np.ndarray,
    H_net: np.ndarray,
    levels: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Add the cascade's flow at each air level that lies inside its range."""
    extra = [
        lvl for lvl in levels if T[-1] < lvl < T[0] and not np.isclose(T, lvl).any()
    ]
    if not extra:
        return T, H_net
    T_all = np.concatenate([T, np.asarray(extra, dtype=float)])
    H_all = np.concatenate([H_net, np.interp(extra, T[::-1], H_net[::-1])])
    order = np.argsort(-T_all, kind="stable")
    return T_all[order], H_all[order]
