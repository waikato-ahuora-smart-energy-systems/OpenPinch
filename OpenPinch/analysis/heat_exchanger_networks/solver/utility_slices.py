"""Slice sliding-temperature utilities at the streams they must reach.

A heater or cooler runs its utility over the utility's whole range: hot
water from 90 to 70 C cannot heat a stream entering at 82 C with a 5 K
approach, though targeting lets it heat that stream to 85 C. Cutting the
utility at each stream's supply plus (minus, for a cooler) its approach
gives slices with fixed inlets and outlets, each an ordinary utility. The
slices run in series: one utility flow passes them all, so each slice's
duty is the level's times its share of the level's temperature range
(``_base/utility_slices.py`` holds the models to that).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .arrays import PreparedSolverArrays

# Only a utility spanning more than this (K) is sliced: steam at 0.1 K is not.
MINIMUM_SLICED_RANGE = 1.0
# Cuts closer than this (K) are merged, so no slice is near empty.
MINIMUM_SLICE_SPAN = 0.5
# A cut within this (K) of a utility's end is no cut.
_END_TOLERANCE = 1e-6

_SIDES = (
    # (side, short prefix, long prefix, matched stream prefix)
    ("hot", "hu", "hot_utility", "c"),
    ("cold", "cu", "cold_utility", "h"),
)


def with_utility_slices(
    arrays: PreparedSolverArrays, *, like: PreparedSolverArrays | None = None
) -> PreparedSolverArrays:
    """``arrays`` with each sliding utility replaced by its slices in series.

    Slices take the parent's place on the utility axis, from its inlet to its
    outlet, named ``"<parent>@<inlet>-<outlet>"`` (degrees Celsius, period
    0). ``preparation["utility_slices"]`` maps each slice to its parent and
    its share of the parent's range, per side. With ``like``, the cuts are
    ``like``'s (a child task keeps its parent's slices). With no cut,
    ``arrays`` itself is returned.
    """

    cuts = _cuts_from(like) if like is not None else _cuts(arrays)
    if not any(cuts[side] for side, *_rest in _SIDES):
        return arrays
    data = {name: values.copy() for name, values in arrays.arrays.items()}
    axis_maps = {axis: dict(names) for axis, names in arrays.axis_maps.items()}
    streams = {axis: list(names) for axis, names in arrays.stream_identities.items()}
    utilities = {axis: list(names) for axis, names in arrays.utility_identities.items()}
    mapping: dict[str, dict[str, list]] = {}
    for side, short, long, _matched in _SIDES:
        axis = f"{side}_utilities"
        names = list(arrays.utility_identities[axis])
        order, fractions, new_names = [], [], []
        mapping[axis] = {}
        for u, name in enumerate(names):
            shares = cuts[side].get(name)
            if not shares:
                order.append(u)
                fractions.append((0.0, 1.0))
                new_names.append(name)
                continue
            for start, end in zip(shares[:-1], shares[1:]):
                inlet, outlet = _between(arrays, short, u, start, end)
                label = f"{name}@{inlet - 273.15:g}-{outlet - 273.15:g}"
                order.append(u)
                fractions.append((start, end))
                new_names.append(label)
                mapping[axis][label] = [name, end - start]
        _take(data, short, long, order, fractions)
        utilities[axis] = new_names
        axis_maps[axis] = {name: index for index, name in enumerate(new_names)}
        segment_names = [f"{name}.S1" for name in new_names]
        data[f"{long}_segment_identities"][:, 0] = segment_names
        axis_maps[f"{long}_segments"] = {
            name: index for index, name in enumerate(segment_names)
        }
        streams[f"{long}_segments"] = segment_names
    preparation = dict(arrays.preparation)
    preparation["utility_slices"] = mapping
    return replace(
        arrays,
        arrays=data,
        axis_maps=axis_maps,
        stream_identities=streams,
        utility_identities=utilities,
        preparation=preparation,
    )


def utility_slices(arrays: PreparedSolverArrays) -> dict[str, dict[str, list]]:
    """Each side's ``{slice: [parent, share]}``; empty when nothing is sliced."""

    preparation = getattr(arrays, "preparation", None) or {}
    return preparation.get("utility_slices") or {}


def sliced_decomposition(decomposition, arrays: PreparedSolverArrays):
    """``decomposition`` on ``arrays``' utility axes, each slice taking its
    share of its parent's level target."""

    slices = utility_slices(arrays)
    if not slices:
        return decomposition
    hot = tuple(arrays.utility_identities["hot_utilities"])
    cold = tuple(arrays.utility_identities["cold_utilities"])

    def levels(names, before, targets, axis):
        duty = dict(zip(before, targets))
        return tuple(
            duty.get(slices[axis].get(name, [name])[0], 0.0)
            * (slices[axis][name][1] if name in slices[axis] else 1.0)
            for name in names
        )

    return decomposition.model_copy(
        update={
            "hot_utility_identities": hot,
            "cold_utility_identities": cold,
            "period_targets": tuple(
                target.model_copy(
                    update={
                        "hot_utility_level_targets": levels(
                            hot,
                            decomposition.hot_utility_identities,
                            target.hot_utility_level_targets,
                            "hot_utilities",
                        ),
                        "cold_utility_level_targets": levels(
                            cold,
                            decomposition.cold_utility_identities,
                            target.cold_utility_level_targets,
                            "cold_utilities",
                        ),
                    }
                )
                for target in decomposition.period_targets
            ),
        }
    )


def _cuts(arrays: PreparedSolverArrays) -> dict[str, dict[str, list[float]]]:
    """Each sliceable utility's cuts, as shares of its range from its inlet.

    A heater's cut is a cold stream's supply plus its approach (the stream's
    and utility's contributions); a cooler's a hot stream's supply less it.
    Shares are pooled over periods, so every period slices alike.
    """

    values = arrays.arrays
    result: dict[str, dict[str, list[float]]] = {}
    for side, short, long, matched in _SIDES:
        result[side] = {}
        names = arrays.utility_identities[f"{side}_utilities"]
        for u, name in enumerate(names):
            if bool(values[f"{long}_parent_segmented"][u]):
                continue
            shares = set()
            sliceable = True
            for n in range(values[f"T_{short}_in_period"].shape[0]):
                inlet = float(values[f"T_{short}_in_period"][n][u])
                outlet = float(values[f"T_{short}_out_period"][n][u])
                span = abs(inlet - outlet)
                if span <= MINIMUM_SLICED_RANGE:
                    sliceable = False
                    break
                contribution = float(values[f"T_{short}_cont_period"][n][u])
                supplies = values[f"T_{matched}_in_period"][n]
                approaches = values[f"T_{matched}_cont_period"][n] + contribution
                sign = 1.0 if side == "hot" else -1.0
                for supply, approach in zip(supplies, approaches):
                    cut = float(supply) + sign * float(approach)
                    share = (inlet - cut) / (inlet - outlet)
                    if _END_TOLERANCE / span < share < 1.0 - _END_TOLERANCE / span:
                        shares.add(share)
            if not sliceable or not shares:
                continue
            kept = _merged(sorted(shares), values, short, u)
            if kept:
                result[side][name] = [0.0, *kept, 1.0]
    return result


def _merged(shares: list[float], values, short: str, u: int) -> list[float]:
    """Shares at least ``MINIMUM_SLICE_SPAN`` apart and from either end."""

    span = float(
        np.min(
            np.abs(
                values[f"T_{short}_in_period"][:, u]
                - values[f"T_{short}_out_period"][:, u]
            )
        )
    )
    gap = MINIMUM_SLICE_SPAN / span
    kept: list[float] = []
    for share in shares:
        if share < gap or share > 1.0 - gap:
            continue
        if kept and share - kept[-1] < gap:
            continue
        kept.append(share)
    return kept


def _cuts_from(like: PreparedSolverArrays) -> dict[str, dict[str, list[float]]]:
    result: dict[str, dict[str, list[float]]] = {side: {} for side, *_r in _SIDES}
    for side, *_rest in _SIDES:
        for _label, (parent, share) in (
            utility_slices(like).get(f"{side}_utilities", {}).items()
        ):
            ends = result[side].setdefault(parent, [0.0])
            ends.append(ends[-1] + share)
    for side in result:
        for parent, ends in result[side].items():
            ends[-1] = 1.0
    return result


def _between(arrays, short: str, u: int, start: float, end: float):
    inlet = float(arrays.arrays[f"T_{short}_in_period"][0][u])
    outlet = float(arrays.arrays[f"T_{short}_out_period"][0][u])
    return inlet + start * (outlet - inlet), inlet + end * (outlet - inlet)


def _take(data, short: str, long: str, order, fractions) -> None:
    """Rebuild every per-utility array of a side along ``order``."""

    count = len(data[f"T_{short}_in_period"][0])
    index = np.asarray(order, dtype=int)
    starts = np.asarray([start for start, _end in fractions])
    ends = np.asarray([end for _start, end in fractions])
    shares = ends - starts
    inlet = data[f"T_{short}_in_period"].copy()
    outlet = data[f"T_{short}_out_period"].copy()
    for key in list(data):
        values = data[key]
        if key.startswith(f"{long}_") or short in key.split("_"):
            axis = 1 if key.endswith("_period") else 0
        else:
            continue
        if values.ndim <= axis or values.shape[axis] != count:
            continue
        data[key] = np.take(values, index, axis=axis)
    new_in = inlet[:, index] + starts * (outlet[:, index] - inlet[:, index])
    new_out = inlet[:, index] + ends * (outlet[:, index] - inlet[:, index])
    data[f"T_{short}_in_period"] = new_in
    data[f"T_{short}_out_period"] = new_out
    data[f"{long}_segment_t_in_period"][:, :, 0] = new_in
    data[f"{long}_segment_t_out_period"][:, :, 0] = new_out
    duty = data[f"{long}_segment_duty_period"][:, :, 0] * shares
    data[f"{long}_segment_duty_period"][:, :, 0] = duty
    data[f"{long}_segment_cumulative_duty_period"][:, :, 1] = duty


__all__ = [
    "MINIMUM_SLICED_RANGE",
    "MINIMUM_SLICE_SPAN",
    "sliced_decomposition",
    "utility_slices",
    "with_utility_slices",
]
