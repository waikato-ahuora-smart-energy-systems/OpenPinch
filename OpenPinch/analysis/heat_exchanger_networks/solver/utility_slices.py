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

from .arrays import MINIMUM_PAIR_APPROACH, PreparedSolverArrays

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
    arrays: PreparedSolverArrays,
    *,
    like: PreparedSolverArrays | None = None,
    cuts: dict[str, dict[str, list[float]]] | None = None,
) -> PreparedSolverArrays:
    """``arrays`` with each sliding utility replaced by its slices in series.

    Slices take the parent's place on the utility axis, from its inlet to its
    outlet, named ``"<parent>@<inlet>-<outlet>"`` (degrees Celsius, period
    0). ``preparation["utility_slices"]`` maps each slice to its parent and
    its share of the parent's range, per side. With ``like``, the cuts are
    ``like``'s (a child task keeps its parent's slices); with ``cuts`` (by
    side and parent, shares of its range from its inlet), those. With no
    cut, ``arrays`` itself is returned.
    """

    if cuts is None:
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
                # Each pair's minimum approach, floored on the pair.
                approaches = np.maximum(
                    values[f"T_{matched}_cont_period"][n] + contribution,
                    MINIMUM_PAIR_APPROACH,
                )
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


def slice_parent(label: str, parents) -> str:
    """The utility in ``parents`` a slice label belongs to: the longest name
    the label starts with, then ``@``; a name in ``parents`` is its own.
    Utility names may hold ``@`` themselves, so the label is not split."""
    if label in parents:
        return label
    owners = [name for name in parents if label.startswith(f"{name}@")]
    return max(owners, key=len) if owners else label


def slice_label(exchanger) -> str | None:
    """The utility slice a heater's or cooler's id may name.

    Synthesis ids read ``hot-utility:<utility>-><stream>`` and
    ``cold-utility:<stream>-><utility>``, with ``:S<stage>`` for a staged
    unit; the stream and stage come from the exchanger itself, so names
    holding ``->`` or ``:S`` do not mislead. A unit named after a slice (as
    the fit's padding names it) is its own candidate. Any ``@`` makes a
    candidate; ``with_named_slices`` keeps only those that slice a utility
    on its axis.
    """
    heater = exchanger.kind.value == "hot_utility"
    utility = exchanger.source_stream if heater else exchanger.sink_stream
    identifier = exchanger.exchanger_id or ""
    stage = getattr(exchanger, "stage", None)
    suffix = f":S{stage}" if stage is not None else ""
    if suffix and identifier.endswith(suffix):
        identifier = identifier[: -len(suffix)]
    if heater:
        prefix, end = "hot-utility:", f"->{exchanger.sink_stream}"
        named = (
            identifier[len(prefix) : -len(end)]
            if identifier.startswith(prefix) and identifier.endswith(end)
            else utility
        )
    else:
        prefix = f"cold-utility:{exchanger.source_stream}->"
        named = identifier[len(prefix) :] if identifier.startswith(prefix) else utility
    return named if "@" in named else None


def slice_labels(exchangers) -> dict[str, set[str]]:
    """The slice labels the heaters (``"hot"``) and coolers name."""
    labels: dict[str, set[str]] = {"hot": set(), "cold": set()}
    for exchanger in exchangers:
        label = slice_label(exchanger)
        if label is not None:
            side = "hot" if exchanger.kind.value == "hot_utility" else "cold"
            labels[side].add(label)
    return labels


def with_named_slices(
    arrays: PreparedSolverArrays, labels: dict[str, set[str]]
) -> PreparedSolverArrays:
    """``arrays`` sliced so each slice label in ``labels`` (by side, as
    ``with_utility_slices`` names them) is a utility on its axis.

    A network designed with slices names each heater's or cooler's slice in
    its exchanger id; the cuts are read back from those labels' inlet and
    outlet (degrees Celsius, period 0).
    """
    cuts: dict[str, dict[str, list[float]]] = {side: {} for side, *_r in _SIDES}
    for side, short, _long, _matched in _SIDES:
        names = list(arrays.utility_identities[f"{side}_utilities"])
        for label in labels.get(side, ()):
            parent = slice_parent(label, names)
            if parent == label or parent not in names:
                continue
            span = label[len(parent) + 1 :]
            try:
                ends_text = _span_ends(span)
                float(ends_text[0]), float(ends_text[1])
            except ValueError:
                continue
            u = names.index(parent)
            inlet = float(arrays.arrays[f"T_{short}_in_period"][0][u])
            outlet = float(arrays.arrays[f"T_{short}_out_period"][0][u])
            ends = cuts[side].setdefault(parent, [0.0, 1.0])
            for text in ends_text:
                share = (float(text) + 273.15 - inlet) / (outlet - inlet)
                if 0.0 < share < 1.0:
                    ends.append(share)
    for side in cuts:
        for parent, ends in list(cuts[side].items()):
            merged = []
            for share in sorted(ends):
                if not merged or share - merged[-1] > 1e-9:
                    merged.append(share)
            # A label spanning the whole utility needs no cut.
            if len(merged) > 2:
                cuts[side][parent] = merged
            else:
                del cuts[side][parent]
    return with_utility_slices(arrays, cuts=cuts)


def _span_ends(span: str) -> tuple[str, str]:
    """A slice label's inlet and outlet texts; either may be negative."""
    for index in range(1, len(span)):
        if span[index] == "-" and span[index - 1] not in "eE":
            return span[:index], span[index + 1 :]
    raise ValueError(f"not a utility slice span: {span!r}")


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
    "slice_label",
    "slice_labels",
    "slice_parent",
    "sliced_decomposition",
    "utility_slices",
    "with_named_slices",
    "with_utility_slices",
]
