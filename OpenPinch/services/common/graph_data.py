"""Graph construction helpers for composite curves and related plots."""

from collections import defaultdict
from typing import Iterable, List, Optional, Tuple

import numpy as np

from ...classes.problem_table import ProblemTable
from ...classes.zone import Zone
from ...lib.config import tol
from ...lib.enums import GT, PT, ArrowHead, LineColour, StreamLoc
from ...lib.schemas.targets import BaseTargetModel
from .graph_series_meta import GRAPH_SERIES_META, GraphSeriesMeta

DECIMAL_PLACES = 2
GCC_VERTICAL_TOL = 1e-3


__all__ = [
    "get_output_graph_data",
    "clean_composite_curve_ends",
    "clean_composite_curve",
]


################################################################################
# Public API
################################################################################


def get_output_graph_data(zone: Zone, graph_sets: Optional[dict] = None) -> dict:
    """Returns Json data points for each process."""
    if graph_sets is None:
        graph_sets = {}

    for target in zone.targets.values():
        graph_sets[target.name] = _create_graph_set(target, zone=zone)

    if len(zone.subzones) > 0:
        for subzone in zone.subzones.values():
            graph_sets = get_output_graph_data(subzone, graph_sets)

    return graph_sets


def clean_composite_curve_ends(
    y_vals: np.ndarray | list, x_vals: np.ndarray | list
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove redundant points in composite curves."""
    y_vals = np.array(y_vals)
    x_vals = np.array(x_vals)

    if np.all(np.isclose(x_vals, 0.0, atol=tol)) or np.abs(x_vals.var()) < tol:
        return np.array([]), np.array([])

    mask_0 = ~np.isclose(x_vals, x_vals[0] * np.ones(len(x_vals)), atol=tol)
    start = np.flatnonzero(mask_0)[0] - 1
    mask_1 = ~np.isclose(x_vals, x_vals[-1] * np.ones(len(x_vals)), atol=tol)
    end = np.flatnonzero(mask_1)[-1] + 1

    x_clean = x_vals[start : end + 1]
    y_clean = y_vals[start : end + 1]
    return y_clean, x_clean


def clean_composite_curve(
    y_array: np.ndarray | list, x_array: np.ndarray | list
) -> Tuple[np.ndarray | list]:
    """Remove redundant points in composite curves."""

    # Round to avoid tiny numerical errors
    y_vals, x_vals = clean_composite_curve_ends(y_array, x_array)

    if len(x_vals) <= 2:
        return y_vals, x_vals

    x_clean, y_clean = [x_vals[0]], [y_vals[0]]

    for i in range(1, len(x_vals) - 1):
        x1, x2, x3 = x_vals[i - 1], x_vals[i], x_vals[i + 1]
        y1, y2, y3 = y_vals[i - 1], y_vals[i], y_vals[i + 1]

        if x1 == x3:
            # All three x are the same; keep x2 only if y2 is different
            if x1 != x2:
                x_clean.append(x2)
                y_clean.append(y2)
        else:
            # Linear interpolation check
            y_interp = y1 + (y3 - y1) * (x2 - x1) / (x3 - x1)
            if abs(y2 - y_interp) > tol:
                x_clean.append(x2)
                y_clean.append(y2)

    x_clean.append(x_vals[-1])
    y_clean.append(y_vals[-1])

    if abs(x_clean[0] - x_clean[1]) < tol:
        x_clean.pop(0)
        y_clean.pop(0)

    i = len(x_clean) - 1
    if abs(x_clean[i] - x_clean[i - 1]) < tol:
        x_clean.pop(i)
        y_clean.pop(i)

    return np.asarray(y_clean), np.asarray(x_clean)


################################################################################
# Helper Functions
################################################################################


def _create_graph_set(t: BaseTargetModel, zone: Optional[Zone] = None) -> dict:
    """Creates Pinch Analysis and total site analysis graphs for a specifc zone."""

    graphs: List[dict] = []
    target_graphs = getattr(t, "graphs", {})
    graph_title = t.name
    zone_name = getattr(zone, "name", None) or getattr(t, "zone_name", None)
    zone_address = getattr(zone, "address", None)

    if GT.CC.value in target_graphs:
        graphs.append(
            _make_composite_graph(
                graph_title=graph_title,
                key=GT.CC.value,
                data=target_graphs[GT.CC.value],
                label="Composite Curve",
                value_field=[
                    PT.H_HOT,
                    PT.H_COLD,
                ],
                stream_type=[
                    StreamLoc.HotS,
                    StreamLoc.ColdS,
                ],
            )
        )

    if GT.SCC.value in target_graphs:
        graphs.append(
            _make_composite_graph(
                graph_title=graph_title,
                key=GT.SCC.value,
                data=target_graphs[GT.SCC.value],
                label="Shifted Composite Curve",
                value_field=[
                    PT.H_HOT,
                    PT.H_COLD,
                ],
                stream_type=[
                    StreamLoc.HotS,
                    StreamLoc.ColdS,
                ],
            )
        )

    if GT.BCC.value in target_graphs:
        graphs.append(
            _make_composite_graph(
                graph_title=graph_title,
                key=GT.BCC.value,
                data=target_graphs[GT.BCC.value],
                label="Balanced Composite Curve",
                value_field=[PT.H_HOT_BAL, PT.H_COLD_BAL],
                stream_type=[StreamLoc.HotS, StreamLoc.ColdS],
                include_arrows=True,
            )
        )

    if GT.GCC.value in target_graphs:
        graphs.append(
            _make_gcc_graph(
                graph_title=graph_title,
                key=GT.GCC.value,
                data=target_graphs[GT.GCC.value],
                label="Grand Composite Curve",
                value_field=[
                    PT.H_NET,
                    PT.H_NET_NP,
                    PT.H_NET_V,
                    PT.H_NET_A,
                    PT.H_NET_UT,
                ],
                is_utility_profile=[False, False, False, False, True],
            )
        )

    if GT.GCC_R.value in target_graphs:
        graphs.append(
            _make_gcc_graph(
                graph_title=graph_title,
                key=GT.GCC_R.value,
                data=target_graphs[GT.GCC_R.value],
                label="Grand Composite Curve (Real)",
                value_field=[
                    PT.H_NET,
                    PT.H_NET_UT,
                ],
                is_utility_profile=[False, True],
            )
        )

    if GT.GCC_X.value in target_graphs:
        graphs.append(
            _make_gcc_graph(
                graph_title=graph_title,
                key=GT.GCC_X.value,
                data=target_graphs[GT.GCC_X.value],
                label="Exergetic Grand Composite Curve",
                value_field=[
                    PT.X_GCC.value,
                ],
                is_utility_profile=[False],
            )
        )

    if GT.NLP.value in target_graphs:
        graphs.append(
            _make_composite_graph(
                graph_title=graph_title,
                key=GT.NLP.value,
                data=target_graphs[GT.NLP.value],
                label="Net Load Curves",
                value_field=[
                    PT.H_NET_HOT,
                    PT.H_NET_COLD,
                    PT.H_HOT_UT,
                    PT.H_COLD_UT,
                    PT.H_HOT_HP,
                    PT.H_COLD_HP,
                ],
                stream_type=[
                    StreamLoc.HotS,
                    StreamLoc.ColdS,
                    StreamLoc.HotU,
                    StreamLoc.ColdU,
                    StreamLoc.HotU,
                    StreamLoc.ColdU,
                ],
                include_arrows=True,
            )
        )

    if GT.NLP_X.value in target_graphs:
        graphs.append(
            _make_composite_graph(
                graph_title=graph_title,
                key=GT.NLP_X.value,
                data=target_graphs[GT.NLP_X.value],
                label="Exergetic Net Load Profiles",
                value_field=[
                    PT.X_SUR,
                    PT.X_DEF,
                ],
                stream_type=[
                    StreamLoc.HotS,
                    StreamLoc.ColdS,
                ],
                include_arrows=True,
            )
        )

    if GT.TSP.value in target_graphs:
        graphs.append(
            _make_composite_graph(
                graph_title=graph_title,
                key=GT.TSP.value,
                data=target_graphs[GT.TSP.value],
                value_field=[
                    PT.H_HOT,
                    PT.H_COLD,
                    PT.H_HOT_UT,
                    PT.H_COLD_UT,
                ],
                stream_type=[
                    StreamLoc.HotS,
                    StreamLoc.ColdS,
                    StreamLoc.HotU,
                    StreamLoc.ColdU,
                ],
                label="Total Site Profiles",
                include_arrows=True,
            )
        )

    if GT.SUGCC.value in target_graphs:
        graphs.append(
            _make_gcc_graph(
                graph_title=graph_title,
                key=GT.SUGCC.value,
                data=target_graphs[GT.SUGCC.value],
                label="Site Utility Grand Composite Curve",
                value_field=[PT.H_NET_UT],
                is_utility_profile=[True],
            )
        )

    if GT.GCC_HP.value in target_graphs:
        graphs.append(
            _make_gcc_graph(
                graph_title=graph_title,
                key=GT.GCC_HP.value,
                data=target_graphs[GT.GCC_HP.value],
                label="Grand Composite Curve with Heat Pump",
                value_field=[
                    PT.H_NET_W_AIR,
                    PT.H_NET_HP,
                ],
                is_utility_profile=[False, True],
            )
        )

    return {
        "name": graph_title,
        "target_type": getattr(t, "type", None),
        "state_id": getattr(t, "state_id", None),
        "zone_name": zone_name,
        "zone_address": zone_address,
        "graphs": graphs,
    }


def _make_composite_graph(
    graph_title: str,
    key: str,
    data,
    label: str,
    *,
    value_field,
    stream_type,
    name: Optional[str] = None,
    include_arrows: bool = True,
    decolour: bool = False,
):
    temperatures = _column_to_list(data, PT.T)
    fields = _normalise_graph_fields(value_field)
    stream_types = _normalise_graph_values(
        stream_type,
        len(fields),
        "`value_field` and `stream_type` must have the same length.",
    )
    segments: List[dict] = []
    for field, stream_loc in zip(fields, stream_types):
        column_key = _column_key(field)
        x_vals = _column_to_list(data, column_key)
        if not _should_plot_series(x_vals):
            continue
        segments.extend(
            _graph_cc(
                key,
                stream_loc,
                temperatures,
                x_vals,
                column_key=column_key,
                include_arrows=include_arrows,
                decolour=decolour,
            )
        )
    return {
        "type": key,
        "name": name or f"{label}: {graph_title}",
        "segments": segments,
    }


def _normalise_graph_fields(value_field) -> List:
    if isinstance(value_field, str) or not hasattr(value_field, "__iter__"):
        return [value_field]
    return list(value_field)


def _normalise_graph_values(value, count: int, mismatch_message: str) -> List:
    if isinstance(value, str) or not hasattr(value, "__iter__"):
        values = [value] * count
    else:
        values = list(value)
    if len(values) != count:
        raise ValueError(mismatch_message)
    return values


def _normalise_gcc_flags(flag, count: int) -> List[bool]:
    flags = _normalise_graph_values(
        flag if flag is not None else False,
        count,
        "`value_field` and `is_utility_profile` must have the same length.",
    )
    try:
        return [bool(item) for item in flags]
    except TypeError as exc:
        raise ValueError(
            "`is_utility_profile` values must be coercible to bool."
        ) from exc


def _column_key(field) -> str:
    return getattr(field, "value", field)


def _series_meta_from_key(column_key: str) -> GraphSeriesMeta:
    meta = GRAPH_SERIES_META.get(column_key)
    if meta is not None:
        return meta
    label = str(column_key)
    return GraphSeriesMeta(label, label)


def _build_gcc_segments(
    y_vals: Iterable[float],
    x_vals: Iterable[float],
    *,
    series_id: str,
    meta: GraphSeriesMeta,
    is_utility_profile: bool,
    decolour: bool,
) -> List[dict]:

    y_vals, x_vals = clean_composite_curve(y_vals, x_vals)
    counts: dict[StreamLoc, int] = defaultdict(int)
    segments: List[dict] = []
    for stream_loc, is_vertical, x_seg, y_seg in _iter_gcc_segment_slices(
        x_vals,
        y_vals,
        is_utility_profile,
        meta.preferred_stream_loc,
    ):
        counts[stream_loc] += 1
        title = _format_segment_title(meta, counts[stream_loc])
        colour = (
            LineColour.Black.value
            if is_vertical or decolour
            else _streamloc_colour(stream_loc)
        )
        segments.append(
            _create_curve(
                title=title,
                colour=colour,
                x_vals=x_seg,
                y_vals=y_seg,
                arrow=ArrowHead.NO_ARROW.value,
                series_label=meta.label,
                series_id=series_id,
                series_description=meta.description,
                is_vertical=is_vertical,
                is_utility_stream=stream_loc in {StreamLoc.HotU, StreamLoc.ColdU},
            )
        )
    return segments


def _iter_gcc_segment_slices(
    x_vals: List[float],
    y_vals: List[float],
    is_utility_profile: bool,
    preferred_stream_loc: Optional[StreamLoc],
):
    start, end = _segment_bounds(x_vals)
    j = start
    while j < end:
        classified = _classify_segment(x_vals[j] - x_vals[j + 1], is_utility_profile)
        next_j = j + 1
        while next_j < end:
            next_class = _classify_segment(
                x_vals[next_j] - x_vals[next_j + 1], is_utility_profile
            )
            if next_class != classified:
                break
            next_j += 1
        raw_loc = _segment_streamloc(classified)
        is_vertical = raw_loc == StreamLoc.Unassigned
        stream_loc = (
            preferred_stream_loc if is_vertical and preferred_stream_loc else raw_loc
        )
        yield stream_loc, is_vertical, x_vals[j : next_j + 1], y_vals[j : next_j + 1]
        j = next_j


def _segment_bounds(x_vals: List[float]) -> Tuple[int, int]:
    start = next(
        (i for i in range(len(x_vals) - 1) if abs(x_vals[i] - x_vals[i + 1]) > tol), 0
    )
    end = next(
        (
            i
            for i in range(len(x_vals) - 1, 0, -1)
            if abs(x_vals[i] - x_vals[i - 1]) > tol
        ),
        len(x_vals) - 1,
    )
    return start, end


def _format_segment_title(meta: GraphSeriesMeta, index: int) -> str:
    base = meta.description or meta.label or "Segment"
    return f"{base} {index}"


def _make_gcc_graph(
    graph_title: str,
    key: str,
    data,
    label: str,
    *,
    value_field,
    name: Optional[str] = None,
    is_utility_profile: bool = False,
    decolour: bool = False,
):
    temperatures = _column_to_list(data, PT.T)
    fields = _normalise_graph_fields(value_field)
    flags = _normalise_gcc_flags(is_utility_profile, len(fields))
    segments: List[dict] = []
    for field, utility_flag in zip(fields, flags):
        column_key = _column_key(field)
        x_vals = _column_to_list(data, column_key)
        if not _should_plot_series(x_vals):
            continue
        meta = _series_meta_from_key(column_key)
        segments.extend(
            _build_gcc_segments(
                temperatures,
                x_vals,
                series_id=f"{key}:{column_key}",
                meta=meta,
                is_utility_profile=utility_flag,
                decolour=decolour,
            )
        )

    return {
        "type": key,
        "name": name or f"{label}: {graph_title}",
        "segments": segments,
    }


def _graph_cc(
    key: str,
    stream_loc,
    y_vals: List[float],
    x_vals: List[float],
    *,
    column_key: Optional[str] = None,
    include_arrows: bool = True,
    decolour: bool = False,
) -> List[dict]:
    """Plots a (shifted) hot or cold composite curve."""

    # Clean composite
    y_vals, x_vals = clean_composite_curve(y_vals, x_vals)

    if not isinstance(stream_loc, StreamLoc):
        candidate = getattr(stream_loc, "value", stream_loc)
        try:
            stream_loc = StreamLoc(candidate)
        except ValueError:
            aliases = {
                "Hot": StreamLoc.HotS,
                "Cold": StreamLoc.ColdS,
            }
            if candidate in aliases:
                stream_loc = aliases[candidate]
            else:
                raise ValueError(
                    "Unrecognised composite curve stream location."
                ) from None

    title_map = {
        StreamLoc.HotS: "Hot CC",
        StreamLoc.ColdS: "Cold CC",
        StreamLoc.HotU: "Hot Utility",
        StreamLoc.ColdU: "Cold Utility",
    }

    if key not in [GT.TSP.value]:
        arrow_map = {
            StreamLoc.HotS: ArrowHead.END.value,
            StreamLoc.HotU: ArrowHead.END.value,
            StreamLoc.ColdS: ArrowHead.START.value,
            StreamLoc.ColdU: ArrowHead.START.value,
        }
    else:
        arrow_map = {
            StreamLoc.HotS: ArrowHead.START.value,
            StreamLoc.HotU: ArrowHead.START.value,
            StreamLoc.ColdS: ArrowHead.END.value,
            StreamLoc.ColdU: ArrowHead.END.value,
        }

    if stream_loc not in title_map:
        raise ValueError("Unrecognised composite curve stream location.")

    base_colour = _streamloc_colour(stream_loc)
    colour = LineColour.Black.value if decolour else base_colour
    arrow = arrow_map[stream_loc] if include_arrows else ArrowHead.NO_ARROW.value
    meta = _composite_series_meta(stream_loc, column_key)

    return [
        _create_curve(
            title=meta.composite_title,
            colour=colour,
            arrow=arrow,
            x_vals=x_vals,
            y_vals=y_vals,
            series_label=meta.label if column_key is not None else None,
            series_id=f"{key}:{column_key}" if column_key is not None else None,
            series_description=meta.description if column_key is not None else None,
        )
    ]


def _composite_series_meta(
    stream_loc: StreamLoc,
    column_key: Optional[str],
) -> GraphSeriesMeta:
    """Return the display metadata for one Composite Curve series."""
    if column_key is not None:
        meta = _series_meta_from_key(column_key)
        if meta.composite_title is not None:
            return meta
    title_map = {
        StreamLoc.HotS: "Hot CC",
        StreamLoc.ColdS: "Cold CC",
        StreamLoc.HotU: "Hot Utility",
        StreamLoc.ColdU: "Cold Utility",
    }
    default_title = title_map[stream_loc]
    return GraphSeriesMeta(
        label=default_title,
        description=default_title,
        composite_title=default_title,
    )


def _should_plot_series(values: List[float]) -> bool:
    """Return ``True`` when a graph series contains meaningful non-zero values."""
    try:
        numeric = np.asarray(values, dtype=float)
    except TypeError, ValueError:
        numeric = np.array(
            [float(value) for value in values if value is not None],
            dtype=float,
        )
    finite = numeric[np.isfinite(numeric)]
    if finite.size == 0:
        return False
    return bool(np.any(np.abs(finite) > tol))


def _column_to_list(data, column_key: str) -> List[float]:
    """Return the requested column from ``data`` as a Python list."""
    if not isinstance(column_key, str):
        column_key = getattr(column_key, "value", column_key)

    try:
        if isinstance(data, ProblemTable):
            column = data[column_key]
        elif (
            hasattr(data, "col")
            and hasattr(data, "columns")
            and column_key in getattr(data, "columns", [])
        ):
            column = data.col[column_key]
        elif isinstance(data, dict):
            column = data[column_key]
        else:
            column = data[column_key]
    except (KeyError, AttributeError, TypeError) as exc:
        raise KeyError(
            f"Column '{column_key}' not found in graph data payload."
        ) from exc

    if hasattr(column, "to_list"):
        return column.to_list()
    if hasattr(column, "tolist"):
        return column.tolist()
    return list(column)


def _classify_segment(enthalpy_diff: float, is_utility_profile: bool) -> str:
    if abs(enthalpy_diff) <= GCC_VERTICAL_TOL:
        return StreamLoc.Unassigned
    if enthalpy_diff > 0:
        return StreamLoc.ColdS if not is_utility_profile else StreamLoc.HotU
    if enthalpy_diff < 0:
        return StreamLoc.HotS if not is_utility_profile else StreamLoc.ColdU
    return StreamLoc.Unassigned


def _segment_streamloc(segment_type: str) -> StreamLoc:
    """Map a segment classification to a :class:`StreamLoc`."""
    if isinstance(segment_type, StreamLoc):
        return segment_type
    if segment_type == StreamLoc.ColdS.value:
        return StreamLoc.ColdS
    if segment_type == StreamLoc.HotS.value:
        return StreamLoc.HotS
    if segment_type == StreamLoc.HotU.value:
        return StreamLoc.HotU
    if segment_type == StreamLoc.ColdU.value:
        return StreamLoc.ColdU
    return StreamLoc.Unassigned


def _streamloc_colour(stream_loc: StreamLoc) -> int:
    """Return the default colour for a given stream location."""
    if stream_loc == StreamLoc.HotS:
        return LineColour.HotS.value
    if stream_loc == StreamLoc.ColdS:
        return LineColour.ColdS.value
    if stream_loc == StreamLoc.HotU:
        return LineColour.HotU.value
    if stream_loc == StreamLoc.ColdU:
        return LineColour.ColdU.value
    return LineColour.Other.value


def _create_curve(
    title: str,
    colour: int,
    x_vals,
    y_vals,
    arrow=ArrowHead.NO_ARROW.value,
    series_label: Optional[str] = None,
    series_id: Optional[str] = None,
    series_description: Optional[str] = None,
    is_vertical: Optional[bool] = None,
    is_utility_stream: Optional[bool] = None,
) -> dict:
    """Creates an individual curve from data points."""
    curve = {"title": title, "colour": colour, "arrow": arrow}
    if series_label is not None:
        curve["series"] = series_label
    if series_id is not None:
        curve["series_id"] = series_id
    if series_description is not None:
        curve["series_description"] = series_description
    if is_vertical is not None:
        curve["is_vertical"] = bool(is_vertical)
    if is_utility_stream is not None:
        curve["is_utility_stream"] = bool(is_utility_stream)
    curve["data_points"] = [
        {"x": round(x, DECIMAL_PLACES), "y": round(y, DECIMAL_PLACES)}
        for x, y in zip(x_vals, y_vals)
        if x is not None and y is not None
    ]
    return curve
