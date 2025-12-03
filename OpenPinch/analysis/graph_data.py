"""Graph construction helpers for composite curves and related plots."""

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, List, Optional, Tuple

from ..classes import *
from ..lib import *
from ..utils import *

DECIMAL_PLACES = 2
GCC_VERTICAL_TOL = 1e-3


@dataclass(frozen=True)
class GCCSeriesMeta:
    label: str
    description: str
    preferred_stream_loc: Optional[StreamLoc] = None


GCC_SERIES_LEGEND: dict[str, GCCSeriesMeta] = {
    PT.H_NET.value: GCCSeriesMeta(LegendSeries.GCC.name, LegendSeries.GCC.value),
    PT.H_NET_NP.value: GCCSeriesMeta(LegendSeries.GCC_N.name, LegendSeries.GCC_N.value),
    PT.H_NET_V.value: GCCSeriesMeta(LegendSeries.GCC_V.name, LegendSeries.GCC_V.value),
    PT.H_NET_A.value: GCCSeriesMeta(LegendSeries.GCC_A.name, LegendSeries.GCC_A.value),
    PT.H_NET_UT.value: GCCSeriesMeta(
        LegendSeries.GCC_U.name,
        LegendSeries.GCC_U.value,
        StreamLoc.HotU,
    ),
}

__all__ = ["get_output_graph_data", "visualise_graphs"]


#######################################################################################################
# Public API
#######################################################################################################


def get_output_graph_data(zone: Zone, graph_sets: dict = {}) -> dict:
    """Returns Json data points for each process."""
    for key, t in zone.targets.items():
        graph_sets[key] = _create_graph_set(t, key)

    if len(zone.subzones) > 0:
        for z in zone.subzones.values():
            graph_sets = get_output_graph_data(z, graph_sets)

    return graph_sets


def visualise_graphs(graph_set: dict, graph) -> None:
    """Adds a graph to the graph_set based on its type."""
    graph_data = graph.data
    graph_type = graph.type

    match graph_type:
        case GT.CC.value | GT.SCC.value | GT.BCC.value:
            if graph_type == GT.BCC.value:
                col_keys = [PT.H_HOT_BAL.value, PT.H_COLD_BAL.value]
                stream_types = [StreamLoc.HotS, StreamLoc.ColdS]
                include_arrows = False
            else:
                col_keys = [PT.H_HOT.value, PT.H_COLD.value]
                stream_types = [StreamLoc.HotS, StreamLoc.ColdS]
                include_arrows = True

            graph_set["graphs"].append(
                _make_composite_graph(
                    graph_title="Graph",
                    key=graph_type,
                    data=graph_data,
                    label=graph_type,
                    name=f"{graph_type} Graph",
                    col_keys=col_keys,
                    stream_types=stream_types,
                    include_arrows=include_arrows,
                )
            )

        case GT.GCC.value:
            graph_set["graphs"].append(
                _make_gcc_graph(
                    graph_title="Graph",
                    key=graph_type,
                    data=graph_data,
                    label=graph_type,
                    name=f"{graph_type} Graph",
                    value_field=[PT.H_NET.value, PT.H_NET_NP.value, PT.H_NET_V.value, PT.H_NET_A.value, PT.H_NET_UT.value],
                    is_utility_profile=[False, False, False, False, True],
                )
            )


#######################################################################################################
# Helper Functions
#######################################################################################################


def _create_graph_set(t: EnergyTarget, graphTitle: str) -> dict:
    """Creates Pinch Analysis and total site analysis graphs for a specifc zone."""

    graphs: List[dict] = []

    if GT.CC.value in t.graphs:
        graphs.append(
            _make_composite_graph(
                graph_title=graphTitle,
                key=GT.CC.value,
                data=t.graphs[GT.CC.value],
                col_keys=[PT.H_HOT.value, PT.H_COLD.value],
                stream_types=[StreamLoc.HotS, StreamLoc.ColdS],
                label="Composite Curve",
            )
        )

    if GT.SCC.value in t.graphs:
        graphs.append(
            _make_composite_graph(
                graph_title=graphTitle,
                key=GT.SCC.value,
                data=t.graphs[GT.SCC.value],
                col_keys=[PT.H_HOT.value, PT.H_COLD.value],
                stream_types=[StreamLoc.HotS, StreamLoc.ColdS],
                label="Shifted Composite Curve",
            )
        )

    if GT.BCC.value in t.graphs:
        graphs.append(
            _make_composite_graph(
                graph_title=graphTitle,
                key=GT.BCC.value,
                data=t.graphs[GT.BCC.value],
                col_keys=[PT.H_HOT_BAL.value, PT.H_COLD_BAL.value],
                stream_types=[StreamLoc.HotS, StreamLoc.ColdS],
                label="Balanced Composite Curve",
                include_arrows=True,
            )
        )

    if GT.GCC.value in t.graphs:
        graphs.append(
            _make_gcc_graph(
                graph_title=graphTitle,
                key=GT.GCC.value,
                data=t.graphs[GT.GCC.value],
                label="Grand Composite Curve",
                value_field=[PT.H_NET.value, PT.H_NET_NP.value, PT.H_NET_V.value, PT.H_NET_A.value, PT.H_NET_UT.value],
                is_utility_profile=[False, False, False, False, True],
            )
        )

    if GT.TSP.value in t.graphs:
        graphs.append(
            _make_composite_graph(
                graph_title=graphTitle,
                key=GT.TSP.value,
                data=t.graphs[GT.TSP.value],
                col_keys=[PT.H_NET_HOT.value, PT.H_NET_COLD.value, PT.H_HOT_UT.value, PT.H_COLD_UT.value],
                stream_types=[StreamLoc.HotS, StreamLoc.ColdS, StreamLoc.HotU, StreamLoc.ColdU],
                label="Total Site Profiles",
                include_arrows=True,
            )
        )

    if GT.SUGCC.value in t.graphs:
        graphs.append(
            _make_gcc_graph(
                graph_title=graphTitle,
                key=GT.SUGCC.value,
                data=t.graphs[GT.SUGCC.value],
                label="Site Utility Grand Composite Curve",
                value_field=[PT.H_NET_UT.value],
                is_utility_profile=[True],
            )
        )
    
    if GT.GCC_HP.value in t.graphs:
        graphs.append(
            _make_gcc_graph(
                graph_title=graphTitle,
                key=GT.GCC_HP.value,
                data=t.graphs[GT.GCC_HP.value],
                label="Grand Composite Curve with Heat Pump",
                value_field=[PT.H_NET_W_AIR.value, PT.H_NET_HP_PRO.value],
                is_utility_profile=[False, True],
            )
        )

    return {"name": graphTitle, "graphs": graphs}


def _make_composite_graph(
    graph_title: str,
    key: str,
    data,
    label: str,
    col_keys: list,
    stream_types: list,
    *,
    name: Optional[str] = None,
    include_arrows: bool = True,
    decolour: bool = False,
):
    temperatures = _column_to_list(data, PT.T.value)
    segments: List[dict] = []
    for stream_loc, col_key in zip(stream_types, col_keys):
        segments.extend(
            _graph_cc(
                key,
                stream_loc,
                temperatures,
                _column_to_list(data, col_key),
                include_arrows=include_arrows,
                decolour=decolour,
            )
        )
    return {
        "type": key,
        "name": name or f"{label}: {graph_title}",
        "segments": segments,
    }


def _normalise_gcc_fields(value_field) -> List:
    if isinstance(value_field, str) or not hasattr(value_field, "__iter__"):
        return [value_field]
    return list(value_field)


def _normalise_gcc_flags(flag, count: int) -> List[bool]:
    if isinstance(flag, bool) or flag is None:
        return [bool(flag)] * count
    flags = list(flag)
    if len(flags) != count:
        raise ValueError("`value_field` and `is_utility_profile` must have the same length.")
    return [bool(item) for item in flags]


def _column_key(field) -> str:
    return getattr(field, "value", field)


def _series_meta_from_key(column_key: str) -> GCCSeriesMeta:
    meta = GCC_SERIES_LEGEND.get(column_key)
    if meta is not None:
        return meta
    label = str(column_key)
    return GCCSeriesMeta(label, label)


def _build_gcc_segments(
    y_vals: Iterable[float],
    x_vals: Iterable[float],
    *,
    series_id: str,
    meta: GCCSeriesMeta,
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
        colour = LineColour.Black.value if decolour else _streamloc_colour(stream_loc)
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
            next_class = _classify_segment(x_vals[next_j] - x_vals[next_j + 1], is_utility_profile)
            if next_class != classified:
                break
            next_j += 1
        raw_loc = _segment_streamloc(classified)
        is_vertical = raw_loc == StreamLoc.Unassigned
        stream_loc = preferred_stream_loc if is_vertical and preferred_stream_loc else raw_loc
        yield stream_loc, is_vertical, x_vals[j : next_j + 1], y_vals[j : next_j + 1]
        j = next_j


def _segment_bounds(x_vals: List[float]) -> Tuple[int, int]:
    start = next((i for i in range(len(x_vals) - 1) if abs(x_vals[i] - x_vals[i + 1]) > tol), 0)
    end = next(
        (i for i in range(len(x_vals) - 1, 0, -1) if abs(x_vals[i] - x_vals[i - 1]) > tol),
        len(x_vals) - 1,
    )
    return start, end


def _format_segment_title(meta: GCCSeriesMeta, index: int) -> str:
    base = meta.description or meta.label or "Segment"
    return f"{base} {index}"


def _graph_gcc(
    y_vals: Iterable[float],
    x_vals: Iterable[float],
    *,
    series_label: str = "GCC",
    series_id: str = "GCC",
    series_description: Optional[str] = None,
    preferred_stream_loc: Optional[StreamLoc] = None,
    is_utility_profile: bool = False,
    decolour: bool = False,
) -> List[dict]:
    """Backward-compatible wrapper retained for tests and legacy callers."""
    meta = GCCSeriesMeta(
        label=series_label,
        description=series_description or series_label,
        preferred_stream_loc=preferred_stream_loc,
    )
    return _build_gcc_segments(
        list(y_vals),
        list(x_vals),
        series_id=series_id,
        meta=meta,
        is_utility_profile=is_utility_profile,
        decolour=decolour,
    )


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
    temperatures = _column_to_list(data, PT.T.value)
    fields = _normalise_gcc_fields(value_field)
    flags = _normalise_gcc_flags(is_utility_profile, len(fields))
    segments: List[dict] = []
    for field, utility_flag in zip(fields, flags):
        column_key = _column_key(field)
        meta = _series_meta_from_key(column_key)
        segments.extend(
            _build_gcc_segments(
                temperatures,
                _column_to_list(data, column_key),
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
                raise ValueError("Unrecognised composite curve stream location.") from None

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

    return [
        _create_curve(
            title=title_map[stream_loc],
            colour=colour,
            arrow=arrow,
            x_vals=x_vals,
            y_vals=y_vals,
        )
    ]


def _column_to_list(data, column_key: str) -> List[float]:
    """Return the requested column from ``data`` as a Python list."""
    if not isinstance(column_key, str):
        column_key = getattr(column_key, "value", column_key)

    try:
        if isinstance(data, ProblemTable):
            column = data.col[column_key]
        elif hasattr(data, "col") and hasattr(data, "columns") and column_key in getattr(
            data, "columns", []
        ):
            column = data.col[column_key]
        elif isinstance(data, dict):
            column = data[column_key]
        else:
            column = data[column_key]
    except (KeyError, AttributeError, TypeError) as exc:
        raise KeyError(f"Column '{column_key}' not found in graph data payload.") from exc

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
        if x != None and y != None
    ]
    return curve


# def Graph_ETD(site: Zone, ETD, ETD_header, zone_name, graph, IncRecoveryHX=True, HCC=None):
#     """Graphs an ETD.
#     """

#     Tot_HX = [0 for i in range(4)]
#     x_seg = [0, 0]
#     y_seg = [0, 0]

#     x_col = len(ETD) - 2
#     y_col = 0

#     j_0 = 1
#     j_1 = len(ETD[0])

#     x_val = [0] * (j_1 - j_0 + 1)
#     y_val = [0] * (j_1 - j_0 + 1)

#     for j in range(j_0, j_1):
#         y_val[j - j_0] = ETD[y_col][j]

#     for i in range(x_col, 0, -3):
#         if not (ETD[i + 1][0] == 'R' and not IncRecoveryHX):
#             for j in range(j_0, j_1 - 1):
#                 x_val[j] = ETD[i][j] + x_val[j]
#         if ETD[i + 1][0] == 'H':
#             Tot_HX[1] += 1
#         elif ETD[i + 1][0] == 'R':
#             Tot_HX[2] += 1
#         elif ETD[i + 1][0] == 'C':
#             Tot_HX[3] += 1
#     Tot_HX[0] = Tot_HX[1] + Tot_HX[2] + Tot_HX[3]

#     if site.config.AUTOREORDER:
#         if site.config.AUTOREORDER_1:
#             stat_row = 1
#         elif site.config.AUTOREORDER_2:
#             stat_row = 2
#         elif site.config.AUTOREORDER_3:
#             stat_row = 3
#         else:
#             stat_row = 4
#     else:
#         stat_row = 4

#     Graph_ETC(ETD, ETD_header, graph, zone_name, stat_row, y_col, x_val, y_val, 'R', IncRecoveryHX, Tot_HX[1])
#     Graph_ETC(ETD, ETD_header, graph, zone_name, stat_row, y_col, x_val, y_val, 'C', IncRecoveryHX, Tot_HX[2])
#     Graph_ETC(ETD, ETD_header, graph, zone_name, stat_row, y_col, x_val, y_val, 'H', IncRecoveryHX, Tot_HX[0])

#     if zone_name[-3:] == 'ACC':
#         _graph_cc(HCC, 4, 7, 0, graph, False, True)

#     site_ret = list(filter(lambda p: p.name == TargetType.ET.value, site.subzones))[0]
#     x_seg[1] = site_ret.cold_utility_target
#     y_seg[0] = ETD[1][len(ETD[1]) - 1]
#     y_seg[1] = ETD[1][len(ETD[1]) - 1]

#     graph['segments'].append(_create_curve(
#         title='Cold Ut Segment',
#         colour=LineColour.Cold.value,
#         x_vals= x_seg,
#         y_vals= y_seg
#     ))

#     y_seg[0] = ETD[0][1]
#     y_seg[1] = ETD[0][1]
#     if zone_name[-3:] != 'ACC':
#         x_seg[1] = site_ret.hot_utility_target
#     else:
#         x_seg[0] = site_ret.cold_utility_target + site_ret.heat_recovery_target
#         x_seg[1] = site_ret.hot_utility_target + x_seg[1]

#     graph['segments'].append(_create_curve(
#         title='Hot Ut Segment',
#         colour=LineColour.Hot.value,
#         x_vals=x_seg,
#         y_vals=y_seg
#     ))

#     return graph


# def Graph_ETC(ETD, ETD_header, graph, zone_name, stat_row, y_col, x_val_base, y_val, HX_Type, IncRecoveryHX, HX_countdown):
#     if HX_countdown == 0:
#         return
#     prev_points = [None, None]
#     x_col = len(ETD) - 2

#     di = 3 if zone_name[-3:] == 'ACC' else 0
#     min_val = (10) ** 12
#     for i in range(len(ETD) - 2, di, -3):
#         if min_val > ETD_header[i - di][stat_row] and ETD[i + 1][0] == HX_Type:
#             min_val = ETD_header[i - di][stat_row]
#             x_col = i - di

#     if x_col <= len(ETD_header) - 1:
#         ETD_header[x_col][stat_row] = (10) ** 12

#     if ETD[x_col + 1][0] == HX_Type and (HX_Type == 'H' or HX_Type == 'C' or (HX_Type == 'R' and IncRecoveryHX)):

#         j_0 = 1
#         j_1 = len(ETD[1]) - 1

#         if HX_Type == 'R' or HX_Type == 'C':
#             for j_0 in range(j_0, len(ETD[0])):
#                 if abs(ETD[x_col - 1][j_0 + 1]) > tol:
#                     break

#         if HX_Type == 'R' or HX_Type == 'H':
#             for j_1 in range(j_1, 1, -1):
#                 if abs(ETD[x_col - 1][j_1]) > tol:
#                     break

#         j = j_0
#         while j < j_1:
#             tmp = ETD[x_col - 1][j + 1]
#             if tmp > tol:
#                 # Find process heat deficit segments and utility heat supply segments
#                 seg_type = 1
#             elif tmp < -tol:
#                 # Find process heat surplus segments and utility heat sink segments
#                 seg_type = 2
#             else:
#                 # Zero heat transfer
#                 seg_type = 3

#             for j in range(j + 1, j_1 - 1):
#                 if seg_type == 1:
#                     if ETD[x_col - 1][j + 1] < tol:
#                         break
#                 elif seg_type == 2:
#                     if ETD[x_col - 1][j + 1] > -tol:
#                         break
#                 elif seg_type == 3:
#                     if abs(ETD[x_col - 1][j + 1]) > tol:
#                         break

#             j_i = j
#             x_seg = [None] * (j_i - j_0 + 3)
#             y_seg = [None] * (j_i - j_0 + 3)

#             for j in range(j_0, j_i + 3):
#                 x_seg[j - j_0] = x_val_base[j]
#                 y_seg[j - j_0] = ETD[y_col][j]

#             j += 1
#             j_0 = j

#             if prev_points[0] is not None:
#                 x_seg.insert(0, prev_points[0])
#                 y_seg.insert(0, prev_points[1])
#                 prev_points[0] = x_seg[-1]
#                 prev_points[1] = y_seg[-1]
#             else:
#                 prev_points[0] = x_seg[-1]
#                 prev_points[1] = y_seg[-1]

#             if seg_type == 1:
#                 graph['segments'].append(_create_curve(
#                     title='Cold Segment',
#                     colour=LineColour.Cold.value,
#                     x_vals=x_seg,
#                     y_vals=y_seg
#                 ))
#             elif seg_type == 2:
#                 graph['segments'].append(_create_curve(
#                     title='Hot Segment',
#                     colour=LineColour.Hot.value,
#                     x_vals=x_seg,
#                     y_vals=y_seg
#                 ))
#             else:
#                 graph['segments'].append(_create_curve(
#                     title='Zero Segment',
#                     colour=LineColour.Other.value,
#                     x_vals=x_seg,
#                     y_vals=y_seg
#                 ))

#         x_val = [0.0 for i in range(len(ETD[0]) - 1)]
#         for j in range(1, len(ETD[0]) - 1):
#             x_val[j] = x_val_base[j] - ETD[x_col][j]
#     else:
#         x_val = x_val_base

#     if HX_countdown > 1 + tol:
#         Graph_ETC(ETD, ETD_header, graph, zone_name, stat_row, y_col, x_val, y_val, HX_Type, IncRecoveryHX, HX_countdown - 1)
#     x_val_base = x_val

#     return graph


# def Create_ERC_Graph_Set(graph_set: dict, site: Zone, process: Zone) -> dict:
#     graph = {'segments': [], 'type': GT.ERC.value, 'name': 'Shifted ETD'}
#     graph_set['graphs'].append(Graph_ETD(site, process.graphs['ETD_star'], process.graphs['ETD_header'], 'Shifted ETD', graph))

#     graph = {'segments': [], 'type': GT.ERC.value, 'name': 'ETD'}
#     graph_set['graphs'].append(Graph_ETD(site, process.graphs['ETD'], process.graphs['ETD_header'], 'ETD', graph))

#     graph = {'segments': [], 'type': GT.ERC.value, 'name': 'Shifted ETD without Recovery'}
#     graph_set['graphs'].append(Graph_ETD(site, process.graphs['ETD_star'], process.graphs['ETD_header'], 'Shifted ETD without recovery', graph, False))

#     graph = {'segments': [], 'type': GT.ERC.value, 'name': 'ETD without Recovery'}
#     graph_set['graphs'].append(Graph_ETD(site, process.graphs['ETD'], process.graphs['ETD_header'], 'ETD without recovery', graph, False))

#     return graph_set
