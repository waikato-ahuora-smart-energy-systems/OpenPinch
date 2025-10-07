"""Graph construction helpers for composite curves and related plots."""

from typing import List, Optional, Tuple

from ..classes import *
from ..lib import *
from ..utils import *

DECIMAL_PLACES = 2

__all__ = ["get_output_graph_data", "visualise_graphs"]


#######################################################################################################
# Public API
#######################################################################################################

@timing_decorator
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
            graph_set["graphs"].append(
                _make_composite_graph(
                    graph_title="Graph",
                    key=graph_type,
                    data=graph_data,
                    label=graph_type,
                    name=f"{graph_type} Graph",
                )
            )

        case GT.GCC.value | GT.GCC_N.value | GT.GCCU.value:
            graph_set["graphs"].append(
                _make_gcc_graph(
                    graph_title="Graph",
                    key=graph_type,
                    data=graph_data,
                    label=graph_type,
                    name=f"{graph_type} Graph",
                    value_field=PT.H_NET.value,
                )
            )
            if graph_type == GT.GCCU.value:
                graph_set["graphs"].append(
                    _make_gcc_graph(
                        graph_title="Utility Graph",
                        key=f"{graph_type}_Utility",
                        data=graph_data,
                        label=f"{graph_type} Utility",
                        name=f"{graph_type} Utility Graph",
                        value_field=PT.H_NET.value,
                        utility_profile=True,
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
                include_arrows=False,
            )
        )

    if GT.GCC.value in t.graphs:
        graphs.append(
            _make_gcc_graph(
                graph_title=graphTitle,
                key=GT.GCC.value,
                data=t.graphs[GT.GCC.value],
                label="Grand Composite Curve",
                value_field=PT.H_NET.value,
            )
        )

    if GT.GCC_A.value in t.graphs:
        graphs.append(
            _make_dual_gcc_graph(
                graph_title=graphTitle,
                key=GT.GCC_A.value,
                data=t.graphs[GT.GCC_A.value],
            )
        )

    if GT.GCC_V.value in t.graphs:
        graphs.append(
            _make_gcc_graph(
                graph_title=graphTitle,
                key=GT.GCC_V.value,
                data=t.graphs[GT.GCC_V.value],
                label="Grand Composite Curve",
                value_field=PT.H_NET_V.value,
            )
        )

    if GT.TSP.value in t.graphs:
        graphs.append(
            _make_composite_graph(
                graph_title=graphTitle,
                key=GT.TSP.value,
                data=t.graphs[GT.TSP.value],
                col_keys=[PT.H_HOT_NET.value, PT.H_COLD_NET.value, PT.H_HOT_UT.value, PT.H_COLD_UT.value],
                stream_types=[StreamLoc.HotS, StreamLoc.ColdS, StreamLoc.HotU, StreamLoc.ColdU],
                label="Shifted Composite Curve",
            )
        )

    if GT.SUGCC.value in t.graphs:
        graphs.append(
            _make_gcc_graph(
                graph_title=graphTitle,
                key=GT.SUGCC.value,
                data=t.graphs[GT.SUGCC.value],
                label="Site Utility Grand Composite Curve",
                value_field=PT.H_UT_NET.value,
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
    temperatures = data[PT.T.value].to_list()
    segments: List[dict] = []
    for stream_loc, col_key in zip(stream_types, col_keys):
        segments.extend(
            _graph_cc(
                stream_loc,
                temperatures,
                data[col_key].to_list(),
                include_arrows=include_arrows,
                decolour=decolour,
            )
        )
    return {
        "type": key,
        "name": name or f"{label}: {graph_title}",
        "segments": segments,
    }


def _make_gcc_graph(
    graph_title: str,
    key: str,
    data,
    label: str,
    *,
    value_field: str,
    name: Optional[str] = None,
    utility_profile: bool = False,
    decolour: bool = False,
):
    segments = _graph_gcc(
        data[PT.T.value].to_list(),
        data[value_field].to_list(),
        utility_profile=utility_profile,
        decolour=decolour,
    )
    return {
        "type": key,
        "name": name or f"{label}: {graph_title}",
        "segments": segments,
    }


def _make_dual_gcc_graph(graph_title: str, key: str, data) -> dict:
    segments = []
    segments += _graph_gcc(
        data[PT.T.value].to_list(),
        data[PT.H_NET_A.value].to_list(),
    )
    segments += _graph_gcc(
        data[PT.T.value].to_list(),
        data[PT.H_UT_NET.value].to_list(),
    )
    return {
        "type": key,
        "name": f"Grand Composite Curve: {graph_title}",
        "segments": segments,
    }


def _graph_cc(
    stream_loc,
    y_vals: List[float],
    x_vals: List[float],
    *,
    include_arrows: bool = True,
    decolour: bool = False,
) -> List[dict]:
    """Plots a (shifted) hot or cold composite curve."""

    # Clean composite
    y_vals, x_vals = _clean_composite(y_vals, x_vals)

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
    colour_map = {
        StreamLoc.HotS: LineColour.HotS.value,
        StreamLoc.ColdS: LineColour.ColdS.value,
        StreamLoc.HotU: LineColour.HotU.value,
        StreamLoc.ColdU: LineColour.ColdU.value,
    }
    arrow_map = {
        StreamLoc.HotS: ArrowHead.END.value,
        StreamLoc.HotU: ArrowHead.END.value,
        StreamLoc.ColdS: ArrowHead.START.value,
        StreamLoc.ColdU: ArrowHead.START.value,
    }

    if stream_loc not in title_map:
        raise ValueError("Unrecognised composite curve stream location.")

    colour = LineColour.Black.value if decolour else colour_map[stream_loc]
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


def _graph_gcc(
    y_vals: List[float],
    x_vals: List[float],
    utility_profile: bool = False,
    decolour: bool = False,
) -> list:
    """Creates segments for a Grand Composite Curve."""

    # Clean composite
    y_vals, x_vals = _clean_composite(y_vals, x_vals)

    # Find start and end indices of useful data
    start_idx = next(
        (j for j in range(len(x_vals) - 1) if abs(x_vals[j] - x_vals[j + 1]) > tol), 0
    )
    end_idx = next(
        (
            j
            for j in range(len(x_vals) - 1, 0, -1)
            if abs(x_vals[j] - x_vals[j - 1]) > tol
        ),
        len(x_vals) - 1,
    )

    curves = []
    j = start_idx

    # Counters for segment naming
    cold_pro_segs, hot_pro_segs, hot_ut_segs, cold_ut_segs, zero_segs = 0, 0, 0, 0, 0

    while j < end_idx:
        segment_type = _classify_segment(
            x_vals[j] - x_vals[j + 1], utility_profile
        )

        next_j = j + 1
        while next_j < end_idx:
            next_type = _classify_segment(
                x_vals[next_j] - x_vals[next_j + 1], utility_profile
            )
            if next_type != segment_type:
                break
            next_j += 1

        x_seg = x_vals[j : next_j + 1]
        y_seg = y_vals[j : next_j + 1]

        title, colour, cold_pro_segs, hot_pro_segs, hot_ut_segs, cold_ut_segs, zero_segs = _segment_style(
            segment_type,
            cold_pro_segs,
            hot_pro_segs,
            hot_ut_segs,
            cold_ut_segs,
            zero_segs,
        )

        curves.append(
            _create_curve(
                title=title,
                colour=colour if not decolour else LineColour.Black.value,
                x_vals=x_seg,
                y_vals=y_seg,
                arrow=ArrowHead.NO_ARROW.value,
            )
        )

        j = next_j

    return curves


def _classify_segment(enthalpy_diff: float, utility_profile: bool) -> str:
    if enthalpy_diff > tol:
        return "cold_pro" if not utility_profile else "hot_ut"
    if enthalpy_diff < -tol:
        return "hot_pro" if not utility_profile else "cold_ut"
    return "zero"


def _segment_style(
    segment_type: str,
    cold_pro_segs: int,
    hot_pro_segs: int,
    hot_ut_segs: int,
    cold_ut_segs: int,
    zero_segs: int,
) -> Tuple[str, int, int, int, int, int, int]:
    if segment_type == "cold_pro":
        cold_pro_segs += 1
        return (
            f"Cold Process Segment {cold_pro_segs}",
            LineColour.ColdS.value,
            cold_pro_segs,
            hot_pro_segs,
            hot_ut_segs,
            cold_ut_segs,
            zero_segs,
        )
    if segment_type == "hot_pro":
        hot_pro_segs += 1
        return (
            f"Hot Process Segment {hot_pro_segs}",
            LineColour.HotS.value,
            cold_pro_segs,
            hot_pro_segs,
            hot_ut_segs,
            cold_ut_segs,
            zero_segs,
        )
    if segment_type == "hot_ut":
        hot_ut_segs += 1
        return (
            f"Hot Utility Segment {hot_ut_segs}",
            LineColour.HotU.value,
            cold_pro_segs,
            hot_pro_segs,
            hot_ut_segs,
            cold_ut_segs,
            zero_segs,
        )
    if segment_type == "cold_ut":
        cold_ut_segs += 1
        return (
            f"Cold Utility Segment {cold_ut_segs}",
            LineColour.ColdU.value,
            cold_pro_segs,
            hot_pro_segs,
            hot_ut_segs,
            cold_ut_segs,
            zero_segs,
        )
    zero_segs += 1
    return (
        f"Vertical Segment {zero_segs}",
        LineColour.Other.value,
        cold_pro_segs,
        hot_pro_segs,
        hot_ut_segs,
        cold_ut_segs,
        zero_segs,
    )


def _clean_composite(
    y_vals: List[float], x_vals: List[float]
) -> Tuple[List[float], List[float]]:
    """Remove redundant points in composite curves."""

    # Round to avoid tiny numerical errors
    x_vals = [round(x, 5) for x in x_vals]
    y_vals = [round(y, 5) for y in y_vals]

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

    return y_clean, x_clean


def _create_curve(
    title: str, colour: int, x_vals, y_vals, arrow=ArrowHead.NO_ARROW.value
) -> dict:
    """Creates an individual curve from data points."""
    curve = {"title": title, "colour": colour, "arrow": arrow}
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
