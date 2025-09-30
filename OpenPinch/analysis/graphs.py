from typing import List, Tuple
from ..lib.enums import *
from ..utils import *
from .support_methods import *
from ..classes import *

DECIMAL_PLACES = 2

__all__ = ["get_output_graphs, visualise_graphs"]


#######################################################################################################
# Public API
#######################################################################################################

def get_output_graphs(zone: Zone, graph_sets: Dict = {}) -> Dict:
    """Returns Json data points for each process."""
    for key, t in zone.targets.items():
        graph_sets[key] = _create_graph_set(t, key)

    if len(zone.subzones) > 0:
        for z in zone.subzones.values():
            graph_sets = get_output_graphs(z, graph_sets)

    return graph_sets


def visualise_graphs(graph_set: dict, graph) -> None:
    """Adds a graph to the graph_set based on its type."""
    graph_data = graph.data
    graph_type = graph.type

    match graph_type:
        case GT.CC.value | GT.SCC.value:
            curves = [
                _graph_cc(StreamType.Hot.value, graph_data[PT.T.value].to_list(), graph_data[PT.H_HOT.value].to_list()),
                _graph_cc(StreamType.Cold.value, graph_data[PT.T.value].to_list(), graph_data[PT.H_COLD.value].to_list())
            ]
            graph_set['graphs'].append({
                'type': graph_type,
                'name': f'{graph_type} Graph',
                'segments': curves[0] + curves[1]
            })

        case GT.BCC.value:
            curves = [
                _graph_cc(StreamType.Hot.value, graph_data[PT.T.value].to_list(), graph_data[PT.H_HOT.value].to_list(), IncludeArrows=False),
                _graph_cc(StreamType.Cold.value, graph_data[PT.T.value].to_list(), graph_data[PT.H_COLD.value].to_list(), IncludeArrows=False)
            ]
            graph_set['graphs'].append({
                'type': graph_type,
                'name': f'{graph_type} Graph',
                'segments': curves[0] + curves[1]
            })

        case GT.GCC.value | GT.GCC_NP.value | GT.GCCU.value:
            segments = _graph_gcc(graph_data[PT.H_NET.value].to_list(), graph_data[PT.T.value].to_list())
            graph_set['graphs'].append({
                'type': graph_type,
                'name': f'{graph_type} Graph',
                'segments': segments
            })
            if graph_type == GT.GCCU.value:
                # Add second set with utility profile style
                segments_ut = _graph_gcc(graph_data[PT.H_NET.value].to_list(), graph_data[PT.T.value].to_list(), utility_profile=True)
                graph_set['graphs'].append({
                    'type': f'{graph_type}_Utility',
                    'name': f'{graph_type} Utility Graph',
                    'segments': segments_ut
                })


#######################################################################################################
# Helper Functions
#######################################################################################################

def _create_graph_set(t: EnergyTarget, graphTitle: str) -> dict:
    """Creates Pinch Analysis and total site analysis graphs for a specifc zone."""
    
    graph_set = {
        'name': graphTitle,
        'graphs': []
    }

    # === Composite Curve ===
    if GT.CC.value in t.graphs:
        key = GT.CC.value
        g = {
            'type': key,
            'name': f'Composite Curve: {graphTitle}',
            'segments': 
                _graph_cc(
                    StreamType.Hot.value,
                    t.graphs[key][PT.T.value].to_list(),
                    t.graphs[key][PT.H_HOT.value].to_list(),
                ) +
                _graph_cc(
                    StreamType.Cold.value,
                    t.graphs[key][PT.T.value].to_list(),
                    t.graphs[key][PT.H_COLD.value].to_list(),
                ),
        }
        graph_set['graphs'].append(g)

    # === Shifted Composite Curve ===
    if GT.SCC.value in t.graphs:
        key = GT.SCC.value
        g = {
            'type': key,
            'name': f'Shifted Composite Curve: {graphTitle}',
            'segments': 
                _graph_cc(
                    StreamType.Hot.value,
                    t.graphs[key][PT.T.value].to_list(),
                    t.graphs[key][PT.H_HOT.value].to_list(),
                ) + 
                _graph_cc(
                    StreamType.Cold.value,
                    t.graphs[key][PT.T.value].to_list(),
                    t.graphs[key][PT.H_COLD.value].to_list(),
                ),
        }
        graph_set['graphs'].append(g)

    # === Balanced Composite Curve ===
    if GT.BCC.value in t.graphs:
        key = GT.BCC.value
        g = {
            'type': key,
            'name': f'Balanced Composite Curve: {graphTitle}',
            'segments':
                _graph_cc(
                    StreamType.Hot.value,
                    t.graphs[key][PT.T.value].to_list(),
                    t.graphs[key][PT.H_HOT.value].to_list(),
                ) +
                _graph_cc(
                    StreamType.Cold.value,
                    t.graphs[key][PT.T.value].to_list(),
                    t.graphs[key][PT.H_COLD.value].to_list(),
                ),
        }
        graph_set['graphs'].append(g)

    # === Grand Composite Curve (GCC) ===
    if GT.GCC.value in t.graphs:
        key = GT.GCC.value
        g = {
            'type': key,
            'name': f'Grand Composite Curve: {graphTitle}',
            'segments': 
                _graph_gcc(
                    t.graphs[key][PT.T.value].to_list(),
                    t.graphs[key][PT.H_NET.value].to_list(),
                ),
        }
        graph_set['graphs'].append(g)

    # === Grand Composite Curve with no pockets (GCC_Act and GCC_Ut_star) ===
    if GT.GCC_Act.value in t.graphs:
        key = GT.GCC_Act.value
        g = {
            'type': key,
            'name': f'Grand Composite Curve: {graphTitle}',
            'segments':
                _graph_gcc(
                    t.graphs[key][PT.T.value].to_list(),
                    t.graphs[key][PT.H_NET_A.value].to_list(),
                ) +
                _graph_gcc(
                    t.graphs[key][PT.T.value].to_list(),
                    t.graphs[key][PT.H_UT_NET.value].to_list(),
                ),
        }
        graph_set['graphs'].append(g)

    # === Grand Composite Curve with vertical CC heat transfer (GCC_Ex and GCC_Ut_star) ===
    if GT.GCC_Ex.value in t.graphs:
        key = GT.GCC_Ex.value
        g = {
            'type': key,
            'name': f'Grand Composite Curve: {graphTitle}',
            'segments':
                _graph_gcc(
                    t.graphs[key][PT.T.value].to_list(),
                    t.graphs[key][PT.H_NET_V.value].to_list(),
                ),
        }
        graph_set['graphs'].append(g)

    # === Total Site Profiles ===
    if GT.TSP.value in t.graphs:
        key = GT.TSP.value
        g = {
            'type': key,
            'name': f'Total Site Profiles: {graphTitle}',
            'segments': 
                _graph_cc(
                    StreamType.Hot.value,
                    t.graphs[key][PT.T.value].to_list(),
                    t.graphs[key][PT.H_HOT_NET.value].to_list(),
                ) +
                _graph_cc(
                    StreamType.Cold.value,
                    t.graphs[key][PT.T.value].to_list(),
                    t.graphs[key][PT.H_COLD_NET.value].to_list(),
                ) +
                _graph_cc(
                    StreamType.Cold.value,
                    t.graphs[key][PT.T.value].to_list(),
                    t.graphs[key][PT.H_COLD_UT.value].to_list(),
                ) +
                _graph_cc(
                    StreamType.Hot.value,
                    t.graphs[key][PT.T.value].to_list(),
                    t.graphs[key][PT.H_HOT_UT.value].to_list(),
                ),
        }
        graph_set['graphs'].append(g)

    # === Site Utility Grand Composite Curves ===
    if GT.SUGCC.value in t.graphs:
        key = GT.SUGCC.value
        g = {
            'type': key,
            'name': f'Site Utility Grand Composite Curve: {graphTitle}',
            'segments': 
                _graph_gcc(
                    t.graphs[key][PT.T.value].to_list(),
                    t.graphs[key][PT.H_UT_NET.value].to_list(),
                ),
        }
        graph_set['graphs'].append(g)

    return graph_set        


def _graph_cc(curve_type: str, y_vals: List[float], x_vals: List[float], IncludeArrows: bool =True, Decolour: bool =False) -> list:
    """Plots a (shifted) hot or cold composite curve."""

    # Clean composite
    y_vals, x_vals = _clean_composite(y_vals, x_vals)

    # Add Hot CC segment
    if curve_type == StreamType.Hot.value:
        return [_create_curve(
            title='Hot CC',
            colour=LineColour.Hot.value if not Decolour else LineColour.Black.value,
            arrow=(ArrowHead.END.value if IncludeArrows else ArrowHead.NO_ARROW.value),
            x_vals=x_vals,
            y_vals=y_vals
        )]

    # Add Cold CC segment
    elif curve_type == StreamType.Cold.value:
        return [_create_curve(
            title='Cold CC',
            colour=LineColour.Cold.value if not Decolour else LineColour.Black.value,
            arrow=(ArrowHead.START.value if IncludeArrows else ArrowHead.NO_ARROW.value),
            x_vals=x_vals,
            y_vals=y_vals
        )]

    else:
        raise ValueError("Unrecognised composite curve type.")


def _graph_gcc(y_vals: List[float], x_vals: List[float], utility_profile: bool = False, decolour: bool = False) -> list:
    """Creates segments for a Grand Composite Curve."""

    # Clean composite
    y_vals, x_vals = _clean_composite(y_vals, x_vals)

    # Find start and end indices of useful data
    start_idx = next((j for j in range(len(x_vals) - 1) if abs(x_vals[j] - x_vals[j + 1]) > ZERO), 0)
    end_idx = next((j for j in range(len(x_vals) - 1, 0, -1) if abs(x_vals[j] - x_vals[j - 1]) > ZERO), len(x_vals) - 1)

    curves = []
    j = start_idx

    # Counters for segment naming
    cold_pro_segs, hot_pro_segs, hot_ut_segs, cold_ut_segs, zero_segs = 0, 0, 0, 0, 0

    while j < end_idx:
        enthalpy_diff = x_vals[j] - x_vals[j + 1]
        segment_type = None

        if enthalpy_diff > ZERO:
            segment_type = 'cold_pro' if not utility_profile else 'hot_ut'
        elif enthalpy_diff < -ZERO:
            segment_type = 'hot_pro'  if not utility_profile else 'cold_ut'
        else:
            segment_type = 'zero'

        # Find where this segment ends
        next_j = j + 1
        while next_j < end_idx:
            next_diff = x_vals[next_j] - x_vals[next_j + 1]
            if (
                (segment_type == 'cold_pro' and next_diff < -ZERO) or
                (segment_type == 'hot_pro'  and next_diff > ZERO) or
                (segment_type == 'hot_ut'   and next_diff < -ZERO) or
                (segment_type == 'cold_ut'  and next_diff > ZERO) or
                (segment_type == 'zero'     and abs(next_diff) > ZERO)
            ):
                break
            next_j += 1

        # Extract segment data
        x_seg = x_vals[j:next_j + 1]
        y_seg = y_vals[j:next_j + 1]

        # Segment title and color
        if segment_type == 'cold_pro':
            cold_pro_segs += 1
            title = f"Cold Process Segment {cold_pro_segs}"
            colour = LineColour.Cold.value
        elif segment_type == 'hot_pro':
            hot_pro_segs += 1
            title = f"Hot Process Segment {hot_pro_segs}"
            colour = LineColour.Hot.value
        elif segment_type == 'hot_ut':
            hot_ut_segs += 1
            title = f"Hot Utility Segment {hot_ut_segs}"
            colour = LineColour.Hot.value
        elif segment_type == 'cold_ut':
            cold_ut_segs += 1
            title = f"Cold Utility Segment {cold_ut_segs}"
            colour = LineColour.Cold.value
        else:  # Zero
            zero_segs += 1
            title = f"Vertical Segment {zero_segs}"
            colour = LineColour.Other.value

        curves.append(_create_curve(
            title=title,
            colour=colour if not decolour else LineColour.Black.value,
            x_vals=x_seg,
            y_vals=y_seg,
            arrow=ArrowHead.NO_ARROW.value
        ))

        j = next_j

    return curves


def _clean_composite(y_vals: List[float], x_vals: List[float]) -> Tuple[List[float], List[float]]:
    """Remove redundant points in composite curves."""

    # Round to avoid tiny numerical errors
    x_vals = [round(x, 5) for x in x_vals]
    y_vals = [round(y, 5) for y in y_vals]

    if len(x_vals) <= 2:
        return y_vals, x_vals

    x_clean, y_clean = [x_vals[0]], [y_vals[0]]

    for i in range(1, len(x_vals) - 1):
        x1, x2, x3 = x_vals[i-1], x_vals[i], x_vals[i+1]
        y1, y2, y3 = y_vals[i-1], y_vals[i], y_vals[i+1]

        if x1 == x3:
            # All three x are the same; keep x2 only if y2 is different
            if x1 != x2:
                x_clean.append(x2)
                y_clean.append(y2)
        else:
            # Linear interpolation check
            y_interp = y1 + (y3 - y1) * (x2 - x1) / (x3 - x1)
            if abs(y2 - y_interp) > ZERO:
                x_clean.append(x2)
                y_clean.append(y2)

    x_clean.append(x_vals[-1])
    y_clean.append(y_vals[-1])

    if abs(x_clean[0] - x_clean[1]) < ZERO:
        x_clean.pop(0)
        y_clean.pop(0)
    
    i = len(x_clean) - 1
    if abs(x_clean[i] - x_clean[i-1]) < ZERO:
        x_clean.pop(i)
        y_clean.pop(i)

    # offset = 0
    # for i in range(len(x_clean) - 1):
    #     x1, x2 = x_clean[i - offset], x_clean[i+1 - offset]        
    #     if abs(x1 - x2) < ZERO:
    #         x_clean.pop(i - offset)
    #         y_clean.pop(i - offset)
    #         offset += 1
    #         if offset > 1:
    #             pass
    #     else:
    #         break

    # offset = 0
    # for i in reversed(range(len(x_clean) - 1)):
    #     x1, x2 = x_clean[i - offset], x_clean[i-1 - offset]     
    #     if abs(x1 - x2) < ZERO:
    #         x_clean.pop(i - offset)
    #         y_clean.pop(i - offset)
    #         offset += 1
    #     else:
    #         break     

    return y_clean, x_clean


def _create_curve(title: str, colour: int, x_vals, y_vals, arrow=ArrowHead.NO_ARROW.value) -> dict:
    """Creates an individual curve from data points."""
    curve = {
        'title': title, 
        'colour': colour, 
        'arrow': arrow
    }
    curve['data_points'] = [
        {'x': round(x, DECIMAL_PLACES), 'y': round(y, DECIMAL_PLACES)} 
        for x, y in zip(x_vals, y_vals) if x != None and y != None
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
#                 if abs(ETD[x_col - 1][j_0 + 1]) > ZERO:
#                     break

#         if HX_Type == 'R' or HX_Type == 'H':
#             for j_1 in range(j_1, 1, -1):
#                 if abs(ETD[x_col - 1][j_1]) > ZERO:
#                     break

#         j = j_0
#         while j < j_1:
#             tmp = ETD[x_col - 1][j + 1]
#             if tmp > ZERO:
#                 # Find process heat deficit segments and utility heat supply segments
#                 seg_type = 1
#             elif tmp < -ZERO:
#                 # Find process heat surplus segments and utility heat sink segments
#                 seg_type = 2
#             else:
#                 # Zero heat transfer
#                 seg_type = 3

#             for j in range(j + 1, j_1 - 1):
#                 if seg_type == 1:
#                     if ETD[x_col - 1][j + 1] < ZERO:
#                         break
#                 elif seg_type == 2:
#                     if ETD[x_col - 1][j + 1] > -ZERO:
#                         break
#                 elif seg_type == 3:
#                     if abs(ETD[x_col - 1][j + 1]) > ZERO:
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
    
#     if HX_countdown > 1 + ZERO:
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