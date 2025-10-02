import pandas as pd
import numpy as np 
from ..utils import *
from ..lib.enums import *
from ..classes import *
from .support_methods import *
from .power_cogeneration_analysis import get_power_cogeneration_above_pinch

__all__ = ["target_area", "min_number_hx", "get_additional_zonal_pinch_analysis"]

#######################################################################################################
# Public API --- TODO
#######################################################################################################

def target_area(z, pt: ProblemTable) -> float:
    """Estimates a heat transfer area target based on counter-current heat transfer using vectorized numpy operations."""
    if abs(pt['HCC'].iloc[0] - pt['CCC'].iloc[0]) > tol:
        raise ValueError("Balanced Composite Curves are imbalanced.")

    # Collect H_val intervals and sort
    h_vals = pd.Series(pt['HCC'].iloc[:-1].tolist() + pt['CCC'].iloc[:-1].tolist()).sort_values().reset_index(drop=True)
    h_start = h_vals[:-1].values
    h_end = h_vals[1:].values
    dh = h_start - h_end

    # Interpolate temperatures for each H at both ends
    t_h1 = np.interp(h_start, pt['HCC'], pt['T'])
    t_h2 = np.interp(h_end, pt['HCC'], pt['T'])
    t_c1 = np.interp(h_start, pt['CCC'], pt['T'])
    t_c2 = np.interp(h_end, pt['CCC'], pt['T'])

    delta_T1 = t_h1 - t_c1
    delta_T2 = t_h2 - t_c2

    t_lmtd = np.where(
        abs(delta_T1 - delta_T2) < 1e-6,
        (delta_T1 + delta_T2) / 2,
        (delta_T1 - delta_T2) / np.log(delta_T1 / delta_T2)
    )

    cp_hot = dh / (t_h1 - t_h2)
    cp_cold = dh / (t_c1 - t_c2)
    cp_min = np.minimum(cp_hot, cp_cold)
    cp_max = np.maximum(cp_hot, cp_cold)

    eff = dh / (cp_min * (t_h1 - t_c2))
    cp_star = cp_min / cp_max

    if z.config.CF_SELECTED:
        arrangement = HX.CF.value
    elif z.config.PF_SELECTED:
        arrangement = HX.PF.value
    else:
        arrangement = HX.ShellTube.value

    ntu = np.vectorize(HX_NTU)(arrangement, eff, cp_star)

    r_hot = np.interp(h_end, pt['HCC'], pt['RH'])
    r_cold = np.interp(h_end, pt['CCC'], pt['RC'])
    u_o = 1 / (r_hot + r_cold)

    area_segments = ntu * cp_min / u_o
    total_area = np.sum(area_segments)

    return float(total_area)


def min_number_hx(z, pt_df: ProblemTable, bcc_star_df: ProblemTable) -> int:
    """
    Estimates the minimum number of heat exchangers required for the pinch problem using vectorized interval logic.

    Args:
        z: Zone with hot/cold streams and utilities.
        pt_df (ProblemTable): Problem table DataFrame with temperature column.
        bcc_star_df (ProblemTable): Balanced Composite Curve data with 'CCC' and 'HCC'.

    Returns:
        int: Minimum number of exchangers.
    """
    T_vals = pt_df.iloc[:, 0].values
    CCC = bcc_star_df['CCC'].values
    HCC = bcc_star_df['HCC'].values

    num_hx = 0
    i = 0
    while i < len(T_vals) - 1:
        if abs(CCC[i + 1] - HCC[i + 1]) > tol:
            break
        i += 1

    i_1 = i
    i += 1

    while i < len(T_vals):
        i_0 = i_1
        if abs(CCC[i] - HCC[i]) < tol or i == len(T_vals) - 1:
            i_1 = i
            T_high, T_low = T_vals[i_0], T_vals[i_1]

            def count_crossing(streams):
                """Count process streams whose adjusted temperatures intersect interval [T_low, T_high]."""
                t_max = np.array([s.t_max_star for s in streams])
                t_min = np.array([s.t_min_star for s in streams])
                return np.sum(
                    ((t_max > T_low + tol) & (t_max <= T_high + tol)) |
                    ((t_min >= T_low - tol) & (t_min < T_high - tol)) |
                    ((t_min < T_low - tol) & (t_max > T_high + tol))
                )

            num_hx += count_crossing(z.hot_streams)
            num_hx += count_crossing(z.cold_streams)

            def count_utility_crossing(utilities):
                """Count utility streams whose adjusted temperatures intersect interval [T_low, T_high]."""
                t_max = np.array([u.t_max_star for u in utilities])
                t_min = np.array([u.t_min_star for u in utilities])
                return np.sum(
                    (t_max > T_low + tol) & (t_max <= T_high + tol) |
                    (t_min >= T_low - tol) & (t_min < T_high - tol)
                )

            num_hx += count_utility_crossing(z.hot_utilities)
            num_hx += count_utility_crossing(z.cold_utilities)
            num_hx -= 1

            j = i_1
            while j < len(T_vals) - 1:
                if abs(CCC[j + 1] - HCC[j + 1]) > tol:
                    break
                j += 1

            i = j
            i_1 = j

        i += 1

    return int(num_hx)





#######################################################################################################
# Helper functions
#######################################################################################################

############# Review and testing needed!!!!!!!!!!!!!!


############# Review and testing needed!!!!!!!!!!!!!!
# def _calc_exergy_gcc(z, pt_real, BCC, GCC_Act):
#     """Determine Exergy Transfer Effectiveness including process and utility streams.
#     """
#     # Exergy Transfer Effectiveness proposed by Marmolejo-Correa, D., Gundersen, T., 2012.
#     # A comparison of exergy efficiency definitions with focus on low temperature processes.
#     # Energy 44, 477–489. https://doi.org/10.1016/j.energy.2012.06.001
#     x_source, x_sink, n_ETE = _calc_total_exergy(BCC)
#     z.exergy_sources = x_source
#     z.exergy_sinks = x_sink
#     z.ETE = n_ETE

#     GCC_X = z.Calc_ExGCC(GCC_Act)
#     x_source, x_sink, n_ETE = _calc_total_exergy(pt_real, Col_T=0, Col_HCC=4, Col_CCC=7)

#     z.exergy_req_min = GCC_X[1][1]
#     z.exergy_des_min = GCC_X[1][-1]

#     return GCC_X

############# Review and testing needed!!!!!!!!!!!!!!
# def _calc_total_exergy(z: Zone, CC, x_source=0, x_sink=0, n_ETE=0, Col_T=0, Col_HCC=2, Col_CCC=4):
#     """Determines the source and sink exergy of a balanced CC."""
#     for i in range(1, len(CC[0])):
#         T_ex1 = compute_exergetic_temperature(CC[Col_T][i - 1], T_ref=z.config.TEMP_REF)
#         T_ex2 = compute_exergetic_temperature(CC[Col_T][i], T_ref=z.config.TEMP_REF)
#         CP_hot = (CC[Col_HCC][i - 1] - CC[Col_HCC][i]) / (CC[Col_T][i - 1] - CC[Col_T][i])
#         CP_cold = (CC[Col_CCC][i - 1] - CC[Col_CCC][i]) / (CC[Col_T][i - 1] - CC[Col_T][i])

#         if T_ex1 > 0:
#             x_source = x_source + CP_hot * T_ex1
#             x_sink = x_sink + CP_cold * T_ex1
#         else:
#             x_source = x_source + CP_cold * T_ex1
#             x_sink = x_sink + CP_hot * T_ex1

#         if T_ex2 > 0:
#             x_source = x_source - CP_hot * T_ex2
#             x_sink = x_sink - CP_cold * T_ex2
#         else:
#             x_source = x_source - CP_cold * T_ex2
#             x_sink = x_sink - CP_hot * T_ex2

#     n_ETE = x_sink / x_source if x_source > tol else 0

#     return x_source, x_sink, n_ETE

# def Calc_ExGCC(z, GCC_Act):
#     """Transposes a normal GCC (T-h) into a exergy GCC (Tx-X).
#     """
#     GCC_X = copy.deepcopy(GCC_Act)
#     Min_X = 0
#     AbovePT = True
#     GCC_X[0][0] = compute_exergetic_temperature(GCC_Act[0][0] + z.config.DTCONT / 2, T_ref=z.config.TEMP_REF)
#     GCC_X[1][0] = 0
#     i_upper = len(GCC_X[0]) + 1

#     # Transpose to exergetic temperature and exergy flow
#     i = 1
#     gcc_act_i = 1
#     while i <= i_upper and gcc_act_i < len(GCC_Act[0]):
#         if AbovePT:
#             GCC_X[0][i] = compute_exergetic_temperature(GCC_Act[0][gcc_act_i] + z.config.DTCONT / 2, T_ref=z.config.TEMP_REF)
#             GCC_X[1][i] = (GCC_Act[1][gcc_act_i - 1] - GCC_Act[1][gcc_act_i]) / (GCC_Act[0][gcc_act_i - 1] - GCC_Act[0][gcc_act_i])
#             GCC_X[1][i] = GCC_X[1][i - 1] - GCC_X[1][i] * (GCC_X[0][i - 1] - GCC_X[0][i])
#             if GCC_Act[1][gcc_act_i] < tol:
#                 Min_X = GCC_X[1][i]
#                 for row in GCC_X:
#                     row += [0, 0]
#                 i += 2
#                 gcc_act_i += 1
#                 GCC_X[0][i] = compute_exergetic_temperature(GCC_Act[0][gcc_act_i - 1] - z.config.DTCONT / 2, T_ref=z.config.TEMP_REF)
#                 GCC_X[1][i] = GCC_X[1][i - 1]
#                 AbovePT = False
#         else:
#             GCC_X[0][i] = compute_exergetic_temperature(GCC_Act[0][gcc_act_i - 1] - z.config.DTCONT / 2, T_ref=z.config.TEMP_REF)
#             GCC_X[1][i] = (GCC_Act[1][gcc_act_i - 2] - GCC_Act[1][gcc_act_i - 1]) / (GCC_Act[0][gcc_act_i - 2] - GCC_Act[0][gcc_act_i - 1])
#             GCC_X[1][i] = GCC_X[1][i - 1] - GCC_X[1][i] * (GCC_X[0][i - 1] - GCC_X[0][i])
#         i += 1
#         gcc_act_i += 1

#     # Shift Exergy GCC appropriately
#     for i in range(1, len(GCC_X[0])):
#         GCC_X[1][i] = GCC_X[1][i] + abs(Min_X)
#         if abs(GCC_X[1][i]) < tol:
#             GCC_X[1][i] = 0

#     return GCC_X



# def Calc_GCC_AI(z, pt_real, gcc_np):
#     """Returns a simplified array for the assisted integration GCC.
#     """
#     GCC_AI = [ [ None for j in range(len(pt_real[0]))] for i in range(2)]
#     for i in range(len(pt_real[0])):
#         GCC_AI[0][i] = pt_real[0][i]
#         GCC_AI[1][i] = pt_real[PT.H_NET.value][i] - gcc_np[1][i]
#     return GCC_AI
