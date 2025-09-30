import pandas as pd
import numpy as np 
from ..utils import *
from ..lib.enums import *
from ..classes import *
from .support_methods import *
from .power_cogeneration_analysis import get_power_cogeneration_above_pinch

__all__ = ["get_additional_zonal_pinch_analysis"]

#######################################################################################################
# Public API --- TODO
#######################################################################################################

def get_additional_zonal_pinch_analysis(pt: ProblemTable, pt_real: ProblemTable, config: Configuration):
    """Calculates additional graphs and targets."""

    # Target heat transfer area and number of exchanger units based on Balanced CC
    if config.AREA_BUTTON:
        area = target_area(pt_real)
        num_units = min_number_hx(pt)
        capital_cost = num_units * config.FC + num_units * config.VC * (area / num_units) ** config.EXP
        annual_capital_cost = capital_cost * capital_recovery_factor(config.DISCOUNT_RATE, config.SERV_LIFE)


    # # TODO: Write analysis. Target exergy supply, rejection, and destruction
    # gcc_x = _calc_exergy_gcc(z, pt_real, bcc, z.graphs[GT.GCC_Act.value]) 
    # z.add_graph(GT.GCC_X.value, gcc_x)

    # # TODO: MOVE to earlier??? Also, requires review and comparision to previous Excel implementation
    # # Determines the assisted heat transfer (pocket cutting) for within a zone
    # GCC_AI = None
    # if z.config.AHT_BUTTON_SELECTED:
    #     z.add_graph('GCC_AI', GCC_AI)

    # # Target co-generation of heat and power
    # if z.config.TURBINE_WORK_BUTTON:
    #     z = get_power_cogeneration_above_pinch(z)

    # # Save data for TS profiles based on HT direction

    return {
        "area": area,
        "num_units": num_units,
        "capital_cost": capital_cost,
        "annual_capital_cost": annual_capital_cost,
    }


#######################################################################################################
# Helper functions
#######################################################################################################

def target_area(z, pt: ProblemTable) -> float:
    """Estimates a heat transfer area target based on counter-current heat transfer using vectorized pandas operations."""
    if abs(pt['HCC'].iloc[0] - pt['CCC'].iloc[0]) > ZERO:
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
        if abs(CCC[i + 1] - HCC[i + 1]) > ZERO:
            break
        i += 1

    i_1 = i
    i += 1

    while i < len(T_vals):
        i_0 = i_1
        if abs(CCC[i] - HCC[i]) < ZERO or i == len(T_vals) - 1:
            i_1 = i
            T_high, T_low = T_vals[i_0], T_vals[i_1]

            def count_crossing(streams):
                t_max = np.array([s.t_max_star for s in streams])
                t_min = np.array([s.t_min_star for s in streams])
                return np.sum(
                    ((t_max > T_low + ZERO) & (t_max <= T_high + ZERO)) |
                    ((t_min >= T_low - ZERO) & (t_min < T_high - ZERO)) |
                    ((t_min < T_low - ZERO) & (t_max > T_high + ZERO))
                )

            num_hx += count_crossing(z.hot_streams)
            num_hx += count_crossing(z.cold_streams)

            def count_utility_crossing(utilities):
                t_max = np.array([u.t_max_star for u in utilities])
                t_min = np.array([u.t_min_star for u in utilities])
                return np.sum(
                    (t_max > T_low + ZERO) & (t_max <= T_high + ZERO) |
                    (t_min >= T_low - ZERO) & (t_min < T_high - ZERO)
                )

            num_hx += count_utility_crossing(z.hot_utilities)
            num_hx += count_utility_crossing(z.cold_utilities)
            num_hx -= 1

            j = i_1
            while j < len(T_vals) - 1:
                if abs(CCC[j + 1] - HCC[j + 1]) > ZERO:
                    break
                j += 1

            i = j
            i_1 = j

        i += 1

    return int(num_hx)


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

#     n_ETE = x_sink / x_source if x_source > ZERO else 0

#     return x_source, x_sink, n_ETE

############# Review and testing needed!!!!!!!!!!!!!!
def Target_Area(z, BCC):
    """Estimates a heat transfer area target for a z based on counter-current heat transfer.
    """
    Area = 0

    # Calculates the area table
    H_val = [0 for i in range(len(BCC[0]) * 2)]

    ColT = 0
    ColRH = 1
    ColHCC = 2
    ColRC = 3
    ColCCC = 4

    # Check the BCC is balanced, if not stop the calculation and return an error
    if abs(BCC[ColHCC][0] - BCC[ColCCC][0]) > ZERO:
        raise Exception('Balanced Composite Curves are imbalanced...')

    # Collate all H intervals
    for i in range(1, len(BCC[0])):
        H_val[i * 2 - 2] = BCC[ColHCC][i - 1]
        H_val[i * 2 - 1] = BCC[ColCCC][i - 1]

    H_val = H_val.sort(reverse=True)

    CalcTable = [ [None for j in range(len(H_val) - 1)] for i in range(10)]

    for i in range(len(H_val) - 1):
        CalcTable[0][i] = H_val[i]
        CalcTable[1][i] = H_val[i + 1]

    r_h = 0
    r_c = 0
    for i in range(len(CalcTable[0])):
        while (CalcTable[0][i] - BCC[ColHCC][r_h + 1]) <= ZERO and r_h + 2 <= len(BCC[0]):
            r_h += 1
        while (CalcTable[0][i] - BCC[ColCCC][r_c + 1]) <= ZERO and r_c + 2 <= len(BCC[0]):
            r_c += 1

        if (CalcTable[0][i] - BCC[ColHCC][r_h + 1] <= ZERO or CalcTable[0][i] - BCC[ColCCC][r_c + 1] <= ZERO) \
                and (r_h + 1 == len(BCC[0]) or r_c + 1 == len(BCC[0])):
            break

        T_h1 = linear_interpolation(CalcTable[0][i], BCC[ColHCC][r_h], BCC[ColHCC][r_h + 1], BCC[ColT][r_h], BCC[ColT][r_h + 1])
        T_h2 = linear_interpolation(CalcTable[1][i], BCC[ColHCC][r_h], BCC[ColHCC][r_h + 1], BCC[ColT][r_h], BCC[ColT][r_h + 1])
        T_c1 = linear_interpolation(CalcTable[0][i], BCC[ColCCC][r_c], BCC[ColCCC][r_c + 1], BCC[ColT][r_c], BCC[ColT][r_c + 1])
        T_c2 = linear_interpolation(CalcTable[1][i], BCC[ColCCC][r_c], BCC[ColCCC][r_c + 1], BCC[ColT][r_c], BCC[ColT][r_c + 1])

        dh = CalcTable[0][i] - CalcTable[1][i]

        T_LMTD = find_LMTD(T_h1, T_h2, T_c1, T_c2)
        CalcTable[2][i] = T_LMTD

        CP_hot = dh / (T_h1 - T_h2)
        CP_cold = dh / (T_c1 - T_c2)

        CP_min = min(CP_hot, CP_cold)
        CP_max = max(CP_hot, CP_cold)
        eff = dh / (CP_min * (T_h1 - T_c2))
        CP_star = CP_min / CP_max

        Arrangement = None
        if z.config.CF_SELECTED:
            Arrangement = HX.CF.value
        elif z.config.PF_SELECTED:
            Arrangement = HX.PF.value
        else:
            Arrangement = HX.ShellTube.value

        Ntu = HX_NTU(Arrangement, eff, CP_star)

        # Heat transfer resistance and coefficient
        R_hot = BCC[ColRH][r_h + 1]
        R_cold = BCC[ColRC][r_c + 1]
        U_o = 1 / (R_hot + R_cold)

        CalcTable[3][i] = Ntu * CP_min / U_o
        CalcTable[4][i] = dh / (U_o * T_LMTD)

        Area = Area + CalcTable[3][i]

    return Area

def MinNumberHX(z, pt, BCC_star):
    """Estimates the minimum number of heat exchanger units for a given Pinch problem.
    """
    Num_HX = 0
    i = 0
    while i < len(pt[0]) - 1:
        if abs(BCC_star[4][i + 1] - BCC_star[2][i + 1]) > ZERO:
            break
        i += 1

    i_1 = i
    i = i + 1
    while i < len(pt[0]):
        i_0 = i_1

        if abs(BCC_star[4][i] - BCC_star[2][i]) < ZERO or i == len(pt[0]) - 1:
            i_1 = i
            T_high = pt[0][i_0]
            T_low = pt[0][i_1]

            for s in z.hot_streams:
                T_max = s.t_max_star
                T_min = s.t_min_star
                if (T_max > T_low + ZERO and T_max <= T_high + ZERO) or (T_min >= T_low - ZERO \
                        and T_min < T_high - ZERO) or (T_min < T_low - ZERO and T_max > T_high + ZERO):
                    Num_HX += 1

            for s in z.cold_streams:
                T_max = s.t_max_star
                T_min = s.t_min_star
                if (T_max > T_low + ZERO and T_max <= T_high + ZERO) or (T_min >= T_low - ZERO \
                        and T_min < T_high - ZERO) or (T_min < T_low - ZERO and T_max > T_high + ZERO):
                    Num_HX += 1

            for utility_k in z.hot_utilities:
                T_max = utility_k.t_max_star
                T_min = utility_k.t_min_star
                if (T_max > T_low + ZERO and T_max <= T_high + ZERO) or (T_min >= T_low - ZERO and T_min < T_high - ZERO):
                    Num_HX += 1

            for utility_k in z.cold_utilities:
                T_max = utility_k.t_max_star
                T_min = utility_k.t_min_star
                if (T_max > T_low + ZERO and T_max <= T_high + ZERO) or (T_min >= T_low - ZERO and T_min < T_high - ZERO):
                    Num_HX += 1

            Num_HX -= 1

            j = i_1
            while j < len(pt[0]) - 1:
                if abs(BCC_star[4][j + 1] - BCC_star[2][j + 1]) > ZERO:
                    break
                j += 1

            i = j
            i_1 = j

        i += 1

    return Num_HX

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
#             if GCC_Act[1][gcc_act_i] < ZERO:
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
#         if abs(GCC_X[1][i]) < ZERO:
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
