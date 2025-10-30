"""Placeholder for legacy energy-transfer targeting routines (currently unused)."""

# import copy
# from typing import Optional
# from ..utils import *
# from ..classes import *
# from ..analysis.support_methods import *


# # TODO: Refactor this entire file.

# __all__ = ["_get_unit_operation_targets"]

# #######################################################################################################
# # Public API --- TODO: Need to restore energy transfer diagram analysis
# #######################################################################################################


# def etd(site: Zone):
#     """
#     Calculates the ETD and retrofit targets.
#     """

#     # Prepares variables and arrays
#     ETD = [ [0, 0] for i in range(1 + len(site.subzones) * 3) ]
#     ETD_star = copy.deepcopy(ETD)

#     for z in site.subzones:
#         # Redefine heat exchanger pockets based on detailed ETD retrofit analysis
#         Req_ut = True if z.hot_utility_target + z.cold_utility_target > tol else False
#         if site.config.GCC_VERT_CUT_KINK_OPTION and not Req_ut:
#             z.graphs['GCC_etc'] = site.Reshape_GCC_Pockets(z.graphs['PT_star'], z.graphs['GCC_etc'])
#         site.Extract_Pro_ETC(ETD_star, z, z.graphs['PT_star'], Req_ut)
#         site.Extract_Pro_ETC(ETD, z, z.graphs['PT'], Req_ut)

#     site_tit: Zone = site.targets[TargetType.DI.value]
#     PT_TIT = site_tit.graphs['PT']
#     PT_star_TIT = site_tit.graphs['PT_star']

#     # Forms a complete set of temperature intervals
#     T_int_star = site.Compile_ETD_T_int(ETD_star, PT_star_TIT)
#     T_int = site.Compile_ETD_T_int(ETD, PT_TIT)

#     # Expands Heat Cascade Table to be based on the complete set of temperature intervals
#     ETD_star = site.Transpose_ETD_T(ETD_star, T_int_star)
#     ETD = site.Transpose_ETD_T(ETD, T_int)

#     # Calculates the ETD
#     ETD_star_header = site.Stack_ETD(ETD_star, PT_star_TIT, 'ETD', True)
#     ETD_header = site.Stack_ETD(ETD, PT_TIT, 'ETD', False)

#     # Shift thermodynamic limiting curve to match the end of the ETD
#     dh = ETD_star[-1][1] - PT_TIT[10][0]
#     PT_TIT = shift_heat_cascade(PT_TIT, dh, 10)

#     # Determines the Advanced Composite Curve that combines conventional CC and the ETD
#     ACC_star = site.Calc_ACCN(ETD_star, PT_star_TIT)
#     ACC = site.Calc_ACCN(ETD, PT_TIT)

#     # Reduces the number of T int to the minimum
#     site.Simplify_ETD(ETD_star)
#     site.Simplify_ETD(ETD)
#     site.Simplify_ETD(ACC_star)
#     site.Simplify_ETD(ACC)

#     # Record retrofit targets
#     Hot_Pinch, Cold_Pinch = get_pinch_temperatures(PT_star_TIT, 10, 0)

#     Retrofit = Zone(name=TargetType.ET.value, config=site.config)
#     Retrofit.hot_pinch = Hot_Pinch
#     Retrofit.hot_utility_target = ETD_star[-1][1]
#     Retrofit.cold_utility_target = ETD_star[-1][-1]
#     Retrofit.retrofit_target = Retrofit.hot_utility_target - site_tit.hot_utility_target
#     Retrofit.heat_recovery_target = site_tit.heat_recovery_target - Retrofit.retrofit_target
#     Retrofit.degree_of_int = Retrofit.heat_recovery_target / site_tit.heat_recovery_limit if site_tit.heat_recovery_limit > 0 else 1
#     Retrofit.add_graph('ETD', ETD)
#     Retrofit.add_graph('ETD_star', ETD_star)
#     Retrofit.add_graph('ACC', ACC)
#     Retrofit.add_graph('ACC_star', ACC_star)
#     Retrofit.add_graph('ETD_header', ETD_header)
#     site.add_zone(Retrofit)


# #######################################################################################################
# # Helper Functions
# #######################################################################################################

# def Reshape_GCC_Pockets(site, PT_star, GCC_etc):
#     """Redefine GCC pockets based on possible HEN retrofit design considerations.
#     """
#     GCC_etc = GCC_etc[:2]
#     for i in range(len(GCC_etc)):
#         GCC_etc[i] = GCC_etc[i][:len(PT_star[0])]

#     min_H_cross = 1E+35

#     for j in range(len(PT_star[0])):
#         GCC_etc[1][j] = PT_star[1][j]
#         if j == 0 or j == len(PT_star[0]):
#             GCC_etc[2][j] = 0
#         else:
#             if abs(PT_star[9][j]) > tol:
#                 if PT_star[3][j] > PT_star[6][j]:
#                     GCC_etc[2][j] = GCC_etc[2][j - 1] + PT_star[2][j] * PT_star[3][j]
#                 else:
#                     GCC_etc[2][j] = GCC_etc[2][j - 1] - PT_star[2][j] * PT_star[6][j]
#             else:
#                 GCC_etc[2][j] = GCC_etc[2][j - 1]
#             if PT_star[3][j] > tol and PT_star[6][j] > tol:
#                 min_H_cross = min(PT_star[11][j], min_H_cross)
#                 min_H_cross = min(PT_star[11][j - 1], min_H_cross)

#     DH_shift = GCC_etc[2][len(GCC_etc[0])]
#     for j in range(len(PT_star[0])):
#         GCC_etc[2][j] = GCC_etc[2][j] - DH_shift

#     for j in range(1, len(PT_star[0])):
#         if min_H_cross > (min(GCC_etc[2][j], GCC_etc[2][j - 1])) + tol and min_H_cross < (max(GCC_etc[2][j], GCC_etc[2][j - 1])) - tol:
#             j_0 = j
#             h_0 = min_H_cross
#             T_new = linear_interpolation(h_0, GCC_etc[2][j_0], GCC_etc[2][j_0 - 1], GCC_etc[1][j_0], GCC_etc[1][j_0 - 1])
#             PT_star = insert_temperature_interval_into_pt(PT_star, T_new, j_0)
#             j_0 = j
#             GCC_etc = insert_temperature_interval_into_pt(GCC_etc, T_new, j_0)

#     for j in range(len(PT_star[0])):
#         if PT_star[11][j] > min_H_cross:
#             PT_star[11][j] = min_H_cross
#     return GCC_etc

# def Write_HSDT(site, ETD, ETD_header, sheet, row=4, col=1, exclude_small_DH=False):
#     """Prints the ETD table to a spreadsheet.
#     """
#     # DoEvents
#     if not sheet.visible:
#         sheet.visible = True

#     k = 1
#     row_0 = row
#     sheet.cells(1, col).value = 'Ti'
#     # sheet.cells(1, col).characters[1].font.defscript = True
#     sheet.cells(3, col).value = chr(176)

#     for i in range(1, len(ETD), 3):
#         sheet.cells(1, col + k).value = ETD[i][0]
#         sheet.cells(2, col + k).value = chr(916) + 'Hnet'
#         # sheet.cells(2, col + k).characters[2:5].font.defscript = True
#         sheet.cells(3, col + k).value = 'kW'
#         k += 1

#     k = 0
#     for j in range(len(ETD[0])):
#         sheet.cells(row + (j - 1), col + k).value = ETD[0][j]

#     k = 1
#     for i in range(1, len(ETD), 3):
#         for j in range(1, len(ETD[0])):
#             sheet.cells(row + (j - 1), col + k).value = ETD[i][j] if ETD_header[i + 1][5] == 0 or exclude_small_DH == False else 0
#         k += 1

#     # With Range(sheet.cells(row_0, 2), sheet.cells(row_0 + len(ETD[0]) - 1, (len(ETD) - 1) / 3 + 3))
#     #     Add conditional formatting to Table
#     #     .FormatConditions.AddColorScale ColorScaleType:=3
#     #     .FormatConditions(.FormatConditions.count).SetFirstPriority
#     #     .FormatConditions(1).ColorScaleCriteria(1).Type = xlConditionValueLowestValue
#     #     With .FormatConditions(1).ColorScaleCriteria(1).FormatColor
#     #         .ThemeColor = xlThemeColorAccent1
#     #         .TintAndShade = 0
#     #     End With
#     #     .FormatConditions(1).ColorScaleCriteria(2).Type = xlConditionValueNumber
#     #     .FormatConditions(1).ColorScaleCriteria(2).Value = 0
#     #     With .FormatConditions(1).ColorScaleCriteria(2).FormatColor
#     #         .ThemeColor = xlThemeColorDark1
#     #         .TintAndShade = 0
#     #     End With
#     #     .FormatConditions(1).ColorScaleCriteria(3).Type = xlConditionValueHighestValue
#     #     With .FormatConditions(1).ColorScaleCriteria(3).FormatColor
#     #         .Color = 255
#     #         .TintAndShade = 0
#     #     End With

#     #     Draw boarders for Table
#     #     With .Borders(xlEdgeLeft)
#     #         .LineStyle = xlContinuous
#     #         .ThemeColor = 1
#     #         .TintAndShade = -0.149998474074526
#     #         .Weight = xlThin
#     #     End With
#     #     With .Borders(xlEdgeTop)
#     #         .LineStyle = xlContinuous
#     #         .ThemeColor = 1
#     #         .TintAndShade = -0.149998474074526
#     #         .Weight = xlThin
#     #     End With
#     #     With .Borders(xlEdgeBottom)
#     #         .LineStyle = xlContinuous
#     #         .ThemeColor = 1
#     #         .TintAndShade = -0.149998474074526
#     #         .Weight = xlThin
#     #     End With
#     #     With .Borders(xlEdgeRight)
#     #         .LineStyle = xlContinuous
#     #         .ThemeColor = 1
#     #         .TintAndShade = -0.149998474074526
#     #         .Weight = xlThin
#     #     End With
#     #     With .Borders(xlInsideVertical)
#     #         .LineStyle = xlContinuous
#     #         .ThemeColor = 1
#     #         .TintAndShade = -0.149998474074526
#     #         .Weight = xlThin
#     #     End With
#     #     With .Borders(xlInsideHorizontal)
#     #         .LineStyle = xlContinuous
#     #         .ThemeColor = 1
#     #         .TintAndShade = -0.149998474074526
#     #         .Weight = xlThin
#     #     End With
#     # End With

# def Compile_ETD_T_int(site, ETD, PT_TIT):
#     """Grabs temperatures from every process operation GCC with a defined system,
#     order, and remove duplicates.
#     """
#     T_int = [ [None for i in range(10000)] ]

#     n = 0
#     j = 1
#     while j < len(ETD):
#         for i in range(1, len(ETD[0])):
#             if ETD[j][i] == None:
#                 break
#             T_int[0][n] = ETD[j][i]
#             n += 1
#         j += 3

#     j_0 = j + 1

#     for j in range(0, len(PT_TIT), 3):
#         for i in range(len(PT_TIT[0])):
#             if PT_TIT[j][i] == None:
#                 break
#             T_int[0][n] = PT_TIT[j][i]
#             n += 1

#     T_int = T_int.sort(reverse=True)

#     # TODO: I think this is unnecessary because the None values would be removed in get_ordered_list.
#     # if T_int[0][len(T_int[0])] < tol or T_int[0][len(T_int[0])] == None:
#     #     for i in range(len(T_int[0]) - 1, -1, -1):
#     #         if T_int[0][i] == None:
#     #             break
#     #     T_int[0][i] = 0
#     return T_int

# def Transpose_ETD_T(site, ETD, T_int):
#     """Transposes temperature intervals from individual GCC cascades to a common set of temperature intervals for the entire system.
#     """
#     ETD_temp = [ [ None for j in range(len(T_int[0]) + 1)] for i in range(len(ETD))]

#     ETD_temp[0][0] = 'T'
#     for i in range(1, len(ETD_temp[0])):
#         ETD_temp[0][i] = T_int[0][i - 1]

#     for j in range(1, len(ETD), 3):
#         k = 2
#         ETD_temp[j][0] = ETD[j][0]
#         ETD_temp[j + 1][0] = None
#         ETD_temp[j + 2][0] = ETD[j + 2][0]

#         ETD_temp[j][1] = None
#         ETD_temp[j + 1][1] = ETD[j + 2][1]
#         ETD_temp[j + 2][1] = 0

#         for i in range(2, len(ETD_temp[0])):
#             if k >= len(ETD[0]):
#                 ETD_temp[j][i] = 0
#                 ETD_temp[j + 1][i] = ETD_temp[j + 1][i - 1]
#                 continue
#             if (ETD_temp[0][i - 1] <= ETD[j][k - 1] + tol) and (ETD_temp[0][i] >= ETD[j][k] - tol):
#                 CPnet = ETD[j + 1][k]
#                 dt = ETD_temp[0][i - 1] - ETD_temp[0][i]
#                 ETD_temp[j][i] = dt * CPnet
#                 ETD_temp[j + 1][i] = ETD_temp[j + 1][i - 1] + dt * CPnet
#                 if abs(ETD_temp[0][i] - ETD[j][k]) < tol:
#                     k += 1
#             else:
#                 ETD_temp[j][i] = 0
#                 ETD_temp[j + 1][i] = ETD_temp[j + 1][i - 1]
#     return ETD_temp

# def Simplify_ETD(site, ETD):
#     """Reduces the ETD (and HSDT) to the minimum number of T intervals by removing all
#     intervals between which there are not changes in CP for all process operations.
#     """
#     # Remove low temperature intervals that exceed the maximum temperatures
#     # for i in range(len(ETD[0]) - 1, 1, -1): # Loop from lowerest to highest temperature
#     #     for j in range(1, len(ETD), 3):
#     #         if ETD[j][i] > tol:
#     #             break # Temperature interval cannot be removed if true
#     #     else:
#     #         continue
#     #     break

#     # if i < len(ETD[0]):
#     #     for j in range(len(ETD)):
#     #         ETD[j] = ETD[j][:i]

#     # # Remove high temperature intervals that exceed the maximum temperatures
#     # for i in range(2, len(ETD[0])): # Loop from highest to lowerest temperature
#     #     for j in range(1, len(ETD), 3):
#     #         if abs(ETD[j][i]) > tol:
#     #             break # Temperature interval cannot be removed if true
#     #     else:
#     #         continue
#     #     break

#     # i = i - 2
#     # if i > 0:
#     #     for n in range(1, len(ETD[0]) - i):
#     #         for m in range(len(ETD)):
#     #             ETD[m][n] = ETD[m][n + i]
#     #     for row in ETD:
#     #         row.pop()

#     # Join two T intervals where CP is constant for all zones
#     for i in range(len(ETD[0]) - 1, 2, -1): # Loop from lowest to highest temperature
#         for j in range(1, len(ETD), 3):
#             CP_0 = ETD[j][i] / (ETD[0][i - 1] - ETD[0][i])
#             CP_1 = ETD[j][i - 1] / (ETD[0][i - 2] - ETD[0][i - 1])
#             if abs(CP_0 - CP_1) > tol:
#                 break # Temperature interval cannot be removed if true
#         else:
#             n = i
#             ETD[0][n - 1] = ETD[0][n]
#             for m in range(1, len(ETD), 3):
#                 ETD[m][n - 1] = ETD[m][n - 1] + ETD[m][n]
#                 ETD[m + 1][n - 1] = ETD[m + 1][n - 1] + ETD[m][n]
#                 ETD[m + 2][n - 1] = ETD[m + 2][n - 1] + ETD[m][n]
#             for n in range(i, len(ETD[0]) - 1):
#                 for m in range(len(ETD)):
#                     ETD[m][n] = ETD[m][n + 1]
#             for row in ETD:
#                 row.pop()

# def Stack_ETD(site, ETD, PT, Diagram_type, is_shifted):
#     """Determine the order and stack individual heat cascades of process operations.
#     """
#     Hot_Pinch, Cold_Pinch = get_pinch_temperatures(PT, 10, 0)

#     ETD_header = [ [None for j in range(11)] for i in range(len(ETD))]

#     for j in range(1, len(ETD), 3):
#         site.Characterise_ETC(ETD, ETD_header, Hot_Pinch, Cold_Pinch, j + 1, ETD[j + 2][0])

#     # Reorder HX in the ETD
#     ETD_temp = copy.deepcopy(ETD)

#     for j in range(1, len(ETD)):
#         ETD[j] = [None for k in range(len(ETD[0]))]

#     ETD_header_temp = copy.deepcopy(ETD_header)

#     col_ETD = 2
#     site.Write_Next_ETC(ETD, ETD_temp, ETD_header, ETD_header_temp, col_ETD)

#     HX_type_1 = 'C'
#     HX_type_2 = 'R'
#     HX_type_3 = 'H'

#     j_0 = 0
#     for j in range(3, len(ETD), 3):
#         if ETD[j][0] == HX_type_1[:1] or \
#                 ETD[j][0] == HX_type_2[:1] or \
#                 ETD[j][0] == HX_type_3[:1]:
#             if j_0 == 0:
#                 j_0 = j
#             for i in range(1, len(ETD[0])):
#                 if j == j_0:
#                     ETD[j][i] = 0 if ETD_header[j - 1][5] == 1 and is_shifted else ETD[j - 1][i]
#                 else:
#                     ETD[j][i] = ETD[j_0][i] if ETD_header[j - 1][5] == 1 and is_shifted else ETD[j - 1][i] + ETD[j_0][i]
#             j_0 = j

#     return ETD_header

# def Calc_ACCN(site, ETD, PT_TIT):
#     """Calculate Adv CC with integrated ETD.
#     """
#     ACC = [
#         [ None for j in range(len(ETD[0])) ] for i in range(len(ETD) + 3)
#     ]

#     i = 0
#     ACC[0][0] = ETD[0][0]
#     ACC[1][0] = 'HCC'

#     for j in range(1, len(ETD), 3):
#         ACC[j + 3][0] = ETD[j][0]
#         ACC[j + 4][0] = ETD[j + 1][0]
#         ACC[j + 5][0] = ETD[j + 2][0]

#     for i in range(1, len(ACC[0])):
#         ACC[0][i] = ETD[0][i]

#         if abs(ACC[0][i] - PT_TIT[0][i]) > tol:
#             PT_TIT = insert_temperature_interval_into_pt(PT_TIT, ACC[0][i], i)

#         if i > 1:
#             ACC[1][i] = PT_TIT[4][i] - PT_TIT[4][i - 1]
#         ACC[2][i] = PT_TIT[4][i]
#         ACC[3][i] = PT_TIT[4][i]

#     for i in range(1, len(ACC[0])):
#         for j in range(1, len(ETD), 3):
#             ACC[j + 3][i] = ETD[j][i]
#             ACC[j + 4][i] = ETD[j + 1][i]
#             ACC[j + 5][i] = ETD[j + 2][i] + ACC[3][i]

#     return ACC

# def Characterise_ETC(site, ETD, ETD_header, Hot_Pinch, Cold_Pinch, Col_j, HX_mode):
#     """Characterises the shape, enclosed area, and temperature driving force of each heat cascade.
#     """
#     TH_tot_area = 0
#     TH_w_tot_area = 0
#     T_h_max = -1000
#     H_max = 0
#     ETD_header[Col_j][0] = 0 # Check for Cross-Pinch Heat Transfer
#     t_const = 99 # tune weghting constant
#     for i in range(2, len(ETD[0])):
#         TH_sub_area = abs(0.5 * (ETD[Col_j][i - 1] + ETD[Col_j][i]) / (ETD[0][i - 1] - ETD[0][i])) # Determine T-H area (row 1)
#         # Determine weighting factor
#         if ETD[0][i] > Hot_Pinch + tol:
#             w = 1 / (abs((ETD[0][i - 1] + ETD[0][i]) / 2 - Hot_Pinch) / t_const + 1)
#         elif ETD[0][i] < Cold_Pinch - tol:
#             w = 1 / (abs((ETD[0][i - 1] + ETD[0][i]) / 2 - Cold_Pinch) / t_const + 1)
#         else:
#             w = 1
#             if abs(ETD[Col_j][i]) > tol:
#                 ETD_header[Col_j][4] = 1
#         TH_tot_area += TH_sub_area
#         TH_w_tot_area += w * TH_sub_area
#         if H_max < abs(ETD[Col_j][i]):
#             H_max = abs(ETD[Col_j][i])
#         if T_h_max == -1000 and abs(ETD[Col_j - 1][i]) > tol:
#             T_h_max = (ETD[0][i - 1] + 273.15) if HX_mode == 'C' else 1 / (ETD[0][i - 1] + 273.15)

#     ETD_header[Col_j][0] = ETD[Col_j - 1][0]
#     ETD_header[Col_j][1] = TH_w_tot_area
#     ETD_header[Col_j][2] = TH_tot_area
#     ETD_header[Col_j][3] = H_max
#     ETD_header[Col_j][4] = T_h_max

#     ETD_header[Col_j][10] = 1 / T_h_max if HX_mode == 'C' else T_h_max

# def Write_Next_ETC(site, ETD, ETD_temp, ETD_header, ETD_header_temp, k):
#     H_thres = site.config.THRESHOLD
#     TH_area_thres = site.config.AREA_THRESHOLD * 1000
#     for HX_Type in ['C', 'R', 'H']:
#         HX_num = 0
#         for i in range(3, len(ETD), 3):
#             if ETD_temp[i][0] == HX_Type:
#                 HX_num += 1

#         if HX_num > 0:
#             k_max = k + (HX_num - 1) * 3 + 1
#             for k in range(k, k_max, 3):
#                 Var_temp = 0
#                 for j in range(2, len(ETD_temp), 3):
#                     if ETD_header_temp[j][10] > Var_temp and ETD_temp[j + 1][0] == HX_Type:
#                         Var_temp = ETD_header_temp[j][10]
#                         k_1 = j
#                 ETD_header_temp[k_1][10] = 0
#                 for i in range(len(ETD_temp[0])):
#                     ETD[k - 1][i] = ETD_temp[k_1 - 1][i]
#                     ETD[k][i] = ETD_temp[k_1][i]

#                 ETD[k + 1][0] = ETD_temp[k_1 + 1][0]
#                 for i in range(len(ETD_header_temp[0])):
#                     ETD_header[k][i] = ETD_header_temp[k_1][i]
#                 ETD_header[k][5] = 0
#                 if (site.config.SET_MIN_DH_THRES and ETD_header[k][3] < H_thres) or (site.config.SET_MIN_TH_AREA and ETD_header[k][1] < TH_area_thres):
#                     if HX_Type[:1] == 'R': # (RetrofitForm.Excl_UE_Option.Value and (HX_Type[:1] = 'H' or HX_Type[:1] = 'C')) Or
#                         ETD_header[k][6] = 1
#             k += 3

# def Extract_Pro_ETC(site, ETD, z, PT, Req_ut):
#         """Save an individual process operation GCC for the ETD.
#         """
#         j = z.zone_num * 3 - 1
#         ETD[j - 1][0] = z.name
#         ETD[j][0] = 'CP net'
#         if Req_ut:
#             ETD[j + 1][0] = 'H' if PT[10][0] > tol else 'C'
#         else:
#             ETD[j + 1][0] = 'R'

#         n = max(len(ETD[0]) - 1, len(PT[0])) + 1
#         if n != len(ETD[0]):
#             for i in range(len(ETD)):
#                 ETD[i] += [None for k in range(n - len(ETD[i]))]
#         for i in range(len(PT[0])):
#             ETD[j - 1][i + 1] = PT[0][i]
#             ETD[j][i + 1] = PT[8][i]

#         ETD[j + 1][1] = PT[10][0]
