"""Legacy placeholders for additional targeting routines pending refactor."""

# import pandas as pd
# import numpy as np
# from ..utils import *
# from ..lib.enums import *
# from ..classes import *
# from .support_methods import *

# __all__ = []

#######################################################################################################
# Public API --- TODO
#######################################################################################################


#######################################################################################################
# Helper functions
#######################################################################################################

############# Review and testing needed!!!!!!!!!!!!!!


############# Review and testing needed!!!!!!!!!!!!!!
# def _calc_exergy_gcc(z, pt_real, BCC, GCC_A):
#     """Determine Exergy Transfer Effectiveness including process and utility streams.
#     """
#     # Exergy Transfer Effectiveness proposed by Marmolejo-Correa, D., Gundersen, T., 2012.
#     # A comparison of exergy efficiency definitions with focus on low temperature processes.
#     # Energy 44, 477–489. https://doi.org/10.1016/j.energy.2012.06.001
#     x_source, x_sink, n_ETE = _calc_total_exergy(BCC)
#     z.exergy_sources = x_source
#     z.exergy_sinks = x_sink
#     z.ETE = n_ETE

#     GCC_X = z.Calc_ExGCC(GCC_A)
#     x_source, x_sink, n_ETE = _calc_total_exergy(pt_real, Col_T=0, Col_HCC=4, Col_CCC=7)

#     z.exergy_req_min = GCC_X[1][1]
#     z.exergy_des_min = GCC_X[1][-1]

#     return GCC_X

############# Review and testing needed!!!!!!!!!!!!!!
# def _calc_total_exergy(z: Zone, CC, x_source=0, x_sink=0, n_ETE=0, Col_T=0, Col_HCC=2, Col_CCC=4):
#     """Determines the source and sink exergy of a balanced CC."""
#     for i in range(1, len(CC[0])):
#         T_ex1 = compute_exergetic_temperature(CC[Col_T][i - 1], T_ref=z.config.T_ENV)
#         T_ex2 = compute_exergetic_temperature(CC[Col_T][i], T_ref=z.config.T_ENV)
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

# def Calc_ExGCC(z, GCC_A):
#     """Transposes a normal GCC (T-h) into a exergy GCC (Tx-X).
#     """
#     GCC_X = copy.deepcopy(GCC_A)
#     Min_X = 0
#     AbovePT = True
#     GCC_X[0][0] = compute_exergetic_temperature(GCC_A[0][0] + z.config.DTCONT / 2, T_ref=z.config.T_ENV)
#     GCC_X[1][0] = 0
#     i_upper = len(GCC_X[0]) + 1

#     # Transpose to exergetic temperature and exergy flow
#     i = 1
#     GCC_A_i = 1
#     while i <= i_upper and GCC_A_i < len(GCC_A[0]):
#         if AbovePT:
#             GCC_X[0][i] = compute_exergetic_temperature(GCC_A[0][GCC_A_i] + z.config.DTCONT / 2, T_ref=z.config.T_ENV)
#             GCC_X[1][i] = (GCC_A[1][GCC_A_i - 1] - GCC_A[1][GCC_A_i]) / (GCC_A[0][GCC_A_i - 1] - GCC_A[0][GCC_A_i])
#             GCC_X[1][i] = GCC_X[1][i - 1] - GCC_X[1][i] * (GCC_X[0][i - 1] - GCC_X[0][i])
#             if GCC_A[1][GCC_A_i] < tol:
#                 Min_X = GCC_X[1][i]
#                 for row in GCC_X:
#                     row += [0, 0]
#                 i += 2
#                 GCC_A_i += 1
#                 GCC_X[0][i] = compute_exergetic_temperature(GCC_A[0][GCC_A_i - 1] - z.config.DTCONT / 2, T_ref=z.config.T_ENV)
#                 GCC_X[1][i] = GCC_X[1][i - 1]
#                 AbovePT = False
#         else:
#             GCC_X[0][i] = compute_exergetic_temperature(GCC_A[0][GCC_A_i - 1] - z.config.DTCONT / 2, T_ref=z.config.T_ENV)
#             GCC_X[1][i] = (GCC_A[1][GCC_A_i - 2] - GCC_A[1][GCC_A_i - 1]) / (GCC_A[0][GCC_A_i - 2] - GCC_A[0][GCC_A_i - 1])
#             GCC_X[1][i] = GCC_X[1][i - 1] - GCC_X[1][i] * (GCC_X[0][i - 1] - GCC_X[0][i])
#         i += 1
#         GCC_A_i += 1

#     # Shift Exergy GCC appropriately
#     for i in range(1, len(GCC_X[0])):
#         GCC_X[1][i] = GCC_X[1][i] + abs(Min_X)
#         if abs(GCC_X[1][i]) < tol:
#             GCC_X[1][i] = 0

#     return GCC_X


# def Calc_GCC_AI(z, pt_real, GCC_N):
#     """Returns a simplified array for the assisted integration GCC.
#     """
#     GCC_AI = [ [ None for j in range(len(pt_real[0]))] for i in range(2)]
#     for i in range(len(pt_real[0])):
#         GCC_AI[0][i] = pt_real[0][i]
#         GCC_AI[1][i] = pt_real[PT.H_NET.value][i] - GCC_N[1][i]
#     return GCC_AI
