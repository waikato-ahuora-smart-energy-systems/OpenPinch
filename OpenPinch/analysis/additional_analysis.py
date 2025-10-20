


# def Calc_GCC_AI(z, pt_real, GCC_N):
#     """Returns a simplified array for the assisted integration GCC.
#     """
#     GCC_AI = [ [ None for j in range(len(pt_real[0]))] for i in range(2)]
#     for i in range(len(pt_real[0])):
#         GCC_AI[0][i] = pt_real[0][i]
#         GCC_AI[1][i] = pt_real[PT.H_NET.value][i] - GCC_N[1][i]
#     return GCC_AI
