"""Private constants used by :mod:`OpenPinch.classes.problem_table`."""

from ...lib.enums import ProblemTableLabel

PT = ProblemTableLabel

INTERPOLATION_KEYS = (
    PT.H_HOT.value,
    PT.H_COLD.value,
    PT.H_NET.value,
    PT.H_NET_NP.value,
    PT.H_NET_A.value,
    PT.H_NET_V.value,
    PT.H_NET_PK.value,
    PT.H_NET_AI.value,
    PT.H_NET_UT.value,
    PT.H_NET_HOT.value,
    PT.H_NET_COLD.value,
    PT.H_NET_W_AIR.value,
    PT.H_NET_HP.value,
    PT.H_NET_RFRG.value,
    PT.H_HOT_UT.value,
    PT.H_COLD_UT.value,
    PT.H_HOT_BAL.value,
    PT.H_COLD_BAL.value,
    PT.H_HOT_HP.value,
    PT.H_COLD_HP.value,
    PT.H_NET_HOT_AFTR_HP.value,
    PT.H_NET_COLD_AFTR_HP.value,
    PT.H_NET_HOT_UT_AFTR_HP.value,
    PT.H_NET_COLD_UT_AFTR_HP.value,
    PT.H_HOT_RFRG.value,
    PT.H_COLD_RFRG.value,
    PT.H_NET_HOT_AFTR_RFRG.value,
    PT.H_NET_COLD_AFTR_RFRG.value,
    PT.H_NET_HOT_UT_AFTR_RFRG.value,
    PT.H_NET_COLD_UT_AFTR_RFRG.value,
)

HEAT_CAPACITY_PAIRS = (
    (PT.CP_HOT.value, PT.DELTA_H_HOT.value),
    (PT.CP_COLD.value, PT.DELTA_H_COLD.value),
    (PT.CP_NET.value, PT.DELTA_H_NET.value),
)
