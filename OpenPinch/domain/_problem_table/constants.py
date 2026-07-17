"""Private constants used by :mod:`OpenPinch.domain.problem_table`."""

from ..enums import ProblemTableLabel

INTERPOLATION_KEYS = (
    ProblemTableLabel.H_HOT.value,
    ProblemTableLabel.H_COLD.value,
    ProblemTableLabel.H_NET.value,
    ProblemTableLabel.H_NET_NP.value,
    ProblemTableLabel.H_NET_A.value,
    ProblemTableLabel.H_NET_V.value,
    ProblemTableLabel.H_NET_PK.value,
    ProblemTableLabel.H_NET_AI.value,
    ProblemTableLabel.H_NET_UT.value,
    ProblemTableLabel.H_NET_HOT.value,
    ProblemTableLabel.H_NET_COLD.value,
    ProblemTableLabel.H_NET_W_AIR.value,
    ProblemTableLabel.H_NET_HP.value,
    ProblemTableLabel.H_NET_RFRG.value,
    ProblemTableLabel.H_HOT_UT.value,
    ProblemTableLabel.H_COLD_UT.value,
    ProblemTableLabel.H_HOT_BAL.value,
    ProblemTableLabel.H_COLD_BAL.value,
    ProblemTableLabel.H_HOT_HP.value,
    ProblemTableLabel.H_COLD_HP.value,
    ProblemTableLabel.H_NET_HOT_AFTR_HP.value,
    ProblemTableLabel.H_NET_COLD_AFTR_HP.value,
    ProblemTableLabel.H_NET_HOT_UT_AFTR_HP.value,
    ProblemTableLabel.H_NET_COLD_UT_AFTR_HP.value,
    ProblemTableLabel.H_HOT_RFRG.value,
    ProblemTableLabel.H_COLD_RFRG.value,
    ProblemTableLabel.H_NET_HOT_AFTR_RFRG.value,
    ProblemTableLabel.H_NET_COLD_AFTR_RFRG.value,
    ProblemTableLabel.H_NET_HOT_UT_AFTR_RFRG.value,
    ProblemTableLabel.H_NET_COLD_UT_AFTR_RFRG.value,
)

HEAT_CAPACITY_PAIRS = (
    (ProblemTableLabel.CP_HOT.value, ProblemTableLabel.DELTA_H_HOT.value),
    (ProblemTableLabel.CP_COLD.value, ProblemTableLabel.DELTA_H_COLD.value),
    (ProblemTableLabel.CP_NET.value, ProblemTableLabel.DELTA_H_NET.value),
)
