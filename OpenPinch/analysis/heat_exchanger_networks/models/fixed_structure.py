"""Fixed-structure StageWise model for heat exchanger duty allocation.

The user supplies a complete network structure: which recovery matches sit in
which stage, which cold streams have a heater and which hot streams have a
cooler. All match binaries are fixed, stream splits stay free (non-isothermal
mixing), and the NLP only distributes duty between the listed exchangers.

Three objectives are supported:

* ``"total utility"`` with minimum approach temperatures per exchanger;
* ``"total area"`` with maximum hot and/or cold utility caps;
* ``"total cost"`` (annualised area cost plus utility cost) with only a small
  positive approach at both ends of every exchanger.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ._base import execution as _execution
from ._base import piecewise as _piecewise
from .stagewise import StageWiseModel

RecoveryKey = tuple[int, int, int]

FIXED_STRUCTURE_GOALS = frozenset({"total utility", "total area", "total cost"})


class FixedStructureStageWiseModel(StageWiseModel):
    """StageWise ESM with fixed binaries, free splits, and duty-only decisions."""

    def __init__(
        self,
        *,
        default_approach: float | None = None,
        recovery_approach: Mapping[RecoveryKey, float] | None = None,
        hot_utility_approach: Mapping[int, float] | None = None,
        cold_utility_approach: Mapping[int, float] | None = None,
        max_hot_utility: float | None = None,
        max_cold_utility: float | None = None,
        initial_recovery_duties: Mapping[RecoveryKey, float] | None = None,
        **kwargs: Any,
    ) -> None:
        goal = kwargs.get("minimisation_goal")
        if goal not in FIXED_STRUCTURE_GOALS:
            raise ValueError(
                "fixed-structure duty optimisation supports minimisation goals "
                f"{sorted(FIXED_STRUCTURE_GOALS)}; got {goal!r}."
            )
        self.default_approach = default_approach
        self.recovery_approach = dict(recovery_approach or {})
        self.hot_utility_approach = dict(hot_utility_approach or {})
        self.cold_utility_approach = dict(cold_utility_approach or {})
        self.max_hot_utility = max_hot_utility
        self.max_cold_utility = max_cold_utility
        self.initial_recovery_duties = dict(initial_recovery_duties or {})
        super().__init__(**kwargs)

    def setup(self) -> None:
        self.set_blank_input_parameters()
        self.get_model_parameters_from_solver_arrays()
        apply_approach_overrides(self)
        self.set_preprocessing()
        self.set_match_restrictions(self.z_restriction)
        self.set_stage_wise_superstructure()
        set_exchanger_approach_constraints(self)
        set_recovery_approach_definitions(self)
        set_utility_approach_constraints(self)
        set_utility_caps(self)
        set_initial_recovery_duties(self)
        self.set_obj()

    def set_obj(self) -> None:
        if self.minimisation_goal == "total area":
            set_total_area_objective(self)
            return
        super().set_obj()


def apply_approach_overrides(owner) -> None:
    """Replace contribution-based approach limits with user minimum approaches.

    A global ``default_approach`` replaces every recovery and utility limit.
    Per-exchanger recovery values set the pair limit to the smallest value on
    that hot/cold pair; stricter per-stage values are added as explicit
    constraints by :func:`set_exchanger_approach_constraints`.
    """

    if owner.default_approach is not None:
        value = float(owner.default_approach)
        owner.dT_r_period[...] = value
        owner.dT_hu_period[...] = value
        owner.dT_cu_period[...] = value

    pair_values: dict[tuple[int, int], float] = {}
    for (i, j, _k), value in owner.recovery_approach.items():
        pair_values[(i, j)] = min(float(value), pair_values.get((i, j), float("inf")))
    for (i, j), value in pair_values.items():
        owner.dT_r_period[:, i, j] = value
    for j, value in owner.hot_utility_approach.items():
        owner.dT_hu_period[:, j] = float(value)
    for i, value in owner.cold_utility_approach.items():
        owner.dT_cu_period[:, i] = float(value)

    owner.dT_r = owner.dT_r_period[0].copy()
    owner.dT_hu = owner.dT_hu_period[0].copy()
    owner.dT_cu = owner.dT_cu_period[0].copy()


def set_exchanger_approach_constraints(owner) -> None:
    """Add per-stage approach limits stricter than the pair-level bound."""

    for (i, j, k), value in owner.recovery_approach.items():
        if owner.z_allowed[i][j][k] <= 0:
            continue
        for n in range(owner.N_periods):
            if float(value) <= float(owner.dT_r_period[n][i][j]):
                continue
            owner.m.Equation(owner.theta_1_by_period[n][i][j][k] >= float(value))
            owner.m.Equation(owner.theta_2_by_period[n][i][j][k] >= float(value))


def set_recovery_approach_definitions(owner) -> None:
    """Tie each approach variable to the actual end temperature difference.

    The superstructure only bounds ``theta`` from above by the temperature
    difference (big-M relaxed). With every binary fixed at one the relaxation
    is exact, so equality makes ``theta`` the real approach, which keeps the
    area expressions and solution verification consistent for every objective.
    """

    for n in range(owner.N_periods):
        for i in range(owner.I):
            for j in range(owner.J):
                for k in range(owner.S):
                    if owner.z_allowed[i][j][k] <= 0:
                        continue
                    if owner.non_isothermal_model:
                        hot_end = (
                            owner.T_h_by_period[n][i][k]
                            - owner.T_c_out_y_by_period[n][j][i][k]
                        )
                        cold_end = (
                            owner.T_h_out_x_by_period[n][i][j][k]
                            - owner.T_c_by_period[n][j][k + 1]
                        )
                    else:
                        hot_end = (
                            owner.T_h_by_period[n][i][k] - owner.T_c_by_period[n][j][k]
                        )
                        cold_end = (
                            owner.T_h_by_period[n][i][k + 1]
                            - owner.T_c_by_period[n][j][k + 1]
                        )
                    owner.m.Equation(owner.theta_1_by_period[n][i][j][k] == hot_end)
                    owner.m.Equation(owner.theta_2_by_period[n][i][j][k] == cold_end)


def set_utility_approach_constraints(owner) -> None:
    """Enforce minimum approaches at both ends of listed utility exchangers.

    Segmented utilities already receive terminal approach constraints from the
    base model. For single-segment utilities the inlet end is fixed by data and
    is checked here; the outlet end depends on the duty split and becomes a
    constraint.
    """

    tol = float(owner.tol)
    for n in range(owner.N_periods):
        if not _piecewise._utility_is_segmented(owner, "hot"):
            for j in range(owner.J):
                if owner.z_hu_allowed[j] <= 0:
                    continue
                approach = float(owner.dT_hu_period[n][j])
                inlet_delta = float(
                    owner.T_hu_in_period[n][0] - owner.T_c_out_period[n][j]
                )
                if inlet_delta + tol < approach:
                    raise ValueError(
                        f"Heater on cold stream {_name(owner, 'cold', j)} cannot "
                        f"meet its {approach:g} K minimum approach: hot utility "
                        f"inlet is only {inlet_delta:g} K above the stream target."
                    )
                owner.m.Equation(
                    float(owner.T_hu_out_period[n][0]) - owner.T_c_by_period[n][j][0]
                    >= approach
                )
        if not _piecewise._utility_is_segmented(owner, "cold"):
            for i in range(owner.I):
                if owner.z_cu_allowed[i] <= 0:
                    continue
                approach = float(owner.dT_cu_period[n][i])
                outlet_delta = float(
                    owner.T_h_out_period[n][i] - owner.T_cu_in_period[n][0]
                )
                if outlet_delta + tol < approach:
                    raise ValueError(
                        f"Cooler on hot stream {_name(owner, 'hot', i)} cannot "
                        f"meet its {approach:g} K minimum approach: cold utility "
                        f"inlet is only {outlet_delta:g} K below the stream target."
                    )
                owner.m.Equation(
                    owner.T_h_by_period[n][i][owner.S]
                    - float(owner.T_cu_out_period[n][0])
                    >= approach
                )


def set_utility_caps(owner) -> None:
    """Bound total hot and cold utility duty in every operating period."""

    if owner.max_hot_utility is not None and any(
        owner.z_hu_allowed[j] > 0 for j in range(owner.J)
    ):
        for n in range(owner.N_periods):
            owner.m.Equation(
                owner.m.sum([owner.Q_h_by_period[n][j] for j in range(owner.J)])
                <= float(owner.max_hot_utility)
            )
    if owner.max_cold_utility is not None and any(
        owner.z_cu_allowed[i] > 0 for i in range(owner.I)
    ):
        for n in range(owner.N_periods):
            owner.m.Equation(
                owner.m.sum([owner.Q_c_by_period[n][i] for i in range(owner.I)])
                <= float(owner.max_cold_utility)
            )


def set_initial_recovery_duties(owner) -> None:
    """Warm-start recovery duties from the user's network where supplied."""

    for (i, j, k), duty in owner.initial_recovery_duties.items():
        if owner.z_allowed[i][j][k] <= 0:
            continue
        for n in range(owner.N_periods):
            _execution._set_value(owner, owner.Q_r_by_period[n][i][j][k], float(duty))


def _chen_lmtd(theta_1, theta_2):
    """Chen (1987) LMTD approximation with the source smoothing constant."""

    return (theta_1 * theta_2 * (theta_1 + theta_2) / 2 + 1e-3) ** (1 / 3)


def set_total_area_objective(owner) -> None:
    """Minimise total heat-transfer area of all listed exchangers."""

    if getattr(owner, "N_periods", 1) > 1:
        raise ValueError(
            "the total area objective supports one operating period; pass "
            "period_id to optimise one period."
        )
    terms = []
    for i in range(owner.I):
        for j in range(owner.J):
            for k in range(owner.S):
                if owner.z_allowed[i][j][k] <= 0:
                    continue
                terms.append(
                    owner.Q_r[i][j][k]
                    / (
                        owner.U_r[i][j]
                        * _chen_lmtd(owner.theta_1[i][j][k], owner.theta_2[i][j][k])
                    )
                )
    for j in range(owner.J):
        if owner.z_hu_allowed[j] <= 0:
            continue
        terms.append(
            owner.Q_h[j]
            / (
                owner.U_hu[j]
                * _chen_lmtd(
                    owner.T_hu_in[0] - owner.T_c_out[j],
                    owner.T_hu_out[0] - owner.T_c[j][0],
                )
            )
        )
    for i in range(owner.I):
        if owner.z_cu_allowed[i] <= 0:
            continue
        terms.append(
            owner.Q_c[i]
            / (
                owner.U_cu[i]
                * _chen_lmtd(
                    owner.T_h[i][owner.S] - owner.T_cu_out[0],
                    owner.T_h_out[i] - owner.T_cu_in[0],
                )
            )
        )
    if not terms:
        raise ValueError("the total area objective requires at least one exchanger.")
    owner.total_area_expr = owner.m.Intermediate(
        sum(terms[1:], terms[0]), name="Total heat transfer area"
    )
    owner.m.Minimize(owner.total_area_expr)


def _name(owner, side: str, index: int) -> str:
    names = getattr(owner, "cold_names" if side == "cold" else "hot_names", None)
    try:
        return str(names[index])
    except TypeError, IndexError:
        return f"{side}[{index}]"


__all__ = ["FIXED_STRUCTURE_GOALS", "FixedStructureStageWiseModel"]
