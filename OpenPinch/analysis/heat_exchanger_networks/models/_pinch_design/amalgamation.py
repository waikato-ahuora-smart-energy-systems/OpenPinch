"""Above- and below-pinch network amalgamation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..stagewise import StageWiseModel

if TYPE_CHECKING:
    from ..pinch_decomposition import PinchDecompModel


def amalgamate_networks(
    owner,
    *,
    below_case: "PinchDecompModel",
    above_case: "PinchDecompModel",
) -> StageWiseModel:
    """Amalgamate solved above/below-pinch side models into one network."""

    above_required = bool(above_case.side_required)
    below_required = bool(below_case.side_required)

    amalgamated = StageWiseModel(
        name="amalgamated",
        framework=owner.framework,
        solver=owner.solver,
        solver_arrays=owner.solver_arrays,
        stages=(above_case.S if above_required else 0)
        + (below_case.S if below_required else 0),
        dTmin=owner.dTmin,
        z_restriction=owner.z_restriction,
        min_dqda=owner.min_dqda,
        minimisation_goal="total utility",
        non_isothermal_model=owner.non_isothermal_model,
        integers=True,
        tol=1e-3,
        solver_options=owner.solver_options,
    )
    if (above_required and above_case.mSuccess == 0) or (
        below_required and below_case.mSuccess == 0
    ):
        raise ValueError(
            "Pinch Decomposition failed: "
            f"Above {above_case.mSuccess} Below {below_case.mSuccess} "
            f"dTmin {owner.dTmin}"
        )

    if above_required and above_case.mSuccess == 1:
        amalgamated.mSuccess = above_case.mSuccess
        amalgamated.TAC = above_case.TAC
        amalgamated.solve_time = above_case.solve_time
        for i in range(owner.I):
            for j in range(owner.J):
                for k in range(above_case.S):
                    owner._copy_recovery_match(amalgamated, above_case, i, j, k, k)
        for i in range(owner.I):
            for k in range(above_case.K):
                for n in range(amalgamated.N_periods):
                    value = (
                        above_case.T_h_by_period[n][i][k][0]
                        if above_case.z_i_active_period[n][i] > 0
                        else amalgamated.T_h_in_period[n][i]
                    )
                    amalgamated.T_h_by_period[n][i][k].VALUE.value = [value]
        for j in range(owner.J):
            for k in range(above_case.K):
                for n in range(amalgamated.N_periods):
                    value = (
                        above_case.T_c_by_period[n][j][k][0]
                        if above_case.z_j_active_period[n][j] > 0
                        else amalgamated.T_c_out_period[n][j]
                    )
                    amalgamated.T_c_by_period[n][j][k].VALUE.value = [value]
        for j in range(owner.J):
            for n in range(amalgamated.N_periods):
                amalgamated.Q_h_by_period[n][j].VALUE.value = [
                    above_case.Q_h_by_period[n][j][0]
                ]
            amalgamated.z_hu[j].VALUE.value = [above_case.z_hu[j][0]]
        if not below_required:
            for i in range(owner.I):
                for n in range(amalgamated.N_periods):
                    amalgamated.Q_c_by_period[n][i].VALUE.value = [0]
                amalgamated.z_cu[i].VALUE.value = [0]
                amalgamated.minimisation_goal = "hot utility"

    if below_required and below_case.mSuccess == 1:
        amalgamated.mSuccess = below_case.mSuccess
        amalgamated.TAC = below_case.TAC
        amalgamated.solve_time = below_case.solve_time
        if not above_required:
            above_case.S = 0
            above_case.K = 0
        for i in range(owner.I):
            for j in range(owner.J):
                for k in range(above_case.S, amalgamated.S):
                    owner._copy_recovery_match(
                        amalgamated,
                        below_case,
                        i,
                        j,
                        k - above_case.S,
                        k,
                    )
        for i in range(owner.I):
            for k in range(above_case.K, amalgamated.K):
                for n in range(amalgamated.N_periods):
                    if below_case.z_i_active_period[n][i] > 0:
                        value = (
                            below_case.T_h_by_period[n][i][k - above_case.K + 1][0]
                            if above_required
                            else round(below_case.T_h_by_period[n][i][k][0], 5)
                        )
                    else:
                        value = amalgamated.T_h_out_period[n][i]
                    amalgamated.T_h_by_period[n][i][k].VALUE.value = [value]
        for j in range(owner.J):
            for k in range(above_case.K, amalgamated.K):
                for n in range(amalgamated.N_periods):
                    if below_case.z_j_active_period[n][j] > 0:
                        value = (
                            below_case.T_c_by_period[n][j][k - above_case.K + 1][0]
                            if above_required
                            else round(below_case.T_c_by_period[n][j][k][0], 5)
                        )
                    else:
                        value = amalgamated.T_c_in_period[n][j]
                    amalgamated.T_c_by_period[n][j][k].VALUE.value = [value]
        for i in range(owner.I):
            for n in range(amalgamated.N_periods):
                amalgamated.Q_c_by_period[n][i].VALUE.value = [
                    round(below_case.Q_c_by_period[n][i][0], 5)
                ]
            amalgamated.z_cu[i].VALUE.value = [below_case.z_cu[i][0]]
        if not above_required:
            for j in range(owner.J):
                for n in range(amalgamated.N_periods):
                    amalgamated.Q_h_by_period[n][j].VALUE.value = [0]
                amalgamated.z_hu[j].VALUE.value = [0]
                amalgamated.minimisation_goal = "cold utility"

    if (
        above_required
        and below_required
        and above_case.mSuccess == 1
        and below_case.mSuccess == 1
    ):
        amalgamated.mSuccess = 1
        amalgamated.TAC = below_case.TAC + above_case.TAC
        amalgamated.solve_time = above_case.solve_time + below_case.solve_time
        amalgamated.S = above_case.S + below_case.S

    amalgamated.K = amalgamated.S + 1
    amalgamated.z_allowed = [
        [
            [
                (
                    1
                    if (
                        max(
                            amalgamated.Q_r_by_period[n][i][j][k][0]
                            for n in range(amalgamated.N_periods)
                        )
                        if hasattr(amalgamated, "Q_r_by_period")
                        else amalgamated.Q_r[i][j][k][0]
                    )
                    > owner.tol
                    else 0
                )
                for k in range(amalgamated.S)
            ]
            for j in range(amalgamated.J)
        ]
        for i in range(amalgamated.I)
    ]
    return amalgamated


def _copy_recovery_match(
    owner,
    target: StageWiseModel,
    source: "PinchDecompModel",
    i: int,
    j: int,
    source_stage: int,
    target_stage: int,
) -> None:
    target.z[i][j][target_stage].VALUE.value = [source.z[i][j][source_stage][0]]
    if target.N_periods != source.N_periods:
        raise ValueError("PDM side period counts must match during amalgamation.")
    for n in range(target.N_periods):
        target.Q_r_by_period[n][i][j][target_stage].VALUE.value = [
            source.Q_r_by_period[n][i][j][source_stage][0]
        ]
        target.theta_1_by_period[n][i][j][target_stage].VALUE.value = [
            source.theta_1_by_period[n][i][j][source_stage][0]
        ]
        target.theta_2_by_period[n][i][j][target_stage].VALUE.value = [
            source.theta_2_by_period[n][i][j][source_stage][0]
        ]
    if source.non_isothermal_model:
        for n in range(target.N_periods):
            target.X_by_period[n][i][j][target_stage].VALUE.value = [
                source.X_by_period[n][i][j][source_stage][0]
            ]
            target.Y_by_period[n][j][i][target_stage].VALUE.value = [
                source.Y_by_period[n][j][i][source_stage][0]
            ]
            target.T_h_out_x_by_period[n][i][j][target_stage].VALUE.value = [
                source.T_h_out_x_by_period[n][i][j][source_stage][0]
            ]
            target.T_c_out_y_by_period[n][j][i][target_stage].VALUE.value = [
                source.T_c_out_y_by_period[n][j][i][source_stage][0]
            ]
