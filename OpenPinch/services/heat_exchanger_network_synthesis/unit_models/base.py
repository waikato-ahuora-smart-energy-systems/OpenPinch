"""Base setup for migrated heat exchanger network equation kernels."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ....utils.heat_exchanger import compute_LMTD_from_dts
from ..common.indexing import build_index_grid
from ..common.solver import backend
from ..common.solver.arrays import PreparedSolverArrays

logger = logging.getLogger(__name__)


class BaseHeatExchangerNetworkModel(ABC):
    """Shared private state for migrated PDM/TDM/ESM equation models.

    The constructor mirrors the source OpenHENS solver defaults, but it accepts
    OpenPinch-prepared solver arrays instead of a CSV path. This layer owns the
    guarded GEKKO backend setup, source-shaped array normalization, inherited
    topology restrictions, common diagnostics, and helper equations that are
    stable across the moved private ``PinchDecompModel`` and ``StageWiseModel``.
    HENS-08 still owns topology evolution and stage-reduction behavior; those
    remain outside the base contract.
    """

    def __init__(
        self,
        name: str,
        framework: Literal["PDM", "TDM", "ESM"],
        solver: Literal["couenne", "ipopt-pyomo", "ipopt-GEKKO", "apopt"],
        solver_arrays: PreparedSolverArrays,
        dTmin: float,
        z_restriction: list | None,
        min_dqda: float,
        minimisation_goal: Literal[
            "hot utility",
            "total utility",
            "utility costs",
            "heat recovery",
            "total cost",
            "variable total cost",
        ],
        non_isothermal_model: bool,
        integers: bool,
        tol: float,
        solver_options: Mapping[str, Any] | Sequence[str] | None = None,
        import_file: Path | None = None,
    ) -> None:
        self.name = name
        self.framework = framework
        self.solver = solver
        self.solver_arrays = solver_arrays
        self.import_file = import_file
        self.dTmin = dTmin
        self.z_restriction = z_restriction
        self.min_dqda = min_dqda
        self.minimisation_goal = minimisation_goal
        self.non_isothermal_model = non_isothermal_model
        self.integers = integers
        self.tol = tol
        self.solver_options = solver_options

        self.solve_time = None
        self.solver_run = None
        self._piecewise_active_mappings: list[dict[str, Any]] = []

        self.setup_model()
        self.setup()

    def setup_model(self) -> None:
        """Create and configure the GEKKO model behind optional guards."""

        self.m = backend.create_gekko_model(remote=False)
        self.mSuccess: int = 0
        self.solver_run = backend.configure_gekko_solver(
            self.m,
            self.solver,
            solver_options=self.solver_options,
        )

    @abstractmethod
    def setup(self) -> None:
        """Create concrete equation variables, constraints, and objective."""

    @abstractmethod
    def set_preprocessing(self) -> None:
        """Populate model dimensions and derived solver constants."""

    @abstractmethod
    def set_stage_wise_superstructure(self) -> None:
        """Create the stage-wise superstructure in concrete model slices."""

    @abstractmethod
    def set_obj(self) -> None:
        """Attach the concrete objective formula unchanged from OpenHENS."""

    @abstractmethod
    def get_post_process(self) -> None:
        """Extract solved arrays after a successful concrete solve."""

    def _solver_value(self, value: Any) -> float:
        try:
            return float(value[0])
        except TypeError, IndexError, KeyError:
            return float(value)

    def _set_value(
        self, variable: Any, value: float, *, brackets: bool = False
    ) -> None:
        """Assign GEKKO values while preserving source bound-clamping behavior."""

        if type(variable).__name__ == "GKVariable":
            if variable.lower is not None:
                value = max(variable.lower, value)
            if variable.upper is not None:
                value = min(variable.upper, value)
            variable.VALUE.value = [value] if brackets else value
            return
        if type(variable).__name__ == "GKParameter":
            variable.VALUE.value = [value] if brackets else value
            return

    def _post_process_lmtd(
        self,
        delta_1: float,
        delta_2: float,
        active: float,
        *,
        formula_allowed: bool,
        fallback_delta: float | None = None,
    ) -> float:
        """Return source-compatible post-process LMTD.

        Heat exchanger network synthesis owns the OpenHENS active-unit and
        dTmin/tolerance gates.
        Once those gates pass, the shared OpenPinch heat-exchanger utility owns
        the positive endpoint logarithmic-mean formula.
        """

        if not formula_allowed:
            return (delta_1 if fallback_delta is None else fallback_delta) * active
        return active * float(compute_LMTD_from_dts(delta_1, delta_2))

    def _apply_segment_recovery_areas(self, q_r) -> None:
        """Replace aggregate-CP recovery areas with ordered local slice totals."""
        if not hasattr(self, "solver_arrays"):
            return
        arrays = self.solver_arrays.arrays
        if not {"hot_segment_count", "cold_segment_count"}.issubset(arrays):
            return
        if not (
            np.any(np.asarray(arrays["hot_segment_count"]) > 1)
            or np.any(np.asarray(arrays["cold_segment_count"]) > 1)
        ):
            return

        from ..common.solver.piecewise import (
            duty_aligned_area_contributions,
            profile_from_solver_arrays,
        )

        contribution_grid = [
            [
                [[() for _k in range(self.S)] for _j in range(self.J)]
                for _i in range(self.I)
            ]
            for _n in range(self.N_periods)
        ]
        area_grid = [
            [
                [[0.0 for _k in range(self.S)] for _j in range(self.J)]
                for _i in range(self.I)
            ]
            for _n in range(self.N_periods)
        ]
        for n in range(self.N_periods):
            period = str(self.period_ids[n])
            for i in range(self.I):
                hot_profile = profile_from_solver_arrays(
                    self.solver_arrays,
                    side="hot",
                    parent_index=i,
                    period_index=n,
                )
                for j in range(self.J):
                    cold_profile = profile_from_solver_arrays(
                        self.solver_arrays,
                        side="cold",
                        parent_index=j,
                        period_index=n,
                    )
                    if len(hot_profile.duties) == len(cold_profile.duties) == 1:
                        for k in range(self.S):
                            area_grid[n][i][j][k] = self.area_r_by_period[n][i][j][k]
                        continue
                    for k in range(self.S):
                        duty = float(q_r[n][i][j][k])
                        if duty <= self.tol:
                            continue
                        contributions = duty_aligned_area_contributions(
                            hot_profile,
                            cold_profile,
                            duty=duty,
                            hot_inlet_temperature=self._active_binary_value(
                                self.T_h_by_period[n][i][k]
                            ),
                            cold_inlet_temperature=self._active_binary_value(
                                self.T_c_by_period[n][j][k + 1]
                            ),
                            period=period,
                            tolerance=self.tol,
                        )
                        contribution_grid[n][i][j][k] = contributions
                        area_grid[n][i][j][k] = sum(
                            contribution.area for contribution in contributions
                        )
        self.segment_area_contributions_by_period = contribution_grid
        self.area_r_by_period = area_grid
        self.area_r = [
            [
                [
                    max(area_grid[n][i][j][k] for n in range(self.N_periods))
                    for k in range(self.S)
                ]
                for j in range(self.J)
            ]
            for i in range(self.I)
        ]

    def _apply_segment_utility_areas(self, q_h, q_c) -> None:
        """Use local process segments for hot- and cold-utility area totals."""
        if not hasattr(self, "solver_arrays"):
            return
        if not (
            any(self._solver_parent_is_segmented("hot", i) for i in range(self.I))
            or any(self._solver_parent_is_segmented("cold", j) for j in range(self.J))
            or self._utility_is_segmented("hot")
            or self._utility_is_segmented("cold")
        ):
            return
        from ..common.solver.piecewise import (
            duty_aligned_area_contributions,
            profile_from_solver_arrays,
            utility_thermal_profile,
        )

        hot_utility_identity = self.solver_arrays.utility_identities["hot_utilities"][0]
        cold_utility_identity = self.solver_arrays.utility_identities["cold_utilities"][
            0
        ]
        self.segment_area_hu_contributions_by_period = [
            [() for _j in range(self.J)] for _n in range(self.N_periods)
        ]
        self.segment_area_cu_contributions_by_period = [
            [() for _i in range(self.I)] for _n in range(self.N_periods)
        ]
        for n in range(self.N_periods):
            period = str(self.period_ids[n])
            for j in range(self.J):
                duty = float(q_h[n][j])
                if duty <= self.tol or not (
                    self._solver_parent_is_segmented("cold", j)
                    or self._utility_is_segmented("hot")
                ):
                    continue
                hot_utility_profile = (
                    profile_from_solver_arrays(
                        self.solver_arrays,
                        side="hot_utility",
                        parent_index=0,
                        period_index=n,
                    )
                    if self._utility_is_segmented("hot")
                    else utility_thermal_profile(
                        identity=hot_utility_identity,
                        inlet_temperature=self.T_hu_in_period[n][0],
                        outlet_temperature=self.T_hu_out_period[n][0],
                        duty=duty,
                        heat_transfer_coefficient=self.htc_hu_period[n][0],
                    )
                )
                contributions = duty_aligned_area_contributions(
                    hot_utility_profile,
                    profile_from_solver_arrays(
                        self.solver_arrays,
                        side="cold",
                        parent_index=j,
                        period_index=n,
                    ),
                    duty=duty,
                    hot_inlet_temperature=self.T_hu_in_period[n][0],
                    cold_inlet_temperature=self._active_binary_value(
                        self.T_c_by_period[n][j][0]
                    ),
                    period=period,
                    tolerance=self.tol,
                )
                self.segment_area_hu_contributions_by_period[n][j] = contributions
                self.area_hu_by_period[n][j] = sum(
                    contribution.area for contribution in contributions
                )

            for i in range(self.I):
                duty = float(q_c[n][i])
                if duty <= self.tol or not (
                    self._solver_parent_is_segmented("hot", i)
                    or self._utility_is_segmented("cold")
                ):
                    continue
                cold_utility_profile = (
                    profile_from_solver_arrays(
                        self.solver_arrays,
                        side="cold_utility",
                        parent_index=0,
                        period_index=n,
                    )
                    if self._utility_is_segmented("cold")
                    else utility_thermal_profile(
                        identity=cold_utility_identity,
                        inlet_temperature=self.T_cu_in_period[n][0],
                        outlet_temperature=self.T_cu_out_period[n][0],
                        duty=duty,
                        heat_transfer_coefficient=self.htc_cu_period[n][0],
                    )
                )
                contributions = duty_aligned_area_contributions(
                    profile_from_solver_arrays(
                        self.solver_arrays,
                        side="hot",
                        parent_index=i,
                        period_index=n,
                    ),
                    cold_utility_profile,
                    duty=duty,
                    hot_inlet_temperature=self._active_binary_value(
                        self.T_h_by_period[n][i][self.S]
                    ),
                    cold_inlet_temperature=self.T_cu_in_period[n][0],
                    period=period,
                    tolerance=self.tol,
                )
                self.segment_area_cu_contributions_by_period[n][i] = contributions
                self.area_cu_by_period[n][i] = sum(
                    contribution.area for contribution in contributions
                )

    def _segment_exact_dqda(
        self,
        *,
        period_index: int,
        hot_parent_index: int,
        cold_parent_index: int,
        duty: float,
        hot_inlet_temperature: float,
        cold_inlet_temperature: float,
    ) -> float | None:
        """Return a local numerical dQ/dA from ordered segment-summed area."""
        if not (
            self._solver_parent_is_segmented("hot", hot_parent_index)
            or self._solver_parent_is_segmented("cold", cold_parent_index)
        ):
            return None

        from ..common.solver.piecewise import (
            duty_aligned_area_contributions,
            profile_from_solver_arrays,
        )

        hot_profile = profile_from_solver_arrays(
            self.solver_arrays,
            side="hot",
            parent_index=hot_parent_index,
            period_index=period_index,
        )
        cold_profile = profile_from_solver_arrays(
            self.solver_arrays,
            side="cold",
            parent_index=cold_parent_index,
            period_index=period_index,
        )
        hot_start = hot_profile.heat_at_temperature(hot_inlet_temperature)
        cold_start = cold_profile.heat_at_temperature(cold_inlet_temperature)
        maximum_duty = min(
            hot_profile.total_duty - hot_start,
            cold_profile.total_duty - cold_start,
        )
        epsilon = max(maximum_duty * 1e-5, self.tol * 10.0, 1e-6)
        lower_duty = max(0.0, duty - epsilon)
        upper_duty = min(maximum_duty, duty + epsilon)
        if upper_duty - lower_duty <= self.tol:
            return None

        def area_at(value: float) -> float:
            if value <= self.tol:
                return 0.0
            contributions = duty_aligned_area_contributions(
                hot_profile,
                cold_profile,
                duty=value,
                hot_inlet_temperature=hot_inlet_temperature,
                cold_inlet_temperature=cold_inlet_temperature,
                period=str(self.period_ids[period_index]),
                tolerance=self.tol,
            )
            return sum(contribution.area for contribution in contributions)

        try:
            area_delta = area_at(upper_duty) - area_at(lower_duty)
        except ValueError:
            return None
        if area_delta <= self.tol:
            return None
        return (upper_duty - lower_duty) / area_delta

    def _register_piecewise_mapping(self, mapping) -> None:
        if mapping is not None:
            self._piecewise_active_mappings.append(mapping)

    def _utility_is_segmented(self, side: str) -> bool:
        if not hasattr(self, "solver_arrays"):
            return False
        values = self.solver_arrays.arrays.get(f"{side}_utility_parent_segmented")
        return bool(values is not None and np.asarray(values, dtype=bool)[0])

    def _utility_cost_expression(
        self,
        side: str,
        period_index: int,
        heat_duty,
        *,
        name: str,
    ):
        """Return the flat or exact piecewise utility-cost solver expression."""
        price_attr = "hu_cost_period" if side == "hot" else "cu_cost_period"
        if not self._utility_is_segmented(side):
            if hasattr(self, price_attr):
                price = getattr(self, price_attr)[period_index][0]
            else:
                price = getattr(self, "hu_cost" if side == "hot" else "cu_cost")[0]
            return price * heat_duty

        from ..common.solver.piecewise import (
            add_piecewise_cost_mapping,
            profile_from_solver_arrays,
        )

        profile = profile_from_solver_arrays(
            self.solver_arrays,
            side=f"{side}_utility",
            parent_index=0,
            period_index=period_index,
        )
        coordinate = self.m.Var(
            value=0.0,
            lb=0.0,
            ub=profile.total_duty,
            name=f"{name}_duty",
        )
        cost = self.m.Var(
            value=0.0,
            lb=0.0,
            ub=float(profile.cumulative_costs[-1]),
            name=f"{name}_cost",
        )
        self.m.Equation(coordinate == heat_duty)
        self._register_piecewise_mapping(
            add_piecewise_cost_mapping(
                self.m,
                coordinate,
                cost,
                profile,
                name=name,
                integer_capable=self.solver in {"apopt", "couenne"},
            )
        )
        return cost

    def _utility_cost_value(
        self,
        side: str,
        period_index: int,
        heat_duty: float,
    ) -> float:
        """Return exact solved utility cost for reporting and verification."""
        duty = max(float(heat_duty), 0.0)
        price_attr = "hu_cost_period" if side == "hot" else "cu_cost_period"
        if not self._utility_is_segmented(side):
            return float(getattr(self, price_attr)[period_index][0]) * duty

        from ..common.solver.piecewise import profile_from_solver_arrays

        profile = profile_from_solver_arrays(
            self.solver_arrays,
            side=f"{side}_utility",
            parent_index=0,
            period_index=period_index,
        )
        if duty > profile.total_duty + self.tol:
            raise ValueError(
                f"Solved {side} utility duty exceeds its segmented profile capacity."
            )
        return profile.cost_at_heat(duty)

    def _update_piecewise_active_segments(self) -> bool:
        changed = False
        for mapping in self._piecewise_active_mappings:
            coordinate = self._solver_value(mapping["heat_coordinate"])
            segment_index_at_heat = mapping.get(
                "segment_index_at_heat",
                mapping["profile"].segment_index_at_heat,
            )
            next_segment = segment_index_at_heat(coordinate)
            if next_segment == mapping["active_segment"]:
                continue
            for index, selector in enumerate(mapping["selectors"]):
                self._set_value(selector, 1.0 if index == next_segment else 0.0)
            mapping["active_segment"] = next_segment
            changed = True
        return changed

    def _set_piecewise_stage_heat_coordinates(self) -> None:
        """Add parent cumulative-Q balances and ordered T(Q) mappings by period."""
        if not hasattr(self, "solver_arrays"):
            self._segmented_hot_parents = np.zeros(self.I, dtype=bool)
            self._segmented_cold_parents = np.zeros(self.J, dtype=bool)
            return
        arrays = self.solver_arrays.arrays
        self._set_segmented_utility_capacity_constraints()
        self._set_piecewise_utility_outlet_states()
        self._segmented_hot_parents = (
            np.asarray(arrays.get("hot_segment_count", np.ones(self.I)), dtype=int) > 1
        )
        self._segmented_cold_parents = (
            np.asarray(arrays.get("cold_segment_count", np.ones(self.J)), dtype=int) > 1
        )
        if not (
            np.any(self._segmented_hot_parents) or np.any(self._segmented_cold_parents)
        ):
            return

        from ..common.solver.piecewise import (
            add_piecewise_temperature_mapping,
            profile_from_solver_arrays,
        )

        integer_capable = self.solver in {"apopt", "couenne"}
        self.Q_coordinate_h_by_period = [
            [[None for _k in range(self.K)] for _i in range(self.I)]
            for _n in range(self.N_periods)
        ]
        self.Q_coordinate_c_by_period = [
            [[None for _k in range(self.K)] for _j in range(self.J)]
            for _n in range(self.N_periods)
        ]
        for n in range(self.N_periods):
            for i in range(self.I):
                if not self._segmented_hot_parents[i]:
                    continue
                if (
                    hasattr(self, "z_i_active_period")
                    and self.z_i_active_period[n][i] <= 0
                ):
                    continue
                profile = profile_from_solver_arrays(
                    self.solver_arrays,
                    side="hot",
                    parent_index=i,
                    period_index=n,
                ).clipped(self.T_h_in_period[n][i], self.T_h_out_period[n][i])
                for k in range(self.K):
                    initial_q = profile.total_duty * k / max(self.S, 1)
                    coordinate = (
                        self.m.Param(value=0.0, name=f"Qcoord_H{i}_B0_period{n}")
                        if k == 0
                        else self.m.Var(
                            value=initial_q,
                            lb=0.0,
                            ub=profile.total_duty,
                            name=f"Qcoord_H{i}_B{k}_period{n}",
                        )
                    )
                    self.Q_coordinate_h_by_period[n][i][k] = coordinate
                    self._register_piecewise_mapping(
                        add_piecewise_temperature_mapping(
                            self.m,
                            coordinate,
                            self.T_h_by_period[n][i][k],
                            profile,
                            name=f"TQ_H{i}_B{k}_period{n}",
                            integer_capable=integer_capable,
                            initial_segment=profile.segment_index_at_heat(initial_q),
                        )
                    )
                self.m.Equations(
                    [
                        self.Q_coordinate_h_by_period[n][i][k + 1]
                        - self.Q_coordinate_h_by_period[n][i][k]
                        - sum(self.Q_r_by_period[n][i][j][k] for j in range(self.J))
                        == 0.0
                        for k in range(self.S)
                    ]
                )

            for j in range(self.J):
                if not self._segmented_cold_parents[j]:
                    continue
                if (
                    hasattr(self, "z_j_active_period")
                    and self.z_j_active_period[n][j] <= 0
                ):
                    continue
                profile = profile_from_solver_arrays(
                    self.solver_arrays,
                    side="cold",
                    parent_index=j,
                    period_index=n,
                ).clipped(self.T_c_in_period[n][j], self.T_c_out_period[n][j])
                for k in range(self.K):
                    initial_q = profile.total_duty * (self.S - k) / max(self.S, 1)
                    coordinate = (
                        self.m.Param(value=0.0, name=f"Qcoord_C{j}_B{self.S}_period{n}")
                        if k == self.S
                        else self.m.Var(
                            value=initial_q,
                            lb=0.0,
                            ub=profile.total_duty,
                            name=f"Qcoord_C{j}_B{k}_period{n}",
                        )
                    )
                    self.Q_coordinate_c_by_period[n][j][k] = coordinate
                    self._register_piecewise_mapping(
                        add_piecewise_temperature_mapping(
                            self.m,
                            coordinate,
                            self.T_c_by_period[n][j][k],
                            profile,
                            name=f"TQ_C{j}_B{k}_period{n}",
                            integer_capable=integer_capable,
                            initial_segment=profile.segment_index_at_heat(initial_q),
                        )
                    )
                self.m.Equations(
                    [
                        self.Q_coordinate_c_by_period[n][j][k]
                        - self.Q_coordinate_c_by_period[n][j][k + 1]
                        - sum(self.Q_r_by_period[n][i][j][k] for i in range(self.I))
                        == 0.0
                        for k in range(self.S)
                    ]
                )

    def _set_segmented_utility_capacity_constraints(self) -> None:
        """Bound selected utility load by each explicit ordered profile."""
        from ..common.solver.piecewise import profile_from_solver_arrays

        for n in range(self.N_periods):
            if self._utility_is_segmented("hot"):
                hot_profile = profile_from_solver_arrays(
                    self.solver_arrays,
                    side="hot_utility",
                    parent_index=0,
                    period_index=n,
                )
                self.m.Equation(sum(self.Q_h_by_period[n]) <= hot_profile.total_duty)
            if self._utility_is_segmented("cold"):
                cold_profile = profile_from_solver_arrays(
                    self.solver_arrays,
                    side="cold_utility",
                    parent_index=0,
                    period_index=n,
                )
                self.m.Equation(sum(self.Q_c_by_period[n]) <= cold_profile.total_duty)

    def _set_piecewise_utility_outlet_states(self) -> None:
        """Map aggregate utility duty to outlet temperature and local ``dt_cont``."""
        from ..common.solver.piecewise import (
            add_piecewise_temperature_contribution_mapping,
            add_piecewise_temperature_mapping,
            profile_from_solver_arrays,
        )

        self.T_hu_solved_out_by_period = [[] for _n in range(self.N_periods)]
        self.T_cu_solved_out_by_period = [[] for _n in range(self.N_periods)]
        self.T_hu_out_cont_by_period = [[] for _n in range(self.N_periods)]
        self.T_cu_out_cont_by_period = [[] for _n in range(self.N_periods)]
        self.T_hu_in_cont_by_period = []
        self.T_cu_in_cont_by_period = []
        for n in range(self.N_periods):
            for side, loads in (
                ("hot", self.Q_h_by_period[n]),
                ("cold", self.Q_c_by_period[n]),
            ):
                scalar_contribution = float(
                    (
                        self.T_hu_cont_period[n][0]
                        if side == "hot"
                        else self.T_cu_cont_period[n][0]
                    )
                )
                inlet_contribution = scalar_contribution
                if self._utility_is_segmented(side):
                    profile = profile_from_solver_arrays(
                        self.solver_arrays,
                        side=f"{side}_utility",
                        parent_index=0,
                        period_index=n,
                    )
                    inlet_contribution = float(profile.temperature_contributions[0])
                getattr(self, f"T_{side[0]}u_in_cont_by_period").append(
                    inlet_contribution
                )

                solved_outlets = getattr(self, f"T_{side[0]}u_solved_out_by_period")[n]
                outlet_contributions = getattr(
                    self, f"T_{side[0]}u_out_cont_by_period"
                )[n]
                for match_index, load in enumerate(loads):
                    if not self._utility_is_segmented(side):
                        solved_outlets.append(
                            self.T_hu_out_period[n][0]
                            if side == "hot"
                            else self.T_cu_out_period[n][0]
                        )
                        outlet_contributions.append(scalar_contribution)
                        continue

                    matched_duty = (
                        self.Qtot_sc_period[n][match_index]
                        if side == "hot"
                        else self.Qtot_sh_period[n][match_index]
                    )
                    initial_duty = min(matched_duty / 2.0, profile.total_duty)
                    coordinate = self.m.Var(
                        value=initial_duty,
                        lb=0.0,
                        ub=profile.total_duty,
                        name=(f"Qcoord_{side}_utility_M{match_index}_period{n}"),
                    )
                    self.m.Equation(coordinate == load)
                    solved_outlet = self.m.Var(
                        value=profile.temperature_at_heat(initial_duty),
                        lb=float(
                            min(
                                profile.temperatures_in.min(),
                                profile.temperatures_out.min(),
                            )
                        ),
                        ub=float(
                            max(
                                profile.temperatures_in.max(),
                                profile.temperatures_out.max(),
                            )
                        ),
                        name=(f"T_{side}_utility_out_M{match_index}_period{n}"),
                    )
                    contribution_values = profile.temperature_contributions
                    outlet_contribution = self.m.Var(
                        value=profile.temperature_contribution_at_heat(initial_duty),
                        lb=float(contribution_values.min()),
                        ub=float(contribution_values.max()),
                        name=(f"dTcont_{side}_utility_out_M{match_index}_period{n}"),
                    )
                    temperature_segment = profile.segment_index_at_heat(initial_duty)
                    contribution_segment = profile.contribution_index_at_heat(
                        initial_duty
                    )
                    self._register_piecewise_mapping(
                        add_piecewise_temperature_mapping(
                            self.m,
                            coordinate,
                            solved_outlet,
                            profile,
                            name=(f"TQ_{side}_utility_M{match_index}_period{n}"),
                            integer_capable=self.solver in {"apopt", "couenne"},
                            initial_segment=temperature_segment,
                        )
                    )
                    self._register_piecewise_mapping(
                        add_piecewise_temperature_contribution_mapping(
                            self.m,
                            coordinate,
                            outlet_contribution,
                            profile,
                            name=(f"dTQ_{side}_utility_M{match_index}_period{n}"),
                            initial_segment=contribution_segment,
                        )
                    )
                    solved_outlets.append(solved_outlet)
                    outlet_contributions.append(outlet_contribution)

    def _hot_parent_segmented(self, index: int) -> bool:
        return bool(getattr(self, "_segmented_hot_parents", [False] * self.I)[index])

    def _cold_parent_segmented(self, index: int) -> bool:
        return bool(getattr(self, "_segmented_cold_parents", [False] * self.J)[index])

    def _solver_parent_is_segmented(self, side: str, index: int) -> bool:
        if not hasattr(self, "solver_arrays"):
            return False
        counts = self.solver_arrays.arrays.get(f"{side}_segment_count")
        return counts is not None and int(counts[index]) > 1

    def _parent_profile_duty(
        self,
        side: str,
        period_index: int,
        parent_index: int,
        supply_temperature: float,
        target_temperature: float,
        aggregate_cp: float,
    ) -> float:
        if not self._solver_parent_is_segmented(side, parent_index):
            return abs(supply_temperature - target_temperature) * aggregate_cp
        from ..common.solver.piecewise import profile_from_solver_arrays

        return (
            profile_from_solver_arrays(
                self.solver_arrays,
                side=side,
                parent_index=parent_index,
                period_index=period_index,
            )
            .clipped(supply_temperature, target_temperature)
            .total_duty
        )

    def _recovery_heat_upper_bound(
        self,
        *,
        period_index: int,
        hot_index: int,
        cold_index: int,
        hot_total_duty: float,
        cold_total_duty: float,
        hot_cp: float,
        cold_cp: float,
    ) -> float:
        temperature_span = max(
            self.T_h_in_period[period_index][hot_index]
            - self.T_c_in_period[period_index][cold_index]
            - self._recovery_approach_temperature(hot_index, cold_index, period_index),
            0.0,
        )
        if self._solver_parent_is_segmented(
            "hot", hot_index
        ) or self._solver_parent_is_segmented("cold", cold_index):
            return min(hot_total_duty, cold_total_duty) if temperature_span > 0 else 0.0
        return temperature_span * min(hot_cp, cold_cp)

    def _set_piecewise_match_outlet_equations(self) -> None:
        """Map non-isothermal branch outlets through parent heat coordinates."""
        if not hasattr(self, "X_by_period") or not hasattr(self, "Y_by_period"):
            return
        if not (
            np.any(getattr(self, "_segmented_hot_parents", []))
            or np.any(getattr(self, "_segmented_cold_parents", []))
        ):
            return

        from ..common.solver.piecewise import (
            add_piecewise_temperature_mapping,
            profile_from_solver_arrays,
        )

        integer_capable = self.solver in {"apopt", "couenne"}
        self.Q_coordinate_h_out_x_by_period = [
            [
                [[None for _k in range(self.S)] for _j in range(self.J)]
                for _i in range(self.I)
            ]
            for _n in range(self.N_periods)
        ]
        self.Q_coordinate_c_out_y_by_period = [
            [
                [[None for _k in range(self.S)] for _i in range(self.I)]
                for _j in range(self.J)
            ]
            for _n in range(self.N_periods)
        ]
        for n in range(self.N_periods):
            for i in range(self.I):
                hot_profile = (
                    profile_from_solver_arrays(
                        self.solver_arrays,
                        side="hot",
                        parent_index=i,
                        period_index=n,
                    ).clipped(self.T_h_in_period[n][i], self.T_h_out_period[n][i])
                    if self._hot_parent_segmented(i)
                    else None
                )
                for j in range(self.J):
                    cold_profile = (
                        profile_from_solver_arrays(
                            self.solver_arrays,
                            side="cold",
                            parent_index=j,
                            period_index=n,
                        ).clipped(self.T_c_in_period[n][j], self.T_c_out_period[n][j])
                        if self._cold_parent_segmented(j)
                        else None
                    )
                    for k in range(self.S):
                        if self.z_allowed[i][j][k] <= 0:
                            continue
                        if hot_profile is not None:
                            q_in = self.Q_coordinate_h_by_period[n][i][k]
                            q_out = self.m.Var(
                                value=hot_profile.total_duty * (k + 1) / max(self.S, 1),
                                lb=0.0,
                                ub=hot_profile.total_duty,
                                name=f"Qcoord_H{i}_out_C{j}_S{k}_period{n}",
                            )
                            self.Q_coordinate_h_out_x_by_period[n][i][j][k] = q_out
                            self.m.Equation(
                                self.Q_r_by_period[n][i][j][k]
                                == self.X_by_period[n][i][j][k] * (q_out - q_in)
                            )
                            self._register_piecewise_mapping(
                                add_piecewise_temperature_mapping(
                                    self.m,
                                    q_out,
                                    self.T_h_out_x_by_period[n][i][j][k],
                                    hot_profile,
                                    name=f"TQ_H{i}_out_C{j}_S{k}_period{n}",
                                    integer_capable=integer_capable,
                                    initial_segment=hot_profile.segment_index_at_heat(
                                        hot_profile.total_duty
                                        * (k + 1)
                                        / max(self.S, 1)
                                    ),
                                )
                            )
                        if cold_profile is not None:
                            q_in = self.Q_coordinate_c_by_period[n][j][k + 1]
                            q_out = self.m.Var(
                                value=cold_profile.total_duty
                                * (self.S - k)
                                / max(self.S, 1),
                                lb=0.0,
                                ub=cold_profile.total_duty,
                                name=f"Qcoord_C{j}_out_H{i}_S{k}_period{n}",
                            )
                            self.Q_coordinate_c_out_y_by_period[n][j][i][k] = q_out
                            self.m.Equation(
                                self.Q_r_by_period[n][i][j][k]
                                == self.Y_by_period[n][j][i][k] * (q_out - q_in)
                            )
                            self._register_piecewise_mapping(
                                add_piecewise_temperature_mapping(
                                    self.m,
                                    q_out,
                                    self.T_c_out_y_by_period[n][j][i][k],
                                    cold_profile,
                                    name=f"TQ_C{j}_out_H{i}_S{k}_period{n}",
                                    integer_capable=integer_capable,
                                    initial_segment=cold_profile.segment_index_at_heat(
                                        cold_profile.total_duty
                                        * (self.S - k)
                                        / max(self.S, 1)
                                    ),
                                )
                            )

    def get_alpha_values(self) -> list:
        """Calculate source alpha flow-on values in a post-optimisation solve."""

        if self.alpha != []:
            return self.alpha

        model = backend.create_gekko_model(remote=False)
        model.options.IMODE = 1
        model.options.SOLVER = 1
        self.set_alpha_dqda_equations(m=model, postoptimisation=True)
        try:
            with backend.suppress_gekko_numpy_array_copy_deprecation():
                model.solve(disp=False)
        except Exception:
            pass
        return self.alpha

    def set_alpha_dqda_equations(
        self,
        *,
        m: Any | None = None,
        postoptimisation: bool = False,
    ) -> None:
        """Move the source alpha and dQ/dA equations without changing formulas."""

        if postoptimisation:
            if m is None:
                raise ValueError("postoptimisation alpha equations require a model.")
        else:
            m = self.m
        recovery_grid_shape = (self.I, self.J, self.S)

        def postoptimisation_denominator(i: int, j: int, k: int) -> float:
            return self.T_h[i][k][0] - self.T_c[j][k + 1][0]

        def model_denominator(i: int, j: int, k: int) -> Any:
            return (self.T_h[i][k] - self.T_c[j][k + 1] - 1) * self.z[i][j][k] + 1

        if postoptimisation:
            if self.non_isothermal_model:
                self.P_h = build_index_grid(
                    lambda i, j, k: (
                        (self.T_h[i][k][0] - self.T_h_out_x[i][j][k][0])
                        / postoptimisation_denominator(i, j, k)
                        if self.T_h[i][k][0] > self.T_c[j][k + 1][0]
                        else 0.0
                    ),
                    recovery_grid_shape,
                )
                self.P_c = build_index_grid(
                    lambda i, j, k: (
                        (self.T_c_out_y[j][i][k][0] - self.T_c[j][k + 1][0])
                        / postoptimisation_denominator(i, j, k)
                        if self.T_h[i][k][0] > self.T_c[j][k + 1][0]
                        else 0.0
                    ),
                    recovery_grid_shape,
                )
            else:
                self.P_h = build_index_grid(
                    lambda i, j, k: (
                        (self.T_h[i][k][0] - self.T_h[i][k + 1][0])
                        / postoptimisation_denominator(i, j, k)
                        if self.T_h[i][k][0] > self.T_c[j][k + 1][0]
                        else 0.0
                    ),
                    recovery_grid_shape,
                )
                self.P_c = build_index_grid(
                    lambda i, j, k: (
                        (self.T_c[j][k][0] - self.T_c[j][k + 1][0])
                        / postoptimisation_denominator(i, j, k)
                        if self.T_h[i][k][0] > self.T_c[j][k + 1][0]
                        else 0.0
                    ),
                    recovery_grid_shape,
                )

            self.Sum_Qr_is = build_index_grid(
                lambda i, k: [sum(self.Q_r[i][j][k][0] for j in range(self.J))],
                (self.I, self.S),
            )
            self.Sum_Qr_js = build_index_grid(
                lambda j, k: [sum(self.Q_r[i][j][k][0] for i in range(self.I))],
                (self.J, self.S),
            )
            self.beta_h = build_index_grid(
                lambda i, j, k: (
                    self.Q_r[i][j][k][0] / self.Sum_Qr_is[i][k][0]
                    if self.Sum_Qr_is[i][k][0] > 0
                    else 0.0
                ),
                recovery_grid_shape,
            )
            self.beta_c = build_index_grid(
                lambda i, j, k: (
                    self.Q_r[i][j][k][0] / self.Sum_Qr_js[j][k][0]
                    if self.Sum_Qr_js[j][k][0] > 0
                    else 0.0
                ),
                recovery_grid_shape,
            )
            self.z_i = build_index_grid(
                lambda j, k: (
                    sum(self.z[i][j][k][0] for i in range(self.I))
                    / (sum(self.z[i][j][k][0] for i in range(self.I)) + 1e-9)
                ),
                (self.J, self.S),
            )
            self.z_j = build_index_grid(
                lambda i, k: (
                    sum(self.z[i][j][k][0] for j in range(self.J))
                    / (sum(self.z[i][j][k][0] for j in range(self.J)) + 1e-9)
                ),
                (self.I, self.S),
            )
        else:
            if self.non_isothermal_model:
                self.P_h = build_index_grid(
                    lambda i, j, k: m.Intermediate(
                        (self.T_h[i][k] - self.T_h_out_x[i][j][k])
                        * self.z[i][j][k]
                        / model_denominator(i, j, k)
                    ),
                    recovery_grid_shape,
                )
                self.P_c = build_index_grid(
                    lambda i, j, k: m.Intermediate(
                        (self.T_c_out_y[j][i][k] - self.T_c[j][k + 1])
                        * self.z[i][j][k]
                        / model_denominator(i, j, k)
                    ),
                    recovery_grid_shape,
                )
            else:
                self.P_h = build_index_grid(
                    lambda i, j, k: m.Intermediate(
                        (self.T_h[i][k] - self.T_h[i][k + 1])
                        * self.z[i][j][k]
                        / model_denominator(i, j, k)
                    ),
                    recovery_grid_shape,
                )
                self.P_c = build_index_grid(
                    lambda i, j, k: m.Intermediate(
                        (self.T_c[j][k] - self.T_c[j][k + 1])
                        * self.z[i][j][k]
                        / model_denominator(i, j, k)
                    ),
                    recovery_grid_shape,
                )

            self.Sum_Qr_j = build_index_grid(
                lambda i, k: m.Intermediate(
                    sum(self.Q_r[i][j][k] for j in range(self.J))
                ),
                (self.I, self.S),
            )
            self.Sum_Qr_i = build_index_grid(
                lambda j, k: m.Intermediate(
                    sum(self.Q_r[i][j][k] for i in range(self.I))
                ),
                (self.J, self.S),
            )
            self.beta_h = build_index_grid(
                lambda i, j, k: m.Intermediate(
                    self.Q_r[i][j][k] / (self.Sum_Qr_j[i][k] + 1 - self.z[i][j][k])
                ),
                recovery_grid_shape,
            )
            self.beta_c = build_index_grid(
                lambda i, j, k: m.Intermediate(
                    self.Q_r[i][j][k] / (self.Sum_Qr_i[j][k] + 1 - self.z[i][j][k])
                ),
                recovery_grid_shape,
            )
            self.z_i = build_index_grid(
                lambda j, k: m.Intermediate(
                    sum(self.z[i][j][k] for i in range(self.I))
                    / (sum(self.z[i][j][k] for i in range(self.I)) + 1e-9)
                ),
                (self.J, self.S),
            )
            self.z_j = build_index_grid(
                lambda i, k: m.Intermediate(
                    sum(self.z[i][j][k] for j in range(self.J))
                    / (sum(self.z[i][j][k] for j in range(self.J)) + 1e-9)
                ),
                (self.I, self.S),
            )

        self.alpha = build_index_grid(
            lambda i, j, k: m.Var(
                value=0.0,
                ub=1.0,
                lb=-1.0,
                name=f"alpha_H{i}_to_C{j}_at_S{k}",
            ),
            recovery_grid_shape,
        )
        self.gamma_h = build_index_grid(
            lambda i, j, k: m.Var(
                value=0.5,
                ub=1.0,
                lb=-1.0,
                name=f"gamma_h_H{i}_to_C{j}_at_S{k}",
            ),
            recovery_grid_shape,
        )
        self.gamma_c = build_index_grid(
            lambda i, j, k: m.Var(
                value=0.5,
                ub=1.0,
                lb=-1.0,
                name=f"gamma_c_H{i}_to_C{j}_at_S{k}",
            ),
            recovery_grid_shape,
        )

        self.gamma_h_eqn = []
        self.gamma_c_eqn = []
        for k in range(self.S):
            for j in range(self.J):
                for i in range(self.I):
                    if k + 1 >= self.S:
                        self.gamma_h_eqn.append(
                            [m.Equation(self.gamma_h[i][j][k] == 0.0)]
                        )
                        self.gamma_c_eqn.append(
                            [
                                m.Equation(
                                    self.gamma_c[i][j][k]
                                    == sum(
                                        self.beta_c[i0][j][k - 1]
                                        * self.P_c[i0][j][k - 1]
                                        * self.alpha[i0][j][k - 1]
                                        for i0 in range(self.I)
                                    )
                                    + (1 - self.z_i[j][k - 1])
                                    * self.gamma_c[i][j][k - 1]
                                )
                            ]
                        )
                    elif k - 1 < 0:
                        self.gamma_h_eqn.append(
                            [
                                m.Equation(
                                    self.gamma_h[i][j][k]
                                    == sum(
                                        self.beta_h[i][j0][k + 1]
                                        * self.P_h[i][j0][k + 1]
                                        * self.alpha[i][j0][k + 1]
                                        for j0 in range(self.J)
                                    )
                                    + (1 - self.z_j[i][k + 1])
                                    * self.gamma_h[i][j][k + 1]
                                )
                            ]
                        )
                        self.gamma_c_eqn.append(
                            [m.Equation(self.gamma_c[i][j][k] == 0.0)]
                        )
                    else:
                        self.gamma_h_eqn.append(
                            [
                                m.Equation(
                                    self.gamma_h[i][j][k]
                                    == sum(
                                        self.beta_h[i][j0][k + 1]
                                        * self.P_h[i][j0][k + 1]
                                        * self.alpha[i][j0][k + 1]
                                        for j0 in range(self.J)
                                    )
                                    + (1 - self.z_j[i][k + 1])
                                    * self.gamma_h[i][j][k + 1]
                                )
                            ]
                        )
                        self.gamma_c_eqn.append(
                            [
                                m.Equation(
                                    self.gamma_c[i][j][k]
                                    == sum(
                                        self.beta_c[i0][j][k - 1]
                                        * self.P_c[i0][j][k - 1]
                                        * self.alpha[i0][j][k - 1]
                                        for i0 in range(self.I)
                                    )
                                    + (1 - self.z_i[j][k - 1])
                                    * self.gamma_c[i][j][k - 1]
                                )
                            ]
                        )

        self.alpha_eqn = [
            m.Equation(
                self.alpha[i][j][k]
                == (1 - 0.5 * (self.gamma_h[i][j][k] + self.gamma_c[i][j][k]))
            )
            for k in range(self.S)
            for j in range(self.J)
            for i in range(self.I)
            if postoptimisation or self.z_allowed[i][j][k] > 0
        ]
        if not postoptimisation:
            self.alpha_dQ_dA_eqn = [
                (
                    m.Equation(
                        (
                            self.min_dqda * (self.T_h[i][k] - self.T_c[j][k + 1])
                            - self.alpha[i][j][k]
                            * self.theta_1[i][j][k]
                            * self.theta_2[i][j][k]
                            * self.U_r[i][j]
                        )
                        * self.z[i][j][k]
                        <= 0.0
                    )
                    if self.z_allowed[i][j][k] > 0
                    else None
                )
                for k in range(self.S)
                for j in range(self.J)
                for i in range(self.I)
            ]

    def set_blank_input_parameters(self) -> None:
        """Initialize the solver-array attributes expected by source equations."""

        self.T_h_in = np.array([], dtype=float)
        self.T_h_out = np.array([], dtype=float)
        self.f_h = np.array([], dtype=float)
        self.htc_h = np.array([], dtype=float)
        self.h_cost = np.array([], dtype=float)
        self.hot_names = np.array([], dtype=str)
        self.T_h_cont = np.array([], dtype=float)

        self.T_c_in = np.array([], dtype=float)
        self.T_c_out = np.array([], dtype=float)
        self.f_c = np.array([], dtype=float)
        self.htc_c = np.array([], dtype=float)
        self.c_cost = np.array([], dtype=float)
        self.cold_names = np.array([], dtype=str)
        self.T_c_cont = np.array([], dtype=float)

        self.hu_cost = np.array([], dtype=float)
        self.hu_unit_cost = np.array([], dtype=float)
        self.hu_coeff = np.array([], dtype=float)
        self.T_hu_in = np.array([], dtype=float)
        self.T_hu_out = np.array([], dtype=float)
        self.T_hu_cont = np.array([], dtype=float)
        self.htc_hu = np.array([], dtype=float)
        self.hu_exp = np.array([], dtype=float)

        self.cu_cost = np.array([], dtype=float)
        self.cu_unit_cost = np.array([], dtype=float)
        self.cu_coeff = np.array([], dtype=float)
        self.T_cu_in = np.array([], dtype=float)
        self.T_cu_out = np.array([], dtype=float)
        self.T_cu_cont = np.array([], dtype=float)
        self.htc_cu = np.array([], dtype=float)
        self.cu_exp = np.array([], dtype=float)

        self.unit_cost = np.array([], dtype=float)
        self.A_coeff = np.array([], dtype=float)
        self.A_exp = np.array([], dtype=float)
        self.period_ids = np.array(["0"], dtype=str)
        self.period_weights = np.array([1.0], dtype=float)
        self.N_periods = 1
        self.period_weight_sum = 1.0

    def get_model_parameters_from_solver_arrays(self) -> None:
        """Populate model attributes from the OpenPinch private array adapter."""

        for name, values in self.solver_arrays.arrays.items():
            setattr(self, name, np.array(values, copy=True))
        self._normalise_state_arrays()
        self._set_minimum_approach_temperatures()

    def _normalise_state_arrays(self) -> None:
        """Validate the explicit operating-period axis used by HEN models."""

        if "period_ids" not in self.solver_arrays.arrays:
            raise ValueError("period_ids is required for HEN model setup.")
        if "period_weights" not in self.solver_arrays.arrays:
            raise ValueError("period_weights is required for HEN model setup.")

        self.period_ids = np.asarray(self.period_ids, dtype=str)
        self.period_weights = np.asarray(self.period_weights, dtype=float)
        self.N_periods = int(len(self.period_ids))
        if self.N_periods <= 0:
            raise ValueError("HEN model construction requires at least one state.")
        if len(self.period_weights) != self.N_periods:
            raise ValueError("HEN period weight count must match period_id count.")
        if not np.isfinite(self.period_weights).all():
            raise ValueError("HEN period weights must be finite.")
        self.period_weight_sum = float(np.sum(self.period_weights))
        if self.period_weight_sum <= 0.0:
            raise ValueError("HEN period weights must have a positive sum.")

        for base_name in (
            "T_h_in",
            "T_h_out",
            "f_h",
            "htc_h",
            "h_cost",
            "T_h_cont",
            "T_c_in",
            "T_c_out",
            "f_c",
            "htc_c",
            "c_cost",
            "T_c_cont",
            "T_hu_in",
            "T_hu_out",
            "htc_hu",
            "hu_cost",
            "T_hu_cont",
            "T_cu_in",
            "T_cu_out",
            "htc_cu",
            "cu_cost",
            "T_cu_cont",
        ):
            period_name = f"{base_name}_period"
            values = np.asarray(getattr(self, period_name, []), dtype=float)
            if values.size == 0:
                raise ValueError(f"{period_name} is required for HEN model setup.")
            if values.ndim != 2:
                raise ValueError(f"{period_name} must be indexed by operating period.")
            if values.shape[0] != self.N_periods:
                raise ValueError(
                    f"{period_name} has {values.shape[0]} state rows; "
                    f"expected {self.N_periods}."
                )
            setattr(self, period_name, values)
            setattr(self, base_name, values[0].copy())

    def _set_minimum_approach_temperatures(self) -> None:
        """Derive pair-specific approach limits from stream contributions."""

        self.dT_r_period = np.array(
            [
                [
                    [
                        self.T_h_cont_period[n][i] + self.T_c_cont_period[n][j]
                        for j in range(len(self.T_c_cont_period[n]))
                    ]
                    for i in range(len(self.T_h_cont_period[n]))
                ]
                for n in range(self.N_periods)
            ],
            dtype=float,
        )
        self.dT_hu_period = np.array(
            [
                [
                    (
                        self.T_hu_cont_period[n][0]
                        if len(self.T_hu_cont_period[n])
                        else self.dTmin / 2.0
                    )
                    + self.T_c_cont_period[n][j]
                    for j in range(len(self.T_c_cont_period[n]))
                ]
                for n in range(self.N_periods)
            ],
            dtype=float,
        )
        self.dT_cu_period = np.array(
            [
                [
                    self.T_h_cont_period[n][i]
                    + (
                        self.T_cu_cont_period[n][0]
                        if len(self.T_cu_cont_period[n])
                        else self.dTmin / 2.0
                    )
                    for i in range(len(self.T_h_cont_period[n]))
                ]
                for n in range(self.N_periods)
            ],
            dtype=float,
        )
        self.dT_r = self.dT_r_period[0].copy()
        self.dT_hu = self.dT_hu_period[0].copy()
        self.dT_cu = self.dT_cu_period[0].copy()

    def _recovery_approach_temperature(
        self,
        i: int,
        j: int,
        period_idx: int = 0,
    ) -> float:
        if not hasattr(self, "dT_r"):
            return float(self.dTmin)
        if hasattr(self, "dT_r_period"):
            return float(self.dT_r_period[period_idx][i][j])
        return float(self.dT_r[i][j])

    def _hot_utility_inlet_approach_temperature(
        self,
        j: int,
        period_idx: int = 0,
    ) -> float:
        contribution = (
            self.T_hu_in_cont_by_period[period_idx]
            if hasattr(self, "T_hu_in_cont_by_period")
            else self.T_hu_cont_period[period_idx][0]
        )
        return float(contribution + self.T_c_cont_period[period_idx][j])

    def _hot_utility_outlet_approach_temperature(
        self,
        j: int,
        period_idx: int = 0,
        heat_duty: float | None = None,
    ):
        contribution = self._utility_outlet_temperature_contribution(
            "hot",
            period_idx,
            match_index=j,
            heat_duty=heat_duty,
        )
        return contribution + self.T_c_cont_period[period_idx][j]

    def _cold_utility_inlet_approach_temperature(
        self,
        i: int,
        period_idx: int = 0,
    ) -> float:
        contribution = (
            self.T_cu_in_cont_by_period[period_idx]
            if hasattr(self, "T_cu_in_cont_by_period")
            else self.T_cu_cont_period[period_idx][0]
        )
        return float(self.T_h_cont_period[period_idx][i] + contribution)

    def _cold_utility_outlet_approach_temperature(
        self,
        i: int,
        period_idx: int = 0,
        heat_duty: float | None = None,
    ):
        contribution = self._utility_outlet_temperature_contribution(
            "cold",
            period_idx,
            match_index=i,
            heat_duty=heat_duty,
        )
        return self.T_h_cont_period[period_idx][i] + contribution

    def _utility_outlet_temperature_contribution(
        self,
        side: str,
        period_idx: int,
        match_index: int,
        heat_duty: float | None = None,
    ):
        if heat_duty is None and hasattr(
            self,
            f"T_{side[0]}u_out_cont_by_period",
        ):
            return getattr(self, f"T_{side[0]}u_out_cont_by_period")[period_idx][
                match_index
            ]
        scalar = (
            self.T_hu_cont_period[period_idx][0]
            if side == "hot"
            else self.T_cu_cont_period[period_idx][0]
        )
        if heat_duty is None or not self._utility_is_segmented(side):
            return float(scalar)

        from ..common.solver.piecewise import profile_from_solver_arrays

        profile = profile_from_solver_arrays(
            self.solver_arrays,
            side=f"{side}_utility",
            parent_index=0,
            period_index=period_idx,
        )
        return profile.temperature_contribution_at_heat(heat_duty)

    def _utility_solved_outlet_temperature(
        self,
        side: str,
        period_idx: int,
        match_index: int,
        heat_duty: float | None = None,
    ):
        if heat_duty is None and hasattr(
            self,
            f"T_{side[0]}u_solved_out_by_period",
        ):
            return getattr(self, f"T_{side[0]}u_solved_out_by_period")[period_idx][
                match_index
            ]
        if heat_duty is None or not self._utility_is_segmented(side):
            return (
                self.T_hu_out_period[period_idx][0]
                if side == "hot"
                else self.T_cu_out_period[period_idx][0]
            )

        from ..common.solver.piecewise import profile_from_solver_arrays

        return profile_from_solver_arrays(
            self.solver_arrays,
            side=f"{side}_utility",
            parent_index=0,
            period_index=period_idx,
        ).temperature_at_heat(heat_duty)

    def _utility_max_temperature_contribution(
        self,
        side: str,
        period_idx: int,
    ) -> float:
        scalar = (
            self.T_hu_cont_period[period_idx][0]
            if side == "hot"
            else self.T_cu_cont_period[period_idx][0]
        )
        if not self._utility_is_segmented(side):
            return float(scalar)
        values = self.solver_arrays.arrays[f"{side}_utility_segment_dt_cont_period"][
            period_idx, 0
        ]
        count = int(self.solver_arrays.arrays[f"{side}_utility_segment_count"][0])
        return float(np.max(values[:count]))

    def _set_multiperiod_utility_approach_equations(self) -> None:
        """Constrain both utility terminals with local segment contributions."""
        for n in range(self.N_periods):
            for j in range(self.J):
                if self.z_hu_allowed[j] <= 0 or not self._utility_is_segmented("hot"):
                    continue
                inlet_approach = self._hot_utility_inlet_approach_temperature(j, n)
                outlet_approach = self._hot_utility_outlet_approach_temperature(j, n)
                maximum_approach = (
                    self._utility_max_temperature_contribution("hot", n)
                    + self.T_c_cont_period[n][j]
                )
                big_m = max(
                    abs(self.T_hu_in_period[n][0] - self.T_c_out_period[n][j]),
                    abs(self.T_hu_in_period[n][0] - self.T_c_in_period[n][j]),
                    abs(self.T_hu_out_period[n][0] - self.T_c_out_period[n][j]),
                    abs(self.T_hu_out_period[n][0] - self.T_c_in_period[n][j]),
                ) + max(maximum_approach, float(self.dTmin))
                inlet_delta = self.T_hu_in_period[n][0] - self.T_c_out_period[n][j]
                if type(self.z_hu[j]).__name__ == "GKParameter":
                    if (
                        self._solver_value(self.z_hu[j].VALUE.value) > self.tol
                        and inlet_delta + self.tol < inlet_approach
                    ):
                        raise ValueError(
                            f"Hot utility match {j} violates its inlet approach "
                            f"temperature in period {self.period_ids[n]!r}."
                        )
                else:
                    self.m.Equation(
                        inlet_delta >= inlet_approach - big_m * (1 - self.z_hu[j])
                    )
                self.m.Equation(
                    self._utility_solved_outlet_temperature("hot", n, j)
                    - self.T_c_by_period[n][j][0]
                    >= outlet_approach - big_m * (1 - self.z_hu[j])
                )

            for i in range(self.I):
                if self.z_cu_allowed[i] <= 0 or not self._utility_is_segmented("cold"):
                    continue
                inlet_approach = self._cold_utility_inlet_approach_temperature(i, n)
                outlet_approach = self._cold_utility_outlet_approach_temperature(i, n)
                maximum_approach = self.T_h_cont_period[n][
                    i
                ] + self._utility_max_temperature_contribution("cold", n)
                big_m = max(
                    abs(self.T_h_in_period[n][i] - self.T_cu_out_period[n][0]),
                    abs(self.T_h_in_period[n][i] - self.T_cu_in_period[n][0]),
                    abs(self.T_h_out_period[n][i] - self.T_cu_out_period[n][0]),
                    abs(self.T_h_out_period[n][i] - self.T_cu_in_period[n][0]),
                ) + max(maximum_approach, float(self.dTmin))
                self.m.Equation(
                    self.T_h_by_period[n][i][self.S]
                    - self._utility_solved_outlet_temperature("cold", n, i)
                    >= outlet_approach - big_m * (1 - self.z_cu[i])
                )
                inlet_delta = self.T_h_out_period[n][i] - self.T_cu_in_period[n][0]
                if type(self.z_cu[i]).__name__ == "GKParameter":
                    if (
                        self._solver_value(self.z_cu[i].VALUE.value) > self.tol
                        and inlet_delta + self.tol < inlet_approach
                    ):
                        raise ValueError(
                            f"Cold utility match {i} violates its inlet approach "
                            f"temperature in period {self.period_ids[n]!r}."
                        )
                else:
                    self.m.Equation(
                        inlet_delta >= inlet_approach - big_m * (1 - self.z_cu[i])
                    )

    def _weighted_state_average(self, values: Sequence[Any]) -> Any:
        """Return ``sum_s(w_s * value_s) / sum_s(w_s)`` for GEKKO expressions."""

        return (
            sum(
                float(self.period_weights[n]) * values[n] for n in range(self.N_periods)
            )
            / self.period_weight_sum
        )

    def set_match_restrictions(self, restrictions) -> None:
        """Apply inherited topology restrictions in the source array shape."""

        if restrictions is None:
            restrictions = [None, None, None]
        z_restriction, zhu_restriction, zcu_restriction = (
            restrictions[0],
            restrictions[1],
            restrictions[2],
        )

        if z_restriction is not None:
            if isinstance(z_restriction[0][0][0], int):
                self.z_allowed = [
                    [
                        [
                            1 if z_restriction[i][j][k] > self.tol else 0
                            for k in range(self.S)
                        ]
                        for j in range(self.J)
                    ]
                    for i in range(self.I)
                ]
            elif isinstance(z_restriction[0][0][0], list):
                self.z_allowed = [
                    [
                        [
                            1 if z_restriction[i][j][k][0] > self.tol else 0
                            for k in range(self.S)
                        ]
                        for j in range(self.J)
                    ]
                    for i in range(self.I)
                ]
            elif type(z_restriction[0][0][0]).__name__ in {
                "GKVariable",
                "GKParameter",
            }:
                self.z_allowed = [
                    [
                        [
                            1 if z_restriction[i][j][k][0] > self.tol else 0
                            for k in range(self.S)
                        ]
                        for j in range(self.J)
                    ]
                    for i in range(self.I)
                ]
            else:
                raise ValueError("Invalid restriction type")
        else:
            self.z_allowed = self.z_feasible

        if zhu_restriction is not None:
            if isinstance(zhu_restriction[0], int):
                self.z_hu_allowed = [
                    1 if zhu_restriction[j] > self.tol else 0 for j in range(self.J)
                ]
            else:
                self.z_hu_allowed = [
                    1 if zhu_restriction[j][0] > self.tol else 0 for j in range(self.J)
                ]
        else:
            self.z_hu_allowed = self.z_hu_feasible

        if zcu_restriction is not None:
            if isinstance(zcu_restriction[0], int):
                self.z_cu_allowed = [
                    1 if zcu_restriction[i] > self.tol else 0 for i in range(self.I)
                ]
            else:
                self.z_cu_allowed = [
                    1 if zcu_restriction[i][0] > self.tol else 0 for i in range(self.I)
                ]
        else:
            self.z_cu_allowed = self.z_cu_feasible

    def optimise(self, print_output: bool) -> None:
        """Solve the concrete model and extract plain result data on success."""

        total_solve_time = 0.0
        piecewise_mappings = getattr(self, "_piecewise_active_mappings", [])
        active_mapping_stable = not piecewise_mappings or getattr(
            self, "integers", False
        )
        for _attempt in range(8):
            self.solver_run = backend.solve_gekko_model(
                self.m,
                solver_name=self.solver,
                disp=False,
                debug=0,
            )
            total_solve_time += float(self.solver_run.solve_time or 0.0)
            if self.solver_run.failure_reason is not None:
                break
            active_mapping_stable = (
                True
                if not piecewise_mappings
                else not self._update_piecewise_active_segments()
            )
            if active_mapping_stable:
                break
        self.solve_time = total_solve_time

        if (
            self.solver_run.failure_reason is not None
            and piecewise_mappings
            and not getattr(self, "integers", False)
        ):
            self.solver_run = backend.SolverRun(
                name=self.solver_run.name,
                extension=self.solver_run.extension,
                status=self.solver_run.status,
                objective_value=self.solver_run.objective_value,
                solve_time=total_solve_time,
                failure_reason=(
                    f"{self.solver_run.failure_reason}; segmented-stream active "
                    "interval solve was unresolved, so use APOPT or Couenne"
                ),
            )

        if self.solver_run.failure_reason is None and not active_mapping_stable:
            self.solver_run = backend.SolverRun(
                name=self.solver_run.name,
                extension=self.solver_run.extension,
                status=self.solver_run.status,
                objective_value=self.solver_run.objective_value,
                solve_time=total_solve_time,
                failure_reason=(
                    "piecewise active segments did not stabilise; use APOPT or "
                    "Couenne for interval-disjunctive segmented-stream solving"
                ),
            )

        if self.solver_run.failure_reason is not None:
            self.mSuccess = 0
            logger.error("[Failed] [model: %s] [path: %s]", self.name, self.m._path)
        elif self.m.options.SOLVESTATUS == 1:
            if self.m.options.objfcnval + self.tol < 0:
                self.mSuccess = 0
                self.solver_run = backend.SolverRun(
                    name=self.solver_run.name,
                    extension=self.solver_run.extension,
                    status=self.solver_run.status,
                    objective_value=self.solver_run.objective_value,
                    solve_time=self.solver_run.solve_time,
                    failure_reason="negative objective value",
                )
                logger.error(
                    "[Failed] [model: %s] [path: %s]",
                    self.name,
                    self.m._path,
                )
            else:
                self.mSuccess = self.m.options.SOLVESTATUS
                logger.info(
                    "[Success] [model: %s] [path: %s]",
                    self.name,
                    self.m._path,
                )
        else:
            self.mSuccess = self.m.options.SOLVESTATUS
            logger.error(
                "[Failed] [model: %s] [path: %s] [status: %s]",
                self.name,
                self.m._path,
                self.m.options.SOLVESTATUS,
            )

        if self.mSuccess:
            self.get_post_process()
            if print_output:
                self.output_to_cmd_line()

    def output_to_cmd_line(self) -> None:
        """Emit the same solved-array diagnostics as the source base model."""

        if self.mSuccess != 1:
            return
        logger.info("Successful Solve.Path %s name %s", self.m._path, self.name)
        logger.info("Objective 0: %s", self.m.options.objfcnval)
        logger.info("Objective 1: %s", self.TAC)
        logger.info("Total Units: %s", self.n_units)
        logger.info("Total Recovery Units: %s", self.n_recovery_units)
        logger.info("T hot: %s", self.T_h)
        logger.info("T cold: %s", self.T_c)
        logger.info("theta 1: %s", self.theta_1)
        logger.info("theta 2: %s", self.theta_2)
        logger.info("Heat recovery Q: %s", self.Q_r)
        logger.info("Heat recovery z: %s", self.z)
        logger.info("Heat recovery LMTD: %s", self.LMTD_r)
        logger.info("Heat recovery area: %s", self.area_r)
        logger.info("Q_r total: %s", self.Q_r_total)
        logger.info("Cold utility Q: %s", self.Q_c)
        logger.info("Cold utility z: %s", self.z_cu)
        logger.info("Cold utility LMTD: %s", self.LMTD_cu)
        logger.info("Cold utility area: %s", self.area_cu)
        logger.info("Q_cu total: %s", self.Q_cu_total)
        logger.info("Hot utility Q: %s", self.Q_h)
        logger.info("Hot utility z: %s", self.z_hu)
        logger.info("Hot utility LMTD: %s", self.LMTD_hu)
        logger.info("Hot utility area: %s", self.area_hu)
        logger.info("Q_hu total: %s", self.Q_hu_total)
