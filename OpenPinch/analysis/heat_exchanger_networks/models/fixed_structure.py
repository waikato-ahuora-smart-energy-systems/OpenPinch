"""Fixed-structure duty allocation model for heat exchanger networks.

The user fixes the network: recovery matches by stage, plus heaters and
coolers. Several utilities may serve one stream, and a utility exchanger may
sit between stages as well as at the stream end. The NLP decides only the duty
on every exchanger and the split fractions where several recovery matches share
a stream in one stage (non-isothermal mixing).

Stage and temperature conventions (0-based stages ``k = 0 .. S-1``):

* hot streams pass stages ``0 -> S-1``; cold streams pass ``S-1 -> 0``;
* ``th_in[k]``/``th_out[k]`` and ``tc_in[k]``/``tc_out[k]`` are the stream
  temperatures entering and leaving stage ``k``;
* a utility exchanger placed ``after_stage = k`` acts on its stream just after
  the stream leaves stage ``k``: a cooler between hot stages ``k`` and ``k+1``
  (``k = S-1`` is the hot stream end), a heater between cold stages ``k`` and
  ``k-1`` (``k = 0`` is the cold stream end);
* several utility exchangers at one position run in series: heaters from the
  coldest to the hottest utility, coolers from the warmest to the coldest.

Objectives (all multi-period aware, weighted by period weights):

* ``"total utility"``: minimise utility duty;
* ``"total area"``: minimise the sum of common exchanger areas. Each exchanger
  has one area shared by all periods; in every period it must provide at
  least the area that period needs (bypass covers any surplus);
* ``"total cost"``: minimise operating cost plus exchanger capital on the
  common areas.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ._base import execution as _execution
from ._base import piecewise as _piecewise
from .base import BaseHeatExchangerNetworkModel

FIXED_STRUCTURE_GOALS = frozenset({"total utility", "total area", "total cost"})
HEATER = "hot"
COOLER = "cold"
_AREA_SMOOTHING = 1e-3


@dataclass(frozen=True)
class RecoveryMatch:
    """Process-to-process exchanger: hot index, cold index, 0-based stage."""

    hot: int
    cold: int
    stage: int


@dataclass(frozen=True)
class UtilityMatch:
    """Utility exchanger on one process stream.

    ``side`` is ``"hot"`` for a heater (hot utility -> cold stream ``stream``)
    and ``"cold"`` for a cooler (hot stream ``stream`` -> cold utility).
    ``after_stage`` is the 0-based stage the stream has just left.
    """

    side: str
    stream: int
    utility: int
    after_stage: int


@dataclass(frozen=True)
class FixedStructureSpec:
    """Solver-index structure and per-exchanger settings."""

    stage_count: int
    recovery: tuple[RecoveryMatch, ...]
    utilities: tuple[UtilityMatch, ...]
    recovery_approach: Mapping[int, float] = field(default_factory=dict)
    utility_approach: Mapping[int, float] = field(default_factory=dict)
    initial_recovery_duties: Mapping[int, float] = field(default_factory=dict)
    initial_utility_duties: Mapping[int, float] = field(default_factory=dict)


@dataclass
class ExchangerPeriodResult:
    """Solved operating state of one exchanger in one period."""

    duty: float
    active: bool
    approach: tuple[float, float]
    source_inlet: float
    source_outlet: float
    sink_inlet: float
    sink_outlet: float
    required_area: float
    source_split: float | None = None
    sink_split: float | None = None


@dataclass
class ExchangerResult:
    """Solved exchanger: per-period states plus the common design area."""

    periods: list[ExchangerPeriodResult]
    area: float
    capital_cost: float

    @property
    def active(self) -> bool:
        return any(period.active for period in self.periods)

    @property
    def max_duty(self) -> float:
        return max(period.duty for period in self.periods)


class FixedStructureModel(BaseHeatExchangerNetworkModel):
    """Duty-only NLP on a fixed network with free stream splits."""

    def __init__(
        self,
        *,
        name: str,
        solver: str,
        solver_arrays,
        spec: FixedStructureSpec,
        minimisation_goal: str,
        default_approach: float | None = None,
        max_hot_utility: float | None = None,
        max_cold_utility: float | None = None,
        tol: float = 1e-3,
        solver_options: Mapping[str, Any] | Sequence[str] | None = None,
    ) -> None:
        if minimisation_goal not in FIXED_STRUCTURE_GOALS:
            raise ValueError(
                "fixed-structure duty optimisation supports minimisation goals "
                f"{sorted(FIXED_STRUCTURE_GOALS)}; got {minimisation_goal!r}."
            )
        self.spec = spec
        self.default_approach = default_approach
        self.max_hot_utility = max_hot_utility
        self.max_cold_utility = max_cold_utility
        super().__init__(
            name=name,
            framework="ESM",
            solver=solver,
            solver_arrays=solver_arrays,
            dTmin=float(default_approach or 1.0),
            z_restriction=None,
            min_dqda=0.0,
            minimisation_goal=minimisation_goal,
            non_isothermal_model=True,
            integers=False,
            tol=tol,
            solver_options=solver_options,
        )

    # -- construction -----------------------------------------------------

    def setup(self) -> None:
        self.set_blank_input_parameters()
        self.get_model_parameters_from_solver_arrays()
        self.set_preprocessing()
        self.set_stage_wise_superstructure()
        self.set_obj()

    def set_preprocessing(self) -> None:
        prepare_fixed_structure(self)

    def set_stage_wise_superstructure(self) -> None:
        build_fixed_structure_equations(self)

    def set_obj(self) -> None:
        set_fixed_structure_objective(self)

    def get_post_process(self) -> None:
        post_process_fixed_structure(self)

    def output_to_cmd_line(self) -> None:
        return None

    # -- solver primitives (overridden in tests) --------------------------

    def _var(self, name: str, value: float, lb: float | None, ub: float | None):
        return self.m.Var(value=value, lb=lb, ub=ub, name=name)

    def _param(self, name: str, value: float):
        return self.m.Param(value=value, name=name)

    def _equal(self, lhs, rhs) -> None:
        self.m.Equation(lhs == rhs)

    def _at_least(self, lhs, rhs) -> None:
        self.m.Equation(lhs >= rhs)

    def _minimise(self, expression) -> None:
        self.m.Minimize(expression)

    def _intermediate(self, expression, name: str):
        return self.m.Intermediate(expression, name=name)


# -- preprocessing ------------------------------------------------------------


def prepare_fixed_structure(owner) -> None:
    """Derive dimensions, loads, heat-transfer coefficients and approach limits."""

    spec: FixedStructureSpec = owner.spec
    owner.S = int(spec.stage_count)
    owner.K = owner.S + 1
    owner.I = owner.f_h_period.shape[1]
    owner.J = owner.f_c_period.shape[1]
    owner.N_hu = owner.T_hu_in_period.shape[1]
    owner.N_cu = owner.T_cu_in_period.shape[1]
    _reject_segmented_profiles(owner)
    _check_indices(owner)

    periods = range(owner.N_periods)
    owner.hot_load = [
        [
            float(owner.f_h_period[n][i])
            * float(owner.T_h_in_period[n][i] - owner.T_h_out_period[n][i])
            for i in range(owner.I)
        ]
        for n in periods
    ]
    owner.cold_load = [
        [
            float(owner.f_c_period[n][j])
            * float(owner.T_c_out_period[n][j] - owner.T_c_in_period[n][j])
            for j in range(owner.J)
        ]
        for n in periods
    ]
    largest = max(
        [abs(value) for row in owner.hot_load + owner.cold_load for value in row]
        or [0.0]
    )
    owner.duty_tolerance = max(float(owner.tol), 1e-4 * largest)

    owner.recovery_U = [
        [
            1.0
            / (
                1.0 / float(owner.htc_h_period[n][match.hot])
                + 1.0 / float(owner.htc_c_period[n][match.cold])
            )
            for match in spec.recovery
        ]
        for n in periods
    ]
    owner.utility_U = [
        [_utility_U(owner, n, match) for match in spec.utilities] for n in periods
    ]
    owner.recovery_dt = [
        [
            _approach_limit(
                owner,
                spec.recovery_approach.get(r),
                float(owner.T_h_cont_period[n][match.hot])
                + float(owner.T_c_cont_period[n][match.cold]),
            )
            for r, match in enumerate(spec.recovery)
        ]
        for n in periods
    ]
    owner.utility_dt = [
        [
            _approach_limit(
                owner,
                spec.utility_approach.get(e),
                _utility_contribution_dt(owner, n, match),
            )
            for e, match in enumerate(spec.utilities)
        ]
        for n in periods
    ]
    owner.utility_chain = utility_chains(owner)
    _check_utility_reach(owner)
    owner.hot_matches_at = {
        (i, k): [r for r, m in enumerate(spec.recovery) if (m.hot, m.stage) == (i, k)]
        for i in range(owner.I)
        for k in range(owner.S)
    }
    owner.cold_matches_at = {
        (j, k): [r for r, m in enumerate(spec.recovery) if (m.cold, m.stage) == (j, k)]
        for j in range(owner.J)
        for k in range(owner.S)
    }
    for n in periods:
        for r, match in enumerate(spec.recovery):
            span = float(
                owner.T_h_in_period[n][match.hot] - owner.T_c_in_period[n][match.cold]
            )
            if span < owner.recovery_dt[n][r]:
                raise ValueError(
                    f"recovery match {_name(owner, 'hot', match.hot)} -> "
                    f"{_name(owner, 'cold', match.cold)} in stage {match.stage + 1} "
                    f"cannot meet its {owner.recovery_dt[n][r]:g} K minimum "
                    f"approach: the hot supply is only {span:g} K above the cold "
                    "supply."
                )


def _check_utility_reach(owner) -> None:
    """Reject utility exchangers that cannot meet their approach at any duty.

    Every listed exchanger keeps its approach constraint even at zero duty, so
    a utility that is too cold (or too hot) for its stream would make the
    whole problem infeasible rather than idle.
    """

    for n in range(owner.N_periods):
        for (side, stream, k), members in owner.utility_chain.items():
            for position, e in enumerate(members):
                match = owner.spec.utilities[e]
                dt = owner.utility_dt[n][e]
                util_in, util_out = utility_side_temperatures(owner, n, e)
                last_at_end = position == len(members) - 1 and (
                    (side == HEATER and k == 0) or (side == COOLER and k == owner.S - 1)
                )
                if side == HEATER:
                    supply = float(owner.T_c_in_period[n][stream])
                    target = float(owner.T_c_out_period[n][stream])
                    reach = util_out - supply
                    end_reach = util_in - target
                    what = f"cold stream {_name(owner, 'cold', stream)}"
                else:
                    supply = float(owner.T_h_in_period[n][stream])
                    target = float(owner.T_h_out_period[n][stream])
                    reach = supply - util_out
                    end_reach = target - util_in
                    what = f"hot stream {_name(owner, 'hot', stream)}"
                label = (
                    f"{'heater' if side == HEATER else 'cooler'} using "
                    f"{_utility_name(owner, side, match.utility)} on {what}"
                )
                if reach + owner.tol < dt:
                    raise ValueError(
                        f"{label} cannot meet its {dt:g} K minimum approach at "
                        "any duty; the utility is on the wrong side of the "
                        "stream's temperature range."
                    )
                if last_at_end and end_reach + owner.tol < dt:
                    raise ValueError(
                        f"{label} is last at the stream end but cannot meet its "
                        f"{dt:g} K minimum approach against the stream target."
                    )


def utility_chains(owner) -> dict[tuple[str, int, int], list[int]]:
    """Return utility exchangers per (side, stream, after_stage) in series order."""

    chains: dict[tuple[str, int, int], list[int]] = {}
    for e, match in enumerate(owner.spec.utilities):
        chains.setdefault((match.side, match.stream, match.after_stage), []).append(e)
    for (side, _stream, _stage), members in chains.items():
        if side == HEATER:
            members.sort(key=lambda e: float(owner.T_hu_in_period[0][_util(owner, e)]))
        else:
            members.sort(key=lambda e: -float(owner.T_cu_in_period[0][_util(owner, e)]))
    return chains


def _util(owner, e: int) -> int:
    return owner.spec.utilities[e].utility


def _utility_U(owner, n: int, match: UtilityMatch) -> float:
    if match.side == HEATER:
        return 1.0 / (
            1.0 / float(owner.htc_hu_period[n][match.utility])
            + 1.0 / float(owner.htc_c_period[n][match.stream])
        )
    return 1.0 / (
        1.0 / float(owner.htc_h_period[n][match.stream])
        + 1.0 / float(owner.htc_cu_period[n][match.utility])
    )


def _utility_contribution_dt(owner, n: int, match: UtilityMatch) -> float:
    if match.side == HEATER:
        return float(owner.T_hu_cont_period[n][match.utility]) + float(
            owner.T_c_cont_period[n][match.stream]
        )
    return float(owner.T_h_cont_period[n][match.stream]) + float(
        owner.T_cu_cont_period[n][match.utility]
    )


def _approach_limit(owner, override: float | None, contribution: float) -> float:
    if override is not None:
        return float(override)
    if owner.default_approach is not None:
        return float(owner.default_approach)
    return float(contribution)


def _reject_segmented_profiles(owner) -> None:
    checks = [("hot", owner.I, "hot stream"), ("cold", owner.J, "cold stream")]
    checks += [
        ("hot_utility", owner.N_hu, "hot utility"),
        ("cold_utility", owner.N_cu, "cold utility"),
    ]
    for side, count, label in checks:
        for index in range(count):
            if _piecewise._solver_parent_is_segmented(owner, side, index):
                raise ValueError(
                    "fixed-structure duty optimisation needs constant heat "
                    f"capacity streams and utilities; {label} {index} has a "
                    "segmented temperature profile."
                )


def _check_indices(owner) -> None:
    spec = owner.spec
    for match in spec.recovery:
        if not (
            0 <= match.hot < owner.I
            and 0 <= match.cold < owner.J
            and 0 <= match.stage < owner.S
        ):
            raise ValueError(f"recovery match {match} is outside the problem axes.")
    for match in spec.utilities:
        streams, utilities = (
            (owner.J, owner.N_hu) if match.side == HEATER else (owner.I, owner.N_cu)
        )
        if match.side not in {HEATER, COOLER} or not (
            0 <= match.stream < streams
            and 0 <= match.utility < utilities
            and 0 <= match.after_stage < owner.S
        ):
            raise ValueError(f"utility match {match} is outside the problem axes.")


# -- initial point ------------------------------------------------------------


def initial_point(owner, n: int) -> dict[str, Any]:
    """Return a heat-balanced starting point for period ``n``."""

    spec = owner.spec
    q_r = []
    for r, match in enumerate(spec.recovery):
        given = spec.initial_recovery_duties.get(r)
        if given is None:
            hot_share = owner.hot_load[n][match.hot] / max(
                1, _count_on_hot(owner, match.hot)
            )
            cold_share = owner.cold_load[n][match.cold] / max(
                1, _count_on_cold(owner, match.cold)
            )
            given = 0.5 * min(hot_share, cold_share)
        q_r.append(max(0.0, min(float(given), _q_max(owner, n, match))))
    q_u = []
    for e, match in enumerate(spec.utilities):
        given = spec.initial_utility_duties.get(e)
        if given is None:
            if match.side == HEATER:
                recovered = sum(
                    q for q, m in zip(q_r, spec.recovery) if m.cold == match.stream
                )
                residual = owner.cold_load[n][match.stream] - recovered
                count = _utilities_on(owner, HEATER, match.stream)
            else:
                recovered = sum(
                    q for q, m in zip(q_r, spec.recovery) if m.hot == match.stream
                )
                residual = owner.hot_load[n][match.stream] - recovered
                count = _utilities_on(owner, COOLER, match.stream)
            given = max(residual, 0.0) / max(count, 1)
        q_u.append(max(0.0, float(given)))

    th_in = [[0.0] * owner.S for _ in range(owner.I)]
    th_out = [[0.0] * owner.S for _ in range(owner.I)]
    for i in range(owner.I):
        f = float(owner.f_h_period[n][i])
        t = float(owner.T_h_in_period[n][i])
        for k in range(owner.S):
            th_in[i][k] = t
            t -= sum(q_r[r] for r in owner.hot_matches_at[(i, k)]) / f
            th_out[i][k] = t
            t -= sum(q_u[e] for e in owner.utility_chain.get((COOLER, i, k), [])) / f
    tc_in = [[0.0] * owner.S for _ in range(owner.J)]
    tc_out = [[0.0] * owner.S for _ in range(owner.J)]
    for j in range(owner.J):
        f = float(owner.f_c_period[n][j])
        t = float(owner.T_c_in_period[n][j])
        for k in reversed(range(owner.S)):
            tc_in[j][k] = t
            t += sum(q_r[r] for r in owner.cold_matches_at[(j, k)]) / f
            tc_out[j][k] = t
            t += sum(q_u[e] for e in owner.utility_chain.get((HEATER, j, k), [])) / f
    return {
        "q_r": q_r,
        "q_u": q_u,
        "th_in": th_in,
        "th_out": th_out,
        "tc_in": tc_in,
        "tc_out": tc_out,
    }


def _count_on_hot(owner, i: int) -> int:
    return sum(1 for m in owner.spec.recovery if m.hot == i)


def _count_on_cold(owner, j: int) -> int:
    return sum(1 for m in owner.spec.recovery if m.cold == j)


def _utilities_on(owner, side: str, stream: int) -> int:
    return sum(1 for m in owner.spec.utilities if (m.side, m.stream) == (side, stream))


def _q_max(owner, n: int, match: RecoveryMatch) -> float:
    return max(0.0, min(owner.hot_load[n][match.hot], owner.cold_load[n][match.cold]))


# -- equations ----------------------------------------------------------------


def build_fixed_structure_equations(owner) -> None:
    """Create variables and constraints for every period."""

    owner.q_r = []
    owner.q_u = []
    owner.th_in = []
    owner.th_out = []
    owner.tc_in = []
    owner.tc_out = []
    owner.x = []
    owner.y = []
    owner.t_hx = []
    owner.t_cy = []
    owner.theta_r = []
    owner.theta_u = []
    owner.utility_temperatures = []
    for n in range(owner.N_periods):
        start = initial_point(owner, n)
        _period_variables(owner, n, start)
        _stream_balances(owner, n)
        _recovery_equations(owner, n, start)
        _utility_equations(owner, n, start)
        _utility_caps(owner, n)
    if owner.minimisation_goal in {"total area", "total cost"}:
        _common_areas(owner)
    else:
        owner.area_r = None
        owner.area_u = None


def _period_variables(owner, n: int, start: Mapping[str, Any]) -> None:
    spec = owner.spec
    owner.q_r.append(
        [
            owner._var(
                f"qr{r}p{n}", start["q_r"][r], 0.0, max(_q_max(owner, n, m), 0.0)
            )
            for r, m in enumerate(spec.recovery)
        ]
    )
    owner.q_u.append(
        [
            owner._var(
                f"qu{e}p{n}",
                start["q_u"][e],
                0.0,
                owner.cold_load[n][m.stream]
                if m.side == HEATER
                else owner.hot_load[n][m.stream],
            )
            for e, m in enumerate(spec.utilities)
        ]
    )

    def hot_temperature(name, i, value):
        return owner._var(
            name,
            value,
            float(owner.T_h_out_period[n][i]),
            float(owner.T_h_in_period[n][i]),
        )

    def cold_temperature(name, j, value):
        return owner._var(
            name,
            value,
            float(owner.T_c_in_period[n][j]),
            float(owner.T_c_out_period[n][j]),
        )

    owner.th_in.append(
        [
            [
                owner._param(f"thi{i}s{k}p{n}", float(owner.T_h_in_period[n][i]))
                if k == 0
                else hot_temperature(f"thi{i}s{k}p{n}", i, start["th_in"][i][k])
                for k in range(owner.S)
            ]
            for i in range(owner.I)
        ]
    )
    owner.th_out.append(
        [
            [
                hot_temperature(f"tho{i}s{k}p{n}", i, start["th_out"][i][k])
                for k in range(owner.S)
            ]
            for i in range(owner.I)
        ]
    )
    owner.tc_in.append(
        [
            [
                owner._param(f"tci{j}s{k}p{n}", float(owner.T_c_in_period[n][j]))
                if k == owner.S - 1
                else cold_temperature(f"tci{j}s{k}p{n}", j, start["tc_in"][j][k])
                for k in range(owner.S)
            ]
            for j in range(owner.J)
        ]
    )
    owner.tc_out.append(
        [
            [
                cold_temperature(f"tco{j}s{k}p{n}", j, start["tc_out"][j][k])
                for k in range(owner.S)
            ]
            for j in range(owner.J)
        ]
    )


def _stream_balances(owner, n: int) -> None:
    q_r = owner.q_r[n]
    q_u = owner.q_u[n]
    for i in range(owner.I):
        f = float(owner.f_h_period[n][i])
        for k in range(owner.S):
            owner._equal(
                f * (owner.th_in[n][i][k] - owner.th_out[n][i][k]),
                _sum([q_r[r] for r in owner.hot_matches_at[(i, k)]]),
            )
            removed = _sum(
                [q_u[e] for e in owner.utility_chain.get((COOLER, i, k), [])]
            )
            downstream = (
                owner.th_in[n][i][k + 1]
                if k < owner.S - 1
                else float(owner.T_h_out_period[n][i])
            )
            owner._equal(f * (owner.th_out[n][i][k] - downstream), removed)
    for j in range(owner.J):
        f = float(owner.f_c_period[n][j])
        for k in range(owner.S):
            owner._equal(
                f * (owner.tc_out[n][j][k] - owner.tc_in[n][j][k]),
                _sum([q_r[r] for r in owner.cold_matches_at[(j, k)]]),
            )
            added = _sum([q_u[e] for e in owner.utility_chain.get((HEATER, j, k), [])])
            downstream = (
                owner.tc_in[n][j][k - 1] if k > 0 else float(owner.T_c_out_period[n][j])
            )
            owner._equal(f * (downstream - owner.tc_out[n][j][k]), added)


def _recovery_equations(owner, n: int, start: Mapping[str, Any]) -> None:
    spec = owner.spec
    x_row, y_row, hx_row, cy_row, theta_row = [], [], [], [], []
    for r, match in enumerate(spec.recovery):
        i, j, k = match.hot, match.cold, match.stage
        hot_branches = owner.hot_matches_at[(i, k)]
        cold_branches = owner.cold_matches_at[(j, k)]
        x = (
            owner._param(f"x{r}p{n}", 1.0)
            if len(hot_branches) == 1
            else owner._var(f"x{r}p{n}", 1.0 / len(hot_branches), 0.0, 1.0)
        )
        y = (
            owner._param(f"y{r}p{n}", 1.0)
            if len(cold_branches) == 1
            else owner._var(f"y{r}p{n}", 1.0 / len(cold_branches), 0.0, 1.0)
        )
        f_h = float(owner.f_h_period[n][i])
        f_c = float(owner.f_c_period[n][j])
        th_in0 = start["th_in"][i][k]
        tc_in0 = start["tc_in"][j][k]
        q0 = start["q_r"][r]
        t_hx = owner._var(
            f"thx{r}p{n}",
            th_in0 - q0 * len(hot_branches) / f_h,
            float(owner.T_h_out_period[n][i]),
            float(owner.T_h_in_period[n][i]),
        )
        t_cy = owner._var(
            f"tcy{r}p{n}",
            tc_in0 + q0 * len(cold_branches) / f_c,
            float(owner.T_c_in_period[n][j]),
            float(owner.T_c_out_period[n][j]),
        )
        dt = owner.recovery_dt[n][r]
        upper = float(owner.T_h_in_period[n][i] - owner.T_c_in_period[n][j])
        theta_1 = owner._var(
            f"tr1{r}p{n}", max(dt, th_in0 - (tc_in0 + q0 / f_c)), dt, upper
        )
        theta_2 = owner._var(
            f"tr2{r}p{n}", max(dt, th_in0 - q0 / f_h - tc_in0), dt, upper
        )
        q = owner.q_r[n][r]
        owner._equal(q, x * f_h * (owner.th_in[n][i][k] - t_hx))
        owner._equal(q, y * f_c * (t_cy - owner.tc_in[n][j][k]))
        owner._equal(theta_1, owner.th_in[n][i][k] - t_cy)
        owner._equal(theta_2, t_hx - owner.tc_in[n][j][k])
        x_row.append(x)
        y_row.append(y)
        hx_row.append(t_hx)
        cy_row.append(t_cy)
        theta_row.append((theta_1, theta_2))
    owner.x.append(x_row)
    owner.y.append(y_row)
    owner.t_hx.append(hx_row)
    owner.t_cy.append(cy_row)
    owner.theta_r.append(theta_row)
    for branches in owner.hot_matches_at.values():
        if len(branches) > 1:
            owner._equal(_sum([x_row[r] for r in branches]), 1.0)
    for branches in owner.cold_matches_at.values():
        if len(branches) > 1:
            owner._equal(_sum([y_row[r] for r in branches]), 1.0)


def utility_stream_temperatures(owner, n: int, values=None) -> list[tuple[Any, Any]]:
    """Return (stream inlet, stream outlet) for every utility exchanger.

    With ``values`` (a callable mapping a solver object to a float) the result
    is numeric; otherwise it is built from solver expressions.
    """

    read = values or (lambda item: item)
    result: list[tuple[Any, Any] | None] = [None] * len(owner.spec.utilities)
    for (side, stream, k), members in owner.utility_chain.items():
        if side == HEATER:
            f = float(owner.f_c_period[n][stream])
            t = read(owner.tc_out[n][stream][k])
            for e in members:
                t_next = t + read(owner.q_u[n][e]) / f
                result[e] = (t, t_next)
                t = t_next
        else:
            f = float(owner.f_h_period[n][stream])
            t = read(owner.th_out[n][stream][k])
            for e in members:
                t_next = t - read(owner.q_u[n][e]) / f
                result[e] = (t, t_next)
                t = t_next
    return result  # type: ignore[return-value]


def utility_side_temperatures(owner, n: int, e: int) -> tuple[float, float]:
    """Return (utility inlet, utility outlet) temperatures for exchanger ``e``."""

    match = owner.spec.utilities[e]
    if match.side == HEATER:
        return (
            float(owner.T_hu_in_period[n][match.utility]),
            float(owner.T_hu_out_period[n][match.utility]),
        )
    return (
        float(owner.T_cu_in_period[n][match.utility]),
        float(owner.T_cu_out_period[n][match.utility]),
    )


def _utility_equations(owner, n: int, start: Mapping[str, Any]) -> None:
    temperatures = utility_stream_temperatures(owner, n)
    owner.utility_temperatures.append(temperatures)
    start_temperatures = _numeric_utility_temperatures(owner, n, start)
    row = []
    for e, match in enumerate(owner.spec.utilities):
        stream_in, stream_out = temperatures[e]
        util_in, util_out = utility_side_temperatures(owner, n, e)
        dt = owner.utility_dt[n][e]
        start_in, start_out = start_temperatures[e]
        if match.side == HEATER:
            hot_end, cold_end = util_in - stream_out, util_out - stream_in
            start_hot, start_cold = util_in - start_out, util_out - start_in
        else:
            hot_end, cold_end = stream_in - util_out, stream_out - util_in
            start_hot, start_cold = start_in - util_out, start_out - util_in
        theta_1 = owner._var(f"tu1{e}p{n}", max(dt, start_hot), dt, None)
        theta_2 = owner._var(f"tu2{e}p{n}", max(dt, start_cold), dt, None)
        owner._equal(theta_1, hot_end)
        owner._equal(theta_2, cold_end)
        row.append((theta_1, theta_2))
    owner.theta_u.append(row)


def _numeric_utility_temperatures(owner, n: int, start) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = [(0.0, 0.0)] * len(owner.spec.utilities)
    for (side, stream, k), members in owner.utility_chain.items():
        if side == HEATER:
            f = float(owner.f_c_period[n][stream])
            t = start["tc_out"][stream][k]
            sign = 1.0
        else:
            f = float(owner.f_h_period[n][stream])
            t = start["th_out"][stream][k]
            sign = -1.0
        for e in members:
            t_next = t + sign * start["q_u"][e] / f
            result[e] = (t, t_next)
            t = t_next
    return result


def _utility_caps(owner, n: int) -> None:
    heaters = [e for e, m in enumerate(owner.spec.utilities) if m.side == HEATER]
    coolers = [e for e, m in enumerate(owner.spec.utilities) if m.side == COOLER]
    if owner.max_hot_utility is not None and heaters:
        owner._at_least(
            float(owner.max_hot_utility), _sum([owner.q_u[n][e] for e in heaters])
        )
    if owner.max_cold_utility is not None and coolers:
        owner._at_least(
            float(owner.max_cold_utility), _sum([owner.q_u[n][e] for e in coolers])
        )


def chen_lmtd(theta_1, theta_2):
    """Chen (1987) LMTD approximation with the source smoothing constant."""

    return (theta_1 * theta_2 * (theta_1 + theta_2) / 2 + _AREA_SMOOTHING) ** (1 / 3)


def _common_areas(owner) -> None:
    """One area per exchanger, at least what every period needs.

    With one period the area is the Chen-LMTD expression itself. With several
    periods it is a variable bounded below by each period's requirement, so
    periods that need less area run with a bypass.
    """

    def area_term(name: str, duties, coefficients, thetas):
        if owner.N_periods == 1:
            return owner._intermediate(
                duties[0] / (coefficients[0] * chen_lmtd(*thetas[0])), name
            )
        start = max(
            _numeric(duties[n]) / (coefficients[n] * _start_lmtd(thetas[n]))
            for n in range(owner.N_periods)
        )
        area = owner._var(name, start, 0.0, None)
        for n in range(owner.N_periods):
            owner._at_least(area * coefficients[n] * chen_lmtd(*thetas[n]), duties[n])
        return area

    owner.area_r = [
        area_term(
            f"ar{r}",
            [owner.q_r[n][r] for n in range(owner.N_periods)],
            [owner.recovery_U[n][r] for n in range(owner.N_periods)],
            [owner.theta_r[n][r] for n in range(owner.N_periods)],
        )
        for r in range(len(owner.spec.recovery))
    ]
    owner.area_u = [
        area_term(
            f"au{e}",
            [owner.q_u[n][e] for n in range(owner.N_periods)],
            [owner.utility_U[n][e] for n in range(owner.N_periods)],
            [owner.theta_u[n][e] for n in range(owner.N_periods)],
        )
        for e in range(len(owner.spec.utilities))
    ]


def _start_lmtd(thetas) -> float:
    theta_1, theta_2 = (_numeric(theta) for theta in thetas)
    return max(chen_lmtd(max(theta_1, 1e-6), max(theta_2, 1e-6)), 1e-6)


# -- objective ----------------------------------------------------------------


def set_fixed_structure_objective(owner) -> None:
    goal = owner.minimisation_goal
    if goal == "total utility":
        owner.objective_expression = _weighted(
            owner, [_sum(owner.q_u[n]) for n in range(owner.N_periods)]
        )
    elif goal == "total area":
        owner.objective_expression = _sum(list(owner.area_r) + list(owner.area_u))
    else:
        owner.objective_expression = operating_cost_expression(
            owner
        ) + capital_cost_expression(owner)
    owner._minimise(owner.objective_expression)


def operating_cost_expression(owner):
    """Weighted utility cost: price of each utility times its duty."""

    return _weighted(
        owner,
        [
            _sum(
                [
                    _utility_price(owner, n, e) * owner.q_u[n][e]
                    for e in range(len(owner.spec.utilities))
                ]
            )
            for n in range(owner.N_periods)
        ],
    )


def capital_cost_expression(owner):
    """Exchanger capital on the common areas (unit costs are constant)."""

    terms = [
        _area_cost(owner.A_coeff[0], owner.A_exp[0], area) for area in owner.area_r
    ]
    for e, area in enumerate(owner.area_u):
        coeff, exponent = _utility_area_cost_parameters(owner, e)
        terms.append(_area_cost(coeff, exponent, area))
    return _sum(terms) + fixed_unit_cost(owner)


def fixed_unit_cost(owner) -> float:
    unit = float(owner.unit_cost[0]) * len(owner.spec.recovery)
    for match in owner.spec.utilities:
        unit += float(
            owner.hu_unit_cost[0] if match.side == HEATER else owner.cu_unit_cost[0]
        )
    return unit


def _area_cost(coeff, exponent, area):
    coeff = float(coeff)
    exponent = float(exponent)
    if exponent == 1.0:
        return coeff * area
    return coeff * (area + 1e-6) ** exponent


def _utility_area_cost_parameters(owner, e: int) -> tuple[float, float]:
    if owner.spec.utilities[e].side == HEATER:
        return float(owner.hu_coeff[0]), float(owner.hu_exp[0])
    return float(owner.cu_coeff[0]), float(owner.cu_exp[0])


def _utility_price(owner, n: int, e: int) -> float:
    match = owner.spec.utilities[e]
    prices = owner.hu_cost_period if match.side == HEATER else owner.cu_cost_period
    return float(prices[n][match.utility])


def _weighted(owner, values):
    weights = [float(w) for w in owner.period_weights]
    total = sum(weights) or 1.0
    return _sum([weights[n] / total * values[n] for n in range(owner.N_periods)])


def _sum(values):
    values = list(values)
    if not values:
        return 0.0
    total = values[0]
    for value in values[1:]:
        total = total + value
    return total


# -- post-processing ----------------------------------------------------------


def _numeric(value: Any) -> float:
    try:
        return _execution._solver_value(None, value.value)
    except AttributeError:
        return float(value)


def exact_lmtd(theta_1: float, theta_2: float) -> float:
    if theta_1 <= 0.0 or theta_2 <= 0.0:
        return 0.0
    if math.isclose(theta_1, theta_2, rel_tol=1e-9, abs_tol=1e-9):
        return theta_1
    return (theta_1 - theta_2) / math.log(theta_1 / theta_2)


def _required_area(duty: float, coefficient: float, theta_1, theta_2) -> float:
    if duty <= 0.0:
        return 0.0
    lmtd = exact_lmtd(theta_1, theta_2)
    if lmtd <= 0.0:
        return math.inf
    return duty / (coefficient * lmtd)


def post_process_fixed_structure(owner) -> None:
    """Read the solution into plain exchanger results and cost totals."""

    value = _numeric
    tolerance = owner.duty_tolerance
    spec = owner.spec

    owner.recovery_results = []
    for r, match in enumerate(spec.recovery):
        periods = []
        for n in range(owner.N_periods):
            duty = value(owner.q_r[n][r])
            theta_1, theta_2 = (value(t) for t in owner.theta_r[n][r])
            active = duty > tolerance
            periods.append(
                ExchangerPeriodResult(
                    duty=duty if active else 0.0,
                    active=active,
                    approach=(theta_1, theta_2),
                    source_inlet=value(owner.th_in[n][match.hot][match.stage]),
                    source_outlet=value(owner.t_hx[n][r]),
                    sink_inlet=value(owner.tc_in[n][match.cold][match.stage]),
                    sink_outlet=value(owner.t_cy[n][r]),
                    required_area=_required_area(
                        duty if active else 0.0,
                        owner.recovery_U[n][r],
                        theta_1,
                        theta_2,
                    ),
                    source_split=value(owner.x[n][r]),
                    sink_split=value(owner.y[n][r]),
                )
            )
        area = _design_area(periods)
        owner.recovery_results.append(
            ExchangerResult(
                periods=periods,
                area=area,
                capital_cost=_capital(
                    area, owner.unit_cost[0], owner.A_coeff[0], owner.A_exp[0]
                ),
            )
        )

    owner.utility_results = []
    stream_temperatures = [
        utility_stream_temperatures(owner, n, value) for n in range(owner.N_periods)
    ]
    for e, match in enumerate(spec.utilities):
        periods = []
        for n in range(owner.N_periods):
            duty = value(owner.q_u[n][e])
            active = duty > tolerance
            theta_1, theta_2 = (value(t) for t in owner.theta_u[n][e])
            stream_in, stream_out = stream_temperatures[n][e]
            util_in, util_out = utility_side_temperatures(owner, n, e)
            if match.side == HEATER:
                source = (util_in, util_out)
                sink = (stream_in, stream_out)
            else:
                source = (stream_in, stream_out)
                sink = (util_in, util_out)
            periods.append(
                ExchangerPeriodResult(
                    duty=duty if active else 0.0,
                    active=active,
                    approach=(theta_1, theta_2),
                    source_inlet=source[0],
                    source_outlet=source[1],
                    sink_inlet=sink[0],
                    sink_outlet=sink[1],
                    required_area=_required_area(
                        duty if active else 0.0,
                        owner.utility_U[n][e],
                        theta_1,
                        theta_2,
                    ),
                )
            )
        area = _design_area(periods)
        unit, coeff, exponent = (
            (owner.hu_unit_cost[0], owner.hu_coeff[0], owner.hu_exp[0])
            if match.side == HEATER
            else (owner.cu_unit_cost[0], owner.cu_coeff[0], owner.cu_exp[0])
        )
        owner.utility_results.append(
            ExchangerResult(
                periods=periods,
                area=area,
                capital_cost=_capital(area, unit, coeff, exponent),
            )
        )

    weights = [float(w) for w in owner.period_weights]
    weight_sum = sum(weights) or 1.0
    owner.operating_cost_by_period = [
        sum(
            _utility_price(owner, n, e) * result.periods[n].duty
            for e, result in enumerate(owner.utility_results)
        )
        for n in range(owner.N_periods)
    ]
    owner.utility_cost_value = (
        sum(w * c for w, c in zip(weights, owner.operating_cost_by_period)) / weight_sum
    )
    owner.capital_cost_value = sum(
        result.capital_cost for result in owner.recovery_results + owner.utility_results
    )
    owner.TAC = owner.utility_cost_value + owner.capital_cost_value
    owner.total_area = sum(
        result.area for result in owner.recovery_results + owner.utility_results
    )


def _design_area(periods: Sequence[ExchangerPeriodResult]) -> float:
    """Common design area: the largest exact-LMTD area any period needs.

    Periods needing less run with a bypass. The solver's Chen-LMTD area is
    only used inside the optimisation, so reported areas agree across
    objectives.
    """

    if not any(p.active for p in periods):
        return 0.0
    return max(p.required_area for p in periods)


def _capital(area: float, unit, coeff, exponent) -> float:
    if area <= 0.0:
        return 0.0
    return float(unit) + float(coeff) * area ** float(exponent)


def _utility_name(owner, side: str, index: int) -> str:
    axis = getattr(owner.solver_arrays, "axis_maps", {}).get(f"{side}_utilities", {})
    for name, position in axis.items():
        if position == index:
            return str(name)
    return f"{side} utility {index}"


def _name(owner, side: str, index: int) -> str:
    names = getattr(owner, "cold_names" if side == "cold" else "hot_names", None)
    try:
        return str(names[index])
    except TypeError, IndexError:
        return f"{side}[{index}]"


__all__ = [
    "COOLER",
    "FIXED_STRUCTURE_GOALS",
    "HEATER",
    "ExchangerPeriodResult",
    "ExchangerResult",
    "FixedStructureModel",
    "FixedStructureSpec",
    "RecoveryMatch",
    "UtilityMatch",
    "chen_lmtd",
    "exact_lmtd",
    "initial_point",
]
