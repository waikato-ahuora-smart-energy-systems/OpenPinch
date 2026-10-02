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

Streams are described by heat-temperature profiles (``ThermalProfile``), so
segmented streams with piecewise heat capacities are handled exactly: energy
balances use ``heat(T)``, the minimum approach is enforced at segment
boundaries inside each exchanger, and reported areas are summed over
duty-aligned slices. A segmented utility keeps its profile shape and segment
prices with a free flow scale.

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

from ..solver.piecewise import (
    duty_aligned_area_contributions,
    profile_from_solver_arrays,
    utility_thermal_profile,
)
from ._base import execution as _execution
from ._base import piecewise as _piecewise
from .base import BaseHeatExchangerNetworkModel
from .thermal_profiles import ThermalProfile, smooth_clip

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
    slices: tuple = ()
    utility_flow: float | None = None


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
    """Build stream and utility profiles, approach limits and series chains."""

    spec: FixedStructureSpec = owner.spec
    owner.S = int(spec.stage_count)
    owner.K = owner.S + 1
    owner.I = owner.T_h_in_period.shape[1]
    owner.J = owner.T_c_in_period.shape[1]
    owner.N_hu = owner.T_hu_in_period.shape[1]
    owner.N_cu = owner.T_cu_in_period.shape[1]
    _check_indices(owner)
    periods = range(owner.N_periods)

    owner.hot_profiles = [
        [_stream_profile(owner, "hot", n, i) for i in range(owner.I)] for n in periods
    ]
    owner.cold_profiles = [
        [_stream_profile(owner, "cold", n, j) for j in range(owner.J)] for n in periods
    ]
    owner.hot_utility_profiles = [
        [_utility_profile(owner, "hot", n, u) for u in range(owner.N_hu)]
        for n in periods
    ]
    owner.cold_utility_profiles = [
        [_utility_profile(owner, "cold", n, u) for u in range(owner.N_cu)]
        for n in periods
    ]
    owner.hot_load = [[p.total for p in row] for row in owner.hot_profiles]
    owner.cold_load = [[p.total for p in row] for row in owner.cold_profiles]
    largest = max(
        [abs(value) for row in owner.hot_load + owner.cold_load for value in row]
        or [0.0]
    )
    owner.duty_tolerance = max(float(owner.tol), 1e-4 * largest)

    owner.recovery_U = [
        [
            _overall(
                owner.hot_profiles[n][m.hot].mean_htc,
                owner.cold_profiles[n][m.cold].mean_htc,
            )
            for m in spec.recovery
        ]
        for n in periods
    ]
    owner.utility_U = [
        [
            _overall(
                _utility_htc(owner, n, m),
                stream_profile(owner, n, m).mean_htc,
            )
            for m in spec.utilities
        ]
        for n in periods
    ]
    owner.recovery_dt = [
        [
            _approach_limit(
                owner,
                spec.recovery_approach.get(r),
                owner.hot_profiles[n][m.hot].max_contribution
                + owner.cold_profiles[n][m.cold].max_contribution,
            )
            for r, m in enumerate(spec.recovery)
        ]
        for n in periods
    ]
    owner.utility_dt = [
        [
            _approach_limit(
                owner,
                spec.utility_approach.get(e),
                _utility_contribution(owner, n, m)
                + stream_profile(owner, n, m).max_contribution,
            )
            for e, m in enumerate(spec.utilities)
        ]
        for n in periods
    ]
    owner.utility_chain = utility_chains(owner)
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
    _check_recovery_spans(owner)
    _check_utility_reach(owner)
    owner.implied_cold_ends = utility_free_components(owner)


def _stream_profile(owner, side: str, n: int, index: int) -> ThermalProfile:
    """Stream profile for one period; a stream that is off has zero flow."""

    prefix = "h" if side == "hot" else "c"
    if _piecewise._solver_parent_is_segmented(owner, side, index) and _has_duty(
        owner, side, n, index
    ):
        return ThermalProfile.from_piecewise(
            side,
            profile_from_solver_arrays(
                owner.solver_arrays, side=side, parent_index=index, period_index=n
            ),
        )
    names = owner.hot_names if side == "hot" else owner.cold_names
    return ThermalProfile.single(
        side,
        _safe_name(names, index, f"{side}{index}"),
        float(getattr(owner, f"T_{prefix}_in_period")[n][index]),
        float(getattr(owner, f"T_{prefix}_out_period")[n][index]),
        float(getattr(owner, f"f_{prefix}_period")[n][index]),
        float(getattr(owner, f"htc_{prefix}_period")[n][index]),
        float(getattr(owner, f"T_{prefix}_cont_period")[n][index]),
    )


def _has_duty(owner, side: str, n: int, index: int) -> bool:
    """Whether a segmented stream carries heat in period ``n`` (it may be off)."""

    duties = owner.solver_arrays.arrays.get(f"{side}_segment_duty_period")
    if duties is None:
        return True
    return float(sum(duties[n][index])) > 0.0


def _utility_profile(owner, side: str, n: int, index: int) -> ThermalProfile | None:
    """Segmented utilities keep their profile shape; others stay ``None``.

    The shape (segment temperatures, relative heat capacities, prices and
    film coefficients) comes from the targeted utility profile. Its flow
    scales freely with use, like any other utility.
    """

    if not _piecewise._solver_parent_is_segmented(owner, f"{side}_utility", index):
        return None
    try:
        profile = profile_from_solver_arrays(
            owner.solver_arrays,
            side=f"{side}_utility",
            parent_index=index,
            period_index=n,
        )
    except ValueError as exc:
        raise ValueError(
            f"segmented {side} utility {_utility_name(owner, side, index)} has no "
            f"targeted duty in period {n}, so its profile shape is unknown; "
            "target with a demand on this utility or give it a single "
            "temperature range."
        ) from exc
    return ThermalProfile.from_piecewise(side, profile)


def _off(profile: ThermalProfile) -> bool:
    return profile.total <= 0.0


def _recovery_off(owner, n: int, match: RecoveryMatch) -> bool:
    return _off(owner.hot_profiles[n][match.hot]) or _off(
        owner.cold_profiles[n][match.cold]
    )


def stream_off(owner, n: int, match: UtilityMatch) -> bool:
    """Whether the process stream of a utility exchanger is off in period n."""

    return _off(stream_profile(owner, n, match))


def stream_profile(owner, n: int, match: UtilityMatch) -> ThermalProfile:
    """Process-stream profile on the stream side of a utility exchanger."""

    if match.side == HEATER:
        return owner.cold_profiles[n][match.stream]
    return owner.hot_profiles[n][match.stream]


def utility_profile(owner, n: int, match: UtilityMatch) -> ThermalProfile | None:
    if match.side == HEATER:
        return owner.hot_utility_profiles[n][match.utility]
    return owner.cold_utility_profiles[n][match.utility]


def _utility_htc(owner, n: int, match: UtilityMatch) -> float:
    profile = utility_profile(owner, n, match)
    if profile is not None:
        return profile.mean_htc
    table = owner.htc_hu_period if match.side == HEATER else owner.htc_cu_period
    return float(table[n][match.utility])


def _utility_contribution(owner, n: int, match: UtilityMatch) -> float:
    profile = utility_profile(owner, n, match)
    if profile is not None:
        return profile.max_contribution
    table = owner.T_hu_cont_period if match.side == HEATER else owner.T_cu_cont_period
    return float(table[n][match.utility])


def utility_side_temperatures(owner, n: int, e: int) -> tuple[float, float]:
    """(inlet, outlet) of a constant-temperature (non-segmented) utility."""

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


def _overall(first: float, second: float) -> float:
    """Series film resistance; a missing coefficient (stream off) counts as 1."""

    first = float(first) if float(first) > 0.0 else 1.0
    second = float(second) if float(second) > 0.0 else 1.0
    return 1.0 / (1.0 / first + 1.0 / second)


def _approach_limit(owner, override: float | None, contribution: float) -> float:
    if override is not None:
        return float(override)
    if owner.default_approach is not None:
        return float(owner.default_approach)
    return float(contribution)


def utility_chains(owner) -> dict[tuple[str, int, int], list[int]]:
    """Return utility exchangers per (side, stream, after_stage) in series order."""

    chains: dict[tuple[str, int, int], list[int]] = {}
    for e, match in enumerate(owner.spec.utilities):
        chains.setdefault((match.side, match.stream, match.after_stage), []).append(e)
    for (side, _stream, _stage), members in chains.items():
        if side == HEATER:
            members.sort(key=lambda e: _utility_supply(owner, e))
        else:
            members.sort(key=lambda e: -_utility_supply(owner, e))
    return chains


def _utility_supply(owner, e: int) -> float:
    match = owner.spec.utilities[e]
    profiles = (
        getattr(owner, "hot_utility_profiles", None)
        if match.side == HEATER
        else getattr(owner, "cold_utility_profiles", None)
    )
    if profiles is not None and profiles[0][match.utility] is not None:
        return profiles[0][match.utility].supply
    table = owner.T_hu_in_period if match.side == HEATER else owner.T_cu_in_period
    return float(table[0][match.utility])


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


def utility_free_components(owner) -> set[int]:
    """Return one cold stream per utility-free group whose end is implied.

    Streams linked only by recovery (no heater or cooler anywhere in the
    group) must balance exactly. Their target equations are then dependent,
    so one cold stream's end condition is dropped (energy balance implies it)
    to keep the system square for the NLP solver.
    """

    parent = {("hot", i): ("hot", i) for i in range(owner.I)}
    parent.update({("cold", j): ("cold", j) for j in range(owner.J)})

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for match in owner.spec.recovery:
        parent[find(("hot", match.hot))] = find(("cold", match.cold))
    with_utility = {
        find(("cold", m.stream) if m.side == HEATER else ("hot", m.stream))
        for m in owner.spec.utilities
    }
    groups: dict[Any, list[tuple[str, int]]] = {}
    for node in parent:
        groups.setdefault(find(node), []).append(node)
    implied: set[int] = set()
    for root, members in groups.items():
        colds = sorted(index for side, index in members if side == "cold")
        hots = sorted(index for side, index in members if side == "hot")
        if root in with_utility or not colds or not hots:
            continue
        for n in range(owner.N_periods):
            imbalance = sum(owner.hot_load[n][i] for i in hots) - sum(
                owner.cold_load[n][j] for j in colds
            )
            if abs(imbalance) > max(owner.duty_tolerance, 1e-6):
                names = [_name(owner, "hot", i) for i in hots] + [
                    _name(owner, "cold", j) for j in colds
                ]
                raise ValueError(
                    "streams " + ", ".join(names) + " have no heater or cooler "
                    f"and their heat loads differ by {imbalance:g} kW in period "
                    f"{n}; add a utility exchanger."
                )
        implied.add(colds[0])
    return implied


def _check_recovery_spans(owner) -> None:
    for n in range(owner.N_periods):
        for r, match in enumerate(owner.spec.recovery):
            span = (
                owner.hot_profiles[n][match.hot].supply
                - owner.cold_profiles[n][match.cold].supply
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
                profile = stream_profile(owner, n, match)
                own = utility_profile(owner, n, match)
                if own is None:
                    util_in, util_out = utility_side_temperatures(owner, n, e)
                else:
                    util_in = util_out = own.supply
                last_at_end = position == len(members) - 1 and (
                    (side == HEATER and k == 0) or (side == COOLER and k == owner.S - 1)
                )
                if side == HEATER:
                    reach = util_out - profile.supply
                    end_reach = util_in - profile.target
                    what = f"cold stream {_name(owner, 'cold', stream)}"
                else:
                    reach = profile.supply - util_out
                    end_reach = profile.target - util_in
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


# -- initial point ------------------------------------------------------------


def initial_point(owner, n: int) -> dict[str, Any]:
    """Return a heat-balanced starting point for period ``n``."""

    spec = owner.spec
    q_r = []
    for r, match in enumerate(spec.recovery):
        given = spec.initial_recovery_duties.get(r)
        if given is None:
            hot_share = owner.hot_load[n][match.hot] / max(
                1, _count_on(owner, "hot", match.hot)
            )
            cold_share = owner.cold_load[n][match.cold] / max(
                1, _count_on(owner, "cold", match.cold)
            )
            given = 0.5 * min(hot_share, cold_share, _q_reach(owner, n, r, match))
        q_r.append(max(0.0, min(float(given), _q_max(owner, n, match))))
    q_u = []
    for e, match in enumerate(spec.utilities):
        given = spec.initial_utility_duties.get(e)
        if given is None:
            key = "cold" if match.side == HEATER else "hot"
            recovered = sum(
                q
                for q, m in zip(q_r, spec.recovery)
                if (m.cold if key == "cold" else m.hot) == match.stream
            )
            load = owner.cold_load if key == "cold" else owner.hot_load
            residual = load[n][match.stream] - recovered
            given = max(residual, 0.0) / max(
                _utilities_on(owner, match.side, match.stream), 1
            )
        q_u.append(max(0.0, float(given)))

    utility_out = [0.0] * len(spec.utilities)
    th_in = [[0.0] * owner.S for _ in range(owner.I)]
    th_out = [[0.0] * owner.S for _ in range(owner.I)]
    for i in range(owner.I):
        profile = owner.hot_profiles[n][i]
        h = 0.0
        for k in range(owner.S):
            th_in[i][k] = _clamp(profile, profile.temperature_at(h))
            h += sum(q_r[r] for r in owner.hot_matches_at[(i, k)])
            th_out[i][k] = _clamp(profile, profile.temperature_at(h))
            for e in owner.utility_chain.get((COOLER, i, k), []):
                h += q_u[e]
                utility_out[e] = _clamp(profile, profile.temperature_at(h))
    tc_in = [[0.0] * owner.S for _ in range(owner.J)]
    tc_out = [[0.0] * owner.S for _ in range(owner.J)]
    for j in range(owner.J):
        profile = owner.cold_profiles[n][j]
        h = 0.0
        for k in reversed(range(owner.S)):
            tc_in[j][k] = _clamp(profile, profile.temperature_at(h))
            h += sum(q_r[r] for r in owner.cold_matches_at[(j, k)])
            tc_out[j][k] = _clamp(profile, profile.temperature_at(h))
            for e in owner.utility_chain.get((HEATER, j, k), []):
                h += q_u[e]
                utility_out[e] = _clamp(profile, profile.temperature_at(h))
    return {
        "q_r": q_r,
        "q_u": q_u,
        "th_in": th_in,
        "th_out": th_out,
        "tc_in": tc_in,
        "tc_out": tc_out,
        "utility_out": utility_out,
    }


def _clamp(profile: ThermalProfile, temperature: float) -> float:
    return min(max(temperature, profile.lowest), profile.highest)


def _count_on(owner, side: str, stream: int) -> int:
    return sum(
        1 for m in owner.spec.recovery if (m.hot if side == "hot" else m.cold) == stream
    )


def _utilities_on(owner, side: str, stream: int) -> int:
    return sum(1 for m in owner.spec.utilities if (m.side, m.stream) == (side, stream))


def _q_reach(owner, n: int, r: int, match: RecoveryMatch) -> float:
    """Largest duty the end temperatures allow for one match on its own."""

    hot = owner.hot_profiles[n][match.hot]
    cold = owner.cold_profiles[n][match.cold]
    dt = owner.recovery_dt[n][r]
    hot_limit = hot.heat(
        min(max(cold.supply + dt, hot.lowest), hot.highest), smooth=False
    )
    cold_limit = cold.heat(
        min(max(hot.supply - dt, cold.lowest), cold.highest), smooth=False
    )
    return max(0.0, min(hot_limit, cold_limit))


def _q_max(owner, n: int, match: RecoveryMatch) -> float:
    return max(0.0, min(owner.hot_load[n][match.hot], owner.cold_load[n][match.cold]))


# -- equations ----------------------------------------------------------------


def build_fixed_structure_equations(owner) -> None:
    """Create variables and constraints for every period."""

    for name in (
        "q_r",
        "q_u",
        "th_in",
        "th_out",
        "tc_in",
        "tc_out",
        "x",
        "y",
        "t_hx",
        "t_cy",
        "theta_r",
        "theta_u",
        "utility_in",
        "utility_out",
        "utility_fraction",
        "utility_outlet",
    ):
        setattr(owner, name, [])
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
    # A stream that is off in a period carries no duty there: its exchangers'
    # duties are fixed at zero and its heat balances are skipped, which keeps
    # the period's equations free of empty or duplicated rows.
    owner.q_r.append(
        [
            owner._param(f"qr{r}p{n}", 0.0)
            if _recovery_off(owner, n, m)
            else owner._var(f"qr{r}p{n}", start["q_r"][r], 0.0, _q_max(owner, n, m))
            for r, m in enumerate(spec.recovery)
        ]
    )
    owner.q_u.append(
        [
            owner._param(f"qu{e}p{n}", 0.0)
            if stream_off(owner, n, m)
            else owner._var(
                f"qu{e}p{n}",
                start["q_u"][e],
                0.0,
                stream_profile(owner, n, m).total,
            )
            for e, m in enumerate(spec.utilities)
        ]
    )

    def temperature(name, profile, value):
        return owner._var(name, value, profile.lowest, profile.highest)

    hot = owner.hot_profiles[n]
    cold = owner.cold_profiles[n]
    owner.th_in.append(
        [
            [
                owner._param(f"thi{i}s{k}p{n}", hot[i].supply)
                if k == 0
                else temperature(f"thi{i}s{k}p{n}", hot[i], start["th_in"][i][k])
                for k in range(owner.S)
            ]
            for i in range(owner.I)
        ]
    )
    owner.th_out.append(
        [
            [
                temperature(f"tho{i}s{k}p{n}", hot[i], start["th_out"][i][k])
                for k in range(owner.S)
            ]
            for i in range(owner.I)
        ]
    )
    owner.tc_in.append(
        [
            [
                owner._param(f"tci{j}s{k}p{n}", cold[j].supply)
                if k == owner.S - 1
                else temperature(f"tci{j}s{k}p{n}", cold[j], start["tc_in"][j][k])
                for k in range(owner.S)
            ]
            for j in range(owner.J)
        ]
    )
    owner.tc_out.append(
        [
            [
                temperature(f"tco{j}s{k}p{n}", cold[j], start["tc_out"][j][k])
                for k in range(owner.S)
            ]
            for j in range(owner.J)
        ]
    )
    # The last exchanger of a chain shares its outlet with what follows: the
    # next stage's inlet, or the fixed stream target at the stream end.
    utility_out: list[Any] = [None] * len(spec.utilities)
    utility_in: list[Any] = [None] * len(spec.utilities)
    for (side, stream, k), members in owner.utility_chain.items():
        profile = cold[stream] if side == HEATER else hot[stream]
        if side == HEATER:
            previous = owner.tc_out[n][stream][k]
            following = (
                owner.tc_in[n][stream][k - 1]
                if k > 0
                else owner._param(f"tce{stream}p{n}", profile.target)
            )
        else:
            previous = owner.th_out[n][stream][k]
            following = (
                owner.th_in[n][stream][k + 1]
                if k < owner.S - 1
                else owner._param(f"the{stream}p{n}", profile.target)
            )
        for position, e in enumerate(members):
            utility_in[e] = previous
            utility_out[e] = (
                following
                if position == len(members) - 1
                else temperature(f"tuo{e}p{n}", profile, start["utility_out"][e])
            )
            previous = utility_out[e]
    owner.utility_out.append(utility_out)
    owner.utility_in.append(utility_in)


def _stream_balances(owner, n: int) -> None:
    q_r = owner.q_r[n]
    q_u = owner.q_u[n]
    for i in range(owner.I):
        profile = owner.hot_profiles[n][i]
        for k in range(owner.S):
            if not _off(profile):
                owner._equal(
                    profile.heat(owner.th_out[n][i][k])
                    - profile.heat(owner.th_in[n][i][k]),
                    _sum([q_r[r] for r in owner.hot_matches_at[(i, k)]]),
                )
            _close_stream_link(
                owner,
                profile,
                owner.th_out[n][i][k],
                owner.utility_chain.get((COOLER, i, k), []),
                n,
                owner.th_in[n][i][k + 1] if k < owner.S - 1 else profile.target,
            )
    for j in range(owner.J):
        profile = owner.cold_profiles[n][j]
        for k in range(owner.S):
            if not _off(profile):
                owner._equal(
                    profile.heat(owner.tc_out[n][j][k])
                    - profile.heat(owner.tc_in[n][j][k]),
                    _sum([q_r[r] for r in owner.cold_matches_at[(j, k)]]),
                )
            _close_stream_link(
                owner,
                profile,
                owner.tc_out[n][j][k],
                owner.utility_chain.get((HEATER, j, k), []),
                n,
                owner.tc_in[n][j][k - 1] if k > 0 else profile.target,
                implied=k == 0 and j in owner.implied_cold_ends,
            )
    del q_u


def _close_stream_link(
    owner, profile, leaving, chain, n, downstream, *, implied: bool = False
) -> None:
    """Utility exchangers in series between a stage and the next one (or end).

    A chain's last outlet already is ``downstream``; without a chain the
    leaving temperature must equal it (unless the energy balance implies it).
    """

    current = leaving
    for e in chain:
        outlet = owner.utility_out[n][e]
        if not _off(profile):
            owner._equal(profile.heat(outlet) - profile.heat(current), owner.q_u[n][e])
        current = outlet
    if not chain and not implied:
        owner._equal(current, downstream)


def _recovery_equations(owner, n: int, start: Mapping[str, Any]) -> None:
    spec = owner.spec
    x_row, y_row, hx_row, cy_row, theta_row = [], [], [], [], []
    for r, match in enumerate(spec.recovery):
        i, j, k = match.hot, match.cold, match.stage
        hot = owner.hot_profiles[n][i]
        cold = owner.cold_profiles[n][j]
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
        th_in0 = start["th_in"][i][k]
        tc_in0 = start["tc_in"][j][k]
        q0 = start["q_r"][r]
        hx0 = _clamp(
            hot,
            hot.temperature_at(hot.heat(th_in0, smooth=False) + q0 * len(hot_branches)),
        )
        cy0 = _clamp(
            cold,
            cold.temperature_at(
                cold.heat(tc_in0, smooth=False) + q0 * len(cold_branches)
            ),
        )
        t_hx = owner._var(f"thx{r}p{n}", hx0, hot.lowest, hot.highest)
        t_cy = owner._var(f"tcy{r}p{n}", cy0, cold.lowest, cold.highest)
        dt = owner.recovery_dt[n][r]
        upper = hot.supply - cold.supply
        theta_1 = owner._var(
            f"tr1{r}p{n}", min(max(dt, th_in0 - cy0), upper), dt, upper
        )
        theta_2 = owner._var(
            f"tr2{r}p{n}", min(max(dt, hx0 - tc_in0), upper), dt, upper
        )
        q = owner.q_r[n][r]
        th_in = owner.th_in[n][i][k]
        tc_in = owner.tc_in[n][j][k]
        if not _off(hot):
            owner._equal(q, x * (hot.heat(t_hx) - hot.heat(th_in)))
        if not _off(cold):
            owner._equal(q, y * (cold.heat(t_cy) - cold.heat(tc_in)))
        owner._equal(theta_1, th_in - t_cy)
        owner._equal(theta_2, t_hx - tc_in)
        if not _recovery_off(owner, n, match):
            _interior_approaches(
                owner,
                f"r{r}p{n}",
                (hot, x, th_in, t_hx),
                (cold, y, tc_in, t_cy),
                q,
                dt,
            )
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


def _interior_approaches(owner, tag: str, hot_side, cold_side, q, dt: float) -> None:
    """Minimum approach at segment boundaries inside a counter-current match.

    Each side is ``(profile, flow fraction, inlet, outlet)``. A boundary of one
    side is clipped (smoothly) into that side's temperature range; the other
    side's temperature at the same duty position follows from its heat
    balance and must stay ``dt`` away.
    """

    hot, a, h_in, h_out = hot_side
    cold, b, c_in, c_out = cold_side
    for index, boundary in enumerate(hot.breakpoints):
        point = smooth_clip(boundary, h_out, h_in)
        given = a * (hot.heat(point) - hot.heat(h_in))
        partner = owner._var(
            f"bh{index}{tag}", _midpoint(cold), cold.lowest, cold.highest
        )
        owner._equal(b * (cold.heat(partner) - cold.heat(c_in)), q - given)
        owner._at_least(point - partner, dt)
    for index, boundary in enumerate(cold.breakpoints):
        point = smooth_clip(boundary, c_in, c_out)
        received = b * (cold.heat(point) - cold.heat(c_in))
        partner = owner._var(f"bc{index}{tag}", _midpoint(hot), hot.lowest, hot.highest)
        owner._equal(a * (hot.heat(partner) - hot.heat(h_in)), q - received)
        owner._at_least(partner - point, dt)


def _midpoint(profile: ThermalProfile) -> float:
    return 0.5 * (profile.supply + profile.target)


def _utility_equations(owner, n: int, start: Mapping[str, Any]) -> None:
    theta_row = []
    fraction_row: list[Any] = []
    outlet_row: list[Any] = []
    for e, match in enumerate(owner.spec.utilities):
        profile = stream_profile(owner, n, match)
        own = utility_profile(owner, n, match)
        stream_in = owner.utility_in[n][e]
        stream_out = owner.utility_out[n][e]
        q = owner.q_u[n][e]
        dt = owner.utility_dt[n][e]
        if own is None:
            util_in, util_out = utility_side_temperatures(owner, n, e)
            fraction = None
            util_outlet = util_out
        else:
            fraction = owner._var(
                f"w{e}p{n}", start["q_u"][e] / max(own.total, 1e-9), 0.0, None
            )
            util_in = own.supply
            util_outlet = owner._var(f"tw{e}p{n}", own.target, own.lowest, own.highest)
            owner._equal(q, fraction * own.heat(util_outlet))
        if match.side == HEATER:
            hot_end = util_in - stream_out
            cold_end = util_outlet - stream_in
        else:
            hot_end = stream_in - util_outlet
            cold_end = stream_out - util_in
        start_in = _start_stream_inlet(owner, n, e, start)
        start_out = start["utility_out"][e]
        start_util_out = own.target if own is not None else util_out
        if match.side == HEATER:
            start_ends = (util_in - start_out, start_util_out - start_in)
        else:
            start_ends = (start_in - start_util_out, start_out - util_in)
        theta_1 = owner._var(f"tu1{e}p{n}", max(dt, start_ends[0]), dt, None)
        theta_2 = owner._var(f"tu2{e}p{n}", max(dt, start_ends[1]), dt, None)
        owner._equal(theta_1, hot_end)
        owner._equal(theta_2, cold_end)
        if stream_off(owner, n, match):
            pass
        elif own is not None:
            utility_side = (own, fraction, util_in, util_outlet)
            stream_side = (profile, 1.0, stream_in, stream_out)
            hot_side, cold_side = (
                (utility_side, stream_side)
                if match.side == HEATER
                else (stream_side, utility_side)
            )
            _interior_approaches(owner, f"u{e}p{n}", hot_side, cold_side, q, dt)
        elif profile.segmented and abs(util_in - util_out) > 1e-9:
            _linear_utility_interior(
                owner, match, profile, stream_in, stream_out, util_in, util_out, q, dt
            )
        theta_row.append((theta_1, theta_2))
        fraction_row.append(fraction)
        outlet_row.append(util_outlet)
    owner.theta_u.append(theta_row)
    owner.utility_fraction.append(fraction_row)
    owner.utility_outlet.append(outlet_row)


def _start_stream_inlet(owner, n: int, e: int, start) -> float:
    """Stream-side inlet temperature of utility exchanger ``e`` at the start."""

    match = owner.spec.utilities[e]
    members = owner.utility_chain[(match.side, match.stream, match.after_stage)]
    position = members.index(e)
    if position > 0:
        return start["utility_out"][members[position - 1]]
    if match.side == HEATER:
        return start["tc_out"][match.stream][match.after_stage]
    return start["th_out"][match.stream][match.after_stage]


def _linear_utility_interior(
    owner, match, profile, stream_in, stream_out, util_in, util_out, q, dt
) -> None:
    """Approach at stream segment boundaries against a sliding utility.

    A constant-flow utility's temperature is linear in duty, so with ``d`` the
    stream-side duty at the boundary the utility temperature there is
    ``util_in + (util_out - util_in) * (q - d) / q``. Multiplying by ``q``
    keeps the constraint smooth and trivially true at zero duty.
    """

    for boundary in profile.breakpoints:
        if match.side == HEATER:
            point = smooth_clip(boundary, stream_in, stream_out)
            d = profile.heat(point) - profile.heat(stream_in)
            owner._at_least(
                q * (util_in - point) - (util_in - util_out) * (q - d), dt * q
            )
        else:
            point = smooth_clip(boundary, stream_out, stream_in)
            d = profile.heat(point) - profile.heat(stream_in)
            owner._at_least(
                q * (point - util_in) - (util_out - util_in) * (q - d), dt * q
            )


def _utility_caps(owner, n: int) -> None:
    heaters = [
        e
        for e, m in enumerate(owner.spec.utilities)
        if m.side == HEATER and not stream_off(owner, n, m)
    ]
    coolers = [
        e
        for e, m in enumerate(owner.spec.utilities)
        if m.side == COOLER and not stream_off(owner, n, m)
    ]
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
    """Weighted utility cost; segmented utilities pay their segment prices."""

    return _weighted(
        owner,
        [
            _sum(
                [
                    _utility_cost_term(owner, n, e)
                    for e in range(len(owner.spec.utilities))
                ]
            )
            for n in range(owner.N_periods)
        ],
    )


def _utility_cost_term(owner, n: int, e: int):
    match = owner.spec.utilities[e]
    own = utility_profile(owner, n, match)
    if own is None:
        return _utility_price(owner, n, e) * owner.q_u[n][e]
    return owner.utility_fraction[n][e] * own.cost(owner.utility_outlet[n][e])


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
    period_ids = [str(p) for p in owner.period_ids]

    owner.recovery_results = []
    for r, match in enumerate(spec.recovery):
        periods = []
        for n in range(owner.N_periods):
            hot = owner.hot_profiles[n][match.hot]
            cold = owner.cold_profiles[n][match.cold]
            duty = value(owner.q_r[n][r])
            active = duty > tolerance
            theta_1, theta_2 = (value(t) for t in owner.theta_r[n][r])
            x = value(owner.x[n][r])
            y = value(owner.y[n][r])
            th_in = value(owner.th_in[n][match.hot][match.stage])
            tc_in = value(owner.tc_in[n][match.cold][match.stage])
            slices: tuple = ()
            if active and (hot.segmented or cold.segmented):
                slices = _slices(
                    hot.piecewise(max(x, 1e-9)),
                    cold.piecewise(max(y, 1e-9)),
                    duty,
                    th_in,
                    tc_in,
                    period_ids[n],
                )
            periods.append(
                ExchangerPeriodResult(
                    duty=duty if active else 0.0,
                    active=active,
                    approach=(theta_1, theta_2),
                    source_inlet=th_in,
                    source_outlet=value(owner.t_hx[n][r]),
                    sink_inlet=tc_in,
                    sink_outlet=value(owner.t_cy[n][r]),
                    required_area=(
                        sum(s.area for s in slices)
                        if slices
                        else _required_area(
                            duty if active else 0.0,
                            owner.recovery_U[n][r],
                            theta_1,
                            theta_2,
                        )
                    ),
                    source_split=x,
                    sink_split=y,
                    slices=slices,
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
    for e, match in enumerate(spec.utilities):
        periods = []
        for n in range(owner.N_periods):
            profile = stream_profile(owner, n, match)
            own = utility_profile(owner, n, match)
            duty = value(owner.q_u[n][e])
            active = duty > tolerance
            theta_1, theta_2 = (value(t) for t in owner.theta_u[n][e])
            stream_in = value(owner.utility_in[n][e])
            stream_out = value(owner.utility_out[n][e])
            fraction = None
            if own is None:
                util_in, util_out = utility_side_temperatures(owner, n, e)
                utility_curve = None
            else:
                fraction = value(owner.utility_fraction[n][e])
                util_in = own.supply
                util_out = value(owner.utility_outlet[n][e])
                utility_curve = own.piecewise(max(fraction, 1e-9))
            slices = ()
            if active and (profile.segmented or own is not None):
                if utility_curve is None:
                    utility_curve = utility_thermal_profile(
                        identity=_utility_name(owner, match.side, match.utility),
                        inlet_temperature=util_in,
                        outlet_temperature=util_out,
                        duty=duty,
                        heat_transfer_coefficient=_utility_htc(owner, n, match),
                    )
                stream_curve = profile.piecewise()
                if match.side == HEATER:
                    slices = _slices(
                        utility_curve,
                        stream_curve,
                        duty,
                        util_in,
                        stream_in,
                        period_ids[n],
                    )
                else:
                    slices = _slices(
                        stream_curve,
                        utility_curve,
                        duty,
                        stream_in,
                        util_in,
                        period_ids[n],
                    )
            source, sink = (
                ((util_in, util_out), (stream_in, stream_out))
                if match.side == HEATER
                else ((stream_in, stream_out), (util_in, util_out))
            )
            periods.append(
                ExchangerPeriodResult(
                    duty=duty if active else 0.0,
                    active=active,
                    approach=(theta_1, theta_2),
                    source_inlet=source[0],
                    source_outlet=source[1],
                    sink_inlet=sink[0],
                    sink_outlet=sink[1],
                    required_area=(
                        sum(s.area for s in slices)
                        if slices
                        else _required_area(
                            duty if active else 0.0,
                            owner.utility_U[n][e],
                            theta_1,
                            theta_2,
                        )
                    ),
                    utility_flow=fraction,
                    slices=slices,
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
            _solved_utility_cost(owner, n, e, result)
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


def _solved_utility_cost(owner, n: int, e: int, result: ExchangerResult) -> float:
    period = result.periods[n]
    if not period.active:
        return 0.0
    match = owner.spec.utilities[e]
    own = utility_profile(owner, n, match)
    if own is None:
        return _utility_price(owner, n, e) * period.duty
    fraction = _numeric(owner.utility_fraction[n][e])
    outlet = _numeric(owner.utility_outlet[n][e])
    return fraction * own.cost(outlet, smooth=False)


def _slices(hot_curve, cold_curve, duty, hot_inlet, cold_inlet, period):
    return duty_aligned_area_contributions(
        hot_curve,
        cold_curve,
        duty=duty,
        hot_inlet_temperature=min(max(hot_inlet, _low(hot_curve)), _high(hot_curve)),
        cold_inlet_temperature=min(
            max(cold_inlet, _low(cold_curve)), _high(cold_curve)
        ),
        period=period,
        tolerance=1e-6,
    )


def _low(curve) -> float:
    return float(min(curve.temperatures_in[0], curve.temperatures_out[-1]))


def _high(curve) -> float:
    return float(max(curve.temperatures_in[0], curve.temperatures_out[-1]))


def _design_area(periods: Sequence[ExchangerPeriodResult]) -> float:
    """Common design area: the largest area any period needs (exact LMTD).

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


def _safe_name(names, index: int, default: str) -> str:
    try:
        return str(names[index])
    except TypeError, IndexError:
        return default


def _name(owner, side: str, index: int) -> str:
    names = getattr(owner, "cold_names" if side == "cold" else "hot_names", None)
    return _safe_name(names, index, f"{side}[{index}]")


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
    "utility_chains",
]
