"""Descriptive heat-exchanger-network design workflows and result views."""

from __future__ import annotations

import json
import threading
import warnings
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

from ..arguments import (
    split_runtime_and_configuration_options,
    temporary_zone_configuration,
)

if TYPE_CHECKING:
    from ....contracts.synthesis.result import HeatExchangerNetworkSynthesisResult
    from ....contracts.synthesis.retrofit import (
        HeatExchangerNetworkRetrofitResult,
        RetrofitHeatPumpTargets,
    )
    from ....contracts.synthesis.view import (
        HeatExchangerNetworkGridLayout,
        HeatExchangerNetworkView,
    )
    from ....domain.heat_exchanger_network import HeatExchangerNetwork
    from ....domain.zone import Zone
    from ...problem import PinchProblem


# Options a search level sets when they are unset (execution/search.py).
_WIDTH_OPTIONS = ("HENS_DT_CONT_MULTIPLIERS",)
# Ranked networks each process offers a site design: enough to hold its best
# at every multiplier of the widest search.
_SITE_POOL = 50
_DEPTH_OPTIONS = (
    "HENS_HEURISTIC_PDM_NETWORKS",
    "HENS_HEURISTIC_PDM_EVOLUTION_SEEDS",
    "HENS_PDM_MILP",
    "HENS_EVM_N_AD_BRANCHES",
    "HENS_EVM_N_RM_BRANCHES",
)
# Design bases synthesised on a derived case, whose networks are then fitted
# to every period.
_FITTED_BASES = ("median", "period")


def _rounded(value):
    if isinstance(value, float):
        return float(f"{value:.9g}")
    if isinstance(value, dict):
        return {key: _rounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rounded(item) for item in value]
    return value


def _finalise_key(network) -> str:
    """What fitting and rating ``network`` depend on, as a lookup key.

    The task labels (``task_id`` and the ``dt_cont_multiplier`` summary
    metric) are left out: neither fit nor rating reads them. Values are
    rounded to 9 significant digits, since solves of one network in
    different tasks agree to about 1e-15 relative, not bit for bit.
    """
    data = network.model_dump(mode="json")
    data.pop("task_id", None)
    data.get("summary_metrics", {}).pop("dt_cont_multiplier", None)
    return json.dumps(_rounded(data), sort_keys=True)


class UtilityChoiceError(ValueError):
    """A heater or cooler names no utility, and its side has several to choose from."""

    def __init__(self, side: str, utilities: Sequence[str]) -> None:
        super().__init__(
            f"pass {side}_utility explicitly; the problem defines "
            f"{len(utilities)} {side} utilities ({', '.join(utilities)})."
        )
        self.side = side
        self.utilities = tuple(utilities)


def _set_if_supplied(values: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        values[key] = value


_RECOVERY_ENTRY_KEYS = frozenset({"hot", "cold", "stage", "area", "id"})
_UTILITY_ENTRY_KEYS = frozenset({"stream", "utility", "stage", "area", "id"})


def _entry_extra(entry: Mapping[str, Any], allowed, kind: str) -> dict[str, Any]:
    unknown = sorted(str(key) for key in entry if key not in allowed)
    if unknown:
        raise ValueError(
            f"a {kind} entry accepts {', '.join(sorted(allowed))}; got "
            + ", ".join(unknown)
            + "."
        )
    extra: dict[str, Any] = {}
    if entry.get("area") is not None:
        extra["area"] = float(entry["area"])
    if entry.get("id") is not None:
        extra["exchanger_id"] = str(entry["id"])
    return extra


def _recovery_entry(entry) -> tuple[Any, dict[str, Any]]:
    """A recovery entry as a ``(hot, cold, stage)`` placement and extras."""

    if not isinstance(entry, Mapping):
        return entry, {}
    extra = _entry_extra(entry, _RECOVERY_ENTRY_KEYS, "recovery")
    return (entry["hot"], entry["cold"], entry["stage"]), extra


def _utility_entry(entry, default: str | None, side: str) -> tuple[Any, dict]:
    """A heater or cooler entry as a placement and extras."""

    if not isinstance(entry, Mapping):
        return entry, {}
    extra = _entry_extra(entry, _UTILITY_ENTRY_KEYS, side[:-1])
    utility = entry.get("utility") or default
    if entry.get("stage") is None:
        return (entry["stream"], utility), extra
    return (entry["stream"], utility, entry["stage"]), extra


def _names_no_utility(entries) -> bool:
    """Whether a heater or cooler entry leaves its utility to the default."""

    return any(
        isinstance(entry, str)
        or (isinstance(entry, Mapping) and entry.get("utility") is None)
        for entry in entries
    )


class HeatExchangerNetworkDesignView:
    """Explicit application behavior around one serializable HEN result."""

    def __init__(
        self,
        result: "HeatExchangerNetworkSynthesisResult",
        *,
        utility_names: Sequence[str] | None = None,
    ) -> None:
        self._result = result
        # The problem's utilities as the network names them ("Hot Utility.HPS"),
        # so ``utility`` can resolve a utility's own name and reject others.
        self._utility_names = None if utility_names is None else tuple(utility_names)

    @property
    def result(self) -> "HeatExchangerNetworkSynthesisResult":
        """Return the complete serializable synthesis result."""
        return self._result

    @property
    def selected_network(self) -> "HeatExchangerNetwork":
        """Return the network selected by the synthesis service."""
        return self.result.network

    def top(self, n: int):
        """Return the first ``n`` ranked successful outcomes.

        Uses the same ranking as ``network(rank=)`` and ``grid(rank=)``, so
        ``top(n)[k - 1]`` is the network at rank ``k``.
        """
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError("n must be a positive integer.")
        from ....analysis.heat_exchanger_networks.results.selection import (
            ranked_networks,
        )

        return ranked_networks(self.result, n)

    def network(self, *, rank: int = 1) -> "HeatExchangerNetwork":
        """Return one ranked network using a one-based rank."""
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise ValueError("rank must be a positive one-based integer.")
        from ....analysis.heat_exchanger_networks.results.selection import (
            network_for_rank,
        )

        return network_for_rank(self.result, rank)

    def grid(
        self,
        *,
        rank: int = 1,
        period_id: str | None = None,
        temperature_scaled: bool = False,
    ):
        """Lazily render one ranked network as a grid diagram."""
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise ValueError("rank must be a positive one-based integer.")
        from ....presentation.network_grid.results import build_result_grid_diagram

        return build_result_grid_diagram(
            self.result,
            solution_rank=rank,
            period_id=period_id,
            temperature_scaled=temperature_scaled,
        )

    def grid_layout(
        self,
        *,
        rank: int = 1,
        period_id: str | None = None,
    ) -> "HeatExchangerNetworkGridLayout":
        """Return one ranked network's grid topology as data (no plotting)."""
        from ....presentation.network_grid.layout import build_grid_layout

        return build_grid_layout(self.network(rank=rank), period_id=period_id)

    def view(
        self,
        *,
        rank: int = 1,
        period_id: str | None = None,
    ) -> "HeatExchangerNetworkView":
        """Return the ranking, and one ranked network's totals, duties and grid."""
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise ValueError("rank must be a positive one-based integer.")
        from ....presentation.network_grid.view import build_network_view

        return build_network_view(self.result, rank=rank, period_id=period_id)

    def views(self, *, rank: int = 1) -> tuple["HeatExchangerNetworkView", ...]:
        """Return ``view`` for each period of one ranked network, in period order.

        A single-period network has one view; a shared multiperiod network
        has one per period, each naming its ``period_id``.
        """
        period_ids = self.network(rank=rank).period_ids or (None,)
        return tuple(
            self.view(rank=rank, period_id=period_id) for period_id in period_ids
        )

    @property
    def total_heat_recovery(self) -> float:
        from ....domain.enums import HeatExchangerKind

        return self.selected_network.total_duty(kind=HeatExchangerKind.RECOVERY)

    @property
    def total_hot_utility(self) -> float:
        from ....domain.enums import HeatExchangerKind

        return self.selected_network.total_duty(kind=HeatExchangerKind.HOT_UTILITY)

    @property
    def total_cold_utility(self) -> float:
        from ....domain.enums import HeatExchangerKind

        return self.selected_network.total_duty(kind=HeatExchangerKind.COLD_UTILITY)

    def _network_utility_name(self, name: str) -> str:
        """The network's stream name for ``name``, a utility or stream name."""
        if self._utility_names is None or name in self._utility_names:
            return name
        matches = [
            utility
            for utility in self._utility_names
            if utility.split(".", 1)[-1] == name
        ]
        if len(matches) > 1:
            raise ValueError(
                f"Utility {name!r} is both a hot and a cold utility; ask for "
                + " or ".join(repr(match) for match in matches)
                + "."
            )
        if not matches:
            raise ValueError(
                f"Unknown utility {name!r}. The problem's utilities are "
                f"{', '.join(self._utility_names)}."
            )
        return matches[0]

    def utility(self, name: str, *, period_id: str | None = None) -> float:
        """Return selected-network utility duty for one named utility."""
        from ....domain.enums import HeatExchangerKind

        if not isinstance(name, str) or not name.strip():
            raise ValueError("utility name must be a non-empty string")
        utility_name = self._network_utility_name(name.strip())
        network = self.selected_network
        resolved_period_id = network.resolve_period_id(period_id)
        total = 0.0
        for exchanger in network.exchangers:
            if exchanger.kind is HeatExchangerKind.HOT_UTILITY:
                stream = exchanger.source_stream
            elif exchanger.kind is HeatExchangerKind.COLD_UTILITY:
                stream = exchanger.sink_stream
            else:
                continue
            if stream == utility_name:
                state = exchanger.state(resolved_period_id)
                if state.active:
                    total += float(state.duty)
        return total


class _DesignAccessor:
    """Explicit HEN design workflows rooted in one problem."""

    def __init__(self, problem: "PinchProblem") -> None:
        self._problem = problem

    def _arguments(
        self,
        *,
        options: Mapping[str, Any] | None,
        period_id: str | None,
        width=None,
        depth=None,
        dt_cont_multipliers=None,
        derivative_thresholds=None,
        stages=None,
        pack_stages: bool | None = None,
        solver=None,
        max_parallel=None,
        best_solutions=None,
        solve_tolerance=None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if options is not None and not isinstance(options, Mapping):
            raise TypeError("design options must be supplied as a mapping.")
        runtime, configuration = split_runtime_and_configuration_options(options)
        named: dict[str, Any] = {}
        for key, value in (
            ("HENS_SEARCH_WIDTH", width),
            ("HENS_SEARCH_DEPTH", depth),
            ("HENS_DT_CONT_MULTIPLIERS", dt_cont_multipliers),
            ("HENS_DERIVATIVE_THRESHOLDS", derivative_thresholds),
            ("HENS_STAGE_SELECTION", stages),
            ("HENS_MAX_PARALLEL", max_parallel),
            ("HENS_BEST_SOLUTIONS_TO_SAVE", best_solutions),
            ("HENS_SOLVE_TOLERANCE", solve_tolerance),
        ):
            _set_if_supplied(named, key, value)
        if pack_stages is not None:
            if not isinstance(pack_stages, bool):
                raise TypeError("pack_stages must be a bool when supplied.")
            named["HENS_STAGE_PACKING"] = "all" if pack_stages else "none"
        if solver is not None:
            for key in ("HENS_SOLVER_PDM", "HENS_SOLVER_TDM", "HENS_SOLVER_EVM"):
                named[key] = solver
        configuration.update(named)
        # A level asked for in the call, as an argument or in its options,
        # sets the options it governs over the stored ones, unless the call
        # sets them too.
        for level, governed in (
            ("HENS_SEARCH_WIDTH", _WIDTH_OPTIONS),
            ("HENS_SEARCH_DEPTH", _DEPTH_OPTIONS),
        ):
            if level in configuration:
                for key in governed:
                    configuration.setdefault(key, None)
        if period_id is not None:
            runtime["period_id"] = period_id
        return runtime, configuration

    def _run(self, service, *, surface, runtime, configuration, **service_kwargs):
        from ....domain.enums import StreamLoc
        from ..targeting.execution import walk_zone_tree
        from ..targeting.provenance import make_provenance
        from ..targeting.state import commit_problem, snapshot_problem

        isolated = snapshot_problem(self._problem)
        root = isolated._build_execution_master_zone()
        stored_configs = {zone.address: zone.config for zone in walk_zone_tree(root)}
        with temporary_zone_configuration(root, configuration):
            result = service(isolated, options=runtime, **service_kwargs)
        # A specialized service may prepare thermal targets transactionally and
        # replace the tree. Restore stored configuration on the committed tree too.
        for zone in walk_zone_tree(isolated._master_zone):
            zone.config = stored_configs[zone.address]
        if hasattr(result, "provenance"):
            period_ids = (
                (runtime["period_id"],)
                if runtime.get("period_id") is not None
                else tuple(root.period_ids)
            )
            settings = {**root.config._values, **configuration, **runtime}
            result.provenance = make_provenance(
                isolated,
                "design." + surface,
                isolated._master_zone,
                period_ids=period_ids,
                settings=settings,
            )
            if isolated._results is not None:
                isolated._results.design = result
        view = HeatExchangerNetworkDesignView(
            deepcopy(result),
            utility_names=[
                f"{location.value}.{utility.name}"
                for location, utilities in (
                    (StreamLoc.HotU, self._problem.hot_utilities),
                    (StreamLoc.ColdU, self._problem.cold_utilities),
                )
                for utility in utilities
            ],
        )
        commit_problem(self._problem, isolated)
        return view

    def heat_exchanger_network(
        self,
        *,
        zone: "str | Zone | None" = None,
        width: int | None = None,
        depth: int | None = None,
        options: Mapping[str, Any] | None = None,
        period_id: str | None = None,
        design_basis: str | None = None,
        basis_clusters: int | None = None,
        overdesign_percentage=None,
        dt_cont_multipliers=None,
        derivative_thresholds=None,
        stages=None,
        pack_stages: bool | None = None,
        solver=None,
        max_parallel=None,
        best_solutions=None,
        solve_tolerance=None,
        record=None,
    ) -> HeatExchangerNetworkDesignView:
        """Run HEN synthesis at a search width and depth.

        On a multiperiod case the network serves every operating period;
        ``design_basis`` (``HENS_DESIGN_BASIS``) is what it is synthesised
        from. ``"median"`` (the default) synthesises on a case of each
        stream's period-weighted median over the operating periods, or over
        the heaviest of ``basis_clusters`` clusters of them (weighted
        k-medoids); ``design_basis()`` previews it. ``"period"`` synthesises
        on ``period_id`` alone. Both then fit the network to every period:
        each exchanger gets one area, with bypass where a period needs less,
        and a heater or cooler is added at a stream end where a period needs
        one. ``"all"`` designs over every period at once, after all-period
        targeting when there is none. Passing ``period_id`` implies
        ``"period"``. On a single-period case the basis changes nothing.

        ``overdesign_percentage`` (``HENS_OVERDESIGN_PERCENTAGE``) enlarges
        every heat-recovery area by that margin and rates the network over
        every period at those areas: each period's duties are re-optimised
        for the least utility cost, with bypass where a period needs less.
        ``"optimise"`` picks the margin, up to ``HENS_OVERDESIGN_MAX`` per
        cent, for the least total annual cost. The result's
        ``selected_network.summary_metrics["overdesign"]`` reports it.

        ``width`` (``HENS_SEARCH_WIDTH``, 0 to 3, default 1) picks the dt_cont
        multipliers: 1.0; 0.75, 1.0 and 1.5; then five and seven values from
        0.5 to 2.0. ``depth`` (``HENS_SEARCH_DEPTH``, 0 to 4, default 1) picks
        the stages and their effort: 0 is the heuristic pinch design method
        alone, 1 adds network evolution, 2 adds the MILP pinch design method
        and the thermal derivative method, and 3 and 4 add more networks and
        wider evolution. An option set explicitly, such as
        ``dt_cont_multipliers`` or ``HENS_HEURISTIC_PDM_NETWORKS``, overrides
        the level.

        Each multiplier scales every stream's and utility's ΔT contribution as
        a zone dt_cont multiplier does in targeting, so 1.0 designs at the
        targeting contributions. The pinch design step keeps those
        contributions; the later thermal derivative and evolution steps choose
        each exchanger's approach by total annual cost, down to 0.1 K, or down
        to ``HENS_MIN_EXCHANGER_APPROACH_SHARE`` (in ``options``) times twice
        the scaled ``THERMAL_DT_CONT`` if that is larger.

        ``record``, a ``HeatExchangerNetworkRecord``, receives every network
        as soon as it is verified and costed in its final form (fitted to
        every period and rated at the margin where the design calls for it),
        so another thread can poll it while the design runs. The result ranks
        exactly the record's networks. A record collects one call.

        ``zone`` is one zone, by name or address as targeting takes it
        (``"Hospital[S]"``) or as a ``Zone`` such as
        ``problem.master_zone.subzones["Hospital[S]"]``; the network then
        serves that zone's streams alone, with every utility of the problem.
        It is designed on a case derived from the zone, so the problem's
        stored results are not changed. The root zone (the default) designs
        on every stream.
        """
        if zone is not None:
            target = self._problem._resolve_target_zone(zone)
            if target.address != self._problem._require_prepared_root_zone().address:
                from ...case_derivation import _zone_case

                return _zone_case(self._problem, target).design.heat_exchanger_network(
                    width=width,
                    depth=depth,
                    options=options,
                    period_id=period_id,
                    design_basis=design_basis,
                    basis_clusters=basis_clusters,
                    overdesign_percentage=overdesign_percentage,
                    dt_cont_multipliers=dt_cont_multipliers,
                    derivative_thresholds=derivative_thresholds,
                    stages=stages,
                    pack_stages=pack_stages,
                    solver=solver,
                    max_parallel=max_parallel,
                    best_solutions=best_solutions,
                    solve_tolerance=solve_tolerance,
                    record=record,
                )
        from ....analysis.heat_exchanger_networks.utility_only import is_one_sided

        if len(self._process_zones()) > 1:
            warnings.warn(
                "heat_exchanger_network() pools the streams of every process "
                "zone, so it allows any inter-process match with no piping "
                "cost and places utilities by temperature for the site. Use "
                "design.total_site_heat_exchanger_network() for a network that "
                "recovers heat within each process and holds each process's "
                "utility levels.",
                UserWarning,
                stacklevel=2,
            )
        hens = self._effective_hens(options)
        margin = self._overdesign_margin(overdesign_percentage, hens)
        basis = self._design_basis(design_basis, period_id, hens)
        if is_one_sided(self._problem._require_prepared_root_zone()):
            # Utility exchangers alone need no fit: design over every period.
            if margin is not None:
                raise ValueError(
                    "overdesign is not available for a network of only hot or "
                    "only cold streams."
                )
            basis, period_id = "all", None
        arguments = dict(
            width=width,
            depth=depth,
            options=options,
            dt_cont_multipliers=dt_cont_multipliers,
            derivative_thresholds=derivative_thresholds,
            stages=stages,
            pack_stages=pack_stages,
            solver=solver,
            max_parallel=max_parallel,
            best_solutions=best_solutions,
            solve_tolerance=solve_tolerance,
        )
        if margin is None and basis not in _FITTED_BASES:
            return self._synthesise(
                basis=basis, period_id=period_id, record=record, **arguments
            )
        return self._design_finalised(
            record,
            basis=basis,
            margin=margin,
            hens=hens,
            period_id=period_id,
            basis_clusters=basis_clusters,
            arguments=arguments,
        )

    def total_site_heat_exchanger_network(
        self,
        *,
        width: int | None = None,
        depth: int | None = None,
        options: Mapping[str, Any] | None = None,
        levels: str | None = None,
        level_counts: Sequence[int] | None = None,
        placement: Mapping[str, Any] | None = None,
    ) -> HeatExchangerNetworkDesignView:
        """Design one network for a site of several process zones.

        Each process zone (an immediate subzone of type ``Process Zone``) is
        designed alone with the pinch design method at the search width's
        dt_cont multipliers, its utilities placed by temperature, so each
        level carries the duty targeting gives it. The process networks are
        then joined, one per process at each multiplier, into site seeds,
        and network evolution refines the seeds as one problem: every
        process keeps each utility level at its duty, and heat is recovered
        only between streams of one process. There is no thermal derivative
        step.

        ``depth`` sets the evolution's breadth as in
        ``heat_exchanger_network``, and at 2 and above the MILP pinch
        design method joins the heuristic for the process networks.

        On a multiperiod site each process is designed over every period at
        once (design basis ``"all"``), so its levels carry their targeted
        duty in each period, and the evolution holds them per period with
        one area per exchanger over all periods; no fit to the periods
        follows.

        ``levels`` (``HENS_SITE_LEVELS``) ``"given"`` designs with the case's
        utilities. ``"placed"`` places the site's levels by utility placement
        (``placement`` holds its options) once per count in ``level_counts``
        (``HENS_SITE_LEVEL_COUNTS``, default 2 to 5), each level serving
        heating and cooling, designs the site with each set, and selects the
        cheapest design. A count whose placement is infeasible or whose
        design fails is skipped. The result ranks each designed count's best
        network, tagged ``summary_metrics["utility_level_count"]``, and its
        provenance records every count (``HENS_SITE_LEVEL_SWEEP``).
        ``HENS_SITE_LEVEL_EARLY_STOP`` ends the sweep once two designed
        counts in a row cost more than the cheapest before them.
        """
        from ....domain.configuration_fields import _site_level_counts
        from ... import site_levels

        processes = self._process_zones()
        if len(processes) < 2:
            raise ValueError(
                "a site network needs more than one process zone; design a "
                "single process with heat_exchanger_network()."
            )
        hens = self._effective_hens(options)
        levels = hens.site_levels if levels is None else levels
        if levels == "given":
            return self._site_design(width=width, depth=depth, options=options)
        if levels != "placed":
            raise ValueError('levels must be "given" or "placed".')
        counts = tuple(
            hens.site_level_counts
            if level_counts is None
            else _site_level_counts("level_counts", level_counts)
        )
        runs = site_levels.run_level_counts(
            self._problem,
            counts,
            design=lambda case: case.design._site_design(
                width=width, depth=depth, options=options
            ),
            early_stop=hens.site_level_early_stop,
            placement=placement,
        )
        return site_levels.level_sweep_view(
            self._problem,
            runs,
            settings={
                **dict(options or {}),
                "HENS_SITE_LEVELS": "placed",
                "HENS_SITE_LEVEL_COUNTS": list(counts),
                "placement": dict(placement or {}),
            },
        )

    def _site_design(self, *, width, depth, options) -> HeatExchangerNetworkDesignView:
        """The site design with the case's utilities (``levels="given"``)."""
        from ....analysis.heat_exchanger_networks.execution.search import (
            resolve_search,
        )
        from ....analysis.heat_exchanger_networks.site import site_seeds

        processes = self._process_zones()
        _runtime, configuration = self._arguments(
            options=options, period_id=None, width=width, depth=depth
        )
        search = resolve_search(self._effective_hens(configuration))
        process_options = {
            **configuration,
            "HENS_SEARCH_DEPTH": 0,
            "HENS_PDM_MILP": search.milp,
            "HENS_UTILITY_PLACEMENT": "temperature",
            # Over every period, so each level holds its duty per period.
            "HENS_DESIGN_BASIS": "all",
        }
        pools = {
            name: tuple(
                outcome.network
                for outcome in self.heat_exchanger_network(
                    zone=name, options=process_options
                ).top(_SITE_POOL)
            )
            for name in processes
        }
        return self.network_evolution(
            site_seeds(pools), options=options, width=width, depth=depth
        )

    def _process_zones(self) -> list[str]:
        """The master zone's immediate subzones of type ``Process Zone``."""
        from ....domain.enums import ZoneType

        return [
            name
            for name, zone in (self._problem.master_zone.subzones or {}).items()
            if str(getattr(zone, "type", "")) == ZoneType.P.value
        ]

    def _effective_hens(self, options):
        """The HENS configuration a call runs with: its options over the stored."""
        from copy import copy

        if options is not None and not isinstance(options, Mapping):
            raise TypeError("design options must be supplied as a mapping.")
        _runtime, configuration = split_runtime_and_configuration_options(options)
        root = self._problem._require_prepared_root_zone()
        with temporary_zone_configuration(root, configuration):
            return copy(root.config.hens)

    def _design_finalised(
        self, record, *, basis, margin, hens, period_id, basis_clusters, arguments
    ) -> HeatExchangerNetworkDesignView:
        """Synthesise, admitting each network once fitted and rated.

        On a derived basis every network is fitted to every period; with a
        margin it is then rated at it. The fitted, rated networks make up
        ``record`` and the result, whose first is selected.
        """
        from ....analysis.heat_exchanger_networks.errors import WorkflowContractError
        from ....analysis.heat_exchanger_networks.results.record import (
            HeatExchangerNetworkRecord,
            running,
        )
        from ....analysis.heat_exchanger_networks.results.selection import (
            select_network,
        )
        from ....analysis.heat_exchanger_networks.solver.dependencies import (
            MissingSynthesisDependencyError,
            MissingSynthesisSolverError,
        )

        record = HeatExchangerNetworkRecord() if record is None else record
        fit = basis in _FITTED_BASES
        call = {key: arguments[key] for key in ("options", "solver", "solve_tolerance")}
        # What the networks admitted now were synthesised from; a derived
        # basis replaces it for each case it synthesises.
        source: dict[str, Any] = {"current": {}}
        finalised: dict[int, tuple] = {}
        failures: list[Exception] = []
        # Each distinct network is fitted and rated once: searches offer the
        # same network again from other tasks. ``None`` marks a failed one.
        finished: dict[str, tuple | None] = {}
        # Pooled runs offer networks from a hand-off thread and the calling
        # thread at once; one finalises at a time so each key is done once.
        finishing = threading.Lock()

        def finalise(outcome):
            key = _finalise_key(outcome.network)
            with finishing:
                if key not in finished:
                    try:
                        finished[key] = self._finalise_network(
                            outcome.network, fit=fit, margin=margin, hens=hens, **call
                        )
                    except (
                        MissingSynthesisDependencyError,
                        MissingSynthesisSolverError,
                    ):
                        raise
                    except (ValueError, RuntimeError) as exc:
                        failures.append(exc)
                        finished[key] = None
                done = finished[key]
            if done is None:
                return None
            view, fitted = done
            network = view.selected_network
            final = outcome.model_copy(
                update={
                    "network": network,
                    "objective_value": network.total_annual_cost,
                }
            )
            finalised[id(final)] = (final, view, fitted, source["current"])
            return final

        with running(record, finalise=finalise):
            if basis == "period":
                self._design_on_basis_period(period_id, source=source, **arguments)
            elif basis == "median":
                self._design_on_median_basis(
                    basis_clusters, hens, source=source, **arguments
                )
            else:
                synthesis = self._synthesise(
                    basis=basis, period_id=period_id, **arguments
                )
                source["current"]["synthesis"] = synthesis
            ranked = record.outcomes()
            if not ranked:
                raise WorkflowContractError(
                    "no synthesised heat exchanger network could be fitted to "
                    "every period and rated."
                ) from (failures[-1] if failures else None)
            _final, view, fitted, origin = finalised[id(ranked[0])]
            view.result.provenance = self._finalised_provenance(
                view, fitted, origin, margin=margin
            )
            view.result.ranked_networks = ranked
            select_network(view.result)
            self._problem._results.design = view.result
        return view

    def _finalise_network(
        self, network, *, fit, margin, hens, options, solver, solve_tolerance
    ):
        """``network`` in its final form: fitted to every period, then rated.

        The one fit and rating path: every network a fitted or overdesigned
        design admits goes through it. Returns the final view and the fit's
        view (``None`` without a fit).
        """
        view = fitted = None
        if fit:
            view = fitted = self._fit_to_all_periods(
                network,
                options=options,
                solver=solver,
                solve_tolerance=solve_tolerance,
            )
            network = view.selected_network
        if margin is not None:
            view = self._overdesigned(
                network,
                margin,
                hens=hens,
                options=options,
                solver=solver,
                solve_tolerance=solve_tolerance,
            )
        return view, fitted

    def _finalised_provenance(self, view, fitted, origin, *, margin):
        """The final view's provenance: the synthesis settings it came from."""
        from ..targeting.provenance import canonical_json

        synthesis = origin["synthesis"].result.provenance
        if fitted is not None:
            fitted = fitted.result.provenance
            settings = (
                dict(fitted.effective_settings)
                | dict(synthesis.effective_settings)
                | origin["basis_settings"]
            )
            synthesis = fitted.model_copy(
                update={
                    "method_id": "design.heat_exchanger_network",
                    "period_ids": tuple(self._problem.period_ids),
                    "effective_settings_json": canonical_json(settings),
                }
            )
        if margin is None:
            return synthesis
        settings = dict(synthesis.effective_settings) | {
            "HENS_OVERDESIGN_PERCENTAGE": margin
        }
        return view.result.provenance.model_copy(
            update={
                "method_id": synthesis.method_id,
                "effective_settings_json": canonical_json(settings),
            }
        )

    def _synthesise(
        self,
        *,
        basis,
        period_id,
        width,
        depth,
        options,
        dt_cont_multipliers,
        derivative_thresholds,
        stages,
        pack_stages,
        solver,
        max_parallel,
        best_solutions,
        solve_tolerance,
        record=None,
    ) -> HeatExchangerNetworkDesignView:
        """Synthesise on this problem: one period, or over every period."""
        from ....analysis.heat_exchanger_networks.service import (
            heat_exchanger_network_open_hens_method_service,
        )

        if basis == "all" and not self._problem.period_results:
            self._problem.target.all_periods.all_heat_integration()
        runtime, configuration = self._arguments(
            options=options,
            period_id=period_id,
            width=width,
            depth=depth,
            dt_cont_multipliers=dt_cont_multipliers,
            derivative_thresholds=derivative_thresholds,
            stages=stages,
            pack_stages=pack_stages,
            solver=solver,
            max_parallel=max_parallel,
            best_solutions=best_solutions,
            solve_tolerance=solve_tolerance,
        )
        return self._run(
            heat_exchanger_network_open_hens_method_service,
            surface="heat_exchanger_network",
            runtime=runtime,
            configuration=configuration,
            workspace_variant=None,
            record=record,
        )

    def _overdesign_margin(self, overdesign_percentage, hens):
        from ....analysis.heat_exchanger_networks.overdesign import validate_margin

        if overdesign_percentage is None:
            overdesign_percentage = hens.overdesign_percentage
        return validate_margin(overdesign_percentage)

    def _apply_overdesign(
        self,
        network,
        margin,
        *,
        basis=None,
        hens=None,
        options=None,
        solver=None,
        solve_tolerance=None,
    ):
        """Rate ``network`` with its recovery areas enlarged by ``margin``.

        The rating solves run with the call's ``options``, ``solver`` and
        ``solve_tolerance``; ``hens`` (default: the stored configuration)
        gives the margin limit.
        """
        from ....analysis.heat_exchanger_networks.overdesign import validate_margin
        from ..targeting.provenance import canonical_json

        margin = validate_margin(margin)
        if hens is None:
            hens = self._effective_hens(options)
        view = self._overdesigned(
            network,
            margin,
            hens=hens,
            options=options,
            solver=solver,
            solve_tolerance=solve_tolerance,
        )
        settings = dict((basis or view).result.provenance.effective_settings) | {
            "HENS_OVERDESIGN_PERCENTAGE": margin
        }
        provenance = view.result.provenance.model_copy(
            update={
                "method_id": (basis or view).result.provenance.method_id,
                "effective_settings_json": canonical_json(settings),
            }
        )
        view.result.provenance = provenance
        self._problem._results.design = view.result
        return view

    def _overdesigned(self, network, margin, *, hens, options, solver, solve_tolerance):
        """The rating of ``network`` at ``margin``, its metrics on the network."""
        from ....analysis.heat_exchanger_networks.overdesign import apply_overdesign

        view, metrics = apply_overdesign(
            network,
            margin,
            maximum=float(hens.overdesign_max),
            rate=lambda installed: self.optimise_duties(
                installed,
                objective="rating",
                options=options,
                solver=solver,
                solve_tolerance=solve_tolerance,
            ),
        )
        view.selected_network.summary_metrics["overdesign"] = metrics
        return view

    def _design_basis(self, design_basis: str | None, period_id: str | None, hens=None):
        """The basis a multiperiod design is synthesised from; None for one period.

        ``hens`` is the call's configuration (default: the stored one).
        """
        from ....domain.configuration_fields import HENS_DESIGN_BASIS_VALUES

        if design_basis is not None and design_basis not in HENS_DESIGN_BASIS_VALUES:
            raise ValueError(
                "design_basis must be one of "
                + ", ".join(repr(value) for value in sorted(HENS_DESIGN_BASIS_VALUES))
                + f"; got {design_basis!r}."
            )
        if design_basis == "all" and period_id is not None:
            raise ValueError(
                "design_basis='all' designs over every period; drop period_id, "
                "or pass design_basis='period' to design on that period."
            )
        if len(self._problem.period_ids) <= 1:
            return None
        if hens is None:
            hens = self._problem._require_prepared_root_zone().config.hens
        basis = design_basis or (
            "period" if period_id is not None else hens.design_basis
        )
        if basis == "period" and period_id is None:
            raise ValueError("design_basis='period' needs a period_id to design on.")
        return basis

    def design_basis(self, *, design_basis="median", basis_clusters=None):
        """Preview the median design basis without synthesising a network.

        Returns the clusters of operating periods (members, weights, weight
        shares and medoids) and the synthetic stream table of the heaviest
        cluster: each stream's period-weighted medians, and whether it runs
        in enough of the cluster's weight to be in.
        """
        from ....contracts.synthesis.basis import (
            HenBasisCluster,
            HenBasisStream,
            HenDesignBasis,
        )

        if design_basis != "median":
            raise ValueError(
                f"only the 'median' design basis has a preview; got {design_basis!r}."
            )
        basis, clusters = self._median_basis(basis_clusters)
        total = sum(cluster.weight for cluster in basis.clusters)
        return HenDesignBasis(
            design_basis="median",
            basis_clusters=clusters,
            clusters=tuple(
                HenBasisCluster(
                    period_ids=tuple(basis.period_ids[i] for i in cluster.members),
                    weights=cluster.weights,
                    weight_share=cluster.weight / total,
                    medoid=basis.period_ids[cluster.medoid],
                )
                for cluster in basis.clusters
            ),
            basis_cluster=basis.basis_cluster,
            streams=tuple(
                HenBasisStream(
                    name=stream.name,
                    zone=stream.zone,
                    included=stream.included,
                    active_share=stream.active_share,
                    quantities=stream.quantities,
                    units=stream.units,
                )
                for stream in basis.streams[basis.basis_cluster]
            ),
        )

    def _median_basis(self, basis_clusters, hens=None):
        import warnings

        from ....analysis.heat_exchanger_networks.basis.synthetic_case import (
            median_basis,
        )

        if hens is None:
            hens = self._problem._require_prepared_root_zone().config.hens
        clusters = int(
            hens.basis_clusters if basis_clusters is None else basis_clusters
        )
        basis = median_basis(
            self._problem.to_problem_json(),
            tuple(self._problem.period_ids),
            tuple(self._problem.master_zone.weights),
            clusters=clusters,
            active_share=float(hens.basis_active_share),
        )
        lone = [
            basis.period_ids[cluster.members[0]]
            for cluster in basis.clusters
            if len(cluster.members) == 1
        ]
        if clusters > 1 and lone:
            warnings.warn(
                "median design basis clusters with one period: "
                + ", ".join(lone)
                + "; their basis is that period alone.",
                RuntimeWarning,
                stacklevel=3,
            )
        return basis, clusters

    def _design_on_median_basis(self, basis_clusters, hens, *, source, **arguments):
        """Synthesise on each candidate cluster's median case.

        Every case's fitted networks join one record, so the cheapest fitted
        network over all of them is selected.
        """
        basis, clusters = self._median_basis(basis_clusters, hens)
        candidates = (
            range(len(basis.clusters))
            if hens.basis_candidates == "all"
            else (basis.basis_cluster,)
        )
        for n in candidates:
            cluster = basis.clusters[n]
            case = type(self._problem)(
                basis.sources[n], project_name=self._problem.project_name
            )
            self._design_on_derived_basis(
                case,
                {
                    "HENS_DESIGN_BASIS": "median",
                    "HENS_BASIS_CLUSTERS": clusters,
                    "basis_period_ids": [basis.period_ids[i] for i in cluster.members],
                    "basis_medoid": basis.period_ids[cluster.medoid],
                },
                source=source,
                **arguments,
            )

    def _design_on_basis_period(self, period_id: str, *, source, **arguments):
        """Synthesise on ``period_id`` alone."""
        from ...case_derivation import single_period_case

        self._design_on_derived_basis(
            single_period_case(self._problem, period_id),
            {"HENS_DESIGN_BASIS": "period", "basis_period_id": str(period_id)},
            source=source,
            **arguments,
        )

    def _design_on_derived_basis(self, case, basis_settings, *, source, **arguments):
        """Synthesise on a derived single-period ``case``.

        Each network it admits is fitted to every period (and rated) on its
        way into the run's record; ``source`` tells the fit which case and
        settings the network came from, for the provenance.
        """
        origin = {"basis_settings": basis_settings}
        source["current"] = origin
        # The margin is applied once, to the fitted network, never to the
        # basis design.
        basis_options = dict(arguments["options"] or {}) | {
            "HENS_OVERDESIGN_PERCENTAGE": None
        }
        origin["synthesis"] = case.design.heat_exchanger_network(
            **(arguments | {"options": basis_options})
        )

    def pinch_design(self, *, options=None, period_id=None, **kwargs):
        """Run the pinch design method only.

        The heuristic (Linnhoff rules) method makes up to
        ``HENS_HEURISTIC_PDM_NETWORKS`` networks per dt_cont multiplier. The
        MILP method adds its networks from search depth 2 or with
        ``HENS_PDM_MILP``, and runs in the heuristic's place at any multiplier
        where the heuristic realises no network.
        """
        from ....analysis.heat_exchanger_networks.service import (
            heat_exchanger_network_pinch_design_method_service,
        )

        runtime, configuration = self._arguments(
            options=options, period_id=period_id, **kwargs
        )
        return self._run(
            heat_exchanger_network_pinch_design_method_service,
            surface="pinch_design",
            runtime=runtime,
            configuration=configuration,
            workspace_variant=None,
        )

    def thermal_derivative(
        self,
        initial_networks=None,
        *,
        options=None,
        period_id=None,
        **kwargs,
    ):
        """Run seeded thermal derivative synthesis only."""
        from ....analysis.heat_exchanger_networks.service import (
            heat_exchanger_network_thermal_derivative_method_service,
        )

        runtime, configuration = self._arguments(
            options=options, period_id=period_id, **kwargs
        )
        return self._run(
            heat_exchanger_network_thermal_derivative_method_service,
            surface="thermal_derivative",
            runtime=runtime,
            configuration=configuration,
            initial_networks=initial_networks,
            workspace_variant=None,
        )

    def network_evolution(
        self,
        initial_networks=None,
        *,
        options=None,
        period_id=None,
        **kwargs,
    ):
        """Run seeded network evolution synthesis only."""
        from ....analysis.heat_exchanger_networks.service import (
            heat_exchanger_network_evolution_method_service,
        )

        runtime, configuration = self._arguments(
            options=options, period_id=period_id, **kwargs
        )
        return self._run(
            heat_exchanger_network_evolution_method_service,
            surface="network_evolution",
            runtime=runtime,
            configuration=configuration,
            initial_networks=initial_networks,
            workspace_variant=None,
        )

    _STRUCTURE_KEYS = frozenset(
        {"recovery", "heaters", "coolers", "hot_utility", "cold_utility", "stage_count"}
    )

    def _network_from_structure(
        self, structure: Mapping[str, Any]
    ) -> "HeatExchangerNetwork":
        from ....domain.heat_exchanger_network import HeatExchangerNetwork

        unknown = sorted(
            str(key) for key in structure if key not in self._STRUCTURE_KEYS
        )
        if unknown:
            raise ValueError(
                "network structure accepts recovery, heaters, coolers, "
                "hot_utility, cold_utility and stage_count; got "
                + ", ".join(unknown)
                + "."
            )
        if "recovery" not in structure:
            raise ValueError("network structure needs a 'recovery' list.")
        hot_utility = structure.get("hot_utility")
        cold_utility = structure.get("cold_utility")
        if hot_utility is None and _names_no_utility(structure.get("heaters", ())):
            hot_utility = self._only_utility("Hot")
        if cold_utility is None and _names_no_utility(structure.get("coolers", ())):
            cold_utility = self._only_utility("Cold")
        recovery = [_recovery_entry(entry) for entry in structure["recovery"]]
        heaters = [
            _utility_entry(entry, hot_utility, "heaters")
            for entry in structure.get("heaters", ())
        ]
        coolers = [
            _utility_entry(entry, cold_utility, "coolers")
            for entry in structure.get("coolers", ())
        ]
        network = HeatExchangerNetwork.from_structure(
            recovery=tuple(placement for placement, _ in recovery),
            heaters=tuple(placement for placement, _ in heaters),
            coolers=tuple(placement for placement, _ in coolers),
            hot_utility=hot_utility,
            cold_utility=cold_utility,
            stage_count=structure.get("stage_count"),
        )
        # from_structure keeps the order: recovery, heaters, then coolers.
        extras = [extra for _, extra in recovery + heaters + coolers]
        return network.model_copy(
            update={
                "exchangers": tuple(
                    exchanger.model_copy(update=extra) if extra else exchanger
                    for exchanger, extra in zip(network.exchangers, extras, strict=True)
                )
            }
        )

    def retrofit(
        self,
        existing: "HeatExchangerNetwork | Mapping[str, Any]",
        *,
        max_modifications: int = 3,
        beam_width: int = 3,
        new_match_candidates: int = 4,
        area_tolerance: float = 0.02,
        max_added_area_fraction: float | None = None,
        min_approach_temperature: float | None = None,
        new_unit_cost: float | None = None,
        area_addition_cost: float = 0.0,
        split_cost: float = 0.0,
        relocation_cost: float | None = None,
        relocation_candidates: int = 8,
        new_utility_units: bool = False,
        strategy: str = "structure",
        ceiling_candidates: int = 12,
        ceiling_relocations: int = 400,
        enhancement_factors: Mapping[str, float] | None = None,
        enhancement_cost: float = 0.0,
        enhancement_area_cost: float = 0.0,
        enhancement_pressure_drops: Mapping[str, tuple[str, float]] | None = None,
        pressure_drop_allowances: Mapping[str, float] | None = None,
        repiping_cost: float = 0.0,
        heat_pump_cycle: str | None = None,
        heat_pump_cost: float = 0.0,
        options: Mapping[str, Any] | None = None,
        solver=None,
        solve_tolerance=None,
    ) -> "HeatExchangerNetworkRetrofitResult":
        """Find cost-effective modifications of an existing network.

        ``existing`` is the network as built: a ``HeatExchangerNetwork`` or a
        mapping as for ``optimise_duties``, in which entries may be mappings
        carrying the installed ``area`` (m2) and an ``id``::

            {
                "recovery": [
                    {"hot": "H1", "cold": "C1", "stage": 1, "area": 120.0,
                     "id": "E1"},
                ],
                "heaters": [{"stream": "C1", "area": 30.0}],
                "coolers": [{"stream": "H1", "utility": "CW", "area": 60.0}],
            }

        Every exchanger needs its installed area. The existing network is
        rated at its areas (widened by ``area_tolerance``), then changed one
        structural step at a time, up to ``max_modifications``: a new match
        (screened by the utility it could cut where it sits) or, optionally,
        a relocated exchanger, with the added area each structure needs
        sized in the same solve. Each step keeps the ``beam_width`` designs
        with the best net annual benefit.

        Added area costs ``COSTING_HX_AREA_COEFF * area **
        COSTING_HX_AREA_EXP``; a new exchanger adds ``new_unit_cost`` (default
        ``COSTING_HX_UNIT_COST``) and ``split_cost`` per stream it splits, and
        added area on an existing exchanger adds ``area_addition_cost``, all
        in $ and annualised with the capital recovery factor.
        ``max_added_area_fraction`` caps the added area on an existing
        exchanger at that fraction of its installed area (None: no cap).
        With ``relocation_cost`` ($ per move, plus ``split_cost`` per new
        split) a step may also move an existing recovery exchanger, with its
        installed area, to a screened match that shares one of its streams
        (None, the default: no relocations). The default search ranks up to
        ``ceiling_relocations`` such moves per design (default 400, the best
        scored first) by their ceiling; the stepwise search tries the
        ``relocation_candidates`` with the most heat potential at the match
        less the unit's present duty. With ``new_utility_units`` each new match and
        relocation is also tried together with a new end heater or cooler on
        a stream that has none, which can free the units a move needs; the
        pair is one modification, costing one more ``new_unit_cost`` and the
        utility exchanger's area. ``enhancement_factors`` (existing exchanger
        id to the factor its heat transfer coefficient can be raised by, with
        tube inserts for example; None, the default: no enhancement) lets a
        step enhance an exchanger, at ``enhancement_cost`` ($) plus
        ``enhancement_area_cost`` ($ per m2 of its installed area).
        ``enhancement_pressure_drops`` (exchanger id to the stream its insert
        is on and the pressure drop, kPa, it adds) and
        ``pressure_drop_allowances`` (stream to the extra pressure drop, kPa,
        it can take) keep the inserts of every design within each stream's
        allowance. ``repiping_cost`` ($ per stream) adds the cost of each
        stream whose piping a new match, relocation or new end heater or
        cooler changes: both streams of a new match, the streams a relocated
        exchanger leaves and joins (not one it stays on in the same stage),
        and the stream of a new heater or cooler. A relocated exchanger may
        also take added area, as an area addition. With ``heat_pump_cycle``
        (None, the default: no heat pumps) a step may add a heat pump: an
        evaporator at the end of a hot stream, evaporating
        ``min_approach_temperature`` below its target, and a condenser at the
        end of a cold stream, condensing as far above its target, rated by
        the HPR service. "carnot" uses its Carnot cycle
        (``HPR_ETA_II_CARNOT``), pricing the compressor and package at
        ``heat_pump_cost`` ($ per kW of condenser capacity) and the two
        exchangers' area by the area law. "vapour_compression" solves a
        single stage on the best of ``HPR_REFRIGERANTS`` within the HPR cycle
        limits (the other ``HPR_*`` cycle options apply) and prices the
        machine with the HPR service's installed capital. The work costs
        ``COSTING_HPR_ELE_PRICE``; the heat pump's duties are not counted as
        utility.

        ``strategy`` "structure" (the default) makes one structural change
        per step: ``ceiling_candidates`` screened changes are ranked
        by the utility cost their structure reaches with unlimited area (its
        network pinch), the ``new_match_candidates`` best are solved with
        every existing exchanger free to grow, and an exchanger keeps added
        area of at least 1 % of its installed area, as an area addition, so
        the frontier counts structural changes. "stepwise" instead makes one
        modification per step, an area addition being one. With
        ``HENS_MAX_PARALLEL`` (in ``options``) above 1 each step's solves
        run on that many threads; the result is the same.

        Returns a ``HeatExchangerNetworkRetrofitResult``: ``frontier`` holds
        the best design after each number of steps (0 is the existing
        network; a step is a structural change with the area additions it
        keeps, or with "stepwise" one modification) with its savings,
        capital, payback and modifications, and ``selected`` the one with
        the best net annual benefit. The problem's cached design is not
        changed. Requires the optional synthesis
        dependencies and the ``HENS_SOLVER_EVM`` solver (Ipopt by default).
        """

        from ....analysis.heat_exchanger_networks.duty_optimisation import (
            DEFAULT_FEASIBILITY_APPROACH,
        )
        from ....analysis.heat_exchanger_networks.retrofit.search import (
            RetrofitSettings,
            run_retrofit_search,
        )
        from ..targeting.execution import walk_zone_tree
        from ..targeting.state import commit_problem, snapshot_problem

        if isinstance(existing, Mapping):
            existing = self._network_from_structure(existing)
        _runtime, configuration = self._arguments(
            options=options, period_id=None, solve_tolerance=solve_tolerance
        )
        if solver is not None:
            configuration["HENS_SOLVER_EVM"] = solver
        isolated = snapshot_problem(self._problem)
        root = isolated._build_execution_master_zone()
        stored_configs = {zone.address: zone.config for zone in walk_zone_tree(root)}
        with temporary_zone_configuration(root, configuration):
            unit_cost = (
                isolated.master_zone.config.costing.hx_unit_cost
                if new_unit_cost is None
                else new_unit_cost
            )
            result = run_retrofit_search(
                isolated,
                existing,
                RetrofitSettings(
                    max_modifications=max_modifications,
                    beam_width=beam_width,
                    new_match_candidates=new_match_candidates,
                    area_tolerance=area_tolerance,
                    max_added_area_fraction=max_added_area_fraction,
                    approach=(
                        DEFAULT_FEASIBILITY_APPROACH
                        if min_approach_temperature is None
                        else min_approach_temperature
                    ),
                    new_unit_cost=float(unit_cost),
                    area_addition_cost=float(area_addition_cost),
                    split_cost=float(split_cost),
                    relocation_cost=(
                        None if relocation_cost is None else float(relocation_cost)
                    ),
                    relocation_candidates=relocation_candidates,
                    new_utility_units=new_utility_units,
                    strategy=strategy,
                    ceiling_candidates=ceiling_candidates,
                    ceiling_relocations=ceiling_relocations,
                    enhancement_factors=(
                        None
                        if enhancement_factors is None
                        else {k: float(v) for k, v in enhancement_factors.items()}
                    ),
                    enhancement_cost=float(enhancement_cost),
                    enhancement_area_cost=float(enhancement_area_cost),
                    enhancement_pressure_drops=(
                        None
                        if enhancement_pressure_drops is None
                        else {
                            k: (str(stream), float(drop))
                            for k, (stream, drop) in enhancement_pressure_drops.items()
                        }
                    ),
                    pressure_drop_allowances=(
                        None
                        if pressure_drop_allowances is None
                        else {k: float(v) for k, v in pressure_drop_allowances.items()}
                    ),
                    repiping_cost=float(repiping_cost),
                    heat_pump_cycle=heat_pump_cycle,
                    heat_pump_cost=float(heat_pump_cost),
                ),
            )
        for zone in walk_zone_tree(isolated._master_zone):
            zone.config = stored_configs[zone.address]
        commit_problem(self._problem, isolated)
        return result

    def retrofit_heat_pumps(
        self,
        existing: "HeatExchangerNetwork",
        *,
        cycle: str = "carnot",
        topology: str | None = None,
        maximum_iterations: int | None = None,
        maximum_evaluations: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> "RetrofitHeatPumpTargets":
        """The ideal and the as-piped heat pump target of ``existing``.

        Both come from the HPR service with the ``cycle`` (``"carnot"`` or
        ``"vapour_compression"``) and ``topology`` (``"cascade"`` or
        ``"parallel"``, the cycle's own by default) of
        ``target.heat_pump_and_refrigeration``, and with any ``HPR_*``
        settings in ``options`` (``HPR_LOAD_FRACTION``, ``HPR_N_COND``,
        ``HPR_REFRIGERANTS`` as a list of fluids, and so on).
        ``maximum_iterations`` and ``maximum_evaluations`` bound each
        target's search. The ideal target serves the process's heating
        target from its cooling target, as a new design could; the as-piped
        target serves only ``existing``'s heaters from only its coolers, at
        their solved temperatures and duties (first period), so it needs no
        process-side repiping.

        Returns a ``RetrofitHeatPumpTargets``: each target's heat, work,
        levels and operating cost before and after, priced at the first hot
        and cold utility and ``COSTING_HPR_ELE_PRICE``, and the gap between
        them split into the network's (``network_gap``) and the heat
        pump's (``heat_pump_gap``). The problem's cached design is not
        changed.
        """

        from ....analysis.heat_exchanger_networks.context import (
            ensure_target_results,
        )
        from ....analysis.heat_exchanger_networks.retrofit.heat_pump_targets import (
            retrofit_heat_pumps,
        )
        from ....analysis.heat_pumps.cycles.registry import (
            cycle_spec,
            family_value,
        )
        from ..targeting.execution import walk_zone_tree
        from ..targeting.state import commit_problem, snapshot_problem

        spec = cycle_spec(cycle)
        _runtime, configuration = self._arguments(options=options, period_id=None)
        configuration["HPR_TYPE"] = family_value(
            spec.topologies[topology or spec.default_topology]
        )
        args = {
            key: value
            for key, value in (
                ("maximum_iterations", maximum_iterations),
                ("maximum_evaluations", maximum_evaluations),
            )
            if value is not None
        }
        isolated = snapshot_problem(self._problem)
        root = isolated._build_execution_master_zone()
        stored_configs = {zone.address: zone.config for zone in walk_zone_tree(root)}
        # The first period's direct target, whose period the as-piped
        # profiles and prices take, not whichever period was targeted last.
        first_period = min(isolated.period_ids.items(), key=lambda item: item[1])[0]
        with temporary_zone_configuration(root, configuration):
            ensure_target_results(isolated, {"period_id": first_period}, first_period)
            result = retrofit_heat_pumps(isolated, existing, args=args)
        for zone in walk_zone_tree(isolated._master_zone):
            zone.config = stored_configs[zone.address]
        commit_problem(self._problem, isolated)
        return result

    def _only_utility(self, kind: str) -> str:
        data = self._problem._validated_data
        names = [
            utility.name
            for utility in getattr(data, "utilities", None) or ()
            if getattr(utility.type, "value", utility.type) == kind
        ]
        if len(names) != 1:
            raise UtilityChoiceError(kind.lower(), names)
        return names[0]

    def optimise_duties(
        self,
        network: "HeatExchangerNetwork | Mapping[str, Any]",
        *,
        objective: str = "utility",
        min_approach_temperature: float | None = None,
        exchanger_approach_temperatures: Mapping[str, float] | None = None,
        max_hot_utility: float | None = None,
        max_cold_utility: float | None = None,
        options: Mapping[str, Any] | None = None,
        period_id: str | None = None,
        solver=None,
        solve_tolerance=None,
    ) -> HeatExchangerNetworkDesignView:
        """Optimise exchanger duties on a fixed, user-defined network structure.

        ``network`` lists every exchanger: recovery matches with a stage, plus
        heaters and coolers. Pass a ``HeatExchangerNetwork`` (for example a
        previous design's ``selected_network``) or a mapping::

            {
                "recovery": [("H1", "C1", 1), ("H2", "C1", 1), ("H1", "C2", 2)],
                "heaters": ["C1", ("C2", "LP steam", 2)],
                "coolers": ["H1", ("H2", "Cooling water"), ("H2", "Chilled water")],
            }

        Recovery entries are ``(hot_stream, cold_stream, stage)`` with
        one-based stages. A heater (cold stream) or cooler (hot stream) entry is
        a stream name, ``(stream, utility)`` or ``(stream, utility, stage)``. A
        stream may have several utility exchangers, one per utility and
        position. Without a stage the exchanger sits at the stream end; with a
        stage it sits just after the stream leaves that stage (a heater with
        stage 2 heats the cold stream between stages 2 and 1). Exchangers at
        the same position run in series, heaters from the coldest utility up,
        coolers from the warmest down. Bare stream names use ``hot_utility`` /
        ``cold_utility`` keys, defaulting to the problem's only hot and cold
        utility; ``stage_count`` is optional.

        Duties and stream split fractions are optimised for one of three
        objectives; matches stay fixed:

        * ``"utility"``: minimise total utility use. Each exchanger keeps at
          least ``min_approach_temperature`` (or its entry in
          ``exchanger_approach_temperatures``; stream temperature
          contributions apply when neither is given) at both ends.
        * ``"area"``: minimise total heat-transfer area with utility capped by
          ``max_hot_utility`` and/or ``max_cold_utility`` (kW, every period).
          Each exchanger has one area shared by all operating periods: the
          largest area any period needs. Periods needing less run with a
          bypass, so every period's duties are achievable with that area.
        * ``"cost"``: minimise total annual cost: exchanger capital on the
          common areas (``COSTING_HX_*`` settings) plus utility cost. Only a
          positive approach is required at both ends of each exchanger,
          ``min_approach_temperature`` defaulting to 1 K.
        * ``"rating"``: minimise utility cost with every recovery exchanger's
          area fixed at the network's ``area`` (each needs one). Every
          period's duty must fit that area; a period needing less runs with
          a bypass. The approach defaults to 1 K, as for ``"cost"``.

        An exchanger left at zero duty in every period is removed, with its
        approach constraint, and the problem is solved again until every
        remaining exchanger carries duty. The result names removed exchangers
        in ``selected_network.summary_metrics["removed_exchangers"]``.

        Requires the optional synthesis dependencies (Pyomo and the configured
        ``HENS_SOLVER_EVM`` solver).
        """
        from ....analysis.heat_exchanger_networks.duty_optimisation import (
            heat_exchanger_network_duty_optimisation_service,
        )

        if isinstance(network, Mapping):
            network = self._network_from_structure(network)
        runtime, configuration = self._arguments(
            options=options,
            period_id=period_id,
            solve_tolerance=solve_tolerance,
        )
        if solver is not None:
            configuration["HENS_SOLVER_EVM"] = solver
        return self._run(
            heat_exchanger_network_duty_optimisation_service,
            surface="optimise_duties",
            runtime=runtime,
            configuration=configuration,
            network=network,
            objective=objective,
            min_approach_temperature=min_approach_temperature,
            exchanger_approach_temperatures=exchanger_approach_temperatures,
            max_hot_utility=max_hot_utility,
            max_cold_utility=max_cold_utility,
            workspace_variant=None,
        )

    def _fit_to_all_periods(
        self,
        network: "HeatExchangerNetwork",
        *,
        options=None,
        solver=None,
        solve_tolerance=None,
    ):
        """Fit a network designed on one basis to every operating period.

        Each stream without a utility at its end every period can use gets
        the cheapest one that can serve it there; the fixed-structure cost
        fit, with the call's ``options``, ``solver`` and ``solve_tolerance``,
        then gives every exchanger one area over all periods, with bypass
        where a period needs less, and removes the exchangers no period uses.
        """
        from ....analysis.heat_exchanger_networks.duty_optimisation import (
            DEFAULT_FEASIBILITY_APPROACH,
        )
        from ....analysis.heat_exchanger_networks.period_fit import (
            with_stream_end_utilities,
        )

        padded = with_stream_end_utilities(
            self._problem, network, approach=DEFAULT_FEASIBILITY_APPROACH
        )
        return self.optimise_duties(
            padded,
            objective="cost",
            options=options,
            solver=solver,
            solve_tolerance=solve_tolerance,
        )


class _DesignAccessorDescriptor:
    """Non-data descriptor exposing design workflows on problem instances."""

    def __get__(self, obj: Optional["PinchProblem"], owner=None):
        if obj is None:
            return self
        return _DesignAccessor(obj)


__all__ = ["HeatExchangerNetworkDesignView"]
