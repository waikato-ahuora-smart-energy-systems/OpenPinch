"""Designing a multiperiod HEN from one basis period, then fitting all periods.

A basis period is a derived single-period case: synthesis runs on it
unchanged, and the fit gives the network one area per exchanger over every
operating period, adding a heater or cooler at a stream end where a period
needs one.
"""

from __future__ import annotations

import pytest

from OpenPinch.analysis.heat_exchanger_networks.solver import backend
from OpenPinch.analysis.heat_exchanger_networks.solver.dependencies import (
    MissingSynthesisDependencyError,
    MissingSynthesisSolverError,
)
from OpenPinch.application.case_derivation import single_period_case
from tests.analysis.heat_exchanger_networks.test_hen_period_handling import (
    _two_period_problem,
)


def _available(solver: str) -> bool:
    try:
        backend.require_solver_backend(solver)
    except (MissingSynthesisDependencyError, MissingSynthesisSolverError):
        return False
    return True


def _stream_values(problem, key: str) -> dict[str, float]:
    return {
        record["name"]: record[key].get("value", None)
        for record in problem.to_problem_json()["streams"]
    }


def test_a_single_period_case_holds_that_periods_data_alone():
    problem = _two_period_problem()

    peak = single_period_case(problem, "peak")

    assert list(peak.period_ids) == ["peak"]
    assert _stream_values(peak, "heat_flow")["Raw Milk"] == pytest.approx(3500.0)
    assert _stream_values(peak, "heat_capacity_flowrate")["Raw Milk"] == (
        pytest.approx(12.5)
    )


def test_a_single_period_case_has_that_periods_targets():
    problem = _two_period_problem()
    by_period = problem.target.all_periods.direct_heat_integration()

    for period_id in ("base", "peak"):
        derived = single_period_case(problem, period_id)
        derived.target.direct_heat_integration()
        expected = by_period[period_id].targets[0]
        actual = derived.results.targets[0]
        assert actual.Qh.value == pytest.approx(expected.Qh.value, rel=1e-9)
        assert actual.Qc.value == pytest.approx(expected.Qc.value, rel=1e-9)


def test_a_single_period_case_rejects_an_unknown_period():
    with pytest.raises(ValueError, match="unknown period 'night'"):
        single_period_case(_two_period_problem(), "night")


@pytest.mark.solver
@pytest.mark.skipif(
    "not (_available('highs') and _available('ipopt-pyomo'))",
    reason="needs HiGHS and Ipopt",
)
def test_a_basis_period_design_fitted_to_all_periods_closes_every_balance():
    problem = _two_period_problem()
    basis = single_period_case(problem, "base")
    network = basis.design.heat_exchanger_network(width=0, depth=0).selected_network

    fitted = problem.design._fit_to_all_periods(network).selected_network

    heat_flows = {
        f"{record['zone'].rsplit('/', 1)[-1]}.{record['name']}": record["heat_flow"][
            "values"
        ]
        for record in problem.to_problem_json()["streams"]
    }
    for period, period_id in enumerate(("base", "peak")):
        duty = dict.fromkeys(heat_flows, 0.0)
        for exchanger in fitted.exchangers:
            state = next(s for s in exchanger.period_states if s.period_id == period_id)
            for stream in (exchanger.source_stream, exchanger.sink_stream):
                if stream in duty:
                    duty[stream] += state.duty
        for stream, values in heat_flows.items():
            assert duty[stream] == pytest.approx(values[period], rel=1e-4), (
                period_id,
                stream,
            )
    # One area per exchanger, for both periods.
    assert all(exchanger.area is not None for exchanger in fitted.exchangers)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"design_basis": "all", "period_id": "base"}, "period_id"),
        ({"design_basis": "period"}, "needs a period_id"),
        ({"design_basis": "busiest"}, "design_basis"),
    ],
)
def test_a_contradictory_or_unknown_design_basis_is_rejected(arguments, message):
    with pytest.raises(ValueError, match=message):
        _two_period_problem().design.heat_exchanger_network(**arguments)


def test_the_all_periods_basis_targets_every_period_first(monkeypatch):
    # The fake executor has no multiperiod networks, so stop at the dispatch.
    from OpenPinch.analysis.heat_exchanger_networks import service

    class Dispatched(Exception):
        pass

    def stop(problem, settings, **kwargs):
        raise Dispatched

    monkeypatch.setattr(service, "execute_open_hens_method", stop)
    problem = _two_period_problem()

    with pytest.raises(Dispatched):
        problem.design.heat_exchanger_network(design_basis="all")

    assert set(problem.period_results) == {"base", "peak"}


def test_the_multiperiod_method_is_gone():
    assert not hasattr(
        _two_period_problem().design, "multiperiod_heat_exchanger_network"
    )


def test_a_saved_multiperiod_design_reads_as_the_all_periods_basis():
    from OpenPinch.domain.analysis import AnalysisProvenance

    provenance = AnalysisProvenance(
        owner_id="owner",
        method_id="design.multiperiod_heat_exchanger_network",
        zone_address="Site",
        period_ids=("base", "peak"),
        input_fingerprint="0" * 64,
        effective_settings_json='{"HENS_SEARCH_DEPTH": 1}',
    )

    assert provenance.method_id == "design.heat_exchanger_network"
    assert provenance.effective_settings["HENS_DESIGN_BASIS"] == "all"
    assert provenance.effective_settings["HENS_SEARCH_DEPTH"] == 1


@pytest.mark.solver
@pytest.mark.skipif(
    "not (_available('highs') and _available('ipopt-pyomo'))",
    reason="needs HiGHS and Ipopt",
)
def test_a_period_id_designs_on_that_period_and_fits_every_period():
    problem = _two_period_problem()

    view = problem.design.heat_exchanger_network(width=0, depth=0, period_id="peak")

    provenance = view.result.provenance
    assert provenance.method_id == "design.heat_exchanger_network"
    assert provenance.period_ids == ("base", "peak")
    assert provenance.effective_settings["HENS_DESIGN_BASIS"] == "period"
    assert provenance.effective_settings["basis_period_id"] == "peak"
    assert problem.results.design.provenance == provenance
    periods = {
        state.period_id
        for exchanger in view.selected_network.exchangers
        for state in exchanger.period_states
    }
    assert periods == {"base", "peak"}


def _periods_problem(raw_milk_scales, weights, cip_heat_flows=None):
    """Yee and Grossmann's four streams over several periods.

    Raw Milk's heat flow and CP scale per period; CIP Water's heat flow can be
    set per period (0 for a period it does not run).
    """
    import json

    from OpenPinch.application.problem import PinchProblem
    from tests.analysis.heat_exchanger_networks.test_hen_period_handling import (
        FOUR_STREAM_JSON,
    )

    payload = json.loads(FOUR_STREAM_JSON.read_text(encoding="utf-8"))
    ids = [f"p{n}" for n in range(len(weights))]
    for record in payload["streams"]:
        for key, value in list(record.items()):
            if isinstance(value, dict) and isinstance(value.get("value"), int | float):
                base = value["value"]
                values = [base] * len(ids)
                if record["name"] == "Raw Milk" and key in (
                    "heat_flow",
                    "heat_capacity_flowrate",
                ):
                    values = [base * scale for scale in raw_milk_scales]
                if (
                    record["name"] == "CIP Water"
                    and key == "heat_flow"
                    and cip_heat_flows
                ):
                    values = list(cip_heat_flows)
                record[key] = {"unit": value["unit"], "values": values}
        if record["name"] == "CIP Water" and cip_heat_flows:
            record["heat_capacity_flowrate"] = None
    payload["options"] = {
        key: value
        for key, value in payload["options"].items()
        if not key.startswith("HENS_SOLVER")
        and key not in ("HENS_DT_CONT_MULTIPLIERS", "HENS_OUTPUT_FOLDER")
    } | {
        "PROBLEM_PERIOD_IDS": ids,
        "PROBLEM_PERIOD_WEIGHTS": list(weights),
        "HENS_OUTPUT_FORMATS": [],
        "HENS_MAX_PARALLEL": 1,
    }
    return PinchProblem(source=payload, project_name="periods")


def _stream(basis, name):
    return next(stream for stream in basis.streams if stream.name == name)


def test_a_median_basis_takes_each_streams_weighted_median():
    # Raw Milk at 1.0, 1.2 and 1.5 times, weighted 0.2, 0.5 and 0.3: the
    # weighted median is the middle period's 1.2.
    problem = _periods_problem([1.0, 1.2, 1.5], [0.2, 0.5, 0.3])

    basis = problem.design.design_basis()

    assert basis.design_basis == "median"
    (cluster,) = basis.clusters
    assert cluster.period_ids == ("p0", "p1", "p2")
    assert cluster.weight_share == pytest.approx(1.0)
    assert _stream(basis, "Raw Milk").quantities["heat_flow"] == pytest.approx(3360.0)
    assert _stream(basis, "HT Flash").quantities["t_supply"] == pytest.approx(590.0)


def test_a_one_member_cluster_is_its_period():
    problem = _periods_problem([1.0, 1.2, 1.5], [0.2, 0.5, 0.3])

    basis = problem.design.design_basis(basis_clusters=3)

    # The heaviest cluster is the heaviest period, p1, on its own.
    chosen = basis.clusters[basis.basis_cluster]
    assert chosen.period_ids == ("p1",) and chosen.medoid == "p1"
    assert _stream(basis, "Raw Milk").quantities["heat_flow"] == pytest.approx(3360.0)
    assert _stream(basis, "Raw Milk").quantities["t_target"] == pytest.approx(370.0)


def test_a_zero_weight_period_joins_no_cluster():
    problem = _periods_problem([1.0, 1.2, 9.0], [0.5, 0.5, 0.0])

    basis = problem.design.design_basis(basis_clusters=2)

    assert {p for cluster in basis.clusters for p in cluster.period_ids} == {
        "p0",
        "p1",
    }


def test_a_stream_off_in_most_of_the_cluster_is_left_out():
    # CIP Water runs only in p0, 30 % of the weight: under the 50 % default.
    problem = _periods_problem(
        [1.0, 1.0, 1.0], [0.3, 0.4, 0.3], cip_heat_flows=[1200.0, 0.0, 0.0]
    )

    basis = problem.design.design_basis()

    cip = _stream(basis, "CIP Water")
    assert not cip.included
    assert cip.active_share == pytest.approx(0.3)


def test_a_multiperiod_design_defaults_to_the_median_basis():
    problem = _two_period_problem()

    assert problem.design._design_basis(None, None) == "median"
    assert problem.design._design_basis(None, "base") == "period"


@pytest.mark.solver
@pytest.mark.skipif(
    "not (_available('highs') and _available('ipopt-pyomo'))",
    reason="needs HiGHS and Ipopt",
)
def test_a_median_basis_design_serves_every_period():
    problem = _periods_problem([1.0, 1.2, 1.5], [0.2, 0.5, 0.3])

    view = problem.design.heat_exchanger_network(width=0, depth=0)

    provenance = view.result.provenance
    assert provenance.effective_settings["HENS_DESIGN_BASIS"] == "median"
    assert provenance.effective_settings["basis_period_ids"] == ["p0", "p1", "p2"]
    assert provenance.period_ids == ("p0", "p1", "p2")
    periods = {
        state.period_id
        for exchanger in view.selected_network.exchangers
        for state in exchanger.period_states
    }
    assert periods == {"p0", "p1", "p2"}


def test_a_cluster_of_one_period_warns():
    problem = _periods_problem([1.0, 1.2, 1.5], [0.2, 0.5, 0.3])

    with pytest.warns(RuntimeWarning, match="clusters with one period"):
        problem.design.design_basis(basis_clusters=3)


@pytest.mark.solver
@pytest.mark.skipif(
    "not (_available('highs') and _available('ipopt-pyomo'))",
    reason="needs HiGHS and Ipopt",
)
def test_every_cluster_as_a_candidate_is_never_costlier_than_the_heaviest():
    def cost(candidates):
        problem = _periods_problem([1.0, 1.1, 1.6, 1.7], [0.3, 0.3, 0.2, 0.2])
        problem.update_options(
            {"HENS_BASIS_CLUSTERS": 2, "HENS_BASIS_CANDIDATES": candidates}
        )
        view = problem.design.heat_exchanger_network(width=0, depth=0)
        return view.selected_network.total_annual_cost

    assert cost("all") <= cost("dominant") * (1 + 1e-9)


def test_a_period_basis_in_the_call_options_needs_a_period_id():
    with pytest.raises(ValueError, match="needs a period_id"):
        _two_period_problem().design.heat_exchanger_network(
            options={"HENS_DESIGN_BASIS": "period"}
        )


def test_an_all_periods_basis_in_the_call_options_is_followed(monkeypatch):
    from OpenPinch.analysis.heat_exchanger_networks import service

    class Dispatched(Exception):
        pass

    def stop(problem, settings, **kwargs):
        raise Dispatched

    monkeypatch.setattr(service, "execute_open_hens_method", stop)
    problem = _two_period_problem()

    with pytest.raises(Dispatched):
        problem.design.heat_exchanger_network(options={"HENS_DESIGN_BASIS": "all"})

    assert set(problem.period_results) == {"base", "peak"}


def test_median_basis_settings_in_the_call_options_are_followed(monkeypatch):
    from OpenPinch.application._problem.accessors.design import _DesignAccessor

    recorded = {}

    class Designed(Exception):
        pass

    def record(self, case, basis_settings, **arguments):
        recorded.update(basis_settings)
        raise Designed

    monkeypatch.setattr(_DesignAccessor, "_design_on_derived_basis", record)
    problem = _periods_problem([1.0, 1.2, 1.5], [0.2, 0.5, 0.3])

    with pytest.warns(RuntimeWarning), pytest.raises(Designed):
        problem.design.heat_exchanger_network(options={"HENS_BASIS_CLUSTERS": 2})

    assert recorded["HENS_BASIS_CLUSTERS"] == 2


def test_the_all_period_fit_runs_with_the_calls_settings(monkeypatch):
    from OpenPinch.application._problem.accessors.design import _DesignAccessor
    from OpenPinch.domain.heat_exchanger_network import HeatExchangerNetwork

    seen = {}

    def capture(self, network, **kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(_DesignAccessor, "_optimise_duties", capture)
    raw, cip = "Process A.Raw Milk", "Process A.CIP Water"
    network = HeatExchangerNetwork.from_structure(
        recovery=[(raw, cip, 1)], stage_count=1
    )

    _two_period_problem().design._fit_to_all_periods(
        network,
        options={"COSTING_HX_UNIT_COST": 9000.0},
        solver="ipopt-pyomo",
        solve_tolerance=1e-4,
    )

    assert seen == {
        "objective": "cost",
        "options": {"COSTING_HX_UNIT_COST": 9000.0},
        "solver": "ipopt-pyomo",
        "solve_tolerance": 1e-4,
        "problem_approach": True,
        "dt_cont_multiplier": 1.0,
    }


@pytest.mark.solver
@pytest.mark.skipif(
    "not (_available('highs') and _available('ipopt-pyomo'))",
    reason="needs HiGHS and Ipopt",
)
def test_a_basis_designs_provenance_keeps_its_synthesis_settings():
    view = _two_period_problem().design.heat_exchanger_network(
        width=0, depth=0, period_id="peak"
    )

    settings = view.result.provenance.effective_settings
    assert settings["HENS_SEARCH_WIDTH"] == 0
    assert settings["HENS_SEARCH_DEPTH"] == 0
    assert settings["HENS_DESIGN_BASIS"] == "period"


def _minimum_recovery_approach(network) -> float:
    from OpenPinch.domain.enums import HeatExchangerKind

    return min(
        min(
            state.source_inlet_temperature - state.sink_outlet_temperature,
            state.source_outlet_temperature - state.sink_inlet_temperature,
        )
        for exchanger in network.exchangers
        if exchanger.kind is HeatExchangerKind.RECOVERY
        for state in exchanger.period_states
        if state.active and state.duty > 1e-6
    )


@pytest.mark.solver
@pytest.mark.skipif(
    "not (_available('highs') and _available('ipopt-pyomo'))",
    reason="needs HiGHS and Ipopt",
)
def test_the_fit_to_every_period_keeps_the_designs_minimum_approach():
    """The default multiperiod design (median basis, then the fit) keeps
    every recovery match at the case's minimum approach (twice
    THERMAL_DT_CONT, 10 K) in every period; it used to fit at 1 K and
    report 8.1 K."""
    network = (
        _two_period_problem()
        .design.heat_exchanger_network(width=0, depth=0)
        .selected_network
    )

    assert _minimum_recovery_approach(network) >= 10.0 - 1e-3


def test_a_fit_that_cannot_keep_the_minimum_approach_warns_and_fits_at_1_k(
    monkeypatch,
):
    from OpenPinch.application._problem.accessors.design import _DesignAccessor
    from OpenPinch.domain.heat_exchanger_network import HeatExchangerNetwork

    calls = []

    def fit(self, network, *, problem_approach=False, **kwargs):
        calls.append(problem_approach)
        if problem_approach:
            raise RuntimeError("infeasible in period peak")
        return "fitted at 1 K"

    monkeypatch.setattr(_DesignAccessor, "_optimise_duties", fit)
    network = HeatExchangerNetwork.from_structure(
        recovery=[("Process A.Raw Milk", "Process A.CIP Water", 1)], stage_count=1
    )

    with pytest.warns(
        UserWarning,
        match="approach the network was designed to.*infeasible in period peak",
    ):
        result = _two_period_problem().design._fit_to_all_periods(network)

    assert result == "fitted at 1 K"
    assert calls == [True, False]


@pytest.mark.solver
@pytest.mark.skipif(
    "not (_available('highs') and _available('ipopt-pyomo'))",
    reason="needs HiGHS and Ipopt",
)
def test_the_fit_to_every_period_holds_the_utility_levels():
    """A network that holds its utility levels in every period (designed
    over all periods with temperature placement) keeps them through the fit
    to every period: the levels are recorded and met, so LPS is not traded
    for HPS by price."""
    from OpenPinch.analysis.heat_exchanger_networks.reporting.ranking import (
        holds_utility_levels,
    )
    from tests.application.test_site_heat_exchanger_network import _site

    problem = _site(periods=True)
    options = {"HENS_UTILITY_PLACEMENT": "temperature"}
    designed = problem.design.heat_exchanger_network(
        width=0, depth=0, design_basis="all", options=options
    ).selected_network

    fitted = problem.design._fit_to_all_periods(
        designed, options=options
    ).selected_network

    held = designed.source_metadata["utility_level_targets"]
    kept = fitted.source_metadata["utility_level_targets"]
    assert kept.keys() == held.keys()
    for name, duties in held.items():
        assert kept[name] == pytest.approx(duties)
    assert holds_utility_levels(fitted)


@pytest.mark.solver
@pytest.mark.skipif(
    "not (_available('highs') and _available('ipopt-pyomo'))",
    reason="needs HiGHS and Ipopt",
)
def test_a_fit_that_cannot_hold_the_levels_says_so():
    """A design on the median period cannot meet every period's level
    targets exactly; the fit warns that it releases the levels instead of
    releasing them silently."""
    from tests.application.test_site_heat_exchanger_network import _site

    with pytest.warns(UserWarning, match="cannot hold the design's utility levels"):
        _site(periods=True).design.heat_exchanger_network(
            width=0,
            depth=0,
            options={"HENS_UTILITY_PLACEMENT": "temperature"},
        )


@pytest.mark.parametrize(
    ("method", "multiplier", "expected"),
    [
        ("PDM", 0.75, {"problem_approach": True, "dt_cont_multiplier": 0.75}),
        ("PDM", 1.0, {"problem_approach": True, "dt_cont_multiplier": 1.0}),
        ("ESM", 1.0, {"min_approach_temperature": 0.1}),
    ],
)
def test_the_fit_keeps_the_approach_the_network_was_designed_to(
    monkeypatch, method, multiplier, expected
):
    """A pinch design network keeps its contributions at its own dt_cont
    multiplier (7.5 K at 0.75x), not the reference 10 K; an evolved network
    keeps the 0.1 K floor it was designed to. The padding uses the same
    approach: the floor, or each pair's contributions at the multiplier."""
    from OpenPinch.analysis.heat_exchanger_networks import period_fit
    from OpenPinch.application._problem.accessors.design import _DesignAccessor
    from OpenPinch.domain.heat_exchanger_network import HeatExchangerNetwork

    seen, padded = [], []

    def fit(self, network, **kwargs):
        seen.append(kwargs)
        return "fitted"

    def pad(problem, network, *, approach, dt_cont_multiplier, levels=frozenset()):
        padded.append((approach, dt_cont_multiplier))
        return network

    monkeypatch.setattr(_DesignAccessor, "_optimise_duties", fit)
    monkeypatch.setattr(period_fit, "with_stream_end_utilities", pad)
    network = HeatExchangerNetwork.from_structure(
        recovery=[("Process A.Raw Milk", "Process A.CIP Water", 1)], stage_count=1
    ).model_copy(
        update={"method": method, "summary_metrics": {"dt_cont_multiplier": multiplier}}
    )

    _two_period_problem().design._fit_to_all_periods(network)

    for name, value in expected.items():
        assert seen[0][name] == pytest.approx(value)
    assert padded == [
        (
            expected.get("min_approach_temperature"),
            pytest.approx(expected.get("dt_cont_multiplier", 1.0)),
        )
    ]


def test_networks_that_differ_in_design_approach_or_levels_are_fitted_apart():
    from OpenPinch.application._problem.accessors.design import _finalise_key
    from OpenPinch.domain.heat_exchanger_network import HeatExchangerNetwork

    base = HeatExchangerNetwork.from_structure(
        recovery=[("Process A.Raw Milk", "Process A.CIP Water", 1)], stage_count=1
    )

    def keyed(multiplier, approach, targets=None):
        return _finalise_key(
            base.model_copy(
                update={
                    "summary_metrics": {"dt_cont_multiplier": multiplier},
                    "source_metadata": (
                        {} if targets is None else {"utility_level_targets": targets}
                    ),
                }
            ),
            approach,
        )

    seven = {"problem_approach": True, "dt_cont_multiplier": 0.75}
    floor = {"min_approach_temperature": 0.1}
    assert keyed(0.75, seven) != keyed(
        1.0, {"problem_approach": True, "dt_cont_multiplier": 1.0}
    )
    # An evolved network's floor does not depend on its multiplier.
    assert keyed(0.75, floor) == keyed(1.5, floor)
    assert keyed(1.0, floor) != keyed(1.0, floor, {"Hot Utility.LPS": [10.0]})
