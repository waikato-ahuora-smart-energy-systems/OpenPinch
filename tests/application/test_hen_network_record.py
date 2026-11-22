"""Networks enter the record in their final form on fitted and rated designs."""

from __future__ import annotations

import json

import pytest

from OpenPinch.analysis.heat_exchanger_networks.results.record import (
    HeatExchangerNetworkRecord,
)
from tests.application.test_hen_design_basis import _available, _periods_problem

_SOLVERS = _available("highs") and _available("ipopt-pyomo")


def _assert_result_is_the_record(view, record):
    assert record.status == "complete"
    assert view.result.ranked_networks == record.outcomes()
    assert view.selected_network == record.outcomes()[0].network
    costs = [entry["total_annual_cost"] for entry in record.snapshot()["entries"]]
    assert costs == sorted(costs)


@pytest.mark.solver
@pytest.mark.skipif(
    not _SOLVERS,
    reason="needs HiGHS and Ipopt",
)
def test_a_median_basis_record_holds_every_network_fitted_to_every_period():
    problem = _periods_problem([1.0, 1.2, 1.5], [0.2, 0.5, 0.3])
    record = HeatExchangerNetworkRecord()

    view = problem.design.heat_exchanger_network(width=0, depth=1, record=record)

    _assert_result_is_the_record(view, record)
    assert len(record.outcomes()) > 1
    for outcome in record.outcomes():
        assert outcome.network.period_ids == ("p0", "p1", "p2")
    assert view.result.provenance.effective_settings["HENS_DESIGN_BASIS"] == "median"
    assert problem.results.design == view.result


@pytest.mark.solver
@pytest.mark.skipif(
    not _SOLVERS,
    reason="needs HiGHS and Ipopt",
)
def test_an_overdesign_record_holds_every_network_rated_at_the_margin():
    problem = _periods_problem([1.0, 1.2], [0.5, 0.5])
    record = HeatExchangerNetworkRecord()

    view = problem.design.heat_exchanger_network(
        width=0,
        depth=0,
        design_basis="all",
        overdesign_percentage=10.0,
        record=record,
    )

    _assert_result_is_the_record(view, record)
    for outcome in record.outcomes():
        assert outcome.network.overdesign.percentage == 10.0
    settings = view.result.provenance.effective_settings
    assert settings["HENS_OVERDESIGN_PERCENTAGE"] == 10.0


def _finalised_networks(monkeypatch):
    """The networks each fit-and-rate call receives, spied on."""
    from OpenPinch.application._problem.accessors.design import _DesignAccessor

    received = []
    finalise = _DesignAccessor._finalise_network

    def spy(self, network, **arguments):
        received.append(network)
        return finalise(self, network, **arguments)

    monkeypatch.setattr(_DesignAccessor, "_finalise_network", spy)
    return received


def _rounded(value):
    if isinstance(value, float):
        return float(f"{value:.9g}")
    if isinstance(value, dict):
        return {key: _rounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rounded(item) for item in value]
    return value


def _rating_input(network):
    """What a fit or rating reads: the network without its task labels.

    Values are rounded to 9 significant digits: solves of one network in
    different tasks agree to about 1e-15 relative, not bit for bit.
    """
    data = network.model_dump(mode="json")
    data.pop("task_id")
    data["summary_metrics"].pop("dt_cont_multiplier", None)
    return json.dumps(_rounded(data), sort_keys=True)


@pytest.mark.solver
@pytest.mark.skipif(
    not _SOLVERS,
    reason="needs HiGHS and Ipopt",
)
def test_a_network_offered_again_is_rated_once(monkeypatch):
    from OpenPinch import PinchProblem
    from tests.support.paths import REPOSITORY_ROOT

    # Ten dt_cont multipliers make pinch design networks that evolve into the
    # same network, so the search offers it many times. That needs evolution
    # to be free of the multiplier, so the share is 0 (every match may close
    # to 0.1 K); at the default quarter the floor follows the multiplier and
    # the same network, designed to different approaches, is rated for each.
    path = (
        REPOSITORY_ROOT
        / "tests/fixtures/openhens/Four-stream-Yee-and-Grossmann-1990-1.json"
    )
    payload = json.loads(path.read_text())
    payload["options"] = {
        key: value
        for key, value in payload["options"].items()
        if key != "HENS_OUTPUT_FOLDER"
    } | {"HENS_OUTPUT_FORMATS": [], "HENS_MIN_EXCHANGER_APPROACH_SHARE": 0.0}
    problem = PinchProblem(source=payload, project_name="Yee")
    received = _finalised_networks(monkeypatch)
    record = HeatExchangerNetworkRecord()

    view = problem.design.heat_exchanger_network(
        overdesign_percentage=10.0, record=record
    )

    _assert_result_is_the_record(view, record)
    inputs = [_rating_input(network) for network in received]
    assert len(inputs) == len(set(inputs))


def test_networks_on_different_slices_of_a_utility_do_not_share_a_fit():
    """A heater's id names the slice of its utility it draws on, and the fit
    holds that slice's temperatures, so two networks alike but for the slice
    are fitted separately."""
    from OpenPinch.application._problem.accessors.design import _fit_key
    from OpenPinch.domain.heat_exchanger_network import HeatExchangerNetwork

    def network(heater_id):
        structure = HeatExchangerNetwork.from_structure(
            recovery=[("H1", "C1", 1)],
            heaters=["C1"],
            coolers=["H1"],
            hot_utility="HU",
            cold_utility="CU",
        )
        return structure.model_copy(
            update={
                "exchangers": tuple(
                    e.model_copy(update={"exchanger_id": heater_id})
                    if e.kind.value == "hot_utility"
                    else e
                    for e in structure.exchangers
                )
            }
        )

    first, second = network("hot-utility:HU@1->C1"), network("hot-utility:HU@2->C1")

    assert _fit_key(first) != _fit_key(second)
    assert _fit_key(first) == _fit_key(network("hot-utility:HU@1->C1"))


def test_networks_holding_levels_at_different_multipliers_do_not_share_a_fit():
    """The fit may target a level-holding network's levels again at the
    network's own dt_cont multiplier, so that multiplier is part of what the
    fit depends on."""
    from OpenPinch.application._problem.accessors.design import _fit_key
    from OpenPinch.domain.heat_exchanger_network import HeatExchangerNetwork

    def holding_levels(multiplier):
        return HeatExchangerNetwork.from_structure(
            recovery=[("H1", "C1", 1)],
            heaters=["C1"],
            coolers=["H1"],
            hot_utility="HU",
            cold_utility="CU",
        ).model_copy(
            update={
                "summary_metrics": {"dt_cont_multiplier": multiplier},
                "source_metadata": {"utility_level_targets": {"hot:HU": [10.0]}},
            }
        )

    assert _fit_key(holding_levels(0.75)) != _fit_key(holding_levels(1.5))


@pytest.mark.solver
@pytest.mark.skipif(
    not _SOLVERS,
    reason="needs HiGHS and Ipopt",
)
def test_a_structure_is_fitted_once_at_each_design_approach(monkeypatch):
    """A fit to every period sets a network's duties and areas afresh, so it
    depends on what it reads of the network (each exchanger's kind, streams,
    stage and id, and the stage count), the approach the network was
    designed to and the levels it holds, not on the duties it arrives with.
    Networks that share those are fitted once (16 fits of 12 such networks
    before)."""

    def structure(network):
        return (
            sorted(
                (e.kind.value, e.source_stream, e.sink_stream, e.stage, e.exchanger_id)
                for e in network.exchangers
            ),
            network.stage_count,
        )

    problem = _periods_problem([1.0, 1.2, 1.5], [0.2, 0.5, 0.3])
    hens = problem.master_zone.config.hens
    received = _finalised_networks(monkeypatch)
    record = HeatExchangerNetworkRecord()

    view = problem.design.heat_exchanger_network(record=record)

    _assert_result_is_the_record(view, record)
    fitted = [
        (
            repr(structure(network)),
            repr(sorted(problem.design._design_approach(network, hens).items())),
            repr((network.source_metadata or {}).get("utility_level_targets")),
        )
        for network in received
    ]
    assert len(fitted) > 1
    assert len(fitted) == len(set(fitted))


@pytest.mark.solver
@pytest.mark.skipif(
    not _SOLVERS,
    reason="needs HiGHS and Ipopt",
)
def test_a_fitted_network_keeps_the_method_that_designed_it(monkeypatch):
    """The fit is a fixed-structure solve, run as a network evolution task,
    but it does not design the network: a heuristic pinch design fitted to
    every period is still a heuristic pinch design."""
    problem = _periods_problem([1.0, 1.2, 1.5], [0.2, 0.5, 0.3])
    received = _finalised_networks(monkeypatch)
    record = HeatExchangerNetworkRecord()

    problem.design.heat_exchanger_network(record=record)

    designed_by = {network.task_id: network.method for network in received}
    ranked = record.outcomes()
    methods = [designed_by[outcome.task.task_id] for outcome in ranked]
    assert "heuristic_pinch_design_method" in methods
    assert [outcome.network.method for outcome in ranked] == methods
