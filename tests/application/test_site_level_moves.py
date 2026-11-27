"""Level-temperature moves on a placed site design.

A move shifts one placed level (its hot and cold utility together) by a
step, keeping the levels in order and within the site's stream
temperatures. Each process's level duties are re-targeted at the new
temperature and the site is re-evolved from the current best network; the
move is adopted only on a lower site cost. With no improving move the step
halves, down to its minimum.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from OpenPinch.analysis.heat_exchanger_networks.errors import WorkflowContractError
from OpenPinch.application import site_level_moves as moves
from OpenPinch.application.problem import PinchProblem
from OpenPinch.domain.enums import HeatExchangerKind
from tests.application.test_site_heat_exchanger_network import (
    _site,
    _solvers_available,
    _utility,
)
from tests.application.test_site_level_sweep import SHORT_PLACEMENT


def _main(name, temperature, price, *, periods=1):
    """A placed level: a hot and a cold utility of one name, 0.01 K apart."""
    hot = _utility(name, "Hot", temperature, temperature - 0.01, price)
    cold = _utility(name, "Cold", temperature - 0.01, temperature, price)
    if periods > 1:
        for utility in (hot, cold):
            for key in ("t_supply", "t_target"):
                value = utility[key].pop("value")
                utility[key]["values"] = [value] * periods
    return [hot, cold]


def _mains_site(temperatures=(180.0, 100.0), *, periods=False) -> PinchProblem:
    """The unit site (streams 50 to 220 C) with placed mains iso_1, iso_2."""
    source = _site(periods=periods).to_problem_json()
    count = 2 if periods else 1
    source["utilities"] = [
        utility
        for level, temperature in enumerate(temperatures, start=1)
        for utility in _main(f"iso_{level}", temperature, 30.0, periods=count)
    ] + [_utility("CU", "Cold", 10.0, 15.0, 5.0)]
    return PinchProblem(source)


def _temperatures(case, name):
    """Each utility of ``name`` as (type, supply, target), supply per period."""
    found = []
    for utility in case.to_problem_json()["utilities"]:
        if utility["name"] != name:
            continue
        supply, target = utility["t_supply"], utility["t_target"]
        found.append(
            (
                utility["type"],
                supply.get("values", [supply.get("value")]),
                target.get("values", [target.get("value")]),
            )
        )
    return sorted(found)


def _hot_supply(case, name) -> float:
    """The supply temperature of ``name``'s hot utility in the first period."""
    return dict((kind, supply) for kind, supply, _ in _temperatures(case, name))["Hot"][
        0
    ]


def test_the_placed_levels_are_the_movable_ones():
    assert moves.movable_levels(_mains_site()) == ("iso_1", "iso_2")
    assert moves.movable_levels(_site()) == ()


def test_a_move_shifts_the_levels_hot_and_cold_utility_together():
    shifted = moves.shifted_case(_mains_site(), "iso_2", 5.0)

    assert _temperatures(shifted, "iso_2") == [
        ("Cold", [pytest.approx(104.99)], [pytest.approx(105.0)]),
        ("Hot", [pytest.approx(105.0)], [pytest.approx(104.99)]),
    ]
    assert _temperatures(shifted, "iso_1") == _temperatures(_mains_site(), "iso_1")
    assert _temperatures(shifted, "CU") == _temperatures(_mains_site(), "CU")


def test_a_move_shifts_every_period():
    shifted = moves.shifted_case(_mains_site(periods=True), "iso_1", -5.0)

    supplies = dict((kind, s) for kind, s, _ in _temperatures(shifted, "iso_1"))
    assert supplies["Hot"] == [pytest.approx(175.0)] * 2


@pytest.mark.parametrize(
    ("level", "shift"),
    [
        ("iso_2", 80.0),  # passes iso_1 at 180 C
        ("iso_1", -80.0),  # onto iso_2 at 100 C
        ("iso_1", 45.0),  # above the hottest stream, 220 C
        ("iso_2", -55.0),  # below the coldest stream, 50 C
    ],
)
def test_a_move_that_breaks_the_level_order_or_leaves_the_streams_is_refused(
    level, shift
):
    assert moves.shifted_case(_mains_site(), level, shift) is None


def test_each_processs_level_duties_are_retargeted_at_the_levels_temperatures():
    """As the level-held pinch design records them: Heating takes 40 kW
    from LPS and Cooling raises 40 kW of LP steam."""
    duties = moves.level_duties(_site(), 1.0)

    assert duties["Heating"]["Hot Utility.LPS"] == pytest.approx([40.0])
    assert duties["Cooling"]["Cold Utility.LPG"] == pytest.approx([40.0])
    assert set(duties) == {"Heating", "Cooling"}


def test_a_one_sided_processs_level_duties_are_retargeted_too():
    """A process of cold streams alone takes all its heat from utilities."""
    source = _site().to_problem_json()
    source["streams"] = [
        stream for stream in source["streams"] if stream["name"] != "H2"
    ] + [
        {**stream, "zone": "Site/Boiling", "name": "C4"}
        for stream in source["streams"]
        if stream["name"] == "C2"
    ]
    source["zone_tree"]["children"].append({"name": "Boiling", "type": "Process Zone"})

    duties = moves.level_duties(PinchProblem(source), 1.0)

    boiling = duties["Boiling"]
    assert sum(d[0] for name, d in boiling.items() if name.startswith("Hot")) == (
        pytest.approx(70.0)
    )
    assert all(d == [0.0] for name, d in boiling.items() if name.startswith("Cold"))


def test_a_move_seed_keeps_the_networks_staged_heaters_and_coolers():
    """As a site seed does: the staged coolers give the task a stage for each
    of H1's levels between its matches. Without them a stream needing two
    staged units between two matches cannot be designed, and every move
    fails, the unshifted baseline included (review finding 14)."""
    from tests.analysis.heat_exchanger_networks.test_site_design import (
        IP,
        LP,
        _raising_network,
    )

    seed = moves.move_seed(_site(), _raising_network())

    assert {e.exchanger_id for e in seed.exchangers} == {
        "recovery:A.H1->A.C1:S1",
        "recovery:A.H1->A.C2:S5",
        f"cold-utility:A.H1->{IP}:S3",
        f"cold-utility:A.H1->{LP}:S4",
    }


def _view(cost):
    return SimpleNamespace(
        selected_network=SimpleNamespace(total_annual_cost=cost, summary_metrics={})
    )


def _cost_by_level(case, best=195.0):
    """A cost falling as iso_1 nears ``best`` C."""
    return 1000.0 + abs(_hot_supply(case, "iso_1") - best)


def test_an_improving_move_is_adopted_and_the_step_halves_when_none_improves():
    evaluated = []

    def evaluate(case, view):
        evaluated.append(_hot_supply(case, "iso_1"))
        return _view(_cost_by_level(case))

    case, view, log = moves.refine_levels(
        _mains_site(),
        _view(_cost_by_level(_mains_site())),
        evaluate=evaluate,
        step=5.0,
        step_min=1.0,
        levels=("iso_1",),
    )

    assert _hot_supply(case, "iso_1") == pytest.approx(195.0)
    assert view.selected_network.total_annual_cost == pytest.approx(1000.0)
    adopted = [move for move in log if move.status == "adopted"]
    assert [move.shift for move in adopted] == [5.0, 5.0, 5.0]
    # After 195 C no 5 K or 2.5 K move improves, and 1.25 K is the last;
    # the first entry is the unshifted baseline.
    assert log[0].status == "baseline"
    assert sorted({abs(move.shift) for move in log[1:]}) == [1.25, 2.5, 5.0]


def test_moves_are_judged_against_the_unshifted_levels_designed_the_same_way():
    """The sweep's best network can come from a seed a move never builds, so
    it can be cheaper than the same levels designed as a move is. Moves are
    compared with the unshifted levels evaluated as a move; the start's
    network is kept only if no move beats it."""

    def evaluate(case, view):
        return _view(_cost_by_level(case))

    case, view, log = moves.refine_levels(
        _mains_site(),
        _view(1001.0),
        evaluate=evaluate,
        step=5.0,
        step_min=5.0,
        levels=("iso_1",),
    )

    assert log[0].status == "baseline"
    assert log[0].shift == 0.0
    assert log[0].total_annual_cost == pytest.approx(1015.0)
    assert _hot_supply(case, "iso_1") == pytest.approx(195.0)
    assert view.selected_network.total_annual_cost == pytest.approx(1000.0)


def test_the_start_is_kept_when_no_move_beats_it():
    start = _view(990.0)

    case, view, log = moves.refine_levels(
        _mains_site(),
        start,
        evaluate=lambda case, view: _view(_cost_by_level(case)),
        step=5.0,
        step_min=5.0,
        levels=("iso_1",),
    )

    assert view is start
    assert _hot_supply(case, "iso_1") == pytest.approx(180.0)
    assert [move.status for move in log].count("adopted") == 3


def test_a_move_whose_design_fails_is_rejected_and_recorded():
    def evaluate(case, view):
        if _hot_supply(case, "iso_1") > 180.0:
            raise WorkflowContractError("no successful task outcomes")
        return _view(_cost_by_level(case, best=100.0))

    case, view, log = moves.refine_levels(
        _mains_site(),
        _view(_cost_by_level(_mains_site(), best=100.0)),
        evaluate=evaluate,
        step=5.0,
        step_min=5.0,
        levels=("iso_1",),
    )

    failed = [move for move in log if move.status == "failed"]
    assert failed and "no successful task outcomes" in failed[0].message
    assert _hot_supply(case, "iso_1") < 180.0


def test_no_moves_run_with_a_zero_step():
    case = _mains_site()
    start = _view(1.0)

    assert moves.refine_levels(
        case, start, evaluate=pytest.fail, step=0.0, step_min=1.0
    ) == (case, start, ())


def test_the_sweep_refines_the_cheapest_counts_design(monkeypatch):
    from OpenPinch.application import site_levels

    refined = []

    def refine(case, view, *, evaluate, step, step_min, expired=None):
        refined.append((case, step, step_min))
        return "case", _view(10.0), (moves.LevelMove("iso_1", 5.0, 10.0, "adopted"),)

    monkeypatch.setattr(moves, "refine_levels", refine)
    runs = (
        site_levels.LevelCountRun(2, 2, 0, "designed", _view(30.0), 30.0, case="c2"),
        site_levels.LevelCountRun(3, 3, 0, "designed", _view(20.0), 20.0, case="c3"),
    )

    refined_runs, log = moves.refine_selected(
        runs, evaluate=None, step=5.0, step_min=1.0
    )

    assert refined == [("c3", 5.0, 1.0)]
    assert refined_runs[1].total_annual_cost == 10.0
    assert refined_runs[1].case == "case"
    assert refined_runs[0] is runs[0]
    assert log[0].row() == {
        "level": "iso_1",
        "shift": 5.0,
        "total_annual_cost": 10.0,
        "status": "adopted",
        "message": None,
    }


def test_level_steps_are_configuration_keys():
    from OpenPinch.domain.configuration_fields import USER_CONFIG_FIELD_SPECS

    assert USER_CONFIG_FIELD_SPECS["HENS_SITE_LEVEL_STEP"].default == 5.0
    assert USER_CONFIG_FIELD_SPECS["HENS_SITE_LEVEL_STEP_MIN"].default == 1.0
    with pytest.raises(ValueError, match="HENS_SITE_LEVEL_STEP_MIN"):
        _site().design.total_site_heat_exchanger_network(
            options={"HENS_SITE_LEVEL_STEP_MIN": 0.0}
        )


def _recovery(network):
    return sorted(
        (e.source_stream, e.sink_stream)
        for e in network.exchangers
        if e.kind is HeatExchangerKind.RECOVERY
    )


@pytest.mark.synthesis
@pytest.mark.solver
@pytest.mark.skipif(not _solvers_available(), reason="needs highspy and ipopt")
def test_level_moves_never_raise_the_placed_designs_cost():
    problem = _site()
    options = {"HENS_EVM_N_AD_BRANCHES": 1}
    arguments = dict(
        levels="placed",
        level_counts=[2],
        placement=SHORT_PLACEMENT,
        width=0,
        depth=1,
    )

    fixed = _site().design.total_site_heat_exchanger_network(
        **arguments, options=options | {"HENS_SITE_LEVEL_STEP": 0.0}
    )
    moved = problem.design.total_site_heat_exchanger_network(
        **arguments, options=options | {"HENS_SITE_LEVEL_STEP_MIN": 2.5}
    )

    log = moved.result.provenance.effective_settings["HENS_SITE_LEVEL_MOVES"]
    assert log
    start = fixed.selected_network.summary_metrics["site_total_annual_cost"]
    cost = start
    for move in log:
        # A cheaper re-design of the unshifted levels is returned too.
        if move["status"] == "baseline" and move["total_annual_cost"] < cost:
            cost = move["total_annual_cost"]
        if move["status"] == "adopted":
            assert move["total_annual_cost"] < cost
            cost = move["total_annual_cost"]
        if move["status"] == "start kept":
            cost = move["total_annual_cost"]
    returned = moved.selected_network.summary_metrics["site_total_annual_cost"]
    assert returned == pytest.approx(cost)
    assert returned <= start + 1e-6 * start


def test_no_level_position_is_designed_twice():
    """Adopting a move makes the next round's opposite move the levels just
    left. Each move designs from the current network, so the seed differs,
    but the levels were designed already: their cached cost is logged
    ``"skipped"`` and they are not designed again."""
    evaluated = []

    def evaluate(case, view):
        evaluated.append(_hot_supply(case, "iso_1"))
        return _seeded_view(_cost_by_level(case), str(_hot_supply(case, "iso_1")))

    _, _, log = moves.refine_levels(
        _mains_site(),
        _seeded_view(_cost_by_level(_mains_site()), "start"),
        evaluate=evaluate,
        step=5.0,
        step_min=1.0,
        levels=("iso_1",),
    )

    assert len(evaluated) == len(set(evaluated))
    skipped = [move for move in log if move.status == "skipped"]
    assert skipped and all(move.message == "already designed" for move in skipped)
    # From 185 C, the move back to 180 C is skipped at the baseline's cost.
    assert skipped[0].shift == -5.0
    assert skipped[0].total_annual_cost == pytest.approx(1015.0)


def test_the_moves_stop_when_the_time_runs_out():
    evaluated = []

    def evaluate(case, view):
        evaluated.append(_hot_supply(case, "iso_1"))
        return _view(_cost_by_level(case))

    case, view, log = moves.refine_levels(
        _mains_site(),
        _view(_cost_by_level(_mains_site())),
        evaluate=evaluate,
        step=5.0,
        step_min=1.0,
        levels=("iso_1",),
        expired=lambda: len(evaluated) >= 3,
    )

    # The baseline (180 C), then 185 C and 175 C: 185 C is kept.
    assert evaluated == [180.0, 185.0, 175.0]
    assert _hot_supply(case, "iso_1") == pytest.approx(185.0)
    assert view.selected_network.total_annual_cost == pytest.approx(1010.0)
    assert log[-1].status == "stopped"
    assert log[-1].message == "time limit"


def test_no_move_is_designed_when_the_time_has_run_out():
    case = _mains_site()
    start = _view(1.0)

    found, view, log = moves.refine_levels(
        case,
        start,
        evaluate=pytest.fail,
        step=5.0,
        step_min=1.0,
        expired=lambda: True,
    )

    assert (found, view) == (case, start)
    assert [move.status for move in log] == ["stopped"]


def test_the_site_time_limit_is_a_configuration_key():
    from OpenPinch.domain.configuration_fields import USER_CONFIG_FIELD_SPECS

    assert USER_CONFIG_FIELD_SPECS["HENS_SITE_TIME_LIMIT"].default is None
    with pytest.raises(ValueError, match="HENS_SITE_TIME_LIMIT"):
        _site().design.total_site_heat_exchanger_network(
            options={"HENS_SITE_LEVELS": "placed", "HENS_SITE_TIME_LIMIT": 0.0}
        )


def _seeded_view(cost, tag):
    """A view whose selected network seeds a move with recovery ``tag``."""
    recovery = SimpleNamespace(
        kind=HeatExchangerKind.RECOVERY,
        stage=1,
        period_states=(SimpleNamespace(active=True),),
        model_dump_json=lambda: tag,
    )
    return SimpleNamespace(
        selected_network=SimpleNamespace(
            total_annual_cost=cost, summary_metrics={}, exchangers=(recovery,)
        )
    )


def test_a_failed_move_is_not_tried_again_at_the_same_levels():
    evaluated = []

    def evaluate(case, view):
        level = _hot_supply(case, "iso_1")
        evaluated.append(level)
        if level == 185.0:
            raise WorkflowContractError("no successful task outcomes")
        return _view(_cost_by_level(case))

    _, _, log = moves.refine_levels(
        _mains_site(),
        _view(_cost_by_level(_mains_site())),
        evaluate=evaluate,
        step=5.0,
        step_min=2.5,
        levels=("iso_1",),
    )

    # 185 C fails at the 5 K step; after 182.5 C is adopted its +2.5 K move
    # is 185 C again, and is skipped with the failure's message.
    assert evaluated.count(185.0) == 1
    again = [move for move in log if move.status == "skipped" and move.shift == 2.5]
    assert len(again) == 1
    assert again[0].total_annual_cost is None
    assert again[0].message == "already failed: no successful task outcomes"


def test_refused_moves_are_logged_with_the_reason():
    """iso_1 at 180 C moved 80 K up leaves the streams (50 to 220 C), and
    down meets iso_2 at 100 C: neither is designed, and the log says why."""
    _, _, log = moves.refine_levels(
        _mains_site(),
        _view(1.0),
        evaluate=lambda case, view: _view(1.0),
        step=80.0,
        step_min=80.0,
        levels=("iso_1",),
    )

    refused = [move for move in log if move.status == "refused"]
    assert [(move.level, move.shift) for move in refused] == [
        ("iso_1", 80.0),
        ("iso_1", -80.0),
    ]
    assert all(move.total_annual_cost is None for move in refused)
    assert "stream temperatures" in refused[0].message
    assert "iso_2" in refused[1].message


def test_a_cheaper_baseline_is_returned_when_no_move_beats_it():
    """The unshifted levels re-designed can beat the sweep's own network;
    with no better move that re-design is returned, not the dearer start."""
    baseline = _view(80.0)

    def evaluate(case, view):
        if _hot_supply(case, "iso_1") == pytest.approx(180.0):
            return baseline
        return _view(90.0)

    case, view, log = moves.refine_levels(
        _mains_site(),
        _view(100.0),
        evaluate=evaluate,
        step=5.0,
        step_min=5.0,
        levels=("iso_1",),
    )

    assert view is baseline
    assert _hot_supply(case, "iso_1") == pytest.approx(180.0)


def test_the_log_says_when_the_start_is_kept_over_an_adopted_move():
    """A move can beat the baseline re-design but not the sweep's own
    network; the start is then returned, and the log ends saying so rather
    than leaving the move as the last word."""
    start = _view(100.0)

    def evaluate(case, view):
        return _view(120.0 if _hot_supply(case, "iso_1") == 180.0 else 110.0)

    case, view, log = moves.refine_levels(
        _mains_site(),
        start,
        evaluate=evaluate,
        step=5.0,
        step_min=5.0,
        levels=("iso_1",),
    )

    assert view is start
    assert "adopted" in [move.status for move in log]
    assert log[-1].status == "start kept"
    assert log[-1].total_annual_cost == pytest.approx(100.0)
