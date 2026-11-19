"""Level holds for a pooled site design in the stage-wise models (TDM and EVM).

``design.heat_exchanger_network`` on a site of several processes pools their
streams. Its pinch design places utility by temperature and holds each level
at its target; the TDM and EVM tasks built on it hold the same levels over
all the pooled streams, so they cannot trade a level for a cheaper one.
"""

from __future__ import annotations

import pytest

from OpenPinch.analysis.heat_exchanger_networks.heuristic.enumeration import (
    LOAD_FRACTION_TOLERANCE,
)
from OpenPinch.analysis.heat_exchanger_networks.models._stagewise.level_targets import (
    UtilityLevelTarget,
)
from OpenPinch.analysis.heat_exchanger_networks.models.problem import (
    InternalHeatExchangerNetworkProblem,
)
from OpenPinch.analysis.heat_exchanger_networks.solver.arrays import (
    problem_to_solver_arrays,
)
from OpenPinch.application.problem import PinchProblem
from OpenPinch.contracts.synthesis.task import HeatExchangerNetworkSynthesisTask
from tests.analysis.heat_exchanger_networks.test_staged_utilities import (
    D_TMIN,
    _highs_available,
    _hot_utility_duties,
    _utility,
)
from tests.analysis.heat_exchanger_networks.test_stagewise_site_holds import (
    _stream,
    _two_heated_plants,
)


def _plants(zones: tuple[str, str], options: dict | None = None) -> PinchProblem:
    """Two heated streams on LPS or HPS, in one process zone or two."""
    a, b = zones
    streams = [
        _stream(a, "C1", 100.0, 130.0, 30.0),
        _stream(b, "C2", 100.0, 130.0, 30.0),
        _stream(a, "H1", 200.0, 190.0, 10.0),
        _stream(b, "H2", 200.0, 190.0, 10.0),
    ]
    return PinchProblem(
        source={
            "streams": streams,
            "utilities": [
                _utility("LPS", "Hot", 140.0, 139.9, 60.0),
                _utility("HPS", "Hot", 250.0, 249.9, 40.0),
                _utility("CW", "Cold", 20.0, 25.0, 5.0),
            ],
            "zone_tree": {
                "name": "Site",
                "type": "Site",
                "children": [
                    {"name": zone, "type": "Process Zone"}
                    for zone in sorted(set(zones))
                ],
            },
            "options": options or {},
        }
    )


pytestmark = pytest.mark.skipif(
    not _highs_available(), reason="needs highspy and pyomo"
)

# LPS is priced above HPS, so by price no stream takes LPS.
LEVELS = {"Hot Utility.LPS": [60.0], "Hot Utility.HPS": [0.0]}


def _pooled(*, hold: bool) -> tuple[InternalHeatExchangerNetworkProblem, list]:
    problem = _two_heated_plants()
    arrays = problem_to_solver_arrays(problem, D_TMIN)
    internal = InternalHeatExchangerNetworkProblem(
        solver_arrays=arrays,
        name="pooled",
        framework="ESM",
        solver="highs",
        dTmin=D_TMIN,
        stages=2,
        minimisation_goal="utility costs",
        integers=True,
        tol=1e-6,
        staged_utilities=True,
        hold_utility_levels=hold,
    )
    # As inherited from a parent pinch design that held the levels.
    internal.utility_level_targets = LEVELS
    return internal, list(arrays.stream_identities["cold_process_streams"])


def test_a_pooled_site_task_holds_its_recorded_levels_over_every_stream():
    internal, cold = _pooled(hold=True)

    internal.load_model()
    case = internal.case

    every = tuple(range(len(cold)))
    # The band is the heuristic's: its networks hold a level within a load
    # fraction of 1e-3, and evolution starts from them.
    assert case.level_targets == (
        UtilityLevelTarget("hot", (0,), every, (60.0,), LOAD_FRACTION_TOLERANCE),
        UtilityLevelTarget("hot", (1,), every, (0.0,), LOAD_FRACTION_TOLERANCE),
    )
    # Pooled: any process stream may still match any other.
    assert set(case.stream_groups[0]) == {""}
    assert set(case.stream_groups[1]) == {""}
    case.optimise(False)
    assert case.mSuccess == 1
    duties = _hot_utility_duties(case)
    # By price the dearer level sits at its band's lower edge.
    assert duties["LPS"] == pytest.approx(60.0 * (1 - LOAD_FRACTION_TOLERANCE))
    assert duties["HPS"] <= LOAD_FRACTION_TOLERANCE + 1e-9


def test_without_the_hold_a_task_keeps_its_cost_based_levels():
    internal, _ = _pooled(hold=False)

    internal.load_model()
    internal.case.optimise(False)

    assert internal.case.level_targets == ()
    assert _hot_utility_duties(internal.case)["LPS"] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize(
    ("method", "site", "placement", "held"),
    [
        ("network_evolution_method", True, None, True),
        ("thermal_derivative_method", True, None, True),
        ("network_evolution_method", True, "price", False),
        ("network_evolution_method", False, None, False),
        # A single process keeps cost-based TDM and EVM even by temperature.
        ("network_evolution_method", False, "temperature", False),
    ],
)
def test_the_executor_holds_levels_in_tdm_and_evm_of_a_pooled_site(
    method, site, placement, held
):
    from OpenPinch.analysis.heat_exchanger_networks.execution.executor import (
        LocalSynthesisExecutor,
    )

    options = {} if placement is None else {"HENS_UTILITY_PLACEMENT": placement}
    problem = _plants(("A", "B") if site else ("A", "A"), options)
    task = HeatExchangerNetworkSynthesisTask(
        run_id="pooled-holds",
        method=method,
        dt_cont_multiplier=1.0,
        stage_count=2,
    )

    built = LocalSynthesisExecutor(print_output=False)._build_problem(
        task, problem=problem, parent_outcomes={}
    )

    assert built.hold_utility_levels is held
