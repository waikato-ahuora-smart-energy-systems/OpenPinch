"""A heat pump is targeted afresh for the design at each multiplier.

A design at dt_cont multiplier ``m`` sees every stream at ``m`` times its
contribution. A heat pump targeted at the problem's own contributions
would then sit at the wrong levels: frozen at 1.0, its streams do not
design at 0.75. So the target is solved on the problem with every
contribution, the heat pump's own included, scaled by ``m``, and its
streams are written into the unscaled problem, where the design scales
them with the rest.
"""

from __future__ import annotations

import pytest

from OpenPinch import PinchProblem
from OpenPinch.application.heat_pump_design import heat_pump_design_case
from tests.application.heat_pump_stream_cases import basic_source

# One start and 100 evaluations: see ``full_load_target``.
SMALL_SEARCH = {"restarts": 1, "evaluations": 100}
ONE_LEVEL = {"condensers": 1, "evaporators": 1}


def test_the_case_for_a_multiplier_targets_at_the_heat_pump_solved_there():
    problem = PinchProblem(source=basic_source(), project_name="basic")

    run = heat_pump_design_case(
        problem, 0.75, cycle="carnot", levels=ONE_LEVEL, search=SMALL_SEARCH
    )

    marked = [
        record
        for record in run.case.to_problem_json()["streams"]
        if record["heat_pump"] is not None
    ]
    # HPR_DT_CONT, unscaled: the design at 0.75 scales it with the rest.
    assert [record["dt_cont"]["value"] for record in marked] == [1.0, 1.0]
    # As the design at 0.75 sees the case: every contribution at 0.75.
    seen = PinchProblem(source=run.case.to_problem_json(), project_name="seen")
    seen.set_dt_cont_multiplier(0.75)
    direct = seen.target.direct_heat_integration()
    assert float(direct.hot_utility_target) == pytest.approx(
        float(run.target.hot_utility_target), abs=0.01
    )
    assert run.multiplier == 0.75
