"""A shared multiperiod heat pump target runs one installed machine.

Design-and-operate (review finding 19, decided 2026-11-28): a vapour
compression heat pump shared by several periods is one installed machine,
which each period runs within its sizes and turndown. Targets still search
the shared design vector, which knows no duty turndown, so this test fails
until targets use design-and-operate: on the crude preheat train (water,
one condenser and one evaporator, a quarter of the load) the turndown
period runs the shared design (13 to 14 kW of work at small budgets) although
HPR_TURNDOWN_DUTY 0.9 leaves the peak-sized machine nothing it can do there.
"""

from __future__ import annotations

from OpenPinch import PinchProblem


def _work(output) -> float:
    return float(output["w_net"] if isinstance(output, dict) else output.w_net)


def test_a_shared_heat_pump_target_is_off_where_it_cannot_turn_down():
    problem = PinchProblem("crude_preheat_train_multiperiod.json", project_name="x")

    target = problem.target.vapour_compression_heat_pump(
        period_id="base",
        options={
            "HPR_MULTIPERIOD_OPTIMIZATION_ENABLED": True,
            "HPR_TURNDOWN_DUTY": 0.9,
            # The duty axis alone: the turndown period's own design lifts at
            # a ratio of 11, which at the default 0.25 would hold the peak's
            # own design (2.5) below the floor too.
            "HPR_TURNDOWN_PRESSURE_RATIO": 0.0,
        },
        refrigerants=["water"],
        load_fraction=0.25,
        condensers=1,
        evaporators=1,
        maximum_restarts=0,
        # Fewer and the peak's own design is still penalised.
        maximum_evaluations=20,
    )

    outputs = target.hpr_details.period_outputs
    assert _work(outputs["turndown"]) == 0.0
    assert _work(outputs["peak"]) > 0.0
