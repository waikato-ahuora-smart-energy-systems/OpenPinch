"""A shared VC+MVR heat pump prices a period where it is off.

Design-and-operate (review finding 19): where the installed machine cannot
run in a period, the period is priced on its utilities with the machine
off. Off is the period's own design with no duty; a VC+MVR cycle at its
lower bounds is infeasible, not off, and the crude preheat train's shared
VC+MVR target raised for want of a cost breakdown (notebook 10). Its
turndown period needs under a quarter of the peak-sized compressor's
volume (HPR_TURNDOWN_DUTY 0.25), so the machine is off there.
"""

from __future__ import annotations

from OpenPinch import PinchProblem


def _work(output) -> float:
    return float(output["w_net"] if isinstance(output, dict) else output.w_net)


def test_a_shared_vc_mvr_heat_pump_prices_a_period_without_it():
    problem = PinchProblem("crude_preheat_train_multiperiod.json", project_name="x")

    target = problem.target.mvr_heat_pump(
        period_id="base",
        mvr_stages=1,
        load_fraction=0.25,
        condensers=1,
        evaporators=1,
        maximum_restarts=0,
        # Fewer and the base and peak operations break a cycle limit.
        maximum_evaluations=50,
        options={"HPR_MULTIPERIOD_OPTIMIZATION_ENABLED": True},
    )

    outputs = target.hpr_details.period_outputs
    assert _work(outputs["turndown"]) == 0.0
