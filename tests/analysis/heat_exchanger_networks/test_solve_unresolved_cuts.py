"""A solve whose last solution still draws a cut is a failed solve.

A model's ``_add_lazy_cuts`` checks each solution and adds constraints it
breaks; the solve loop then solves again. If the loop runs out of solves
while cuts are still being added, the last solution breaks a cut that was
never solved, so it is not accepted.
"""

from __future__ import annotations

from OpenPinch.analysis.heat_exchanger_networks.models._base import (
    execution as base_execution,
)
from tests.analysis.heat_exchanger_networks.test_base_model_helpers import (
    _FakeModel,
    _model,
    _set_success_diagnostics,
    _solve_reporting_fake_options,
)


def test_cuts_still_pending_after_the_last_solve_fail_the_solve(monkeypatch):
    model = _model()
    model.get_post_process = lambda: setattr(model, "post_processed", True)
    _set_success_diagnostics(model)
    monkeypatch.setattr(
        base_execution.backend, "solve_pyomo_model", _solve_reporting_fake_options
    )
    model.m = _FakeModel(solve_status=1, objective=1.0)
    model._add_lazy_cuts = lambda: True

    model.optimise(print_output=False)

    assert model.mSuccess == 0
    assert "cut" in model.solver_run.failure_reason
