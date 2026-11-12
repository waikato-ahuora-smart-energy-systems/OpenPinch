"""Evolution candidates start from their parent's solution."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from OpenPinch.analysis.heat_exchanger_networks.models._stagewise import (
    evolution,
)


class _Value:
    """A Pyomo-shaped variable."""

    def __init__(self, value: float = 0.0) -> None:
        self.value = value
        self.fixed = False
        self.lb = None
        self.ub = None

    def is_variable_type(self) -> bool:
        return True

    def set_value(self, value, skip_validation: bool = False) -> None:
        self.value = value


class _RecordingModel:
    events: list = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.Q_r = [[[_Value()]]]
        self.z = [[[_Value()]]]
        self.theta_1 = [[[_Value()]]]
        self.theta_2 = [[[_Value()]]]

    def set_initial_values_for_variables(self, parent) -> None:
        # The parent had the match off: copying it resets the move's values.
        self.Q_r[0][0][0].value = 0.0
        self.z[0][0][0].value = 0
        self.events.append(("warm start", parent))

    def optimise(self, print_output: bool) -> None:
        self.events.append(("solve", self.z[0][0][0].value))


@pytest.mark.parametrize(("kind", "z_after_move"), [("plus", 1), ("minus", 0)])
def test_evolution_candidate_is_warm_started_from_its_parent_before_the_move(
    monkeypatch, kind: str, z_after_move: int
):
    _RecordingModel.events = []
    monkeypatch.setattr(evolution, "_stagewise_model_class", lambda: _RecordingModel)
    monkeypatch.setattr(
        evolution._approach, "_recovery_approach_temperature", lambda *_: 10.0
    )
    owner = SimpleNamespace(name="root", solver="ipopt-pyomo", solver_options=None)
    parent = SimpleNamespace(
        framework="ESM",
        solver_arrays=None,
        stages=1,
        dTmin=10.0,
        min_dqda=0.0,
        minimisation_goal="total cost",
        non_isothermal_model=True,
    )

    evolution._build_and_solve_evolution(
        owner,
        kind,
        print_output=False,
        unit=0,
        prev_case=parent,
        position=(0, 0, 0),
        z_allowed=[[[1]]],
    )

    assert _RecordingModel.events == [("warm start", parent), ("solve", z_after_move)]
