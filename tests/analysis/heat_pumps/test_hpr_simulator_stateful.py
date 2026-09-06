"""Stateful properties for the HPR point-simulator lifecycle."""

from __future__ import annotations

import pytest
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, rule

from OpenPinch.analysis.heat_pumps.performance_maps.context import (
    build_hpr_map_generation_context,
)
from OpenPinch.analysis.heat_pumps.performance_maps.points import (
    iter_hpr_operating_points,
)
from tests.analysis.heat_pumps.hpr_map_fakes import FakeHprPointSimulator
from tests.analysis.heat_pumps.test_hpr_map_generation import _basis, _request


class HprSimulatorLifecycleMachine(RuleBasedStateMachine):
    """Exercise valid and invalid commands against one generated session."""

    @initialize()
    def initialize_session(self):
        self.simulator = FakeHprPointSimulator()
        self.context = build_hpr_map_generation_context(_basis(), _request())
        self.point = next(iter_hpr_operating_points(self.context))
        self.model_state = "fresh"

    @rule()
    def prepare_or_reject(self):
        if self.model_state == "fresh":
            metadata = self.simulator.prepare(self.context)
            assert metadata.design_converged is True
            self.model_state = "prepared"
        else:
            with pytest.raises(RuntimeError):
                self.simulator.prepare(self.context)

    @rule()
    def simulate_or_reject(self):
        if self.model_state == "prepared":
            result = self.simulator.simulate(self.point)
            assert result.converged is True
        else:
            with pytest.raises(RuntimeError):
                self.simulator.simulate(self.point)

    @rule()
    def close_or_reject(self):
        if self.model_state != "closed":
            self.simulator.close()
            self.model_state = "closed"
        else:
            with pytest.raises(RuntimeError):
                self.simulator.close()

    @invariant()
    def lifecycle_event_counts_are_bounded(self):
        assert self.simulator.events.count(("prepare", None)) <= 1
        assert self.simulator.events.count(("close", None)) <= 1
        if self.model_state == "closed":
            assert self.simulator.events[-1] == ("close", None)


TestHprSimulatorLifecycle = HprSimulatorLifecycleMachine.TestCase
