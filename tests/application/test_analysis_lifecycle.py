"""Shrinking state-machine and generated input checks for owned analyses."""

from copy import deepcopy
from unittest.mock import patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from OpenPinch import PinchProblem
from OpenPinch.application._problem.targeting.state import snapshot_problem
from OpenPinch.contracts.output import TargetOutput
from tests.application.test_hpr_period_batch_boundaries import _two_period_payload


class AnalysisLifecycle(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.problem = PinchProblem("process_mvr.json", project_name="Site")
        self.component = self.problem.components.add_process_mvr(
            "Evaporator vapour", mvr_stage_t_lift=10.0
        )

    @rule(workers=st.integers(1, 3))
    def solve_batch(self, workers):
        self.problem.target.all_periods.direct_heat_integration(workers=workers)
        assert set(self.problem._period_states) == set(self.problem.period_ids)

    @rule(active=st.booleans())
    def set_component_active(self, active):
        if active:
            self.component.activate()
        else:
            self.component.deactivate()
        assert not self.problem._period_states
        assert not self.problem.period_results
        assert self.problem.results is None

    @rule(ambient=st.integers(10, 40))
    def configure(self, ambient):
        self.problem.update_options({"ENV_TEMPERATURE": float(ambient)})
        assert not self.problem._period_states
        assert not self.problem.period_results
        # Input rebuilding discards memory-only components; explicitly recreate.
        self.component = self.problem.components.add_process_mvr(
            "Evaporator vapour", mvr_stage_t_lift=10.0
        )

    @rule()
    def snapshot(self):
        clone = snapshot_problem(self.problem)
        cloned_component = next(iter(clone._process_components.values()))
        assert cloned_component.problem is clone
        assert cloned_component is not self.component
        record = cloned_component.stream_records[0]
        assert any(
            stream is record.original_stream
            for zone in clone._walk_zone_tree(clone._master_zone)
            for stream in zone.hot_streams
        )
        assert (
            record.original_stream
            is not self.component.stream_records[0].original_stream
        )
        before = self.problem.to_problem_json()
        clone._master_zone.name = "isolated"
        assert self.problem.to_problem_json() == before
        assert self.problem.master_zone.name == "Site"

    @invariant()
    def observations_and_serialization_are_detached(self):
        before = {
            sid: output.model_dump(mode="json")
            for sid, output in self.problem.period_results.items()
        }
        for sid, output in self.problem.period_results.items():
            assert (
                TargetOutput.model_validate_json(output.model_dump_json()).model_dump(
                    mode="json"
                )
                == before[sid]
            )
            for row in output.targets:
                assert row.period_id == sid
                assert row.provenance.period_ids == (sid,)
            output.targets.clear()
        assert {
            sid: output.model_dump(mode="json")
            for sid, output in self.problem.period_results.items()
        } == before


TestAnalysisLifecycle = AnalysisLifecycle.TestCase
TestAnalysisLifecycle.settings = settings(
    max_examples=8, stateful_step_count=8, deadline=None
)


@settings(max_examples=8, deadline=None)
@given(scale=st.floats(0.5, 2, allow_nan=False), workers=st.integers(2, 4))
def test_generated_streams_have_identical_worker_results(scale, workers):
    payload = _two_period_payload()
    for stream in payload["streams"]:
        stream["heat_flow"]["values"] = [
            value * scale for value in stream["heat_flow"]["values"]
        ]
    problem = PinchProblem(payload, project_name="Site")
    serial = problem.target.all_periods.direct_heat_integration(workers=1)
    parallel = problem.target.all_periods.direct_heat_integration(workers=workers)
    assert {sid: result.model_dump(mode="json") for sid, result in serial.items()} == {
        sid: result.model_dump(mode="json") for sid, result in parallel.items()
    }


@pytest.mark.parametrize("workers", [1, 2])
def test_failed_complete_batch_preserves_every_previous_period(workers):
    problem = PinchProblem(_two_period_payload(), project_name="Site")
    problem.target.all_periods.direct_heat_integration()
    before = deepcopy(problem._period_results)
    states = dict(problem._period_states)
    from OpenPinch.application._problem.accessors import target as module

    original = module.direct_heat_integration_service

    def fail_peak(zone, args):
        if args["period_id"] == "peak":
            zone.targets.clear()
            raise RuntimeError("peak failed")
        return original(zone, args)

    with patch.object(module, "direct_heat_integration_service", fail_peak):
        with pytest.raises(RuntimeError, match="peak failed"):
            problem.target.all_periods.direct_heat_integration(workers=workers)
    assert all(problem._period_states[sid] is state for sid, state in states.items())
    assert {
        sid: output.model_dump(mode="json")
        for sid, output in problem.period_results.items()
    } == {sid: output.model_dump(mode="json") for sid, output in before.items()}


def test_child_targets_are_not_reported_when_traversal_is_disabled():
    problem = PinchProblem(_two_period_payload(), project_name="Site")
    problem.target.all_heat_integration(include_subzones=True)
    child = problem.master_zone.get_subzone("AreaA").targets
    problem.target.indirect_heat_integration(include_subzones=False)
    assert all(row.scope == "Site" for row in problem.results.targets)
    assert list(problem.master_zone.get_subzone("AreaA").targets) == list(child)


def test_incompatible_thermal_settings_require_new_exergy_prerequisite():
    problem = PinchProblem(_two_period_payload(), project_name="Site")
    problem.target.direct_heat_integration(options={"THERMAL_DT_CONT": 30.0})
    before = problem.results.model_dump(mode="json")
    with pytest.raises(RuntimeError, match="existing|compatible"):
        problem.target.exergy()
    assert problem.results.model_dump(mode="json") == before
