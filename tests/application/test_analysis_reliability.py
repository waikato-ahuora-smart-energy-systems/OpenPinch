"""Behavioral regressions for isolated, composable analysis execution."""

from copy import deepcopy

import pytest

from OpenPinch import PinchProblem
from tests.application.test_hpr_period_batch_boundaries import _two_period_payload


def problem():
    return PinchProblem(_two_period_payload(), project_name="Site")


def test_mvr_serial_parallel_equivalence():
    p = PinchProblem("OpenPinch/tutorials/sample_cases/process_mvr.json")
    p.components.add_process_mvr("Evaporator vapour", mvr_stage_t_lift=10.0)
    serial = p.target.all_periods.direct_heat_integration(workers=1)
    parallel = p.target.all_periods.direct_heat_integration(workers=2)
    for sid in serial:
        a, b = serial[sid].targets[0], parallel[sid].targets[0]
        assert a.Qr.value == pytest.approx(b.Qr.value)
        assert a.process_component_work_target.value == pytest.approx(
            b.process_component_work_target.value
        )


def test_selected_period_never_relabels_stale_targets():
    p = problem()
    p.target.all_heat_integration(period_id="base")
    p.target.direct_heat_integration(zone="AreaA", period_id="peak")
    assert p.results.targets
    assert all(t.period_id == "peak" and t.period_idx == 1 for t in p.results.targets)


def test_exergy_uses_invocation_environment():
    p = problem()
    p.target.direct_heat_integration(zone="AreaA", period_id="base")
    observed = p.target.exergy(
        zone="AreaA", period_id="base", options={"ENV_TEMPERATURE": 80.0}
    )
    independent = problem()
    independent.target.direct_heat_integration(
        zone="AreaA", period_id="base", options={"ENV_TEMPERATURE": 80.0}
    )
    expected = independent.target.exergy(
        zone="AreaA", period_id="base", options={"ENV_TEMPERATURE": 80.0}
    )
    assert observed.exergy_sources == pytest.approx(expected.exergy_sources)


def test_foreign_zone_is_resolved_in_local_problem():
    local, foreign = problem(), problem()
    selector = foreign.master_zone.get_subzone("AreaA")
    local.target.direct_heat_integration(zone=selector, period_id="base")
    assert local.results.targets
    assert not selector.targets
    assert not foreign.master_zone.get_subzone("AreaA").targets


def test_all_integration_without_subzones_does_not_commit_child_targets():
    p = problem()
    p.target.all_heat_integration(include_subzones=False)
    assert not p.master_zone.get_subzone("AreaA").targets
    assert all(row.scope == "Site" for row in p.results.targets)


@pytest.mark.parametrize("workers", [1, 2])
def test_period_exergy_reuses_prepared_thermal_targets(workers, monkeypatch):
    p = problem()
    p.target.all_periods.direct_heat_integration(zone="AreaA", workers=workers)

    def unexpected(*args, **kwargs):
        pytest.fail("Compatible thermal prerequisites must not be recomputed")

    monkeypatch.setattr(
        "OpenPinch.application.targeting.compute_direct_integration_targets", unexpected
    )
    outputs = p.target.all_periods.exergy(zone="AreaA", workers=workers)
    assert list(outputs) == ["base", "peak"]
    assert all(
        output.targets[0].exergy_sources is not None for output in outputs.values()
    )


def test_failed_targeting_preserves_previous_success(monkeypatch):
    p = problem()
    p.target.direct_heat_integration(zone="AreaA", period_id="base")
    before = p.results.model_dump(mode="json")
    original = deepcopy(p._master_zone)

    def fail(zone, args):
        zone.targets.clear()
        raise RuntimeError("injected failure")

    monkeypatch.setattr(
        "OpenPinch.application._problem.accessors.target.direct_heat_integration_service",
        fail,
    )
    with pytest.raises(RuntimeError, match="injected failure"):
        p.target.direct_heat_integration(zone="AreaA", period_id="peak")
    assert p.results.model_dump(mode="json") == before
    assert list(p._master_zone.get_subzone("AreaA").targets) == list(
        original.get_subzone("AreaA").targets
    )


def test_component_changes_invalidate_period_outputs():
    p = PinchProblem("OpenPinch/tutorials/sample_cases/process_mvr.json")
    component = p.components.add_process_mvr("Evaporator vapour", mvr_stage_t_lift=10.0)
    p.target.all_periods.direct_heat_integration()
    component.deactivate()
    assert not p.period_results


def test_public_observations_are_detached():
    p = problem()
    p.target.direct_heat_integration(zone="AreaA", period_id="base")
    p.results.targets.clear()
    p.master_zone.get_subzone("AreaA").targets.clear()
    assert p.results.targets
    assert p.master_zone.get_subzone("AreaA").targets


def test_brayton_rejects_before_any_state_change():
    p = problem()
    with pytest.raises(NotImplementedError, match="Brayton"):
        p.target.brayton_heat_pump(zone="AreaA")
    assert p.results is None
    assert not p.master_zone.get_subzone("AreaA").targets
