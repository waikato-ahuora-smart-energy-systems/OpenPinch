"""Scalar-period and workspace-batch HPR backend propagation contracts."""

from __future__ import annotations

from unittest.mock import patch

from hypothesis import HealthCheck, given, seed, settings
from hypothesis import strategies as st

from OpenPinch import PinchProblem, PinchWorkspace
from OpenPinch.application._problem.accessors.target import _TargetAccessor
from OpenPinch.contracts.output import TargetOutput
from OpenPinch.contracts.reporting import PinchTemp, TargetResults
from tests.analysis.heat_pumps.test_hpr_target_basis import _target
from tests.contracts.test_hpr_target_simulation_record import _record


def _two_period_payload() -> dict:
    return {
        "streams": [
            {
                "zone": "Site/AreaA",
                "name": "HotA",
                "t_supply": {"values": [160.0, 170.0], "unit": "degC"},
                "t_target": {"values": [80.0, 90.0], "unit": "degC"},
                "heat_flow": {"values": [100.0, 120.0], "unit": "kW"},
                "dt_cont": 10.0,
                "htc": 1.0,
            },
            {
                "zone": "Site/AreaA",
                "name": "ColdA",
                "t_supply": {"values": [30.0, 35.0], "unit": "degC"},
                "t_target": {"values": [120.0, 130.0], "unit": "degC"},
                "heat_flow": {"values": [80.0, 90.0], "unit": "kW"},
                "dt_cont": 10.0,
                "htc": 1.0,
            },
        ],
        "utilities": [],
        "zone_tree": {
            "name": "Site",
            "type": "Site",
            "children": [{"name": "AreaA", "type": "Process Zone"}],
        },
        "options": {"PROBLEM_PERIOD_IDS": ["base", "peak"]},
    }


def _report_row(record):
    return TargetResults(
        scope="Site/AreaA",
        zone_type="Process Zone",
        integration_type="Process",
        target_method="Heat Pump",
        period_idx=0 if record.period_id == "base" else 1,
        period_id=record.period_id,
        Qh=0.0,
        Qc=0.0,
        Qr=0.0,
        pinch_temp=PinchTemp(),
        hpr_cycle="Cascade vapour compression cycles",
        hpr_simulation_backend=record.simulation_backend,
        hpr_target_simulation_record=record,
        hpr_success=True,
    )


def test_independent_all_periods_preserve_order_backend_and_one_record_per_call(
    monkeypatch,
) -> None:
    problem = PinchProblem(_two_period_payload(), project_name="Site")
    sessions: list[str] = []

    def fake_target(self, *, period_id=None, simulation_backend="coolprop", **_kwargs):
        sessions.append(period_id)
        record = _record(
            simulation_backend=simulation_backend,
            mode="heat_pump",
            period_id=period_id,
        )
        self._problem._results = TargetOutput(
            name="Site",
            period_id=period_id,
            targets=[_report_row(record)],
        )
        return _target(record)

    monkeypatch.setattr(_TargetAccessor, "vapour_compression_heat_pump", fake_target)

    outputs = problem.target.all_periods.vapour_compression_heat_pump(
        simulation_backend="tespy",
        workers=1,
    )

    assert list(outputs) == ["base", "peak"]
    assert sessions == ["base", "peak"]
    records = [
        outputs[period].targets[0].hpr_target_simulation_record for period in outputs
    ]
    assert [record.period_id for record in records] == ["base", "peak"]
    assert [record.simulation_backend for record in records] == ["tespy", "tespy"]
    assert records[0] is not records[1]


@seed(20260715)
@settings(
    max_examples=4,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(backend=st.sampled_from(("coolprop", "tespy")))
def test_generated_scalar_period_backend_records_are_ordered_and_nonmutating(
    monkeypatch,
    backend: str,
) -> None:
    problem = PinchProblem(_two_period_payload(), project_name="Site")
    original_zone = problem._master_zone
    original_results = problem._results
    original_spec = problem._last_target_run_spec

    def fake_target(self, *, period_id=None, simulation_backend="coolprop", **_kwargs):
        record = _record(
            simulation_backend=simulation_backend,
            mode="heat_pump",
            period_id=period_id,
        )
        self._problem._results = TargetOutput(
            name="Site",
            period_id=period_id,
            targets=[_report_row(record)],
        )
        return _target(record)

    monkeypatch.setattr(_TargetAccessor, "vapour_compression_heat_pump", fake_target)
    outputs = problem.target.all_periods.vapour_compression_heat_pump(
        simulation_backend=backend,
        workers=1,
    )

    assert list(outputs) == ["base", "peak"]
    assert [
        outputs[period].targets[0].hpr_target_simulation_record.simulation_backend
        for period in outputs
    ] == [backend, backend]
    assert problem._master_zone is original_zone
    assert problem._results is original_results
    assert problem._last_target_run_spec is original_spec


@seed(20260715)
@settings(max_examples=6, deadline=None)
@given(order=st.permutations(("baseline", "scenario", "third")))
def test_workspace_batch_preserves_generated_order_backend_and_failure_isolation(
    order,
) -> None:
    workspace = PinchWorkspace(_two_period_payload(), project_name="Site")
    workspace.scenario("scenario", activate=False)
    workspace.scenario("third", activate=False)
    active = workspace.active_case_name
    owners = {id(workspace.case(name)): name for name in workspace.list_cases()}
    seen: list[tuple[str, str]] = []

    def fake_target(self, *, simulation_backend="coolprop", **_kwargs):
        name = owners[id(self._problem)]
        seen.append((name, simulation_backend))
        if name == "scenario":
            raise RuntimeError("isolated case failure")
        return _target(
            _record(
                simulation_backend=simulation_backend,
                mode="heat_pump",
                period_id="base",
            )
        )

    with patch.object(_TargetAccessor, "vapour_compression_heat_pump", fake_target):
        outcome = workspace.cases(order).target.vapour_compression_heat_pump(
            simulation_backend="tespy",
            period_id="base",
        )

    expected_results = tuple(name for name in order if name != "scenario")
    assert tuple(outcome.results) == expected_results
    assert tuple(outcome.errors) == ("scenario",)
    assert all(
        target.hpr_details.target_simulation_record.simulation_backend == "tespy"
        for target in outcome.results.values()
    )
    assert seen == [(name, "tespy") for name in order]
    assert workspace.active_case_name == active
