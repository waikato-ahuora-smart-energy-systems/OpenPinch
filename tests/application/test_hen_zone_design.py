"""Designing a heat exchanger network for one zone of a problem."""

import pytest

from OpenPinch.analysis.heat_exchanger_networks.execution.fake_executor import (
    FakeSynthesisExecutor,
)
from OpenPinch.analysis.heat_exchanger_networks.solver import backend
from OpenPinch.analysis.heat_exchanger_networks.solver.dependencies import (
    MissingSynthesisDependencyError,
    MissingSynthesisSolverError,
)
from OpenPinch.application.problem import PinchProblem
from tests.analysis.heat_exchanger_networks.test_design_workflow import (
    _use_fake_default_executor,
)
from tests.support.paths import REPOSITORY_ROOT

ZONAL_SITE = REPOSITORY_ROOT / "OpenPinch/tutorials/sample_cases/zonal_site.json"


def _available(solver: str) -> bool:
    try:
        backend.require_solver_backend(solver)
    except (MissingSynthesisDependencyError, MissingSynthesisSolverError):
        return False
    return True


def _zonal_site() -> PinchProblem:
    return PinchProblem(str(ZONAL_SITE))


def _keys(zone) -> set[str]:
    return {key for key, _ in zone.hot_streams.items()} | {
        key for key, _ in zone.cold_streams.items()
    }


@pytest.fixture
def synthesised_keys(monkeypatch) -> list[set[str]]:
    """The stream keys of each problem the fake executor synthesises on."""
    _use_fake_default_executor(monkeypatch)
    seen: list[set[str]] = []
    execute = FakeSynthesisExecutor.execute

    def spy(self, tasks, *, problem, **kwargs):
        seen.append(_keys(problem.master_zone))
        return execute(self, tasks, problem=problem, **kwargs)

    monkeypatch.setattr(FakeSynthesisExecutor, "execute", spy)
    return seen


def test_a_zone_design_synthesises_on_that_zones_streams_alone(synthesised_keys):
    problem = _zonal_site()
    hospital = {
        key for key in _keys(problem.master_zone) if key.startswith("Hospital[S].")
    }

    problem.design.heat_exchanger_network(zone="Hospital[S]", width=0, depth=0)

    assert synthesised_keys
    assert all(keys == hospital for keys in synthesised_keys)


@pytest.mark.parametrize("zone", [None, "Site"])
def test_a_root_zone_design_synthesises_on_every_stream(synthesised_keys, zone):
    problem = _zonal_site()

    problem.design.heat_exchanger_network(zone=zone, width=0, depth=0)

    assert synthesised_keys
    assert all(keys == _keys(problem.master_zone) for keys in synthesised_keys)


def test_a_zone_design_leaves_the_problem_unchanged(synthesised_keys):
    problem = _zonal_site()

    problem.design.heat_exchanger_network(zone="Hospital[S]", width=0, depth=0)

    assert problem.results is None


def test_a_zone_design_rejects_an_unknown_zone():
    with pytest.raises(ValueError, match="Bakery"):
        _zonal_site().design.heat_exchanger_network(zone="Bakery", width=0, depth=0)


def test_a_zone_design_rejects_a_problem_with_process_components():
    problem = _zonal_site()
    problem._process_components = {"mvr_1": object()}

    with pytest.raises(ValueError, match="process components"):
        problem.design.heat_exchanger_network(zone="Hospital[S]", width=0, depth=0)


@pytest.mark.solver
@pytest.mark.skipif(
    "not (_available('highs') and _available('ipopt-pyomo'))",
    reason="needs HiGHS and Ipopt",
)
def test_a_zone_network_matches_only_that_zones_streams():
    problem = _zonal_site()

    network = problem.design.heat_exchanger_network(
        zone="Hospital[S]", width=0, depth=0
    ).selected_network

    process_streams = {
        stream
        for exchanger in network.exchangers
        for stream, role in (
            (exchanger.source_stream, exchanger.source_stream_role),
            (exchanger.sink_stream, exchanger.sink_stream_role),
        )
        if role.value == "Process"
    }
    assert process_streams
    assert all(stream.startswith("Hospital[S].") for stream in process_streams)
