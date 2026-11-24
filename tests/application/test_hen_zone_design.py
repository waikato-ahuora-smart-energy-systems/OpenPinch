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


def test_a_subzone_of_the_master_zone_designs_on_its_own_streams(synthesised_keys):
    problem = _zonal_site()
    subzone = problem.master_zone.subzones["Hospital[S]"]

    problem.design.heat_exchanger_network(zone=subzone, width=0, depth=0)

    assert synthesised_keys
    assert all(
        keys
        == {key for key in _keys(problem.master_zone) if key.startswith("Hospital[S].")}
        for keys in synthesised_keys
    )


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


def _two_zone_source(root: str) -> dict:
    """A site with no utilities: A runs hotter and colder than B."""

    def quantity(value, unit):
        return {"value": value, "unit": unit}

    def stream(zone, name, supply, target, heat_flow):
        return {
            "zone": f"{root}/{zone}",
            "name": name,
            "t_supply": quantity(supply, "degC"),
            "t_target": quantity(target, "degC"),
            "heat_flow": quantity(heat_flow, "kW"),
            "dt_cont": quantity(5.0, "delta_degC"),
            "htc": quantity(1.0, "kW/m^2/delta_degC"),
        }

    return {
        "streams": [
            stream("A", "H1", 300.0, 100.0, 100.0),
            stream("A", "C1", 20.0, 250.0, 100.0),
            stream("B", "H2", 120.0, 60.0, 50.0),
            stream("B", "C2", 70.0, 110.0, 40.0),
        ],
        "utilities": [],
        "zone_tree": {
            "name": root,
            "type": "Site",
            "children": [
                {"name": "A", "type": "Process Zone"},
                {"name": "B", "type": "Process Zone"},
            ],
        },
    }


def _utility_temperatures(problem: PinchProblem) -> dict[str, tuple[str, str]]:
    zone = problem.master_zone
    return {
        key: (str(stream.supply_temperature), str(stream.target_temperature))
        for key, stream in (*zone.hot_utilities.items(), *zone.cold_utilities.items())
    }


def test_a_zone_case_keeps_the_sites_default_utilities():
    """Defaults are placed from the whole site's temperatures, not the zone's."""
    from OpenPinch.analysis.targeting.utilities import is_default_utility
    from OpenPinch.application.case_derivation import _zone_case

    site = PinchProblem(source=_two_zone_source("Site"))

    case = _zone_case(site, site._resolve_target_zone("Site/B"))

    assert _utility_temperatures(case) == _utility_temperatures(site)
    assert all(
        is_default_utility(stream)
        for _key, stream in (
            *case.master_zone.hot_utilities.items(),
            *case.master_zone.cold_utilities.items(),
        )
    )


def test_a_zone_case_keeps_a_root_name_with_a_slash():
    from OpenPinch.application.case_derivation import _zone_case

    site = PinchProblem(source=_two_zone_source("Site A/2026"))

    case = _zone_case(site, site._resolve_target_zone("Site A/2026/B"))

    assert _keys(case.master_zone) == {
        key for key in _keys(site.master_zone) if key.startswith("B.")
    }
