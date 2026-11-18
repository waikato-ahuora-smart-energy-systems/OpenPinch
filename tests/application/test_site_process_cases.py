"""Per-process cases derived from a Total Site case."""

import pytest

from OpenPinch.application.case_derivation import process_case, site_processes
from OpenPinch.application.problem import PinchProblem
from tests.support.paths import REPOSITORY_ROOT

ZONAL_SITE = REPOSITORY_ROOT / "OpenPinch/tutorials/sample_cases/zonal_site.json"


def _zonal_site() -> PinchProblem:
    return PinchProblem(str(ZONAL_SITE))


def _stream(zone: str, name: str, t_supply: float, t_target: float) -> dict:
    return {
        "zone": zone,
        "name": name,
        "t_supply": t_supply,
        "t_target": t_target,
        "heat_flow": 100.0,
        "dt_cont": 5.0,
        "htc": 1.0,
    }


def _site(children: list[dict], streams: list[dict]) -> PinchProblem:
    return PinchProblem(
        {
            "streams": streams,
            "utilities": [
                {"name": "HU", "type": "Hot", "t_supply": 250.0, "t_target": 249.0},
                {"name": "CU", "type": "Cold", "t_supply": 10.0, "t_target": 11.0},
            ],
            "zone_tree": {"name": "Site", "type": "Site", "children": children},
        }
    )


def _keys(collection) -> set[str]:
    return {key for key, _ in collection.items()}


def test_site_processes_lists_each_process_zone_with_its_streams():
    processes = site_processes(_zonal_site())

    assert [process.name for process in processes] == [
        "Chemical Plant[S]",
        "Food Plant[S]",
        "Hospital[S]",
        "Residential[S]",
    ]
    assert [process.address for process in processes] == [
        f"Site/{process.name}" for process in processes
    ]
    assert [
        (len(process.hot_streams), len(process.cold_streams)) for process in processes
    ] == [(3, 3), (3, 4), (2, 9), (0, 3)]
    assert [process.one_sided for process in processes] == [False, False, False, True]


def test_site_process_stream_keys_are_the_site_zone_keys():
    problem = _zonal_site()
    hospital = site_processes(problem)[2]

    site = problem.master_zone
    assert set(hospital.hot_streams) == {
        key for key in _keys(site.hot_streams) if key.startswith("Hospital[S].")
    }
    assert set(hospital.cold_streams) == {
        key for key in _keys(site.cold_streams) if key.startswith("Hospital[S].")
    }


def test_site_processes_requires_a_site_zone():
    with pytest.raises(ValueError, match="'Site/Hospital\\[S\\]' is a Process Zone"):
        site_processes(_zonal_site(), "Hospital[S]")


def test_site_processes_rejects_a_site_subzone_that_is_not_a_process_zone():
    problem = _site(
        [
            {"name": "A", "type": "Process Zone"},
            {"name": "Boilers", "type": "Utility Zone"},
        ],
        [
            _stream("Site/A", "H1", 150.0, 60.0),
            _stream("Site/A", "C1", 40.0, 120.0),
        ],
    )

    with pytest.raises(ValueError, match="'Site/Boilers' is a Utility Zone"):
        site_processes(problem)


def test_a_stream_placed_on_the_site_zone_is_a_process_of_its_own():
    problem = _site(
        [{"name": "A", "type": "Process Zone"}],
        [
            _stream("Site/A", "H1", 150.0, 60.0),
            _stream("Site", "C1", 40.0, 120.0),
        ],
    )

    processes = site_processes(problem)

    assert [(process.address, process.cold_streams) for process in processes] == [
        ("Site/A", ()),
        ("Site/C1", ("C1.C1",)),
    ]


def test_a_process_case_holds_that_process_alone_with_the_site_utilities():
    problem = _zonal_site()
    hospital = site_processes(problem)[2]

    case = process_case(problem, "Hospital[S]")

    root = case.master_zone
    assert [zone.address for zone in root.subzones.values()] == ["Site/Hospital[S]"]
    assert _keys(root.hot_streams) == set(hospital.hot_streams)
    assert _keys(root.cold_streams) == set(hospital.cold_streams)
    assert [record["name"] for record in case.to_problem_json()["utilities"]] == [
        record["name"] for record in problem.to_problem_json()["utilities"]
    ]
    assert list(case.period_ids) == list(problem.period_ids)


def test_a_process_case_has_the_site_process_targets():
    problem = _zonal_site()
    expected = problem.target.direct_heat_integration(zone="Hospital[S]")

    case = process_case(problem, "Hospital[S]")
    actual = case.target.direct_heat_integration(zone="Hospital[S]")

    assert actual.hot_utility_target == pytest.approx(
        expected.hot_utility_target, rel=1e-9
    )
    assert actual.cold_utility_target == pytest.approx(
        expected.cold_utility_target, rel=1e-9
    )
    assert actual.heat_recovery_target == pytest.approx(
        expected.heat_recovery_target, rel=1e-9
    )


@pytest.mark.parametrize("zone", ["Site", "Hospital[S]/O1"])
def test_a_process_case_requires_a_process_zone(zone):
    with pytest.raises(ValueError, match="Process Zone"):
        process_case(_zonal_site(), zone)


def test_a_process_case_rejects_a_problem_with_process_components():
    problem = _zonal_site()
    problem._process_components = {"mvr_1": object()}

    with pytest.raises(ValueError, match="process components"):
        process_case(problem, "Hospital[S]")
