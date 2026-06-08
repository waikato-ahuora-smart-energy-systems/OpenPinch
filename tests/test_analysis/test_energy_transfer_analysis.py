"""Tests for energy-transfer diagram and surplus/deficit table helpers."""

from types import SimpleNamespace

from OpenPinch.classes.problem_table import ProblemTable
from OpenPinch.lib.enums import GT, PT, TT
from OpenPinch.services.energy_transfer_analysis.energy_transfer_entry import (
    create_energy_transfer_diagram,
    create_heat_surplus_deficit_table,
)


def _sample_problem_table() -> ProblemTable:
    return ProblemTable(
        {
            PT.T: [200.0, 150.0, 100.0],
            PT.CP_NET: [0.0, 1.0, -1.2],
            PT.H_NET: [15.0, 65.0, 5.0],
        }
    )


def test_create_heat_surplus_deficit_table_reports_operation_delta_hnet():
    table = create_heat_surplus_deficit_table(_sample_problem_table())

    assert table == [
        {
            "temperature": 200.0,
            "interval": 0.0,
            "Process": 0.0,
            "total_delta_hnet": 0.0,
            "heat_surplus": 0.0,
            "heat_deficit": 0.0,
        },
        {
            "temperature": 150.0,
            "interval": 1.0,
            "Process": 50.0,
            "total_delta_hnet": 50.0,
            "heat_surplus": 0.0,
            "heat_deficit": 50.0,
        },
        {
            "temperature": 100.0,
            "interval": 2.0,
            "Process": -60.0,
            "total_delta_hnet": -60.0,
            "heat_surplus": 60.0,
            "heat_deficit": 0.0,
        },
    ]


def test_create_energy_transfer_diagram_builds_stacked_operation_profile():
    pt = _sample_problem_table()
    diagram = create_energy_transfer_diagram(pt)

    assert diagram["temperatures"] == [200.0, 150.0, 100.0]
    assert len(diagram["operations"]) == 1
    operation = diagram["operations"][0]
    assert operation["interval_heat"] == [50.0, -60.0]
    assert operation["cascade_heat"] == [15.0, 65.0, 5.0]
    assert operation["stacked_heat"] == [15.0, 65.0, 5.0]


def test_create_energy_transfer_diagram_prefers_saved_gcc_payload():
    fallback_pt = ProblemTable(
        {
            PT.T: [200.0, 100.0],
            PT.CP_NET: [0.0, 0.0],
            PT.H_NET: [999.0, 999.0],
        }
    )
    gcc = ProblemTable(
        {
            PT.T: [200.0, 150.0, 100.0],
            PT.H_NET: [0.0, 25.0, -10.0],
        }
    )
    target = SimpleNamespace(
        zone_name="NestedUnit",
        type=TT.DI.value,
        pt=fallback_pt,
        graphs={GT.GCC.value: gcc},
        hot_utility_target=0.0,
        cold_utility_target=0.0,
    )

    diagram = create_energy_transfer_diagram([target])

    assert diagram["temperatures"] == [200.0, 150.0, 100.0]
    assert diagram["operations"][0]["name"] == "NestedUnit"
    assert diagram["operations"][0]["interval_heat"] == [25.0, -35.0]
    assert diagram["operations"][0]["cascade_heat"] == [0.0, 25.0, -10.0]
